// NNUE incremental para la evaluacion hibrida.
//
// La primera capa usa features binarias dispersas. El acumulador guarda
// bias + suma de las columnas activas y, al avanzar una posicion, solo suma o
// resta las features que cambiaron. Los pesos actuales conservan el formato
// previo 770 -> 256 -> 32 -> 1, por lo que siguen siendo compatibles.

use crate::board::Board;
use crate::types::{ALL_PIECE_TYPES, Color};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

// N_ENTRADA=5378: 770 base (igual que antes) + 4608 de amenazas (quien
// ataca a cual pieza, ver features_threat.py: 2 colores x 6 tipos
// atacantes x 6 tipos victima x 64 casillas). Candidato experimental --
// sin actualizacion incremental de verdad para las features de amenaza
// (cambian aunque la pieza no se mueva, ej. al abrir una diagonal), asi
// que el acumulador se recalcula COMPLETO en cada jugada en vez de solo
// sumar/restar lo que cambio. Mas lento que el 770 incremental, pero
// correcto por construccion -- optimizarlo queda para despues si esta
// arquitectura demuestra que vale la pena en h2h.
pub const N_ENTRADA: usize = 5378;
const N_OCULTA1: usize = 512;
const N_OCULTA2: usize = 32;

// Cuantizacion de la capa de entrada (W1/b1): esta es la capa que se
// recalcula en CADA nodo de la busqueda (sumar_feature, llamada una vez por
// pieza que cambia al hacer/deshacer una jugada dentro del arbol) -- el
// camino realmente caliente, a diferencia de w2/w3 que solo se usan una vez
// por evaluacion completa (salida()). Guardar W1 en i16 en vez de f32
// permite SIMD entero (8 valores por registro NEON de 128 bits en vez de 4
// con f32) para esa parte especifica. QA=1024 da resolucion de ~0.001 por
// unidad y deja margen de sobra para los pesos reales entrenados (magnitud
// maxima observada ~19, 19*1024=19456, muy por debajo del limite i16 de
// 32767) -- se satura de todos modos por seguridad ante un archivo de pesos
// distinto con valores mas grandes.
const QA: f32 = 1024.0;

// Cuantizacion de la capa 2 (W2/b2, 32x256): perfilado antes confirmo que
// ESTA es la funcion que de verdad consume ~80% del tiempo con NNUE activa
// (se ejecuta completa una vez por CADA evaluacion, a diferencia de la capa
// de entrada que solo actualiza 1-2 columnas por jugada). h1 (salida de la
// capa de entrada, post-ReLU) se cuantiza directo a i8 con QH=2 -- rango
// real observado con datos reales: maximo ~49, holgado bajo el limite de
// 127/QH=63.5. W2 se cuantiza a i8 con QW2=16 -- maximo real observado
// ~7.3, 7.3*16=~117, tambien holgado bajo 127. QA/QH=512=2^9 es potencia de
// 2 exacta, asi que pasar del acumulador (escala QA) a h1 (escala QH) es un
// simple shift a la derecha, sin nada de coma flotante en el camino
// caliente.
const QH: f32 = 2.0;
const QW2: f32 = 16.0;
const QA_SOBRE_QH_SHIFT: u32 = 9; // 1024/2 = 512 = 2^9

fn checksum_fnv1a(datos: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in datos {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub struct RedNeural {
    // Columna j = contribucion de la feature j a las 256 neuronas de la
    // primera capa. Esta disposicion permite actualizar el acumulador con un
    // bloque contiguo al hacer una jugada. Cuantizada a i16 (ver QA arriba);
    // w2/w3/b2/b3 se quedan en f32 (capa chica, se ejecuta una sola vez por
    // evaluacion, no vale la pena el riesgo de cuantizarla tambien).
    w1_col: Vec<i16>,
    b1: Vec<i32>,
    // W2 cuantizado a i8 (fila = neurona de salida, igual que el formato de
    // archivo original) y b2 pre-escalado a la MISMA escala que el producto
    // punto entero (QH*QW2), para poder sumarlos directo en i32 sin
    // reconvertir. w3/b3 se quedan en f32 (capa de 32->1, trivial).
    w2_i8: Vec<i8>,
    b2_i32: Vec<i32>,
    w3: Vec<f32>,
    b3: f32,
}

/// Producto punto acumulado i8*i8 -> i32 con la instruccion `sdot` (extension
/// ARM dotprod). El intrinseco `vdotq_s32` esta detras de una feature
/// inestable de `std::arch`, asi que se emite la instruccion por ensamblador
/// inline -- disponible en Rust estable. `sdot Vd.4s, Vn.16b, Vm.16b`:
/// para cada uno de los 4 carriles i32 de Vd, suma 4 productos de bytes con
/// signo de Vn/Vm y lo ACUMULA sobre Vd. 16 elementos i8 por instruccion,
/// acumulando directo en i32: exacto (sin redondeo), asi que la evaluacion es
/// bit-identica a vmull_s8 + vpadalq_s16. dotprod esta siempre presente en
/// Apple Silicon (hw.optional.arm.FEAT_DotProd = 1).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_s32(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let out;
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc => out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// Lote de features pendientes de aplicar al acumulador.
///
/// Antes, cada feature que cambiaba al hacer una jugada ejecutaba su propia
/// pasada NEON sobre las 512 posiciones del acumulador: 512 lecturas + 512
/// escrituras del acumulador POR FEATURE. Con 20-40 features cambiando por
/// jugada eso es ~100-200 KB de trafico de memoria por nodo, y el bucle
/// quedaba limitado por operaciones de memoria (5 accesos por cada 8 valores:
/// 1 de pesos, 2 del acumulador, 2 de escritura), no por aritmetica.
///
/// Ahora se recolectan primero TODOS los indices que hay que sumar y restar,
/// y al final se hace UNA sola pasada: el acumulador se lee y escribe una vez
/// (en registros NEON, por bloques de 32 valores) y de cada feature solo se
/// leen sus pesos. La suma entera es asociativa y conmutativa, asi que el
/// resultado es BIT A BIT IDENTICO al de antes, sin importar el orden.
// 128 basta con enorme margen: medido sobre ~750 mil actualizaciones reales de
// busqueda, la media es 12 features por jugada y el maximo observado 36. Si
// alguna posicion patologica lo llenara, `empujar` vacia el lote y sigue (dos
// pasadas dan el mismo resultado exacto), asi que el tope es solo de memoria,
// nunca de correccion.
const LOTE_CAP: usize = 128;

struct Lote {
    add: [std::mem::MaybeUninit<u16>; LOTE_CAP],
    n_add: usize,
    sub: [std::mem::MaybeUninit<u16>; LOTE_CAP],
    n_sub: usize,
}

impl Lote {
    #[inline]
    fn nuevo() -> Lote {
        Lote {
            // Array de MaybeUninit: no se inicializa nada (seria 1 KB de
            // escrituras inutiles por nodo); solo se leen las posiciones que
            // de verdad se escribieron, contadas por n_add/n_sub.
            add: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            n_add: 0,
            sub: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            n_sub: 0,
        }
    }
}

impl RedNeural {
    /// Encola una feature. Si el lote se lleno, lo aplica y sigue: partir el
    /// lote en dos pasadas da el mismo resultado exacto (sumas enteras).
    #[inline]
    fn empujar(
        &self,
        acumulador: &mut [i32; N_OCULTA1],
        lote: &mut Lote,
        feature: usize,
        sumar: bool,
    ) {
        debug_assert!(feature < N_ENTRADA);
        if lote.n_add == LOTE_CAP || lote.n_sub == LOTE_CAP {
            self.aplicar_lote(acumulador, lote);
        }
        if sumar {
            lote.add[lote.n_add].write(feature as u16);
            lote.n_add += 1;
        } else {
            lote.sub[lote.n_sub].write(feature as u16);
            lote.n_sub += 1;
        }
    }

    /// Aplica el lote completo en una sola pasada por el acumulador.
    fn aplicar_lote(&self, acumulador: &mut [i32; N_OCULTA1], lote: &mut Lote) {
        if lote.n_add == 0 && lote.n_sub == 0 {
            return;
        }
        let adds: &[u16] =
            unsafe { std::slice::from_raw_parts(lote.add.as_ptr() as *const u16, lote.n_add) };
        let subs: &[u16] =
            unsafe { std::slice::from_raw_parts(lote.sub.as_ptr() as *const u16, lote.n_sub) };

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // Bloque de 32 valores del acumulador = 8 registros NEON de i32.
            // Deja registros libres para los pesos y para el desenrollado; el
            // acumulador entero (512 i32 = 2 KB) no cabe en los 32 registros
            // disponibles, de ahi el recorrido por bloques.
            const TILE: usize = 32;
            let w1 = self.w1_col.as_ptr();
            let mut t = 0;
            while t < N_OCULTA1 {
                let p = acumulador.as_mut_ptr().add(t);
                let mut r0 = vld1q_s32(p);
                let mut r1 = vld1q_s32(p.add(4));
                let mut r2 = vld1q_s32(p.add(8));
                let mut r3 = vld1q_s32(p.add(12));
                let mut r4 = vld1q_s32(p.add(16));
                let mut r5 = vld1q_s32(p.add(20));
                let mut r6 = vld1q_s32(p.add(24));
                let mut r7 = vld1q_s32(p.add(28));
                for &f in adds {
                    let w = w1.add(f as usize * N_OCULTA1 + t);
                    let w0 = vld1q_s16(w);
                    let w1b = vld1q_s16(w.add(8));
                    let w2 = vld1q_s16(w.add(16));
                    let w3 = vld1q_s16(w.add(24));
                    // saddw/saddw2: ensancha i16 -> i32 Y suma en una sola
                    // instruccion.
                    r0 = vaddw_s16(r0, vget_low_s16(w0));
                    r1 = vaddw_high_s16(r1, w0);
                    r2 = vaddw_s16(r2, vget_low_s16(w1b));
                    r3 = vaddw_high_s16(r3, w1b);
                    r4 = vaddw_s16(r4, vget_low_s16(w2));
                    r5 = vaddw_high_s16(r5, w2);
                    r6 = vaddw_s16(r6, vget_low_s16(w3));
                    r7 = vaddw_high_s16(r7, w3);
                }
                for &f in subs {
                    let w = w1.add(f as usize * N_OCULTA1 + t);
                    let w0 = vld1q_s16(w);
                    let w1b = vld1q_s16(w.add(8));
                    let w2 = vld1q_s16(w.add(16));
                    let w3 = vld1q_s16(w.add(24));
                    r0 = vsubw_s16(r0, vget_low_s16(w0));
                    r1 = vsubw_high_s16(r1, w0);
                    r2 = vsubw_s16(r2, vget_low_s16(w1b));
                    r3 = vsubw_high_s16(r3, w1b);
                    r4 = vsubw_s16(r4, vget_low_s16(w2));
                    r5 = vsubw_high_s16(r5, w2);
                    r6 = vsubw_s16(r6, vget_low_s16(w3));
                    r7 = vsubw_high_s16(r7, w3);
                }
                vst1q_s32(p, r0);
                vst1q_s32(p.add(4), r1);
                vst1q_s32(p.add(8), r2);
                vst1q_s32(p.add(12), r3);
                vst1q_s32(p.add(16), r4);
                vst1q_s32(p.add(20), r5);
                vst1q_s32(p.add(24), r6);
                vst1q_s32(p.add(28), r7);
                t += TILE;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for &f in adds {
                let col = &self.w1_col[f as usize * N_OCULTA1..(f as usize + 1) * N_OCULTA1];
                for (v, &peso) in acumulador.iter_mut().zip(col) {
                    *v += peso as i32;
                }
            }
            for &f in subs {
                let col = &self.w1_col[f as usize * N_OCULTA1..(f as usize + 1) * N_OCULTA1];
                for (v, &peso) in acumulador.iter_mut().zip(col) {
                    *v -= peso as i32;
                }
            }
        }
        lote.n_add = 0;
        lote.n_sub = 0;
    }
}

impl RedNeural {
    fn cargar_de_bytes(datos: &[u8]) -> Option<RedNeural> {
        let esperado =
            (N_OCULTA1 * N_ENTRADA + N_OCULTA1 + N_OCULTA2 * N_OCULTA1 + N_OCULTA2 + N_OCULTA2 + 1)
                * 4;
        if datos.len() != esperado {
            eprintln!(
                "info string NNUE: tamano de archivo inesperado ({} bytes, se esperaban {})",
                datos.len(),
                esperado
            );
            return None;
        }
        let mut cursor = 0usize;
        let leer_f32_vec = |n: usize, cursor: &mut usize| -> Vec<f32> {
            let mut valores = Vec::with_capacity(n);
            for _ in 0..n {
                let bytes: [u8; 4] = datos[*cursor..*cursor + 4].try_into().unwrap();
                valores.push(f32::from_le_bytes(bytes));
                *cursor += 4;
            }
            valores
        };
        let w1_fila = leer_f32_vec(N_OCULTA1 * N_ENTRADA, &mut cursor);
        let b1 = leer_f32_vec(N_OCULTA1, &mut cursor);
        let w2 = leer_f32_vec(N_OCULTA2 * N_OCULTA1, &mut cursor);
        let b2 = leer_f32_vec(N_OCULTA2, &mut cursor);
        let w3 = leer_f32_vec(N_OCULTA2, &mut cursor);
        let b3 = leer_f32_vec(1, &mut cursor)[0];

        let todos_finitos = w1_fila
            .iter()
            .chain(b1.iter())
            .chain(w2.iter())
            .chain(b2.iter())
            .chain(w3.iter())
            .chain(std::iter::once(&b3))
            .all(|v| v.is_finite() && v.abs() <= 1.0e6);
        if !todos_finitos {
            eprintln!("info string NNUE: pesos invalidos (NaN, infinito o magnitud absurda)");
            return None;
        }

        let cuantizar =
            |v: f32| -> i16 { (v * QA).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16 };

        let mut w1_col = vec![0i16; N_ENTRADA * N_OCULTA1];
        for fila in 0..N_OCULTA1 {
            for columna in 0..N_ENTRADA {
                w1_col[columna * N_OCULTA1 + fila] = cuantizar(w1_fila[fila * N_ENTRADA + columna]);
            }
        }
        let b1_i32: Vec<i32> = b1.iter().map(|&v| (v * QA).round() as i32).collect();

        let cuantizar_i8 =
            |v: f32| -> i8 { (v * QW2).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8 };
        let w2_i8: Vec<i8> = w2.iter().map(|&v| cuantizar_i8(v)).collect();
        let escala_combinada = QH * QW2;
        let b2_i32: Vec<i32> = b2
            .iter()
            .map(|&v| (v * escala_combinada).round() as i32)
            .collect();

        Some(RedNeural {
            w1_col,
            b1: b1_i32,
            w2_i8,
            b2_i32,
            w3,
            b3,
        })
    }

    fn sumar_feature(&self, acumulador: &mut [i32; N_OCULTA1], feature: usize, sumar: bool) {
        let columna = &self.w1_col[feature * N_OCULTA1..(feature + 1) * N_OCULTA1];
        // Entero (i16 pesos, acumulador i32): 8 valores por registro NEON de
        // 128 bits en vez de 4 con f32 -- el "ancho de banda" real de la
        // parte que se ejecuta en CADA nodo de la busqueda. Suma exacta (sin
        // redondeo intermedio), asi que incremental == recalculo desde cero
        // siempre, igual que antes con +-1.0 en f32.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let mut i = 0;
            while i < N_OCULTA1 {
                let w16 = vld1q_s16(columna.as_ptr().add(i));
                let w_lo = vmovl_s16(vget_low_s16(w16));
                let w_hi = vmovl_s16(vget_high_s16(w16));
                let a_lo = vld1q_s32(acumulador.as_ptr().add(i));
                let a_hi = vld1q_s32(acumulador.as_ptr().add(i + 4));
                if sumar {
                    vst1q_s32(acumulador.as_mut_ptr().add(i), vaddq_s32(a_lo, w_lo));
                    vst1q_s32(acumulador.as_mut_ptr().add(i + 4), vaddq_s32(a_hi, w_hi));
                } else {
                    vst1q_s32(acumulador.as_mut_ptr().add(i), vsubq_s32(a_lo, w_lo));
                    vst1q_s32(acumulador.as_mut_ptr().add(i + 4), vsubq_s32(a_hi, w_hi));
                }
                i += 8;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (valor, &peso) in acumulador.iter_mut().zip(columna) {
            *valor += if sumar { peso as i32 } else { -(peso as i32) };
        }
    }

    fn salida(&self, acumulador: &[i32; N_OCULTA1]) -> f32 {
        // Perfilado con `sample`: esta funcion era ~80% del tiempo de busqueda
        // con NNUE activada -- se ejecuta COMPLETA una vez por cada
        // evaluacion (a diferencia de sumar_feature, que solo actualiza 1-2
        // columnas por jugada). Camino cuantizado: h1 pasa de acumulador i32
        // (escala QA) a i8 (escala QH) con un simple shift entero (ReLU +
        // truncar a 127), W2 ya viene cuantizado a i8 (escala QW2) desde la
        // carga. El producto punto de 256 terminos entero i8*i8 -> i32 usa
        // multiplicacion ensanchada NEON (vmull_s8, 8 carriles i8 por
        // instruccion) acumulada en i32 (vpadalq_s16), sin overflow posible
        // (max 127*127*256 =~4.1M, muy por debajo del limite i32). w3/b3
        // (32->1) se quedan en f32, capa trivial, no vale la pena cuantizarla.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // ReLU + cuantizar a i8 directo desde el acumulador entero, sin
            // pasar por f32 en ningun momento de este paso.
            //
            // Vectorizado: antes era un bucle ESCALAR de 512 iteraciones por
            // cada evaluacion (max, shift, min, truncar a i8, uno por uno).
            // Con NEON son 32 iteraciones de 16 valores. El estrechamiento
            // saturante (vqmovn) hace exactamente lo mismo que el `.min(127)`
            // del original: tras `max(0)` el valor es >= 0, asi que saturar a
            // i16 y luego a i8 da 127 cuando se pasa y el valor exacto cuando
            // no. Resultado BIT A BIT IDENTICO.
            let mut h1 = [0i8; N_OCULTA1];
            {
                let cero = vdupq_n_s32(0);
                let src = acumulador.as_ptr();
                let dst = h1.as_mut_ptr();
                let mut i = 0;
                while i < N_OCULTA1 {
                    let a0 = vshrq_n_s32::<{ QA_SOBRE_QH_SHIFT as i32 }>(vmaxq_s32(
                        vld1q_s32(src.add(i)),
                        cero,
                    ));
                    let a1 = vshrq_n_s32::<{ QA_SOBRE_QH_SHIFT as i32 }>(vmaxq_s32(
                        vld1q_s32(src.add(i + 4)),
                        cero,
                    ));
                    let a2 = vshrq_n_s32::<{ QA_SOBRE_QH_SHIFT as i32 }>(vmaxq_s32(
                        vld1q_s32(src.add(i + 8)),
                        cero,
                    ));
                    let a3 = vshrq_n_s32::<{ QA_SOBRE_QH_SHIFT as i32 }>(vmaxq_s32(
                        vld1q_s32(src.add(i + 12)),
                        cero,
                    ));
                    let m0 = vcombine_s16(vqmovn_s32(a0), vqmovn_s32(a1));
                    let m1 = vcombine_s16(vqmovn_s32(a2), vqmovn_s32(a3));
                    vst1q_s8(dst.add(i), vcombine_s8(vqmovn_s16(m0), vqmovn_s16(m1)));
                    i += 16;
                }
            }
            let mut h2 = [0.0f32; N_OCULTA2];
            for row in 0..N_OCULTA2 {
                let fila = self.w2_i8.as_ptr().add(row * N_OCULTA1);
                let mut j = 0;
                // Producto punto entero i8*i8 -> i32 con la extension dotprod
                // (sdot / vdotq_s32): 16 carriles i8 por instruccion,
                // acumulando directo en i32 -- el doble de ancho que
                // vmull_s8+vpadalq (8 carriles y dos instrucciones). Suma
                // entera exacta idéntica, sin redondeo intermedio, asi que la
                // evaluacion es bit-identica al camino anterior. dotprod esta
                // siempre presente en Apple Silicon (FEAT_DotProd) y en el
                // conjunto de features por defecto de aarch64-apple-darwin.
                //
                // CUATRO acumuladores independientes en vez de uno. Con uno
                // solo, las 32 instrucciones sdot de cada fila forman una
                // CADENA DE DEPENDENCIAS: cada una tiene que esperar a que
                // termine la anterior (~3 ciclos de latencia cada una), asi
                // que la CPU no puede lanzar varias a la vez aunque tenga
                // unidades libres. Con cuatro cadenas separadas se solapan y
                // el bucle pasa a estar limitado por RENDIMIENTO en vez de por
                // latencia. La suma entera es asociativa, asi que sumar las
                // cuatro parciales al final da EXACTAMENTE el mismo i32
                // (imposible desbordar: max 127*127*512 = 8.26M).
                let mut acc0 = vdupq_n_s32(0);
                let mut acc1 = vdupq_n_s32(0);
                let mut acc2 = vdupq_n_s32(0);
                let mut acc3 = vdupq_n_s32(0);
                while j < N_OCULTA1 {
                    acc0 = sdot_s32(
                        acc0,
                        vld1q_s8(h1.as_ptr().add(j)),
                        vld1q_s8(fila.add(j)),
                    );
                    acc1 = sdot_s32(
                        acc1,
                        vld1q_s8(h1.as_ptr().add(j + 16)),
                        vld1q_s8(fila.add(j + 16)),
                    );
                    acc2 = sdot_s32(
                        acc2,
                        vld1q_s8(h1.as_ptr().add(j + 32)),
                        vld1q_s8(fila.add(j + 32)),
                    );
                    acc3 = sdot_s32(
                        acc3,
                        vld1q_s8(h1.as_ptr().add(j + 48)),
                        vld1q_s8(fila.add(j + 48)),
                    );
                    j += 64;
                }
                let acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
                let combinado = self.b2_i32[row] + vaddvq_s32(acc);
                h2[row] = (combinado as f32 / (QH * QW2)).max(0.0);
            }
            // w3 (32) por h2: 1 producto punto de longitud 32.
            let mut acc = vdupq_n_f32(0.0);
            let mut j = 0;
            while j < N_OCULTA2 {
                acc = vfmaq_f32(
                    acc,
                    vld1q_f32(self.w3.as_ptr().add(j)),
                    vld1q_f32(h2.as_ptr().add(j)),
                );
                j += 4;
            }
            return (self.b3 + vaddvq_f32(acc)) * 100.0;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut h1 = [0i32; N_OCULTA1];
            for (v, &a) in h1.iter_mut().zip(acumulador.iter()) {
                *v = (a.max(0) >> QA_SOBRE_QH_SHIFT).min(127);
            }
            let mut h2 = [0.0; N_OCULTA2];
            for (i, fila) in self.w2_i8.chunks_exact(N_OCULTA1).enumerate() {
                let dot: i32 = fila
                    .iter()
                    .zip(h1.iter())
                    .map(|(&w, &v)| w as i32 * v)
                    .sum();
                let combinado = self.b2_i32[i] + dot;
                h2[i] = (combinado as f32 / (QH * QW2)).max(0.0);
            }
            let dot: f32 = self
                .w3
                .iter()
                .zip(h2.iter())
                .map(|(&peso, &valor)| peso * valor)
                .sum();
            (self.b3 + dot) * 100.0
        }
    }
}

#[derive(Clone)]
pub struct NnueAccumulator {
    // Referencia estatica en vez de Arc: el acumulador se CLONA una vez por
    // nodo de la busqueda, y con Arc cada clon era un incremento atomico del
    // contador de referencias y cada destruccion un decremento. Con Lazy SMP
    // los 4+ hilos golpeaban la MISMA linea de cache del contador millones de
    // veces por segundo (true sharing), serializandose entre ellos sin ninguna
    // necesidad: la red se carga una vez y no se libera nunca mientras el
    // proceso vive. Ahora clonar el acumulador es solo copiar un puntero.
    red: &'static RedNeural,
    primera_capa: [i32; N_OCULTA1],
}

impl NnueAccumulator {
    fn desde_tablero(red: &'static RedNeural, b: &Board) -> NnueAccumulator {
        let mut primera_capa = [0i32; N_OCULTA1];
        primera_capa.copy_from_slice(&red.b1);
        for (color_idx, color) in [(0usize, Color::White), (1usize, Color::Black)] {
            for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
                let mut piezas = b.pieces[color as usize][pt as usize];
                while piezas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut piezas);
                    red.sumar_feature(
                        &mut primera_capa,
                        feature_pieza(color_idx, pt_idx, sq as usize),
                        true,
                    );
                }
            }
        }
        if b.turn == Color::White {
            red.sumar_feature(&mut primera_capa, 768, true);
        }
        if enroque_del_bando(b) {
            red.sumar_feature(&mut primera_capa, 769, true);
        }
        for idx in indices_amenaza(b) {
            red.sumar_feature(&mut primera_capa, idx, true);
        }
        NnueAccumulator { red, primera_capa }
    }

    /// Incremental real. Idea: en vez de recalcular las 5378 features desde
    /// cero, se detectan las casillas cuyo contenido cambio (comparando
    /// piece_at en las 64 casillas entre antes/despues -- barato, sin
    /// calculo de ataques) y, a partir de esas casillas, se determina el
    /// conjunto ACOTADO de piezas cuyas features de amenaza pueden haber
    /// cambiado:
    ///   1. Cualquier pieza que este EN una casilla cambiada (la que se
    ///      movio, la capturada, la del enroque, el peon comido al paso...)
    ///      -- se le resta su aporte viejo (con el tablero/ocupacion de
    ///      antes) y se le suma el nuevo (con el de despues).
    ///   2. Cualquier pieza deslizante (alfil/torre/dama) de CUALQUIER color
    ///      que este en la misma fila/columna/diagonal que alguna casilla
    ///      cambiada -- su alcance de ataque puede haberse extendido o
    ///      recortado aunque ella misma no se haya movido (se abrio/cerro
    ///      una linea). Mismo tratamiento: restar aporte con antes, sumar
    ///      con despues.
    ///   3. Piezas NO deslizantes (peon/caballo/rey) que SI atacan una
    ///      casilla cambiada pero ellas mismas no se movieron: su propio
    ///      conjunto de ataque no cambia, pero la feature "amenazo a la
    ///      pieza que hay en esa casilla" si, porque cambio lo que hay ahi.
    ///      Se ajusta solo esa feature puntual (quitar la de antes, poner
    ///      la de despues), sin tocar el resto de sus amenazas.
    /// Todo lo demas (piezas que no se movieron, no son deslizantes, y no
    /// atacan ninguna casilla cambiada) queda exactamente igual -- no se
    /// toca. Verificado por test contra el recalculo completo en muchas
    /// posiciones/jugadas (ver tests, comprobar_incremental_amenazas).
    pub fn despues_de_jugada(&self, antes: &Board, despues: &Board) -> NnueAccumulator {
        let mut nuevo = self.clone();
        self.red
            .actualizar_amenazas_incremental(&mut nuevo.primera_capa, antes, despues);
        nuevo
    }

    pub fn evaluar(&self) -> f32 {
        self.red.salida(&self.primera_capa)
    }
}

#[inline]
fn feature_pieza(color: usize, pieza: usize, sq: usize) -> usize {
    (color * 6 + pieza) * 64 + sq
}

const N_BASE_AMENAZA: usize = 770;

/// Features de "quien ataca a cual pieza" -- ver features_threat.py (mismo
/// dataset con el que se entreno esta red). OJO: la convencion de color
/// AQUI es la inversa de feature_pieza (1=blanco atacante, 0=negro
/// atacante) porque asi quedo definida en el script de entrenamiento
/// original -- hay que igualarla exacto, no "corregirla", o los pesos
/// entrenados quedarian desalineados con los indices en inferencia.
fn indices_amenaza(b: &Board) -> Vec<usize> {
    let mut indices = Vec::new();
    let ocupado =
        b.pieces[0].iter().fold(0u64, |a, &x| a | x) | b.pieces[1].iter().fold(0u64, |a, &x| a | x);
    for (color_idx, color) in [(0usize, Color::White), (1usize, Color::Black)] {
        for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
            let mut piezas = b.pieces[color as usize][pt as usize];
            while piezas != 0 {
                let sq = crate::bitboard::pop_lsb(&mut piezas);
                let ataques = match pt {
                    crate::types::PieceType::Pawn => crate::bitboard::pawn_attacks(color, sq),
                    crate::types::PieceType::Knight => crate::bitboard::knight_attacks(sq),
                    crate::types::PieceType::Bishop => crate::bitboard::bishop_attacks(sq, ocupado),
                    crate::types::PieceType::Rook => crate::bitboard::rook_attacks(sq, ocupado),
                    crate::types::PieceType::Queen => crate::bitboard::queen_attacks(sq, ocupado),
                    crate::types::PieceType::King => crate::bitboard::king_attacks(sq),
                };
                let mut victimas = ataques & ocupado;
                while victimas != 0 {
                    let sq_v = crate::bitboard::pop_lsb(&mut victimas);
                    let (_, tipo_v) = b.piece_at(sq_v).expect("casilla ocupada sin pieza");
                    let color_conv = if color_idx == 0 { 1 } else { 0 };
                    let idx = N_BASE_AMENAZA
                        + ((color_conv * 6 + pt_idx) * 6 + tipo_v as usize) * 64
                        + sq_v as usize;
                    indices.push(idx);
                }
            }
        }
    }
    indices
}

/// Casillas donde antes/después difieren en su ocupante. La unión de los
/// XOR por color/tipo es exactamente equivalente a revisar `piece_at` en las
/// 64 casillas, pero evita 64 búsquedas y no asigna un `Vec` por hijo.
/// Cubre normal, captura, enroque, en-passant y promoción sin necesitar el
/// tipo de jugada.
#[inline]
fn mascara_casillas_cambiadas(antes: &Board, despues: &Board) -> u64 {
    let mut cambiadas = 0u64;
    for color in 0..2 {
        for pt in 0..ALL_PIECE_TYPES.len() {
            cambiadas |= antes.pieces[color][pt] ^ despues.pieces[color][pt];
        }
    }
    cambiadas
}

#[inline]
fn ataques_de_pieza(pt: crate::types::PieceType, color: Color, sq: u8, ocupado: u64) -> u64 {
    match pt {
        crate::types::PieceType::Pawn => crate::bitboard::pawn_attacks(color, sq),
        crate::types::PieceType::Knight => crate::bitboard::knight_attacks(sq),
        crate::types::PieceType::Bishop => crate::bitboard::bishop_attacks(sq, ocupado),
        crate::types::PieceType::Rook => crate::bitboard::rook_attacks(sq, ocupado),
        crate::types::PieceType::Queen => crate::bitboard::queen_attacks(sq, ocupado),
        crate::types::PieceType::King => crate::bitboard::king_attacks(sq),
    }
}

/// Entre los deslizantes geométricamente alineados con una casilla cambiada,
/// conserva solo los que de verdad pueden haber cambiado su conjunto de
/// features. Muchos rayos candidatos quedan bloqueados antes de llegar a la
/// casilla modificada; antes se restaban/sumaban igual 256 pesos para ellos.
#[inline]
fn mascara_slider_con_amenaza_cambiante(
    antes: &Board,
    despues: &Board,
    pt: crate::types::PieceType,
    linea_geometrica: u64,
    cambiadas: u64,
) -> u64 {
    let piezas_antes = antes.pieces[Color::White as usize][pt as usize]
        | antes.pieces[Color::Black as usize][pt as usize];
    let piezas_despues = despues.pieces[Color::White as usize][pt as usize]
        | despues.pieces[Color::Black as usize][pt as usize];
    // Las piezas que se movieron ya están cubiertas por `cambiadas`. Para una
    // que permanece en la misma casilla, basta comparar sus rayos reales.
    let mut comunes = piezas_antes & piezas_despues & linea_geometrica & !cambiadas;
    let mut necesarias = cambiadas;
    while comunes != 0 {
        let sq = crate::bitboard::pop_lsb(&mut comunes);
        let (color, encontrado) = antes.piece_at(sq).expect("bitboard inconsistente");
        debug_assert_eq!(encontrado, pt);
        let ataques_antes = ataques_de_pieza(pt, color, sq, antes.occupied);
        let ataques_despues = ataques_de_pieza(pt, color, sq, despues.occupied);
        // Si los rayos no cambiaron y tampoco alcanzan una casilla modificada,
        // sus víctimas y por tanto todas sus features son idénticas.
        if ataques_antes != ataques_despues || ((ataques_antes | ataques_despues) & cambiadas != 0)
        {
            necesarias |= 1u64 << sq;
        }
    }
    necesarias
}

impl RedNeural {
    /// Aplica directamente las features de amenaza de una pieza. La versión
    /// anterior devolvía un `Vec<usize>` temporal para cada atacante afectado;
    /// en una búsqueda NNUE eso producía miles de asignaciones por segundo.
    /// Emitirlas aquí conserva exactamente las mismas features y su signo.
    #[inline]
    fn aplicar_amenazas_de_pieza(
        &self,
        acumulador: &mut [i32; N_OCULTA1],
        lote: &mut Lote,
        tablero: &Board,
        color_idx: usize,
        pt: crate::types::PieceType,
        sq: usize,
        sumar: bool,
    ) {
        let color = if color_idx == 0 {
            Color::White
        } else {
            Color::Black
        };
        let ataques = ataques_de_pieza(pt, color, sq as u8, tablero.occupied);
        let victimas = ataques & tablero.occupied;
        if victimas == 0 {
            return;
        }
        // Convención heredada del dataset: 1=atacante blanco, 0=negro.
        let color_conv = if color_idx == 0 { 1 } else { 0 };
        let base_atacante = N_BASE_AMENAZA + (color_conv * 6 + pt as usize) * 6 * 64;
        // En vez de un `piece_at` por victima (que recorre hasta 6 bitboards
        // buscando el tipo), se agrupan las victimas POR TIPO con una simple
        // interseccion de bitboards: 6 operaciones fijas en vez de N busquedas.
        // Exactamente las mismas features, mismo signo.
        for tipo_v in 0..6 {
            let mut bb = victimas
                & (tablero.pieces[0][tipo_v] | tablero.pieces[1][tipo_v]);
            let base = base_atacante + tipo_v * 64;
            while bb != 0 {
                let sq_v = crate::bitboard::pop_lsb(&mut bb) as usize;
                self.empujar(acumulador, lote, base + sq_v, sumar);
            }
        }
    }

    /// Actualiza las amenazas de las piezas que están en `mascara`. Para
    /// deslizantes la máscara incluye además las líneas abiertas/cerradas por
    /// la jugada; para las demás piezas solo las casillas que cambiaron.
    #[inline]
    fn aplicar_amenazas_en_mascaras(
        &self,
        acumulador: &mut [i32; N_OCULTA1],
        lote: &mut Lote,
        tablero: &Board,
        cambiadas: u64,
        mascara_alfil: u64,
        mascara_torre: u64,
        mascara_dama: u64,
        sumar: bool,
    ) {
        for (color_idx, color) in [(0usize, Color::White), (1usize, Color::Black)] {
            for pt in ALL_PIECE_TYPES {
                let mascara = match pt {
                    crate::types::PieceType::Bishop => mascara_alfil,
                    crate::types::PieceType::Rook => mascara_torre,
                    crate::types::PieceType::Queen => mascara_dama,
                    _ => cambiadas,
                };
                let mut piezas = tablero.pieces[color as usize][pt as usize] & mascara;
                while piezas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut piezas) as usize;
                    self.aplicar_amenazas_de_pieza(
                        acumulador, lote, tablero, color_idx, pt, sq, sumar,
                    );
                }
            }
        }
    }

    /// Delta de amenazas para deslizantes que NO cambiaron de casilla pero
    /// cuya vision cambio (una linea propia se abrio o cerro). Aplica solo
    /// las features de victima que difieren entre `antes` y `despues`,
    /// saltando las identicas. Equivalente exacto a restar todas las viejas y
    /// sumar todas las nuevas, pero sin las que se cancelan.
    #[inline]
    fn aplicar_delta_sliders_estables(
        &self,
        acumulador: &mut [i32; N_OCULTA1],
        lote: &mut Lote,
        antes: &Board,
        despues: &Board,
        cambiadas: u64,
        pt: crate::types::PieceType,
        mut piezas: u64,
    ) {
        while piezas != 0 {
            let sq = crate::bitboard::pop_lsb(&mut piezas);
            // Estable => misma pieza/color en antes y despues.
            let (color, _) = antes.piece_at(sq).expect("bitboard inconsistente");
            let color_idx = color as usize;
            let color_conv = if color_idx == 0 { 1 } else { 0 };
            let base_atacante = N_BASE_AMENAZA + (color_conv * 6 + pt as usize) * 6 * 64;

            let att_antes = ataques_de_pieza(pt, color, sq, antes.occupied);
            let att_despues = ataques_de_pieza(pt, color, sq, despues.occupied);
            let v_antes = att_antes & antes.occupied;
            let v_despues = att_despues & despues.occupied;
            // Una casilla-victima en ambos conjuntos solo cambia de feature si
            // esta en `cambiadas` (ahi difiere el ocupante). El resto es
            // identico y se salta.
            let comunes_cambiados = v_antes & v_despues & cambiadas;
            let restar = (v_antes & !v_despues) | comunes_cambiados;
            let sumar = (v_despues & !v_antes) | comunes_cambiados;

            // Agrupacion por tipo de victima con interseccion de bitboards,
            // en vez de un `piece_at` por casilla (mismas features exactas).
            if restar != 0 {
                for tipo_v in 0..6 {
                    let mut bb = restar & (antes.pieces[0][tipo_v] | antes.pieces[1][tipo_v]);
                    let base = base_atacante + tipo_v * 64;
                    while bb != 0 {
                        let sq_v = crate::bitboard::pop_lsb(&mut bb) as usize;
                        self.empujar(acumulador, lote, base + sq_v, false);
                    }
                }
            }
            if sumar != 0 {
                for tipo_v in 0..6 {
                    let mut bb = sumar & (despues.pieces[0][tipo_v] | despues.pieces[1][tipo_v]);
                    let base = base_atacante + tipo_v * 64;
                    while bb != 0 {
                        let sq_v = crate::bitboard::pop_lsb(&mut bb) as usize;
                        self.empujar(acumulador, lote, base + sq_v, true);
                    }
                }
            }
        }
    }

    /// Ver `NnueAccumulator::despues_de_jugada` para la explicacion del
    /// algoritmo. Aqui solo la mecanica: recolectar piezas afectadas y
    /// restar/sumar sus aportes.
    fn actualizar_amenazas_incremental(
        &self,
        acumulador: &mut [i32; N_OCULTA1],
        antes: &Board,
        despues: &Board,
    ) {
        // Todas las features de esta jugada se acumulan primero en `lote` y se
        // aplican de una sola pasada al final (ver `Lote`).
        let mut lote = Lote::nuevo();
        let lote = &mut lote;

        // --- 1. features base (770 piece-square + turno + enroque) -----
        actualizar_booleano(
            self,
            acumulador,
            lote,
            768,
            antes.turn == Color::White,
            despues.turn == Color::White,
        );
        actualizar_booleano(
            self,
            acumulador,
            lote,
            769,
            enroque_del_bando(antes),
            enroque_del_bando(despues),
        );

        let cambiadas = mascara_casillas_cambiadas(antes, despues);

        let mut scan_cambiadas = cambiadas;
        while scan_cambiadas != 0 {
            let sq = crate::bitboard::pop_lsb(&mut scan_cambiadas) as usize;
            if let Some((c, pt)) = antes.piece_at(sq as u8) {
                self.empujar(
                    acumulador,
                    lote,
                    feature_pieza(c as usize, pt as usize, sq),
                    false,
                );
            }
            if let Some((c, pt)) = despues.piece_at(sq as u8) {
                self.empujar(
                    acumulador,
                    lote,
                    feature_pieza(c as usize, pt as usize, sq),
                    true,
                );
            }
        }

        // --- 2. atacantes afectados por casillas/lineas que cambiaron ---
        // `bishop_attacks/rook_attacks(sq, 0)` representan exactamente las
        // casillas geométricamente alineadas que antes se encontraban con
        // `misma_linea`, sin escanear todas las piezas ni construir Vecs.
        let mut lineas_alfil = 0u64;
        let mut lineas_torre = 0u64;
        let mut lineas_desde = cambiadas;
        while lineas_desde != 0 {
            let sq = crate::bitboard::pop_lsb(&mut lineas_desde);
            lineas_alfil |= crate::bitboard::bishop_attacks(sq, 0);
            lineas_torre |= crate::bitboard::rook_attacks(sq, 0);
        }
        let mascara_alfil = mascara_slider_con_amenaza_cambiante(
            antes,
            despues,
            crate::types::PieceType::Bishop,
            lineas_alfil,
            cambiadas,
        );
        let mascara_torre = mascara_slider_con_amenaza_cambiante(
            antes,
            despues,
            crate::types::PieceType::Rook,
            lineas_torre,
            cambiadas,
        );
        let mascara_dama = mascara_slider_con_amenaza_cambiante(
            antes,
            despues,
            crate::types::PieceType::Queen,
            lineas_alfil | lineas_torre,
            cambiadas,
        );
        // Piezas que se MOVIERON (su casilla esta en `cambiadas`): antes y
        // despues procesan piezas distintas, no se pueden emparejar, asi que
        // se hace la pasada completa restar-antes / sumar-despues. Incluye
        // las no deslizantes (mascara = cambiadas) y los deslizantes que
        // realmente cambiaron de casilla.
        self.aplicar_amenazas_en_mascaras(
            acumulador,
            lote,
            antes,
            cambiadas,
            mascara_alfil & cambiadas,
            mascara_torre & cambiadas,
            mascara_dama & cambiadas,
            false,
        );
        self.aplicar_amenazas_en_mascaras(
            acumulador,
            lote,
            despues,
            cambiadas,
            mascara_alfil & cambiadas,
            mascara_torre & cambiadas,
            mascara_dama & cambiadas,
            true,
        );
        // Deslizantes ESTABLES (misma casilla en antes y despues, pero una
        // linea propia se abrio/cerro): en vez de restar TODAS sus features
        // viejas y sumar TODAS las nuevas -- la mayoria identicas -- se aplica
        // solo el delta de victimas que de verdad cambiaron. Esta era la
        // funcion mas caliente de todo el motor (perfilado: ~26% del tiempo);
        // la victima idle tipica de una dama que gana/pierde vision de una
        // sola casilla ya no repite 8-10 pasadas NEON de 256 pesos que se
        // cancelaban entre si. Evaluacion identica por construccion.
        self.aplicar_delta_sliders_estables(
            acumulador,
            lote,
            antes,
            despues,
            cambiadas,
            crate::types::PieceType::Bishop,
            mascara_alfil & !cambiadas,
        );
        self.aplicar_delta_sliders_estables(
            acumulador,
            lote,
            antes,
            despues,
            cambiadas,
            crate::types::PieceType::Rook,
            mascara_torre & !cambiadas,
        );
        self.aplicar_delta_sliders_estables(
            acumulador,
            lote,
            antes,
            despues,
            cambiadas,
            crate::types::PieceType::Queen,
            mascara_dama & !cambiadas,
        );

        // --- 3. piezas NO deslizantes (peon/caballo/rey) que atacan una
        // casilla cambiada como "victima", sin haberse movido ellas mismas
        // (su propio conjunto de ataque no depende de ocupacion, pero la
        // pieza que hay en la casilla que atacan si cambio).
        let mut victimas_cambiadas = cambiadas;
        while victimas_cambiadas != 0 {
            let sq = crate::bitboard::pop_lsb(&mut victimas_cambiadas) as usize;
            let sqb = sq as u8;
            let mut atacantes = 0u64;
            atacantes |= crate::bitboard::knight_attacks(sqb)
                & (antes.pieces[0][crate::types::PieceType::Knight as usize]
                    | antes.pieces[1][crate::types::PieceType::Knight as usize]);
            atacantes |= crate::bitboard::king_attacks(sqb)
                & (antes.pieces[0][crate::types::PieceType::King as usize]
                    | antes.pieces[1][crate::types::PieceType::King as usize]);
            atacantes |= crate::bitboard::pawn_attacks(Color::Black, sqb)
                & antes.pieces[0][crate::types::PieceType::Pawn as usize];
            atacantes |= crate::bitboard::pawn_attacks(Color::White, sqb)
                & antes.pieces[1][crate::types::PieceType::Pawn as usize];

            let mut resto = atacantes;
            while resto != 0 {
                let asq = crate::bitboard::pop_lsb(&mut resto) as usize;
                if cambiadas & (1u64 << asq) != 0 {
                    // ya cubierta en el paso 2 (esa pieza tambien se movio)
                    continue;
                }
                let (c, pt) = antes.piece_at(asq as u8).expect("bitboard inconsistente");
                let color_idx = c as usize;
                let color_conv = if color_idx == 0 { 1 } else { 0 };
                let pt_idx = pt as usize;
                if let Some((_, tipo_v)) = antes.piece_at(sqb) {
                    let idx = N_BASE_AMENAZA
                        + ((color_conv * 6 + pt_idx) * 6 + tipo_v as usize) * 64
                        + sq;
                    self.empujar(acumulador, lote, idx, false);
                }
                if let Some((_, tipo_v)) = despues.piece_at(sqb) {
                    let idx = N_BASE_AMENAZA
                        + ((color_conv * 6 + pt_idx) * 6 + tipo_v as usize) * 64
                        + sq;
                    self.empujar(acumulador, lote, idx, true);
                }
            }
        }

        // Una sola pasada por el acumulador con todo lo acumulado.
        self.aplicar_lote(acumulador, lote);
    }
}

#[inline]
fn enroque_del_bando(b: &Board) -> bool {
    let derechos = match b.turn {
        Color::White => crate::board::CASTLE_WK | crate::board::CASTLE_WQ,
        Color::Black => crate::board::CASTLE_BK | crate::board::CASTLE_BQ,
    };
    b.castling_rights & derechos != 0
}

fn actualizar_booleano(
    red: &RedNeural,
    acumulador: &mut [i32; N_OCULTA1],
    lote: &mut Lote,
    feature: usize,
    antes: bool,
    despues: bool,
) {
    match (antes, despues) {
        (false, true) => red.empujar(acumulador, lote, feature, true),
        (true, false) => red.empujar(acumulador, lote, feature, false),
        _ => {}
    }
}

static RED: OnceLock<RwLock<Option<&'static RedNeural>>> = OnceLock::new();
static ACTIVA: AtomicBool = AtomicBool::new(false);
// UCI puede recibir UseNNUE antes que NNUEPath. Conservamos la solicitud
// pendiente y activamos la red cuando los pesos finalmente se cargan.
static SOLICITADA: AtomicBool = AtomicBool::new(false);

fn almacenamiento() -> &'static RwLock<Option<&'static RedNeural>> {
    RED.get_or_init(|| RwLock::new(None))
}

/// Carga o reemplaza los pesos. Si la ruta o el contenido son invalidos se
/// conserva la red anterior, evitando que un error de escritura apague una
/// NNUE que ya estaba funcionando. Devuelve el checksum FNV-1a del archivo.
pub fn cargar_detallado(path: &str) -> Result<u64, String> {
    let datos = std::fs::read(path).map_err(|e| format!("no se pudo leer: {e}"))?;
    let checksum = checksum_fnv1a(&datos);
    let red = RedNeural::cargar_de_bytes(&datos)
        .ok_or_else(|| "formato o valores de pesos invalidos".to_string())?;
    // `Box::leak`: la red (11 MB) queda viva hasta que el proceso termina.
    // Recargar los pesos por UCI (raro: cero o una vez por partida) deja la
    // anterior sin liberar, precio aceptable a cambio de quitar los atomicos
    // del camino caliente.
    let estatica: &'static RedNeural = Box::leak(Box::new(red));
    *almacenamiento().write().expect("candado NNUE envenenado") = Some(estatica);
    ACTIVA.store(SOLICITADA.load(Ordering::Relaxed), Ordering::Relaxed);
    Ok(checksum)
}

pub fn cargar(path: &str) -> bool {
    cargar_detallado(path).is_ok()
}

pub fn set_activa(valor: bool) {
    SOLICITADA.store(valor, Ordering::Relaxed);
    ACTIVA.store(valor && hay_red_cargada(), Ordering::Relaxed);
}

pub fn esta_activa() -> bool {
    ACTIVA.load(Ordering::Relaxed)
}

pub fn hay_red_cargada() -> bool {
    almacenamiento()
        .read()
        .expect("candado NNUE envenenado")
        .is_some()
}

pub fn crear_acumulador(b: &Board) -> Option<NnueAccumulator> {
    if !ACTIVA.load(Ordering::Relaxed) {
        return None;
    }
    let red = *almacenamiento()
        .read()
        .expect("candado NNUE envenenado")
        .as_ref()?;
    Some(NnueAccumulator::desde_tablero(red, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal;

    fn red_prueba() -> &'static RedNeural {
        let datos = include_bytes!("../pesos_amenazas_prueba.bin");
        Box::leak(Box::new(
            RedNeural::cargar_de_bytes(datos).expect("pesos validos"),
        ))
    }

    fn comprobar_incremental(fen: &str, uci: &str) {
        let red = red_prueba();
        let antes = Board::from_fen(fen).unwrap();
        let mv = generate_legal(&antes)
            .into_iter()
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("jugada no encontrada: {uci}"));
        let despues = antes.make_move(&mv);
        let incremental = NnueAccumulator::desde_tablero(red, &antes)
            .despues_de_jugada(&antes, &despues);
        let recalculado = NnueAccumulator::desde_tablero(red, &despues);
        assert_eq!(
            incremental.primera_capa, recalculado.primera_capa,
            "acumulador distinto tras {} desde {}",
            uci, fen
        );
        assert!(
            (incremental.evaluar() - recalculado.evaluar()).abs() < 0.01,
            "diferencia tras {} desde {}: incremental={} recalculado={}",
            uci,
            fen,
            incremental.evaluar(),
            recalculado.evaluar()
        );
    }

    #[test]
    fn acumulador_incremental_movimientos_especiales() {
        comprobar_incremental(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e4",
        );
        comprobar_incremental("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5");
        comprobar_incremental("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6");
        comprobar_incremental("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1");
        comprobar_incremental("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q");
        // Casos pensados para las lineas abiertas/cerradas de deslizantes
        // que NO se mueven (el motivo de todo este incremental): torre
        // detras de un peon que avanza, alfil detras de una pieza que se
        // interpone/despeja, dama que gana/pierde vision al abrirse una
        // columna.
        comprobar_incremental("4k3/8/8/8/4P3/8/8/R3K3 w Q - 0 1", "e4e5");
        comprobar_incremental("4k3/8/5n2/8/8/8/1B6/4K3 w - - 0 1", "b2f6");
        comprobar_incremental("4k3/8/8/8/3n4/8/3Q4/4K3 w - - 0 1", "d2d4");
    }

    #[test]
    fn acumulador_incremental_amenazas_fuzz_determinista() {
        let red = red_prueba();
        let mut semilla = 0x5EED_CAFE_D00Du64;
        for _partida in 0..24 {
            let mut tablero = Board::startpos();
            let mut acumulador = NnueAccumulator::desde_tablero(red, &tablero);
            for _ply in 0..80 {
                let legales = generate_legal(&tablero);
                if legales.is_empty() {
                    break;
                }
                semilla ^= semilla << 7;
                semilla ^= semilla >> 9;
                let mv = legales[(semilla as usize) % legales.len()];
                let siguiente = tablero.make_move(&mv);
                let incremental = acumulador.despues_de_jugada(&tablero, &siguiente);
                let recalculado = NnueAccumulator::desde_tablero(red, &siguiente);
                assert_eq!(
                    incremental.primera_capa,
                    recalculado.primera_capa,
                    "acumulador distinto tras {} en partida aleatoria",
                    mv.to_uci()
                );
                assert!(
                    (incremental.evaluar() - recalculado.evaluar()).abs() < 0.01,
                    "diferencia tras {} en partida aleatoria: incremental={} recalculado={}",
                    mv.to_uci(),
                    incremental.evaluar(),
                    recalculado.evaluar()
                );
                tablero = siguiente;
                acumulador = incremental;
            }
        }
    }

    #[test]
    fn rechaza_nan_sin_panico() {
        let mut datos = include_bytes!("../pesos_v1.bin").to_vec();
        datos[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(RedNeural::cargar_de_bytes(&datos).is_none());
    }

    #[test]
    fn checksum_es_estable() {
        let datos = include_bytes!("../pesos_v1.bin");
        assert_eq!(checksum_fnv1a(datos), checksum_fnv1a(datos));
        assert_ne!(checksum_fnv1a(datos), 0);
    }
}
