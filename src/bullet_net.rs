// NNUE con la arquitectura ESTANDAR de la industria, entrenada con la
// libreria `bullet` (https://github.com/jw1912/bullet):
//
//     (768 -> 256)x2 -> 1,  doble perspectiva,  activacion SCReLU
//
// Diferencias clave con la red "de amenazas" que vive en neural.rs:
//   * Entrada 768 = 2 colores x 6 tipos x 64 casillas (features clasicas de
//     pieza-casilla), sin features de amenaza ni de enroque.
//   * DOS acumuladores: uno visto desde las blancas y otro desde las negras.
//     La capa de salida recibe [perspectiva_del_que_mueve | la_del_rival],
//     por lo que la red aprende sola el valor del turno.
//   * Cuantizacion entera de bullet: QA=255 en la capa de entrada, QB=64 en
//     la de salida, y la salida se escala por SCALE=400 para dar centipeones.
//
// FORMATO DEL ARCHIVO (quantised.bin). Verificado byte a byte contra el
// raw.bin sin cuantizar del mismo checkpoint:
//   4 bloques consecutivos de i16 little-endian, en este orden
//     1) l0w: 768*256 = 196608 valores, dispuestos POR FEATURE
//        (feature 0 -> sus 256 pesos, feature 1 -> sus 256 pesos, ...),
//        que es justo lo que hace falta para el acumulador incremental.
//     2) l0b: 256 valores
//     3) l1w: 512 valores (los primeros 256 para la perspectiva del que
//        mueve, los ultimos 256 para la del rival)
//     4) l1b: 1 valor
//   Total util = 197377 * 2 = 394754 bytes. El archivo mide 394816, es
//   decir 62 bytes MAS: no es una cabecera, es RELLENO AL FINAL para
//   alinear el archivo a 64 bytes (394816 = 64 * 6169). Los bytes de
//   relleno son la palabra ASCII "bullet" repetida. Se validan, no se
//   ignoran a ciegas.

use crate::board::Board;
use crate::types::{ALL_PIECE_TYPES, Color};

/// Tamano maximo de capa oculta soportado. El acumulador usa arrays de
/// este tamano y se rellena solo `h` posiciones, para no volver la
/// estructura dinamica (se CLONA una vez por nodo de busqueda).
pub const H_MAX: usize = 512;
const N_ENTRADA: usize = 768;
const QA: i32 = 255;
const QB: i32 = 64;

/// Escala de salida. Debe coincidir con el `eval_scale` con el que se
/// entreno la red: la red aprende `salida ~= score / eval_scale`, asi que
/// para recuperar centipeones hay que multiplicar por ese mismo numero. El
/// archivo de pesos NO lleva este dato dentro, asi que se configura con
/// MIMOTOR_BULLET_SCALE (por defecto 400, que es lo que usaron todas las
/// redes entrenadas hasta ahora).
fn scale() -> i32 {
    use std::sync::OnceLock;
    static CACHE: OnceLock<i32> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("MIMOTOR_BULLET_SCALE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| (16..=4096).contains(v))
            .unwrap_or(400)
    })
}

const ALINEACION: usize = 64;

/// Bytes de pesos para una capa oculta de `h` neuronas y `b` "output
/// buckets" (capas de salida por material). El formato viejo es b = 1.
const fn bytes_utiles(h: usize, b: usize) -> usize {
    (N_ENTRADA * h + h + b * 2 * h + b) * 2
}

/// Arquitecturas admitidas: (capa oculta, output buckets). Cada combinacion
/// da un tamano de archivo DISTINTO, asi que el tamano identifica el
/// formato sin necesidad de cabecera (los archivos viejos siguen cargando).
const ARQUITECTURAS_SOPORTADAS: [(usize, usize); 3] = [(256, 1), (512, 1), (512, 8)];

/// Maximo de output buckets soportado.
pub const B_MAX: usize = 8;

pub struct RedBullet {
    /// Neuronas de la capa oculta de ESTA red (256 o 512).
    h: usize,
    /// Output buckets de ESTA red (1 = formato viejo, 8 = por material).
    buckets: usize,
    /// [feature][neurona] -- 768 bloques contiguos de `h` pesos.
    l0w: Vec<i16>,
    l0b: [i16; H_MAX],
    /// Capa de salida: `buckets` bloques de 2*H_MAX pesos. Dentro de cada
    /// bloque, la mitad "stm" empieza en 0 y la mitad "ntm" en H_MAX.
    l1w: Vec<i16>,
    /// Un sesgo de salida por bucket.
    l1b: [i16; B_MAX],
}

/// Deduce (oculta, buckets) a partir del tamano del archivo, o None si no
/// corresponde a ninguna arquitectura bullet conocida.
pub fn arquitectura_para_tamano(n: usize) -> Option<(usize, usize)> {
    ARQUITECTURAS_SOPORTADAS
        .into_iter()
        .find(|&(h, b)| n >= bytes_utiles(h, b) && n - bytes_utiles(h, b) < ALINEACION)
}

/// True si el tamano del archivo corresponde a esta arquitectura.
pub fn tamano_plausible(n: usize) -> bool {
    arquitectura_para_tamano(n).is_some()
}

impl RedBullet {
    pub fn cargar_de_bytes(datos: &[u8]) -> Option<RedBullet> {
        let (h, buckets) = arquitectura_para_tamano(datos.len())?;
        let utiles = bytes_utiles(h, buckets);
        // El relleno debe ser la firma "bullet" repetida; si no lo es, el
        // archivo no es lo que creemos y preferimos rechazarlo antes que
        // cargar pesos desalineados.
        let relleno = &datos[utiles..];
        for (i, &b) in relleno.iter().enumerate() {
            if b != b"bullet"[i % 6] {
                eprintln!(
                    "info string NNUE bullet: relleno final inesperado ({} bytes), se rechaza",
                    relleno.len()
                );
                return None;
            }
        }

        let leer = |desde: usize, n: usize| -> Vec<i16> {
            (0..n)
                .map(|k| {
                    let p = desde + k * 2;
                    i16::from_le_bytes([datos[p], datos[p + 1]])
                })
                .collect()
        };
        let mut cursor = 0usize;
        let l0w = leer(cursor, N_ENTRADA * h);
        cursor += N_ENTRADA * h * 2;
        let l0b_v = leer(cursor, h);
        cursor += h * 2;
        // l1w: `buckets` bloques consecutivos de [stm (h) | ntm (h)] (con
        // buckets=1 es el formato viejo tal cual; con buckets>1 es el orden
        // que produce SavedFormat::id("l1w").transpose() en bullet: los 2h
        // pesos de cada bucket contiguos). Al copiarlo a bloques de 2*H_MAX
        // hay que separar las mitades para que "ntm" empiece en H_MAX.
        let mut l1w = vec![0i16; buckets * 2 * H_MAX];
        for bk in 0..buckets {
            let v = leer(cursor, 2 * h);
            cursor += 2 * h * 2;
            let base = bk * 2 * H_MAX;
            l1w[base..base + h].copy_from_slice(&v[..h]);
            l1w[base + H_MAX..base + H_MAX + h].copy_from_slice(&v[h..]);
        }
        let l1b_v = leer(cursor, buckets);

        let mut l0b = [0i16; H_MAX];
        l0b[..h].copy_from_slice(&l0b_v);
        let mut l1b = [0i16; B_MAX];
        l1b[..buckets].copy_from_slice(&l1b_v);

        eprintln!(
            "info string NNUE bullet: capa oculta de {h} neuronas, {buckets} output bucket(s)"
        );
        Some(RedBullet {
            h,
            buckets,
            l0w,
            l0b,
            l1w,
            l1b,
        })
    }

    #[inline(always)]
    fn columna(&self, feature: usize) -> &[i16] {
        &self.l0w[feature * self.h..(feature + 1) * self.h]
    }

    #[inline(always)]
    fn sumar(&self, acc: &mut [i16; H_MAX], feature: usize) {
        let col = self.columna(feature);
        for (a, &w) in acc[..self.h].iter_mut().zip(col) {
            *a = a.wrapping_add(w);
        }
    }

    #[inline(always)]
    fn restar(&self, acc: &mut [i16; H_MAX], feature: usize) {
        let col = self.columna(feature);
        for (a, &w) in acc[..self.h].iter_mut().zip(col) {
            *a = a.wrapping_sub(w);
        }
    }
}

/// Indice de feature en Chess768 visto desde `persp` (0=blancas, 1=negras).
/// Convencion identica a `bullet_lib::game::inputs::Chess768`: las piezas
/// PROPIAS van al bloque 0..384 y las del rival al 384..768; ademas, desde
/// las negras la casilla se refleja verticalmente (sq ^ 56) para que la
/// red vea siempre "su" primera fila abajo.
#[inline(always)]
fn feature(persp: usize, color: usize, pieza: usize, sq: usize) -> usize {
    let propia = if color == persp { 0 } else { 384 };
    let casilla = if persp == 0 { sq } else { sq ^ 56 };
    propia + pieza * 64 + casilla
}

#[derive(Clone)]
pub struct AcumBullet {
    red: &'static RedBullet,
    /// [0] = acumulador visto por las blancas, [1] = por las negras.
    /// Solo las primeras `red.h` posiciones son significativas.
    persp: [[i16; H_MAX]; 2],
    negras_mueven: bool,
    /// Piezas totales en el tablero (reyes incluidos). Solo se usa para
    /// elegir el output bucket; se mantiene incrementalmente.
    piezas: u8,
}

/// Piezas totales (reyes incluidos) de un tablero.
#[inline(always)]
fn contar_piezas(b: &Board) -> u8 {
    let mut n = 0u32;
    for color in 0..2usize {
        for &pt in ALL_PIECE_TYPES.iter() {
            n += b.pieces[color][pt as usize].count_ones();
        }
    }
    n as u8
}

impl AcumBullet {
    pub fn desde_tablero(red: &'static RedBullet, b: &Board) -> AcumBullet {
        let mut persp = [red.l0b, red.l0b];
        for color in 0..2usize {
            for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
                let mut piezas = b.pieces[color][pt as usize];
                while piezas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut piezas) as usize;
                    red.sumar(&mut persp[0], feature(0, color, pt_idx, sq));
                    red.sumar(&mut persp[1], feature(1, color, pt_idx, sq));
                }
            }
        }
        AcumBullet {
            red,
            persp,
            negras_mueven: b.turn == Color::Black,
            piezas: contar_piezas(b),
        }
    }

    /// Actualizacion incremental: con features 768 puras basta con mirar,
    /// bitboard por bitboard, que casillas se ocuparon y cuales se
    /// liberaron. Cubre jugada normal, captura, enroque, al paso y
    /// promocion sin necesitar el tipo de jugada.
    pub fn despues_de_jugada(&self, antes: &Board, despues: &Board) -> AcumBullet {
        let mut nuevo = self.clone();
        nuevo.aplicar_jugada(antes, despues);
        nuevo
    }

    /// Version in-place de `despues_de_jugada`. Es su propia inversa al
    /// intercambiar los argumentos: `aplicar_jugada(d, a)` deshace
    /// exactamente `aplicar_jugada(a, d)`, porque (1) las casillas que se
    /// ocuparon pasan a ser las que se liberaron y viceversa, de modo que
    /// cada `sumar` se compensa con su `restar` sobre la misma feature, y
    /// (2) `negras_mueven` se recalcula desde el tablero destino. La
    /// aritmetica es wrapping sobre i16, asi que sumar y restar son
    /// inversos exactos incluso si hubiera desbordamiento.
    #[inline]
    pub fn aplicar_jugada(&mut self, antes: &Board, despues: &Board) {
        let red = self.red;
        let nuevo = self;
        nuevo.negras_mueven = despues.turn == Color::Black;
        nuevo.piezas = contar_piezas(despues);
        for color in 0..2usize {
            for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
                let a = antes.pieces[color][pt as usize];
                let d = despues.pieces[color][pt as usize];
                if a == d {
                    continue;
                }
                let mut anadidas = d & !a;
                while anadidas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut anadidas) as usize;
                    red.sumar(&mut nuevo.persp[0], feature(0, color, pt_idx, sq));
                    red.sumar(&mut nuevo.persp[1], feature(1, color, pt_idx, sq));
                }
                let mut quitadas = a & !d;
                while quitadas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut quitadas) as usize;
                    red.restar(&mut nuevo.persp[0], feature(0, color, pt_idx, sq));
                    red.restar(&mut nuevo.persp[1], feature(1, color, pt_idx, sq));
                }
            }
        }
    }

    /// Salida en centipeones desde la perspectiva del lado que mueve
    /// (misma convencion que la evaluacion clasica del motor).
    pub fn evaluar(&self) -> f32 {
        let (yo, rival) = if self.negras_mueven {
            (&self.persp[1], &self.persp[0])
        } else {
            (&self.persp[0], &self.persp[1])
        };
        let bk = self.bucket();
        let suma = self.producto_punto(yo, rival, bk);
        let salida = suma / QA as i64 + self.red.l1b[bk] as i64;
        (salida * scale() as i64) as f32 / (QA * QB) as f32
    }

    /// Indice del output bucket segun el material. Misma formula que
    /// `MaterialCount<N>` de bullet: (piezas_totales - 2) / ceil(32/N),
    /// contando TODAS las piezas, reyes incluidos. Con redes de 1 bucket
    /// (formato viejo) siempre es 0.
    #[inline(always)]
    fn bucket(&self) -> usize {
        let b = self.red.buckets;
        if b == 1 {
            return 0;
        }
        let divisor = 32usize.div_ceil(b);
        let idx = (self.piezas.max(2) as usize - 2) / divisor;
        idx.min(b - 1)
    }

    /// Capa de salida: SCReLU (clamp 0..QA, al cuadrado) por los pesos l1w,
    /// sumando las dos perspectivas. Es el mismo papel que `Red::salida` en
    /// neural.rs, que el perfilado senalo como ~80% del tiempo de busqueda:
    /// se ejecuta COMPLETA (h*2 = 1024 terminos) en cada evaluacion.
    #[inline(always)]
    fn producto_punto(&self, yo: &[i16; H_MAX], rival: &[i16; H_MAX], bucket: usize) -> i64 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // RANGOS (verificados sobre los pesos reales y razonados para el
            // caso peor posible del formato):
            //   v = clamp(acumulador, 0, 255)   ->  0..255
            //   v*v                             ->  0..65025   (cabe en i32)
            //   w es i16                        ->  -32768..32767
            //   v*v*w                           ->  |.| <= 65025*32768 = 2.13e9
            // Ese producto cabe justo en i32 (limite 2.147e9), asi que cada
            // termino se calcula en i32 con vmulq_s32 sin desbordar NUNCA,
            // sea cual sea la red. Lo que SI desbordaria es la suma de los
            // 1024 terminos (con los pesos reales, max |l1w| = 84, ya son
            // ~5.6e9), asi que los terminos se acumulan ENSANCHANDO a i64 con
            // vpadalq_s32 (suma por pares de i32 -> i64, una instruccion).
            // La suma entera es asociativa: el i64 final es EXACTAMENTE el
            // mismo que el del bucle escalar, sin ninguna aproximacion.
            //
            // CUATRO acumuladores independientes, por el mismo motivo que en
            // neural.rs: con uno solo las multiplicaciones forman una cadena
            // de dependencias y el bucle queda limitado por latencia.
            let cero = vdupq_n_s16(0);
            let tope = vdupq_n_s16(QA as i16);
            let w = self.red.l1w.as_ptr().add(bucket * 2 * H_MAX);
            let mut acc0 = vdupq_n_s64(0);
            let mut acc1 = vdupq_n_s64(0);
            let mut acc2 = vdupq_n_s64(0);
            let mut acc3 = vdupq_n_s64(0);
            // Cierre que procesa 8 neuronas: clamp -> ensanchar a i32 ->
            // v*v -> por el peso -> acumular ensanchando a i64.
            macro_rules! bloque {
                ($src:expr, $peso:expr, $a:expr, $b:expr) => {{
                    let v16 = vminq_s16(vmaxq_s16(vld1q_s16($src), cero), tope);
                    let w16 = vld1q_s16($peso);
                    let v_lo = vmovl_s16(vget_low_s16(v16));
                    let v_hi = vmovl_s16(vget_high_s16(v16));
                    let w_lo = vmovl_s16(vget_low_s16(w16));
                    let w_hi = vmovl_s16(vget_high_s16(w16));
                    let p_lo = vmulq_s32(vmulq_s32(v_lo, v_lo), w_lo);
                    let p_hi = vmulq_s32(vmulq_s32(v_hi, v_hi), w_hi);
                    $a = vpadalq_s32($a, p_lo);
                    $b = vpadalq_s32($b, p_hi);
                }};
            }
            let mut j = 0;
            while j + 8 <= self.red.h {
                bloque!(yo.as_ptr().add(j), w.add(j), acc0, acc1);
                bloque!(rival.as_ptr().add(j), w.add(H_MAX + j), acc2, acc3);
                j += 8;
            }
            let acc = vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3));
            let mut suma = vgetq_lane_s64::<0>(acc) + vgetq_lane_s64::<1>(acc);
            // Cola escalar por si algun dia hay una `h` que no sea multiplo
            // de 8 (hoy solo hay redes de 256 y 512, asi que no se ejecuta).
            let base = bucket * 2 * H_MAX;
            for k in j..self.red.h {
                let v = yo[k].clamp(0, QA as i16) as i64;
                suma += v * v * self.red.l1w[base + k] as i64;
                let u = rival[k].clamp(0, QA as i16) as i64;
                suma += u * u * self.red.l1w[base + H_MAX + k] as i64;
            }
            return suma;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.producto_punto_escalar(yo, rival, bucket)
        }
    }

    /// Camino ESCALAR original de la capa de salida. Se conserva como
    /// referencia de correccion: el test `salida_neon_igual_que_escalar`
    /// compara termino a termino contra la version vectorizada.
    ///
    /// SCReLU: clamp(x, 0, QA)^2. El cuadrado deja la escala en QA^2*QB,
    /// por eso quien llama divide una vez entre QA para volver a QA*QB, que
    /// es la escala en la que bullet guardo l1b.
    #[inline(always)]
    fn producto_punto_escalar(&self, yo: &[i16; H_MAX], rival: &[i16; H_MAX], bucket: usize) -> i64 {
        let w = &self.red.l1w[bucket * 2 * H_MAX..(bucket + 1) * 2 * H_MAX];
        let mut suma: i64 = 0;
        for j in 0..self.red.h {
            let v = yo[j].clamp(0, QA as i16) as i64;
            suma += v * v * w[j] as i64;
            let u = rival[j].clamp(0, QA as i16) as i64;
            suma += u * u * w[H_MAX + j] as i64;
        }
        suma
    }

    /// Igual que `evaluar` pero forzando el camino escalar. Solo para tests.
    #[cfg(test)]
    pub fn evaluar_escalar(&self) -> f32 {
        let (yo, rival) = if self.negras_mueven {
            (&self.persp[1], &self.persp[0])
        } else {
            (&self.persp[0], &self.persp[1])
        };
        let bk = self.bucket();
        let salida = self.producto_punto_escalar(yo, rival, bk) / QA as i64 + self.red.l1b[bk] as i64;
        (salida * scale() as i64) as f32 / (QA * QB) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal;

    fn red() -> Option<&'static RedBullet> {
        // Por defecto, la red que ESTA DESPLEGADA en produccion (512
        // neuronas); MIMOTOR_RED_BULLET permite apuntar a otra.
        let ruta = std::env::var("MIMOTOR_RED_BULLET").unwrap_or_else(|_| {
            "/Users/Tavito/mi-motor-rust-produccion/pesos_amenazas_prueba.bin".to_string()
        });
        let datos = std::fs::read(ruta).ok()?;
        Some(Box::leak(Box::new(RedBullet::cargar_de_bytes(&datos)?)))
    }

    #[test]
    fn incremental_igual_que_recalculo() {
        let Some(red) = red() else {
            eprintln!("sin MIMOTOR_RED_BULLET: test omitido");
            return;
        };
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ];
        for fen in fens {
            let antes = crate::board::Board::from_fen(fen).unwrap();
            let acc = AcumBullet::desde_tablero(red, &antes);
            for mv in generate_legal(&antes) {
                let despues = antes.make_move(&mv);
                let inc = acc.despues_de_jugada(&antes, &despues);
                let re = AcumBullet::desde_tablero(red, &despues);
                assert_eq!(inc.persp, re.persp, "acumulador difiere tras {}", mv.to_uci());
                assert_eq!(inc.negras_mueven, re.negras_mueven);
                assert_eq!(inc.evaluar(), re.evaluar());
            }
        }
    }

    /// Misma propiedad que en `neural.rs`: la busqueda muta el acumulador
    /// in-place y lo deshace invirtiendo los argumentos. Aqui ademas se
    /// comprueba `negras_mueven`, que tambien tiene que volver a su valor.
    #[test]
    fn aplicar_jugada_se_deshace_bit_a_bit() {
        let Some(red) = red() else {
            eprintln!("sin MIMOTOR_RED_BULLET: test omitido");
            return;
        };
        let raiz = crate::board::Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let mut acumulador = AcumBullet::desde_tablero(red, &raiz);
        let original = acumulador.persp;
        let original_turno = acumulador.negras_mueven;
        let mut semilla = 0xBEEF_1234_5678u64;
        let mut pila: Vec<(crate::board::Board, crate::board::Board)> = Vec::new();
        let mut tablero = raiz;
        for _ply in 0..24 {
            let legales = generate_legal(&tablero);
            if legales.is_empty() {
                break;
            }
            semilla ^= semilla << 7;
            semilla ^= semilla >> 9;
            let mv = legales[(semilla as usize) % legales.len()];
            let siguiente = tablero.make_move(&mv);
            acumulador.aplicar_jugada(&tablero, &siguiente);
            let re = AcumBullet::desde_tablero(red, &siguiente);
            assert_eq!(acumulador.persp, re.persp, "in-place difiere tras {}", mv.to_uci());
            assert_eq!(acumulador.negras_mueven, re.negras_mueven);
            pila.push((tablero, siguiente));
            tablero = siguiente;
        }
        while let Some((antes, despues)) = pila.pop() {
            acumulador.aplicar_jugada(&despues, &antes);
            let re = AcumBullet::desde_tablero(red, &antes);
            assert_eq!(acumulador.persp, re.persp, "undo no restauro el padre");
            assert_eq!(acumulador.negras_mueven, re.negras_mueven);
        }
        assert_eq!(acumulador.persp, original);
        assert_eq!(acumulador.negras_mueven, original_turno);
    }

    /// EQUIVALENCIA NUMERICA de la capa de salida vectorizada con NEON
    /// contra el bucle escalar original. Es el test critico del cambio: un
    /// error de indice o un desbordamiento aqui no romperia nada de forma
    /// visible, solo daria evaluaciones sutilmente mal. Se recorren muchas
    /// decenas de posiciones (aperturas conocidas, finales, posiciones
    /// tacticas y partidas aleatorias legales) exigiendo igualdad EXACTA.
    #[test]
    fn salida_neon_igual_que_escalar() {
        let Some(red) = red() else {
            eprintln!("sin red bullet: test omitido");
            return;
        };
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            "rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "8/8/8/8/8/8/6k1/4K2R w K - 0 1",
            "4k3/8/8/8/8/8/8/4K2R b K - 0 1",
            "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
            "2rq1rk1/pp1bppbp/3p1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R b - - 0 1",
            "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
        ];
        let mut comprobadas = 0usize;
        let mut semilla = 0x1234_ABCD_9876u64;
        for fen in fens {
            let mut tablero = crate::board::Board::from_fen(fen).unwrap();
            // La posicion de partida y despues una partida aleatoria legal
            // de hasta 12 jugadas a partir de ella.
            for _ply in 0..12 {
                let acc = AcumBullet::desde_tablero(red, &tablero);
                assert_eq!(
                    acc.evaluar(),
                    acc.evaluar_escalar(),
                    "NEON != escalar en {}",
                    tablero.to_fen()
                );
                comprobadas += 1;
                let legales = generate_legal(&tablero);
                if legales.is_empty() {
                    break;
                }
                semilla ^= semilla << 7;
                semilla ^= semilla >> 9;
                let mv = legales[(semilla as usize) % legales.len()];
                tablero = tablero.make_move(&mv);
            }
        }
        assert!(comprobadas >= 60, "solo se comprobaron {comprobadas} posiciones");
    }

    #[test]
    fn simetria_de_colores() {
        let Some(red) = red() else { return };
        // La misma posicion con los colores invertidos y el tablero
        // reflejado debe dar la MISMA evaluacion (la red es simetrica por
        // construccion: solo ve "yo" y "el rival").
        let pares = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            ),
            (
                "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
                "3qk3/8/8/8/8/8/8/4K3 b - - 0 1",
            ),
        ];
        for (a, b) in pares {
            let ba = crate::board::Board::from_fen(a).unwrap();
            let bb = crate::board::Board::from_fen(b).unwrap();
            let ea = AcumBullet::desde_tablero(red, &ba).evaluar();
            let eb = AcumBullet::desde_tablero(red, &bb).evaluar();
            assert!(
                (ea - eb).abs() < 1.0,
                "asimetria: {a} = {ea}, {b} = {eb}"
            );
        }
    }
}
