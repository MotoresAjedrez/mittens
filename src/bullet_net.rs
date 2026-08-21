// NNUE con la arquitectura ESTANDAR de la industria, entrenada con la
// libreria `bullet` (https://github.com/jw1912/bullet):
//
//     (768 -> H)x2 -> 1,  doble perspectiva,  activacion SCReLU
//     (H = 256, 512 u 1024 segun los pesos cargados; se detecta por el
//     tamano del archivo, ver ARQUITECTURAS_SOPORTADAS mas abajo)
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
///
/// PENDIENTE_ENTRENAR_1024: subido de 512 a 1024 para dejar sitio a la
/// proxima red (mismo formato 768->1024x2->1, 8 output buckets) una vez
/// que el dataset de Lichess eval-db este listo y se entrene con
/// ~/bullet-training. Los pesos de 512 SIGUEN cargando tal cual: el
/// formato se detecta por el TAMANO del archivo (ver
/// `ARQUITECTURAS_SOPORTADAS` y `arquitectura_para_tamano`), asi que subir
/// este maximo no cambia el comportamiento con la red de produccion
/// actual, solo reserva espacio para la que viene. Ver
/// PENDIENTE_ENTRENAR_1024.md en la raiz del repo.
pub const H_MAX: usize = 1024;
const N_ENTRADA: usize = 768;
const QA: i32 = 255;
const QB: i32 = 64;

/// Escala de salida. Debe coincidir con el `eval_scale` con el que se
/// entreno la red: la red aprende `salida ~= score / eval_scale`, asi que
/// para recuperar centipeones hay que multiplicar por ese mismo numero. El
/// archivo de pesos NO lleva este dato dentro, asi que se configura con
/// MITTENS_BULLET_SCALE (por defecto 400, que es lo que usaron todas las
/// redes entrenadas hasta ahora).
fn scale() -> i32 {
    use std::sync::OnceLock;
    static CACHE: OnceLock<i32> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("MITTENS_BULLET_SCALE")
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
///
/// (1024, 8) es PENDIENTE_ENTRENAR_1024: la red de 1024 neuronas aun no
/// existe (falta el dataset + entrenamiento), pero ya se reconoce por
/// tamano de archivo para que, en cuanto aparezcan los pesos entrenados,
/// carguen sin tocar mas codigo. (256, 1) y (512, 1)/(512, 8) son formatos
/// viejos/actual, se conservan intactos.
const ARQUITECTURAS_SOPORTADAS: [(usize, usize); 4] =
    [(256, 1), (512, 1), (512, 8), (1024, 8)];

/// Maximo de output buckets soportado.
pub const B_MAX: usize = 8;

pub struct RedBullet {
    /// Neuronas de la capa oculta de ESTA red (256, 512 o 1024; ver
    /// ARQUITECTURAS_SOPORTADAS).
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

    /// Aplica de UNA SOLA PASADA todas las features que cambian en una
    /// jugada: las de `add` se suman y las de `sub` se restan.
    ///
    /// Antes cada feature hacia su propio recorrido del acumulador: H_MAX
    /// lecturas + H_MAX escrituras POR FEATURE. Una jugada normal cambia 2
    /// features y una captura 3, por cada una de las DOS perspectivas, o sea
    /// entre 4 y 6 pasadas completas de 1 KB cada una por nodo de busqueda.
    /// Aqui el acumulador se lee y escribe UNA vez (en registros NEON, por
    /// bloques de 32 valores) y de cada feature solo se leen sus pesos.
    ///
    /// El resultado es BIT A BIT IDENTICO al de antes: la suma envolvente de
    /// i16 es asociativa y conmutativa, asi que ni el orden ni el
    /// agrupamiento cambian nada, ni siquiera si hubiera desbordamiento.
    /// Igual que `aplicar_features` pero LEYENDO de `src` y ESCRIBIENDO en
    /// `dst`, en una sola pasada. Es lo que permite que la pila calcule el
    /// hijo sin copiar antes al padre: sin esto habria que hacer
    /// `dst = src` (memcpy de H_MAX*2*2 bytes) y despues aplicar in-place,
    /// que es mas trafico de memoria que la pasada fusionada.
    ///
    /// Resultado bit-identico a copiar y aplicar.
    fn aplicar_features_desde(
        &self,
        src: &[i16; H_MAX],
        dst: &mut [i16; H_MAX],
        add: &[u16],
        sub: &[u16],
    ) {
        debug_assert!(
            add.iter().chain(sub).all(|&f| (f as usize) < N_ENTRADA),
            "feature fuera de rango en aplicar_features_desde"
        );
        #[cfg(target_arch = "aarch64")]
        if self.h % 32 == 0 {
            unsafe {
                use std::arch::aarch64::*;
                let w0 = self.l0w.as_ptr();
                let h = self.h;
                let mut t = 0;
                while t < h {
                    let ps = src.as_ptr().add(t);
                    let pd = dst.as_mut_ptr().add(t);
                    let mut r0 = vld1q_s16(ps);
                    let mut r1 = vld1q_s16(ps.add(8));
                    let mut r2 = vld1q_s16(ps.add(16));
                    let mut r3 = vld1q_s16(ps.add(24));
                    for &f in add {
                        let w = w0.add(f as usize * h + t);
                        r0 = vaddq_s16(r0, vld1q_s16(w));
                        r1 = vaddq_s16(r1, vld1q_s16(w.add(8)));
                        r2 = vaddq_s16(r2, vld1q_s16(w.add(16)));
                        r3 = vaddq_s16(r3, vld1q_s16(w.add(24)));
                    }
                    for &f in sub {
                        let w = w0.add(f as usize * h + t);
                        r0 = vsubq_s16(r0, vld1q_s16(w));
                        r1 = vsubq_s16(r1, vld1q_s16(w.add(8)));
                        r2 = vsubq_s16(r2, vld1q_s16(w.add(16)));
                        r3 = vsubq_s16(r3, vld1q_s16(w.add(24)));
                    }
                    vst1q_s16(pd, r0);
                    vst1q_s16(pd.add(8), r1);
                    vst1q_s16(pd.add(16), r2);
                    vst1q_s16(pd.add(24), r3);
                    t += 32;
                }
            }
            return;
        }
        // Camino portable: copiar y aplicar in-place da exactamente lo mismo.
        dst[..self.h].copy_from_slice(&src[..self.h]);
        self.aplicar_features(dst, add, sub);
    }

    fn aplicar_features(&self, acc: &mut [i16; H_MAX], add: &[u16], sub: &[u16]) {
        // El camino escalar indexa por `columna()`, que hace slicing y
        // PANICA si la feature se sale de rango. El camino NEON usa punteros
        // crudos, asi que una feature fuera de rango leeria memoria ajena en
        // silencio. Por construccion `feature()` devuelve 0..768 siempre, y
        // este assert deja constancia de esa precondicion para que un cambio
        // futuro en el esquema de features falle en los tests en vez de
        // corromper la evaluacion.
        debug_assert!(
            add.iter().chain(sub).all(|&f| (f as usize) < N_ENTRADA),
            "feature fuera de rango en aplicar_features"
        );
        // Bloques de 32 i16 = 4 registros NEON. Las redes reales tienen
        // h = 256, 512 o 1024 (todas multiplos de 32); si algun dia hubiera
        // otra, se cae al camino escalar, que da exactamente lo mismo.
        #[cfg(target_arch = "aarch64")]
        if self.h % 32 == 0 {
            unsafe {
                use std::arch::aarch64::*;
                let w0 = self.l0w.as_ptr();
                let h = self.h;
                let mut t = 0;
                while t < h {
                    let p = acc.as_mut_ptr().add(t);
                    let mut r0 = vld1q_s16(p);
                    let mut r1 = vld1q_s16(p.add(8));
                    let mut r2 = vld1q_s16(p.add(16));
                    let mut r3 = vld1q_s16(p.add(24));
                    for &f in add {
                        let w = w0.add(f as usize * h + t);
                        r0 = vaddq_s16(r0, vld1q_s16(w));
                        r1 = vaddq_s16(r1, vld1q_s16(w.add(8)));
                        r2 = vaddq_s16(r2, vld1q_s16(w.add(16)));
                        r3 = vaddq_s16(r3, vld1q_s16(w.add(24)));
                    }
                    for &f in sub {
                        let w = w0.add(f as usize * h + t);
                        r0 = vsubq_s16(r0, vld1q_s16(w));
                        r1 = vsubq_s16(r1, vld1q_s16(w.add(8)));
                        r2 = vsubq_s16(r2, vld1q_s16(w.add(16)));
                        r3 = vsubq_s16(r3, vld1q_s16(w.add(24)));
                    }
                    vst1q_s16(p, r0);
                    vst1q_s16(p.add(8), r1);
                    vst1q_s16(p.add(16), r2);
                    vst1q_s16(p.add(24), r3);
                    t += 32;
                }
            }
            return;
        }
        for &f in add {
            self.sumar(acc, f as usize);
        }
        for &f in sub {
            self.restar(acc, f as usize);
        }
    }
}

/// Buffer de features que cambian en una jugada. Cota holgada: el maximo
/// real es 4 casillas (enroque: rey y torre, origen y destino), y con las
/// dos perspectivas separadas cada lista no pasa de 4 entradas. 8 deja
/// margen y sigue cabiendo en registros/pila sin coste.
const MAX_CAMBIOS: usize = 8;

#[derive(Default)]
struct Cambios {
    add: arrayvec::ArrayVec<u16, MAX_CAMBIOS>,
    sub: arrayvec::ArrayVec<u16, MAX_CAMBIOS>,
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

/// Un nivel de la pila: el estado completo del acumulador en un ply.
#[derive(Clone)]
struct Nivel {
    /// [0] = acumulador visto por las blancas, [1] = por las negras.
    /// Solo las primeras `red.h` posiciones son significativas.
    persp: [[i16; H_MAX]; 2],
    negras_mueven: bool,
    /// Piezas totales en el tablero (reyes incluidos). Solo se usa para
    /// elegir el output bucket; se mantiene incrementalmente.
    piezas: u8,
}

/// Acumulador NNUE con PILA POR PLY.
///
/// Antes era un unico buffer mutado in-place: bajar a un hijo aplicaba el
/// delta y volver aplicaba el delta invertido, o sea DOS pasadas completas
/// del acumulador por jugada. Perfilando, `aplicar_features` era el 34,5%
/// del tiempo de busqueda, asi que esa segunda pasada sola costaba ~17%.
///
/// Con la pila, el hijo se escribe en el slot siguiente leyendo del padre en
/// una sola pasada (`entrar`) y volver es decrementar un indice (`salir`).
/// El resultado es BIT-IDENTICO: son los mismos valores, guardados en otro
/// lado.
#[derive(Clone)]
pub struct AcumBullet {
    red: &'static RedBullet,
    pila: Vec<Nivel>,
    nivel: usize,
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
        let inicial = Self::nivel_desde_tablero(red, b);
        // Capacidad inicial holgada para la profundidad tipica; `entrar`
        // crece la pila sola si hiciera falta.
        let mut pila = Vec::with_capacity(96);
        pila.push(inicial);
        AcumBullet { red, pila, nivel: 0 }
    }

    /// Construye UN nivel recalculando el acumulador entero desde el tablero.
    fn nivel_desde_tablero(red: &'static RedBullet, b: &Board) -> Nivel {
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
        Nivel {
            persp,
            negras_mueven: b.turn == Color::Black,
            piezas: contar_piezas(b),
        }
    }

    /// Nivel actual de la pila (el acumulador vigente).
    #[inline(always)]
    fn actual(&self) -> &Nivel {
        &self.pila[self.nivel]
    }

    /// Lado que mueve en el nivel vigente (lo usan los tests).
    #[inline]
    pub fn negras_mueven(&self) -> bool {
        self.actual().negras_mueven
    }

    /// Piezas contadas en el nivel vigente (lo usan los tests).
    #[inline]
    pub fn piezas(&self) -> u8 {
        self.actual().piezas
    }

    /// Acceso de solo lectura al acumulador vigente (lo usan los tests de
    /// equivalencia incremental-vs-recalculado).
    #[inline]
    pub fn persp(&self) -> &[[i16; H_MAX]; 2] {
        &self.pila[self.nivel].persp
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
    /// Calcula las features que cambian entre dos tableros, sin tocar
    /// ningun acumulador. Se extrajo de `aplicar_jugada` para que la version
    /// in-place y la version con pila (`entrar`) compartan exactamente la
    /// misma logica de deltas en vez de duplicarla.
    ///
    /// Devuelve (cambios perspectiva 0, cambios perspectiva 1, delta de
    /// piezas, desbordado). `desbordado` solo puede darse si el llamador pasa
    /// dos tableros que no se diferencian en una jugada legal.
    fn calcular_cambios(antes: &Board, despues: &Board) -> (Cambios, Cambios, i32, bool) {
        let mut delta_piezas: i32 = 0;
        let mut c0 = Cambios::default();
        let mut c1 = Cambios::default();
        let mut desbordado = false;

        for color in 0..2usize {
            for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
                let a = antes.pieces[color][pt as usize];
                let d = despues.pieces[color][pt as usize];
                if a == d {
                    continue;
                }
                let mut anadidas = d & !a;
                delta_piezas += anadidas.count_ones() as i32;
                while anadidas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut anadidas) as usize;
                    let f0 = feature(0, color, pt_idx, sq) as u16;
                    let f1 = feature(1, color, pt_idx, sq) as u16;
                    if c0.add.try_push(f0).is_err() || c1.add.try_push(f1).is_err() {
                        desbordado = true;
                    }
                }
                let mut quitadas = a & !d;
                delta_piezas -= quitadas.count_ones() as i32;
                while quitadas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut quitadas) as usize;
                    let f0 = feature(0, color, pt_idx, sq) as u16;
                    let f1 = feature(1, color, pt_idx, sq) as u16;
                    if c0.sub.try_push(f0).is_err() || c1.sub.try_push(f1).is_err() {
                        desbordado = true;
                    }
                }
            }
        }
        (c0, c1, delta_piezas, desbordado)
    }

    /// Actualizacion incremental IN-PLACE sobre el nivel actual.
    pub fn aplicar_jugada(&mut self, antes: &Board, despues: &Board) {
        let red = self.red;
        let (c0, c1, delta_piezas, desbordado) = Self::calcular_cambios(antes, despues);
        if desbordado {
            // Imposible con jugadas legales (maximo 4 casillas cambian), pero
            // si un llamador pasara dos tableros arbitrarios se recalcula
            // entero en vez de aplicar un delta incompleto.
            let n = AcumBullet::nivel_desde_tablero(red, despues);
            self.pila[self.nivel] = n;
            return;
        }
        let nivel = &mut self.pila[self.nivel];
        nivel.negras_mueven = despues.turn == Color::Black;
        nivel.piezas = (nivel.piezas as i32 + delta_piezas) as u8;
        debug_assert_eq!(nivel.piezas, contar_piezas(despues));
        red.aplicar_features(&mut nivel.persp[0], &c0.add, &c0.sub);
        red.aplicar_features(&mut nivel.persp[1], &c1.add, &c1.sub);
    }

    /// Baja un nivel: calcula el acumulador del hijo LEYENDO el del padre y
    /// ESCRIBIENDO en el slot siguiente de la pila, en una sola pasada.
    ///
    /// Este es el punto de la pila: antes, volver al padre exigia aplicar el
    /// delta invertido (`aplicar_jugada(despues, antes)`), o sea que cada
    /// jugada pagaba DOS pasadas completas del acumulador. Con la pila el
    /// padre nunca se toca, asi que `salir` es solo decrementar un indice.
    pub fn entrar(&mut self, antes: &Board, despues: &Board) {
        let red = self.red;
        let (c0, c1, delta_piezas, desbordado) = Self::calcular_cambios(antes, despues);
        if self.nivel + 1 >= self.pila.len() {
            let copia = self.pila[self.nivel].clone();
            self.pila.push(copia);
        }
        if desbordado {
            let n = AcumBullet::nivel_desde_tablero(red, despues);
            self.pila[self.nivel + 1] = n;
            self.nivel += 1;
            return;
        }
        let (izq, der) = self.pila.split_at_mut(self.nivel + 1);
        let padre = &izq[self.nivel];
        let hijo = &mut der[0];
        hijo.negras_mueven = despues.turn == Color::Black;
        hijo.piezas = (padre.piezas as i32 + delta_piezas) as u8;
        debug_assert_eq!(hijo.piezas, contar_piezas(despues));
        red.aplicar_features_desde(&padre.persp[0], &mut hijo.persp[0], &c0.add, &c0.sub);
        red.aplicar_features_desde(&padre.persp[1], &mut hijo.persp[1], &c1.add, &c1.sub);
        self.nivel += 1;
    }

    /// Vuelve al nivel del padre. Coste cero: el padre quedo intacto.
    #[inline]
    pub fn salir(&mut self) {
        debug_assert!(self.nivel > 0, "salir sin entrar");
        self.nivel -= 1;
    }

    /// Salida en centipeones desde la perspectiva del lado que mueve
    /// (misma convencion que la evaluacion clasica del motor).
    pub fn evaluar(&self) -> f32 {
        let n = self.actual();
        let (yo, rival) = if n.negras_mueven {
            (&n.persp[1], &n.persp[0])
        } else {
            (&n.persp[0], &n.persp[1])
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
        let idx = (self.actual().piezas.max(2) as usize - 2) / divisor;
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
            // sea cual sea la red. Lo que SI desbordaria en i32 es la suma
            // de los 2*h terminos (con la red de 512 y sus pesos reales,
            // max |l1w| = 84, la suma de los 1024 terminos ya da ~5.6e9;
            // con h=1024 serian hasta 2048 terminos, hasta el doble), asi
            // que los terminos se acumulan ENSANCHANDO a i64 con
            // vpadalq_s32 (suma por pares de i32 -> i64, una instruccion).
            // i64 tiene margen de sobra incluso en el peor caso teorico
            // (2*H_MAX terminos de hasta 2.13e9 cada uno = ~4.4e12 con
            // H_MAX=1024, muy por debajo de i64::MAX ~9.2e18), asi que el
            // margen no depende de cuanto crezca `h`. La suma entera es
            // asociativa: el i64 final es EXACTAMENTE el mismo que el del
            // bucle escalar, sin ninguna aproximacion.
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
            // REORDENAMIENTO EXACTO del producto: v*v*w se calcula como
            // (v*w)*v en vez de (v*v)*w. Aritmetica entera: la
            // multiplicacion es asociativa y conmutativa, asi que el i32
            // resultante es BIT A BIT el mismo. La ventaja es que `v*w`
            // ensanchando de i16 a i32 son las instrucciones SMULL/SMULL2
            // (`vmull_s16` / `vmull_high_s16`): UNA operacion que hace el
            // ensanchado y el producto de una, en vez de ensanchar `v` y `w`
            // por separado (dos SXTL) y multiplicar despues. Quedan 10
            // operaciones aritmeticas por cada 8 neuronas en vez de 12
            // (-17%), y esta es la capa que se ejecuta COMPLETA (2*h = 2048
            // terminos) en cada evaluacion: el 22% del tiempo de busqueda.
            //
            // Rangos (peor caso teorico, no de los pesos reales):
            //   v = clamp(acumulador, 0, 255)  ->  0..255
            //   v*w  con |w| <= 32768          ->  |.| <= 8_355_840  (cabe en i32)
            //   (v*w)*v                        ->  |.| <= 2.13e9     (cabe en i32)
            // Los mismos limites que el orden anterior; ninguno de los dos
            // desborda nunca, sea cual sea la red.
            macro_rules! bloque {
                ($src:expr, $peso:expr, $a:expr, $b:expr) => {{
                    let v16 = vminq_s16(vmaxq_s16(vld1q_s16($src), cero), tope);
                    let w16 = vld1q_s16($peso);
                    let vw_lo = vmull_s16(vget_low_s16(v16), vget_low_s16(w16));
                    let vw_hi = vmull_high_s16(v16, w16);
                    let v_lo = vmovl_s16(vget_low_s16(v16));
                    let v_hi = vmovl_high_s16(v16);
                    let p_lo = vmulq_s32(vw_lo, v_lo);
                    let p_hi = vmulq_s32(vw_hi, v_hi);
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
        let n = self.actual();
        let (yo, rival) = if n.negras_mueven {
            (&n.persp[1], &n.persp[0])
        } else {
            (&n.persp[0], &n.persp[1])
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
        // Por defecto, la red que ESTA DESPLEGADA en produccion: los MISMOS
        // bytes que `include_bytes!` embebe en el binario (ver neural.rs,
        // PESOS_EMBEBIDOS), no una copia suelta en una ruta absoluta de una
        // maquina concreta -- si esa ruta no existia, todos estos tests se
        // "omitian" en silencio y no verificaban nada.
        // MITTENS_RED_BULLET permite apuntar a otra red.
        static RED: std::sync::OnceLock<Option<&'static RedBullet>> = std::sync::OnceLock::new();
        *RED.get_or_init(|| {
            let datos = match std::env::var("MITTENS_RED_BULLET") {
                Ok(ruta) => std::fs::read(ruta).ok()?,
                Err(_) => include_bytes!("../pesos_bullet_1024_buckets8.bin").to_vec(),
            };
            Some(Box::leak(Box::new(RedBullet::cargar_de_bytes(&datos)?)))
        })
    }

    #[test]
    fn incremental_igual_que_recalculo() {
        let Some(red) = red() else {
            eprintln!("sin MITTENS_RED_BULLET: test omitido");
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
                assert_eq!(inc.persp(), re.persp(), "acumulador difiere tras {}", mv.to_uci());
                assert_eq!(inc.negras_mueven(), re.negras_mueven());
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
            eprintln!("sin MITTENS_RED_BULLET: test omitido");
            return;
        };
        let raiz = crate::board::Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let mut acumulador = AcumBullet::desde_tablero(red, &raiz);
        let original = *acumulador.persp();
        let original_turno = acumulador.negras_mueven();
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
            assert_eq!(acumulador.persp(), re.persp(), "in-place difiere tras {}", mv.to_uci());
            assert_eq!(acumulador.negras_mueven(), re.negras_mueven());
            pila.push((tablero, siguiente));
            tablero = siguiente;
        }
        while let Some((antes, despues)) = pila.pop() {
            acumulador.aplicar_jugada(&despues, &antes);
            let re = AcumBullet::desde_tablero(red, &antes);
            assert_eq!(acumulador.persp(), re.persp(), "undo no restauro el padre");
            assert_eq!(acumulador.negras_mueven(), re.negras_mueven());
        }
        assert_eq!(*acumulador.persp(), original);
        assert_eq!(acumulador.negras_mueven(), original_turno);
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

    /// CONSISTENCIA MASIVA del acumulador incremental (ronda2).
    ///
    /// Los tests anteriores cubrian unas pocas posiciones y una linea
    /// aleatoria corta. Este recorre ~20 mil posiciones distintas, con
    /// mutacion in-place + undo (exactamente el patron que usa la busqueda:
    /// `entrar_hijo` / `salir_hijo` en search.rs) y, EN CADA UNA, compara:
    ///   * las dos perspectivas del acumulador, bit a bit, contra el
    ///     recalculo completo desde el tablero,
    ///   * `negras_mueven` y `piezas` (este ultimo elige el output bucket:
    ///     si se desincroniza, la evaluacion usa la capa de salida
    ///     equivocada sin que nada mas se note),
    ///   * la evaluacion en centipeones, con igualdad EXACTA.
    /// Al desandar cada linea se vuelve a comprobar todo, y al final se exige
    /// que el acumulador haya vuelto bit a bit al de la raiz.
    ///
    /// Las posiciones semilla estan elegidas para que la caminata aleatoria
    /// tenga que pasar por enroques, capturas al paso, promociones (a las
    /// cuatro piezas) y capturas; el test lo VERIFICA al final en vez de
    /// confiar en que salieron.
    #[test]
    fn consistencia_masiva_incremental_vs_recalculo() {
        let Some(red) = red() else {
            panic!("la red embebida deberia cargar siempre");
        };
        let semillas_fen = [
            // Apertura estandar: enroques por ambos lados, capturas al paso.
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            // "Kiwipete": enroques disponibles, mucha captura y deslizantes.
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // Final de peones: promociones garantizadas en pocas jugadas.
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            // Peones a un paso de coronar, para ambos bandos.
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            // Con casilla al paso ya disponible en la raiz.
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            // Muchos peones a punto de promocionar en las dos alas.
            "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
            // Final escaso: fuerza buckets de material bajos.
            "8/8/4k3/8/8/3K4/8/7R w - - 0 1",
        ];

        let mut semilla = 0x0BAD_C0DE_5EED_1234u64;
        let mut rand = move || {
            semilla ^= semilla << 13;
            semilla ^= semilla >> 7;
            semilla ^= semilla << 17;
            semilla
        };

        let mut posiciones = 0usize;
        let mut vistos_enroque = 0usize;
        let mut vistos_al_paso = 0usize;
        let mut vistas_promociones = [0usize; 4];
        let mut vistas_capturas = 0usize;

        for fen in semillas_fen {
            let raiz = crate::board::Board::from_fen(fen).unwrap();
            let mut acumulador = AcumBullet::desde_tablero(red, &raiz);
            let raiz_persp = *acumulador.persp();
            let raiz_piezas = acumulador.piezas();
            let raiz_turno = acumulador.negras_mueven();

            for _linea in 0..60 {
                let mut pila: Vec<(crate::board::Board, crate::board::Board)> = Vec::new();
                let mut tablero = raiz;
                for _ply in 0..50 {
                    let legales = generate_legal(&tablero);
                    if legales.is_empty() {
                        break;
                    }
                    let mv = legales[(rand() >> 1) as usize % legales.len()];
                    match mv.flag {
                        crate::types::MoveFlag::CastleKing
                        | crate::types::MoveFlag::CastleQueen => vistos_enroque += 1,
                        crate::types::MoveFlag::EnPassant => {
                            vistos_al_paso += 1;
                            vistas_capturas += 1;
                        }
                        crate::types::MoveFlag::Capture => vistas_capturas += 1,
                        _ => {}
                    }
                    if let Some(p) = mv.promotion {
                        // Knight..Queen -> 0..3 (el rey y el peon no promocionan).
                        let idx = match p {
                            crate::types::PieceType::Knight => 0,
                            crate::types::PieceType::Bishop => 1,
                            crate::types::PieceType::Rook => 2,
                            _ => 3,
                        };
                        vistas_promociones[idx] += 1;
                    }

                    let siguiente = tablero.make_move(&mv);
                    acumulador.aplicar_jugada(&tablero, &siguiente);
                    comparar(&acumulador, red, &siguiente, "bajada", &mv);
                    posiciones += 1;

                    pila.push((tablero, siguiente));
                    tablero = siguiente;
                }
                // Desandar la linea completa, igual que `salir_hijo`.
                while let Some((antes, despues)) = pila.pop() {
                    acumulador.aplicar_jugada(&despues, &antes);
                    comparar(
                        &acumulador,
                        red,
                        &antes,
                        "undo",
                        &crate::types::Move::new(0, 0, None, crate::types::MoveFlag::Quiet),
                    );
                    posiciones += 1;
                }
                assert_eq!(*acumulador.persp(), raiz_persp, "la raiz no se restauro: {fen}");
                assert_eq!(acumulador.piezas(), raiz_piezas);
                assert_eq!(acumulador.negras_mueven(), raiz_turno);
            }
        }

        assert!(
            posiciones >= 20_000,
            "cobertura insuficiente: solo {posiciones} posiciones"
        );
        assert!(vistos_enroque > 0, "ningun enroque en la muestra");
        assert!(vistos_al_paso > 0, "ninguna captura al paso en la muestra");
        assert!(vistas_capturas > 100, "pocas capturas: {vistas_capturas}");
        for (i, n) in vistas_promociones.iter().enumerate() {
            assert!(*n > 0, "ninguna promocion del tipo {i} en la muestra");
        }
    }

    /// Compara el acumulador incremental con el recalculo completo desde
    /// `tablero`, en todos sus campos observables.
    fn comparar(
        acumulador: &AcumBullet,
        red: &'static RedBullet,
        tablero: &crate::board::Board,
        fase: &str,
        mv: &crate::types::Move,
    ) {
        let re = AcumBullet::desde_tablero(red, tablero);
        assert_eq!(
            acumulador.persp()[0], re.persp()[0],
            "[{fase}] perspectiva blanca difiere tras {} en {}",
            mv.to_uci(),
            tablero.to_fen()
        );
        assert_eq!(
            acumulador.persp()[1], re.persp()[1],
            "[{fase}] perspectiva negra difiere tras {} en {}",
            mv.to_uci(),
            tablero.to_fen()
        );
        assert_eq!(
            acumulador.negras_mueven(), re.negras_mueven(),
            "[{fase}] turno desincronizado en {}",
            tablero.to_fen()
        );
        assert_eq!(
            acumulador.piezas(), re.piezas(),
            "[{fase}] conteo de piezas (output bucket) desincronizado en {}",
            tablero.to_fen()
        );
        assert_eq!(
            acumulador.bucket(),
            re.bucket(),
            "[{fase}] output bucket distinto en {}",
            tablero.to_fen()
        );
        assert_eq!(
            acumulador.evaluar(),
            re.evaluar(),
            "[{fase}] evaluacion distinta en {}",
            tablero.to_fen()
        );
        assert_eq!(
            acumulador.evaluar(),
            acumulador.evaluar_escalar(),
            "[{fase}] NEON != escalar en {}",
            tablero.to_fen()
        );
    }

    /// El acumulador tambien tiene que sobrevivir a la jugada NULA (pasar
    /// turno), que la busqueda aplica por la misma via `aplicar_jugada`: las
    /// piezas no cambian pero SI el lado que mueve, y con el la mitad del
    /// acumulador que va a "yo" y la que va al "rival" en la capa de salida.
    #[test]
    fn jugada_nula_consistente_y_reversible() {
        let Some(red) = red() else {
            panic!("la red embebida deberia cargar siempre");
        };
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        for fen in fens {
            let b = crate::board::Board::from_fen(fen).unwrap();
            let mut acumulador = AcumBullet::desde_tablero(red, &b);
            let original = *acumulador.persp();
            let nulo = b.make_null_move();
            acumulador.aplicar_jugada(&b, &nulo);
            let re = AcumBullet::desde_tablero(red, &nulo);
            assert_eq!(acumulador.persp(), re.persp(), "nula: acumulador difiere en {fen}");
            assert_eq!(acumulador.negras_mueven(), re.negras_mueven());
            assert_eq!(acumulador.piezas(), re.piezas());
            assert_eq!(acumulador.evaluar(), re.evaluar(), "nula: eval difiere en {fen}");
            acumulador.aplicar_jugada(&nulo, &b);
            assert_eq!(*acumulador.persp(), original, "nula: undo no restauro {fen}");
            assert_eq!(acumulador.negras_mueven(), b.turn == Color::Black);
        }
    }

    /// Microbench AISLADO de la actualizacion del acumulador: la version de
    /// una sola pasada (`aplicar_features`) contra la de una pasada completa
    /// del acumulador POR FEATURE (`sumar`/`restar`, como estaba antes).
    /// Se mide aqui y no con el bench de la busqueda porque en la busqueda
    /// esta funcion es ~6% del tiempo total y el ruido de la maquina se lo
    /// come. Ademas se exige que ambos caminos den el MISMO acumulador.
    #[test]
    fn microbench_actualizacion_una_pasada_vs_por_feature() {
        let Some(red) = red() else { return };
        let b = crate::board::Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let base = AcumBullet::desde_tablero(red, &b);
        // Caso tipico de captura: 3 features cambian (origen, destino, pieza
        // capturada).
        let add: [u16; 1] = [feature(0, 0, 4, 27) as u16];
        let sub: [u16; 2] = [feature(0, 0, 4, 21) as u16, feature(0, 1, 3, 27) as u16];

        let repeticiones = 200_000;
        let mut a = base.persp()[0];
        let t0 = std::time::Instant::now();
        for _ in 0..repeticiones {
            red.aplicar_features(&mut a, &add, &sub);
            red.aplicar_features(&mut a, &sub, &add); // deshacer
            std::hint::black_box(&a);
        }
        let t_lote = t0.elapsed();

        let mut c = base.persp()[0];
        let t0 = std::time::Instant::now();
        for _ in 0..repeticiones {
            for &f in &add {
                red.sumar(&mut c, f as usize);
            }
            for &f in &sub {
                red.restar(&mut c, f as usize);
            }
            for &f in &add {
                red.restar(&mut c, f as usize);
            }
            for &f in &sub {
                red.sumar(&mut c, f as usize);
            }
            std::hint::black_box(&c);
        }
        let t_feature = t0.elapsed();

        // Igualdad exacta: el agrupamiento no cambia el resultado.
        let mut x = base.persp()[0];
        let mut y = base.persp()[0];
        red.aplicar_features(&mut x, &add, &sub);
        for &f in &add {
            red.sumar(&mut y, f as usize);
        }
        for &f in &sub {
            red.restar(&mut y, f as usize);
        }
        assert_eq!(x, y, "una pasada != por feature");

        eprintln!(
            "acumulador: una pasada {:?} vs por feature {:?} ({:.2}x)",
            t_lote,
            t_feature,
            t_feature.as_secs_f64() / t_lote.as_secs_f64().max(1e-9)
        );
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

    /// El camino NUEVO (pila) tiene que dar exactamente lo mismo que
    /// recalcular desde cero, y `salir` tiene que devolver el acumulador del
    /// padre BIT A BIT. Los tests viejos solo cubrian `aplicar_jugada`
    /// (in-place), que ya no es lo que usa la busqueda.
    #[test]
    fn pila_entrar_salir_coincide_con_recalculo() {
        let Some(red) = red() else { return };
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pp2nppp/3p1n2/2pP4/4P3/2N2N2/PP3PPP/R1BQKB1R w KQkq - 0 8",
            // Con derechos de enroque y al paso, que mueven varias features.
            "r3k2r/pppq1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPPBQPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        ];
        for fen in fens {
            let b = Board::from_fen(fen).unwrap();
            let mut acum = AcumBullet::desde_tablero(red, &b);
            let raiz = *acum.persp();
            let raiz_piezas = acum.piezas();
            let raiz_turno = acum.negras_mueven();

            let mut jugadas: arrayvec::ArrayVec<crate::types::Move, 256> =
                arrayvec::ArrayVec::new();
            crate::movegen::generate_legal_into(&b, &mut jugadas);
            for mv in jugadas.iter().take(12) {
                let mut hijo = b.clone();
                hijo.make_move(mv);

                acum.entrar(&b, &hijo);
                let re = AcumBullet::desde_tablero(red, &hijo);
                assert_eq!(
                    acum.persp(),
                    re.persp(),
                    "pila: acumulador difiere tras {} en {fen}",
                    mv.to_uci()
                );
                assert_eq!(acum.piezas(), re.piezas(), "pila: piezas mal tras {}", mv.to_uci());
                assert_eq!(
                    acum.negras_mueven(),
                    re.negras_mueven(),
                    "pila: turno mal tras {}",
                    mv.to_uci()
                );

                acum.salir();
                assert_eq!(*acum.persp(), raiz, "pila: salir no restauro la raiz en {fen}");
                assert_eq!(acum.piezas(), raiz_piezas, "pila: salir no restauro piezas");
                assert_eq!(acum.negras_mueven(), raiz_turno, "pila: salir no restauro turno");
            }
        }
    }

    /// La pila tiene que aguantar bajar muchos plies seguidos (crece sola) y
    /// devolver la raiz intacta al desandar todo.
    #[test]
    fn pila_soporta_profundidad_y_desanda_bien() {
        let Some(red) = red() else { return };
        let fen = "r1bqkb1r/pp2nppp/3p1n2/2pP4/4P3/2N2N2/PP3PPP/R1BQKB1R w KQkq - 0 8";
        let mut b = Board::from_fen(fen).unwrap();
        let mut acum = AcumBullet::desde_tablero(red, &b);
        let raiz = *acum.persp();

        let mut historia = Vec::new();
        // Baja 120 plies siguiendo siempre la primera jugada legal: mas que
        // la capacidad inicial de la pila, asi que fuerza el crecimiento.
        for _ in 0..120 {
            let mut jugadas: arrayvec::ArrayVec<crate::types::Move, 256> =
                arrayvec::ArrayVec::new();
            crate::movegen::generate_legal_into(&b, &mut jugadas);
            let Some(&mv) = jugadas.first() else { break };
            let mut hijo = b.clone();
            hijo.make_move(&mv);
            acum.entrar(&b, &hijo);
            historia.push(b.clone());
            b = hijo;
        }
        let re = AcumBullet::desde_tablero(red, &b);
        assert_eq!(acum.persp(), re.persp(), "pila profunda: hoja mal");

        while historia.pop().is_some() {
            acum.salir();
        }
        assert_eq!(*acum.persp(), raiz, "pila profunda: no volvio a la raiz");
    }
}
