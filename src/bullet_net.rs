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

/// Bytes de pesos para una capa oculta de `h` neuronas.
const fn bytes_utiles(h: usize) -> usize {
    (N_ENTRADA * h + h + 2 * h + 1) * 2
}

/// Tamanos de capa oculta admitidos. Se distinguen por el tamano del
/// archivo, que es distinto para cada uno.
const OCULTAS_SOPORTADAS: [usize; 2] = [256, 512];

pub struct RedBullet {
    /// Neuronas de la capa oculta de ESTA red (256 o 512).
    h: usize,
    /// [feature][neurona] -- 768 bloques contiguos de `h` pesos.
    l0w: Vec<i16>,
    l0b: [i16; H_MAX],
    l1w: [i16; 2 * H_MAX],
    l1b: i16,
}

/// Deduce el tamano de capa oculta a partir del tamano del archivo, o None
/// si no corresponde a ninguna arquitectura bullet conocida.
pub fn oculta_para_tamano(n: usize) -> Option<usize> {
    OCULTAS_SOPORTADAS
        .into_iter()
        .find(|&h| n >= bytes_utiles(h) && n - bytes_utiles(h) < ALINEACION)
}

/// True si el tamano del archivo corresponde a esta arquitectura.
pub fn tamano_plausible(n: usize) -> bool {
    oculta_para_tamano(n).is_some()
}

impl RedBullet {
    pub fn cargar_de_bytes(datos: &[u8]) -> Option<RedBullet> {
        let h = oculta_para_tamano(datos.len())?;
        let utiles = bytes_utiles(h);
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
        let l1w_v = leer(cursor, 2 * h);
        cursor += 2 * h * 2;
        let l1b = leer(cursor, 1)[0];

        let mut l0b = [0i16; H_MAX];
        l0b[..h].copy_from_slice(&l0b_v);
        // l1w se guarda como [stm (h) | ntm (h)]; al copiarlo a un array de
        // H_MAX hay que separar los dos bloques para que la mitad "ntm"
        // empiece siempre en H_MAX, no en h.
        let mut l1w = [0i16; 2 * H_MAX];
        l1w[..h].copy_from_slice(&l1w_v[..h]);
        l1w[H_MAX..H_MAX + h].copy_from_slice(&l1w_v[h..]);

        eprintln!("info string NNUE bullet: capa oculta de {h} neuronas");
        Some(RedBullet {
            h,
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
        }
    }

    /// Actualizacion incremental: con features 768 puras basta con mirar,
    /// bitboard por bitboard, que casillas se ocuparon y cuales se
    /// liberaron. Cubre jugada normal, captura, enroque, al paso y
    /// promocion sin necesitar el tipo de jugada.
    pub fn despues_de_jugada(&self, antes: &Board, despues: &Board) -> AcumBullet {
        let mut nuevo = self.clone();
        nuevo.negras_mueven = despues.turn == Color::Black;
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
                    self.red
                        .sumar(&mut nuevo.persp[0], feature(0, color, pt_idx, sq));
                    self.red
                        .sumar(&mut nuevo.persp[1], feature(1, color, pt_idx, sq));
                }
                let mut quitadas = a & !d;
                while quitadas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut quitadas) as usize;
                    self.red
                        .restar(&mut nuevo.persp[0], feature(0, color, pt_idx, sq));
                    self.red
                        .restar(&mut nuevo.persp[1], feature(1, color, pt_idx, sq));
                }
            }
        }
        nuevo
    }

    /// Salida en centipeones desde la perspectiva del lado que mueve
    /// (misma convencion que la evaluacion clasica del motor).
    pub fn evaluar(&self) -> f32 {
        let (yo, rival) = if self.negras_mueven {
            (&self.persp[1], &self.persp[0])
        } else {
            (&self.persp[0], &self.persp[1])
        };
        let w = &self.red.l1w;
        let mut suma: i64 = 0;
        // SCReLU: clamp(x, 0, QA)^2. El cuadrado deja la escala en QA^2*QB,
        // por eso se divide una vez entre QA para volver a QA*QB, que es la
        // escala en la que bullet guardo l1b.
        for j in 0..self.red.h {
            let v = yo[j].clamp(0, QA as i16) as i64;
            suma += v * v * w[j] as i64;
            let u = rival[j].clamp(0, QA as i16) as i64;
            suma += u * u * w[H_MAX + j] as i64;
        }
        let salida = suma / QA as i64 + self.l1b_i64();
        (salida * scale() as i64) as f32 / (QA * QB) as f32
    }

    #[inline(always)]
    fn l1b_i64(&self) -> i64 {
        self.red.l1b as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal;

    fn red() -> Option<&'static RedBullet> {
        let ruta = std::env::var("MIMOTOR_RED_BULLET").ok()?;
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
