// Libro de aperturas en formato Polyglot estandar (binario, entradas de 16
// bytes: clave Zobrist propia del formato (8) + jugada codificada (2) + peso
// (2) + "learn" (4), todo big-endian). El hash de Polyglot es TOTALMENTE
// independiente del zobrist interno del motor -- usa su propia tabla fija de
// 781 numeros aleatorios (POLYGLOT_RANDOM, en polyglot_random.rs, extraida
// programaticamente del codigo fuente de referencia de python-chess para
// evitar errores de transcripcion).

use crate::board::{Board, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};
use crate::movegen::generate_legal;
use crate::polyglot_random::POLYGLOT_RANDOM;
use crate::types::{Color, Move, PieceType, file_of, make_square, rank_of};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
struct Entrada {
    key: u64,
    raw_move: u16,
    weight: u16,
}

static LIBRO: OnceLock<Vec<Entrada>> = OnceLock::new();
// Activado por defecto (si hay libro cargado); "setoption name OwnBook value
// false" lo apaga sin necesidad de descargar el libro. Global en vez de un
// campo por Searcher para no tener que hilarlo por los dos caminos de
// busqueda (un solo hilo y Lazy SMP).
static ACTIVO: AtomicBool = AtomicBool::new(true);

pub fn set_activo(v: bool) {
    ACTIVO.store(v, Ordering::Relaxed);
}

pub fn init(path: &str) -> Result<usize, String> {
    if LIBRO.get().is_some() {
        return Err("ya hay un libro cargado; reinicia el motor para cambiar BookPath".to_string());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.is_empty() || bytes.len() % 16 != 0 {
        return Err(
            "tamano de archivo invalido para un libro polyglot (debe ser multiplo de 16 bytes)"
                .to_string(),
        );
    }
    let mut entradas: Vec<Entrada> = bytes
        .chunks_exact(16)
        .map(|c| Entrada {
            key: u64::from_be_bytes(c[0..8].try_into().unwrap()),
            raw_move: u16::from_be_bytes(c[8..10].try_into().unwrap()),
            weight: u16::from_be_bytes(c[10..12].try_into().unwrap()),
        })
        .collect();
    entradas.sort_by_key(|e| e.key);
    let n = entradas.len();
    LIBRO
        .set(entradas)
        .map_err(|_| "no se pudo instalar el libro de aperturas".to_string())?;
    Ok(n)
}

/// Clave Zobrist segun el estandar Polyglot -- distinta del zobrist interno
/// del motor. "pivot" en la formula original es 0=negras, 1=blancas (ojo:
/// invertido respecto al Color::White=0 de este motor).
fn polyglot_key(b: &Board) -> u64 {
    let mut key = 0u64;

    for pt in crate::types::ALL_PIECE_TYPES {
        let python_pt = pt as usize + 1; // Polyglot numera 1=Peon .. 6=Rey
        for (color, pivot) in [(Color::White, 1usize), (Color::Black, 0usize)] {
            let mut bb = b.pieces[color as usize][pt as usize];
            while bb != 0 {
                let sq = crate::bitboard::pop_lsb(&mut bb);
                let piece_index = (python_pt - 1) * 2 + pivot;
                key ^= POLYGLOT_RANDOM[64 * piece_index + sq as usize];
            }
        }
    }

    if b.castling_rights & CASTLE_WK != 0 {
        key ^= POLYGLOT_RANDOM[768];
    }
    if b.castling_rights & CASTLE_WQ != 0 {
        key ^= POLYGLOT_RANDOM[769];
    }
    if b.castling_rights & CASTLE_BK != 0 {
        key ^= POLYGLOT_RANDOM[770];
    }
    if b.castling_rights & CASTLE_BQ != 0 {
        key ^= POLYGLOT_RANDOM[771];
    }

    // Al paso: solo cuenta si de verdad hay un peon propio en columna
    // adyacente listo para capturar (no alcanza con que el FEN tenga la
    // casilla marcada) -- mismo criterio que usa el estandar Polyglot.
    if let Some(ep) = b.ep_square {
        let ep_file = file_of(ep) as i32;
        let fila_atacante = if b.turn == Color::White {
            rank_of(ep) as i32 - 1
        } else {
            rank_of(ep) as i32 + 1
        };
        let mut hay_atacante = false;
        for df in [-1i32, 1] {
            let f = ep_file + df;
            if (0..8).contains(&f) && (0..8).contains(&fila_atacante) {
                let sq = make_square(f as u8, fila_atacante as u8);
                if b.pieces[b.turn as usize][PieceType::Pawn as usize] & (1u64 << sq) != 0 {
                    hay_atacante = true;
                }
            }
        }
        if hay_atacante {
            key ^= POLYGLOT_RANDOM[772 + ep_file as usize];
        }
    }

    if b.turn == Color::White {
        key ^= POLYGLOT_RANDOM[780];
    }

    key
}

/// Decodifica una jugada cruda de Polyglot contra la posicion actual,
/// incluida la rareza historica del formato: el enroque se codifica como
/// "el rey captura su propia torre" (ej. e1h1 para el enroque corto blanco),
/// no con la casilla final real del rey.
fn decodificar_jugada(b: &Board, raw: u16) -> Option<Move> {
    let to_sq = (raw & 0x3f) as u8;
    let from_sq = ((raw >> 6) & 0x3f) as u8;
    let promo_part = (raw >> 12) & 0x7;
    let promotion = match promo_part {
        1 => Some(PieceType::Knight),
        2 => Some(PieceType::Bishop),
        3 => Some(PieceType::Rook),
        4 => Some(PieceType::Queen),
        _ => None,
    };

    let es_rey = b.pieces[b.turn as usize][PieceType::King as usize] & (1u64 << from_sq) != 0;
    let to_real = if es_rey {
        match (b.turn, from_sq, to_sq) {
            (Color::White, 4, 7) => 6,    // e1h1 -> g1 (enroque corto)
            (Color::White, 4, 0) => 2,    // e1a1 -> c1 (enroque largo)
            (Color::Black, 60, 63) => 62, // e8h8 -> g8
            (Color::Black, 60, 56) => 58, // e8a8 -> c8
            _ => to_sq,
        }
    } else {
        to_sq
    };

    generate_legal(b)
        .into_iter()
        .find(|m| m.from == from_sq && m.to == to_real && m.promotion == promotion)
}

/// Jugada del libro para la posicion actual (blancas O negras, ambos colores
/// se consultan igual -- la clave Polyglot ya codifica de quien es el turno).
/// Eleccion aleatoria ponderada por el peso de cada entrada, como hacen la
/// mayoria de los motores UCI con libro. None si la posicion no esta en el
/// libro o no hay libro cargado.
pub fn probe(b: &Board) -> Option<Move> {
    if !ACTIVO.load(Ordering::Relaxed) {
        return None;
    }
    let libro = LIBRO.get()?;
    let key = polyglot_key(b);
    let inicio = libro.partition_point(|e| e.key < key);
    let mut candidatos: Vec<(Move, u32)> = Vec::new();
    let mut i = inicio;
    while i < libro.len() && libro[i].key == key {
        if let Some(mv) = decodificar_jugada(b, libro[i].raw_move) {
            candidatos.push((mv, libro[i].weight as u32));
        }
        i += 1;
    }
    if candidatos.is_empty() {
        return None;
    }
    let total: u32 = candidatos.iter().map(|(_, w)| (*w).max(1)).sum();
    let mut r = rand::random_range(0..total);
    for (mv, w) in &candidatos {
        let w = (*w).max(1);
        if r < w {
            return Some(*mv);
        }
        r -= w;
    }
    candidatos.last().map(|(mv, _)| *mv)
}
