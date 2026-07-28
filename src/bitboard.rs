// Bitboards de 64 bits y tablas de ataques precalculadas.
// Piezas deslizantes: ray-casting clásico (no magic bitboards) -- documentado
// así explícitamente por decisión de tiempo: es más simple de verificar
// correctamente con perft, al costo de menos nodos/s que magic bitboards.
// Si hace falta más velocidad más adelante, este es el punto a optimizar.

use crate::types::{Square, file_of, make_square, rank_of};
use std::sync::OnceLock;

pub type Bitboard = u64;

pub const EMPTY: Bitboard = 0;

#[inline(always)]
pub fn bit(sq: Square) -> Bitboard {
    1u64 << sq
}

#[inline(always)]
pub fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

#[inline(always)]
pub fn lsb(bb: Bitboard) -> Square {
    bb.trailing_zeros() as Square
}

#[inline(always)]
pub fn msb(bb: Bitboard) -> Square {
    63 - bb.leading_zeros() as Square
}

#[inline(always)]
pub fn pop_lsb(bb: &mut Bitboard) -> Square {
    let s = lsb(*bb);
    *bb &= *bb - 1;
    s
}

// 8 direcciones para piezas deslizantes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    N,
    S,
    E,
    W,
    NE,
    NW,
    SE,
    SW,
}

pub const ROOK_DIRS: [Dir; 4] = [Dir::N, Dir::S, Dir::E, Dir::W];
pub const BISHOP_DIRS: [Dir; 4] = [Dir::NE, Dir::NW, Dir::SE, Dir::SW];

struct Tables {
    knight: [Bitboard; 64],
    king: [Bitboard; 64],
    pawn_attacks: [[Bitboard; 64]; 2], // [color][square]
    rays: [[Bitboard; 64]; 8],         // [dir][square]
}

static TABLES: OnceLock<Tables> = OnceLock::new();

fn dir_index(dir: Dir) -> usize {
    match dir {
        Dir::N => 0,
        Dir::S => 1,
        Dir::E => 2,
        Dir::W => 3,
        Dir::NE => 4,
        Dir::NW => 5,
        Dir::SE => 6,
        Dir::SW => 7,
    }
}

fn step(file: i32, rank: i32, dir: Dir) -> Option<(i32, i32)> {
    let (df, dr) = match dir {
        Dir::N => (0, 1),
        Dir::S => (0, -1),
        Dir::E => (1, 0),
        Dir::W => (-1, 0),
        Dir::NE => (1, 1),
        Dir::NW => (-1, 1),
        Dir::SE => (1, -1),
        Dir::SW => (-1, -1),
    };
    let (nf, nr) = (file + df, rank + dr);
    if (0..8).contains(&nf) && (0..8).contains(&nr) {
        Some((nf, nr))
    } else {
        None
    }
}

fn build_tables() -> Tables {
    let mut knight = [0u64; 64];
    let mut king = [0u64; 64];
    let mut pawn_attacks = [[0u64; 64]; 2];
    let mut rays = [[0u64; 64]; 8];

    for sq in 0..64u8 {
        let f = file_of(sq) as i32;
        let r = rank_of(sq) as i32;

        // Caballo
        let knight_deltas = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ];
        for (df, dr) in knight_deltas {
            let (nf, nr) = (f + df, r + dr);
            if (0..8).contains(&nf) && (0..8).contains(&nr) {
                knight[sq as usize] |= bit(make_square(nf as u8, nr as u8));
            }
        }

        // Rey
        for df in -1..=1i32 {
            for dr in -1..=1i32 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let (nf, nr) = (f + df, r + dr);
                if (0..8).contains(&nf) && (0..8).contains(&nr) {
                    king[sq as usize] |= bit(make_square(nf as u8, nr as u8));
                }
            }
        }

        // Peones (ataques diagonales, no incluye avance)
        for (color_idx, dr) in [(0usize, 1i32), (1usize, -1i32)] {
            for df in [-1i32, 1i32] {
                let (nf, nr) = (f + df, r + dr);
                if (0..8).contains(&nf) && (0..8).contains(&nr) {
                    pawn_attacks[color_idx][sq as usize] |= bit(make_square(nf as u8, nr as u8));
                }
            }
        }

        // Rayos para piezas deslizantes
        for &dir in ROOK_DIRS.iter().chain(BISHOP_DIRS.iter()) {
            let mut ray = 0u64;
            let (mut cf, mut cr) = (f, r);
            while let Some((nf, nr)) = step(cf, cr, dir) {
                ray |= bit(make_square(nf as u8, nr as u8));
                cf = nf;
                cr = nr;
            }
            rays[dir_index(dir)][sq as usize] = ray;
        }
    }

    Tables {
        knight,
        king,
        pawn_attacks,
        rays,
    }
}

fn tables() -> &'static Tables {
    TABLES.get_or_init(build_tables)
}

pub fn knight_attacks(sq: Square) -> Bitboard {
    tables().knight[sq as usize]
}

pub fn king_attacks(sq: Square) -> Bitboard {
    tables().king[sq as usize]
}

pub fn pawn_attacks(color: crate::types::Color, sq: Square) -> Bitboard {
    tables().pawn_attacks[color as usize][sq as usize]
}

/// Recorta un rayo precalculado justo después del primer bloqueo. Recibir la
/// tabla por referencia evita consultar el OnceLock por cada dirección; los
/// índices y el sentido son constantes en rook_attacks/bishop_attacks, así el
/// compilador puede integrar y especializar por completo este camino caliente.
/// Perfilado (Codex, `sample`): bishop_attacks+rook_attacks eran ~24% del
/// tiempo de busqueda base -- este cambio es bit a bit equivalente al
/// anterior (ray(dir,sq) + is_positive(dir)), solo evita el lookup repetido.
#[inline(always)]
fn truncate_ray(
    rays: &[[Bitboard; 64]; 8],
    dir_idx: usize,
    sq: Square,
    occupied: Bitboard,
    positive: bool,
) -> Bitboard {
    let full_ray = rays[dir_idx][sq as usize];
    let blockers = full_ray & occupied;
    if blockers == 0 {
        return full_ray;
    }
    let blocker_sq = if positive {
        lsb(blockers)
    } else {
        msb(blockers)
    };
    full_ray & !rays[dir_idx][blocker_sq as usize]
}

// (dir_idx en la tabla de rayos, es_positivo) para cada deslizante.
const ROOK_RAYS: [(usize, bool); 4] = [(0, true), (1, false), (2, true), (3, false)];
const BISHOP_RAYS: [(usize, bool); 4] = [(4, true), (5, true), (6, false), (7, false)];

/// Implementación de referencia por ray-casting (la original). Se usa para
/// construir y verificar las tablas mágicas y en los tests de equivalencia.
fn slider_attacks_ref(sq: Square, occupied: Bitboard, dirs: &[(usize, bool); 4]) -> Bitboard {
    let rays = &tables().rays;
    let mut attacks = 0u64;
    for &(dir_idx, positive) in dirs.iter() {
        attacks |= truncate_ray(rays, dir_idx, sq, occupied, positive);
    }
    attacks
}

// ==================== Magic bitboards ====================
// Ataques de torre/alfil en O(1): ((occ & mask) * magic) >> shift indexa una
// tabla precalculada. Los números mágicos se BUSCAN al arrancar (xorshift con
// semilla fija, determinista, <100ms) y cada entrada se construye con el
// ray-casting de referencia de arriba, verificando TODOS los subconjuntos de
// bloqueos de cada casilla: un mágico con colisión destructiva se descarta,
// así que el resultado es idéntico por construcción al código clásico.

struct MagicEntry {
    mask: Bitboard,
    magic: u64,
    shift: u32,
    offset: usize,
}

struct Magics {
    rook: [MagicEntry; 64],
    bishop: [MagicEntry; 64],
    table: Vec<Bitboard>,
}

static MAGICS: OnceLock<Magics> = OnceLock::new();

fn relevant_mask(sq: Square, dirs: &[(usize, bool); 4]) -> Bitboard {
    // El rayo completo sin la última casilla de cada dirección: un bloqueo en
    // el borde no cambia el ataque, así que no hace falta indexarlo.
    let rays = &tables().rays;
    let mut mask = 0u64;
    for &(dir_idx, positive) in dirs.iter() {
        let r = rays[dir_idx][sq as usize];
        if r != 0 {
            let last = if positive { msb(r) } else { lsb(r) };
            mask |= r & !bit(last);
        }
    }
    mask
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn build_magics() -> Magics {
    let mut table: Vec<Bitboard> = Vec::new();
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rook: Vec<MagicEntry> = Vec::with_capacity(64);
    let mut bishop: Vec<MagicEntry> = Vec::with_capacity(64);

    for (dirs, out) in [(&ROOK_RAYS, &mut rook), (&BISHOP_RAYS, &mut bishop)] {
        for sq in 0..64u8 {
            let mask = relevant_mask(sq, dirs);
            let bits = popcount(mask);
            let size = 1usize << bits;
            let shift = 64 - bits;

            // Todos los subconjuntos de la máscara (truco Carry-Rippler) con
            // su ataque de referencia.
            let mut occs = Vec::with_capacity(size);
            let mut refs = Vec::with_capacity(size);
            let mut sub: Bitboard = 0;
            loop {
                occs.push(sub);
                refs.push(slider_attacks_ref(sq, sub, dirs));
                sub = sub.wrapping_sub(mask) & mask;
                if sub == 0 {
                    break;
                }
            }

            let offset = table.len();
            table.resize(offset + size, 0);
            'busqueda: loop {
                // Candidato disperso (AND de tres aleatorios) -- converge rápido.
                let magic = xorshift(&mut rng) & xorshift(&mut rng) & xorshift(&mut rng);
                if popcount(mask.wrapping_mul(magic) & 0xFF00_0000_0000_0000) < 6 {
                    continue;
                }
                for e in table[offset..offset + size].iter_mut() {
                    *e = 0;
                }
                let mut usado = vec![false; size];
                for (i, &occ) in occs.iter().enumerate() {
                    let idx = (occ.wrapping_mul(magic) >> shift) as usize;
                    if usado[idx] {
                        if table[offset + idx] != refs[i] {
                            continue 'busqueda; // colisión destructiva: probar otro
                        }
                    } else {
                        usado[idx] = true;
                        table[offset + idx] = refs[i];
                    }
                }
                out.push(MagicEntry {
                    mask,
                    magic,
                    shift,
                    offset,
                });
                break;
            }
        }
    }

    let rook: [MagicEntry; 64] = match rook.try_into() {
        Ok(a) => a,
        Err(_) => unreachable!(),
    };
    let bishop: [MagicEntry; 64] = match bishop.try_into() {
        Ok(a) => a,
        Err(_) => unreachable!(),
    };
    Magics {
        rook,
        bishop,
        table,
    }
}

fn magics() -> &'static Magics {
    MAGICS.get_or_init(build_magics)
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let ms = magics();
    let m = &ms.rook[sq as usize];
    ms.table[m.offset + ((occupied & m.mask).wrapping_mul(m.magic) >> m.shift) as usize]
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let ms = magics();
    let m = &ms.bishop[sq as usize];
    ms.table[m.offset + ((occupied & m.mask).wrapping_mul(m.magic) >> m.shift) as usize]
}

pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)
}

// ==================== Tabla "entre" (BETWEEN) ====================
// BETWEEN[a][b]: casillas estrictamente entre a y b si estan alineadas
// (misma fila, columna o diagonal), sin incluir a ni b; 0 si no estan
// alineadas. Se usa para detectar piezas clavadas: construida una sola vez
// con ray-casting de referencia (independiente de las tablas mágicas).
static BETWEEN: OnceLock<Vec<Bitboard>> = OnceLock::new();

fn build_between() -> Vec<Bitboard> {
    let mut t = vec![0u64; 64 * 64];
    for a in 0..64u8 {
        for &dir in ROOK_DIRS.iter().chain(BISHOP_DIRS.iter()) {
            let mut acc = 0u64;
            let (mut cf, mut cr) = (file_of(a) as i32, rank_of(a) as i32);
            while let Some((nf, nr)) = step(cf, cr, dir) {
                let b = make_square(nf as u8, nr as u8);
                t[a as usize * 64 + b as usize] = acc;
                acc |= bit(b);
                cf = nf;
                cr = nr;
            }
        }
    }
    t
}

pub fn between(a: Square, b: Square) -> Bitboard {
    BETWEEN.get_or_init(build_between)[a as usize * 64 + b as usize]
}

/// Piezas propias clavadas contra su rey por una torre/dama/alfil enemiga en
/// linea recta (con exactamente una pieza propia entre el rey y el atacante).
/// Se calcula UNA vez por nodo (no por jugada) y sustituye, para las jugadas
/// que no son de rey/al paso/enroque, la alternativa de copiar el tablero
/// completo y verificar jaque jugada por jugada.
pub fn pinned_pieces(
    king_sq: Square,
    own: Bitboard,
    enemy_rook_like: Bitboard,
    enemy_bishop_like: Bitboard,
    occupied: Bitboard,
) -> Bitboard {
    let mut pinned = 0u64;
    // Rayos "a traves" de las piezas propias: si se ignoran por completo
    // (occupied & !own), un atacante deslizante enemigo alineado con el rey
    // solo puede estar clavando si hay EXACTAMENTE una pieza propia entre
    // ambos (si hubiera una pieza enemiga o mas de una propia interpuesta,
    // no hay clavada real).
    let occ_sin_propias = occupied & !own;
    let mut candidatos = rook_attacks(king_sq, occ_sin_propias) & enemy_rook_like;
    while candidatos != 0 {
        let atacante = pop_lsb(&mut candidatos);
        let interpuestas = between(king_sq, atacante) & own;
        if popcount(interpuestas) == 1 {
            pinned |= interpuestas;
        }
    }
    let mut candidatos = bishop_attacks(king_sq, occ_sin_propias) & enemy_bishop_like;
    while candidatos != 0 {
        let atacante = pop_lsb(&mut candidatos);
        let interpuestas = between(king_sq, atacante) & own;
        if popcount(interpuestas) == 1 {
            pinned |= interpuestas;
        }
    }
    pinned
}

#[cfg(test)]
mod pin_tests {
    use super::*;

    #[test]
    fn between_casillas_alineadas() {
        // a1-h8: entre son b2..g7
        let a1 = 0u8;
        let h8 = 63u8;
        let bb = between(a1, h8);
        assert_eq!(popcount(bb), 6);
        // a1-a8: entre son a2..a7
        let a8 = make_square(0, 7);
        assert_eq!(popcount(between(a1, a8)), 6);
        // a1-b3: no alineadas
        let b3 = make_square(1, 2);
        assert_eq!(between(a1, b3), 0);
    }
}

#[cfg(test)]
mod magic_tests {
    use super::*;

    #[test]
    fn magias_equivalen_a_ray_casting() {
        // Ocupaciones pseudoaleatorias (densas y dispersas) sobre las 64
        // casillas: el lookup mágico debe coincidir bit a bit con el
        // ray-casting de referencia.
        let mut rng: u64 = 0xC0FF_EE12_3456_789A;
        for i in 0..3000 {
            let a = xorshift(&mut rng);
            let b = xorshift(&mut rng);
            let occ = if i % 3 == 0 { a } else { a & b };
            for sq in 0..64u8 {
                assert_eq!(
                    rook_attacks(sq, occ),
                    slider_attacks_ref(sq, occ, &ROOK_RAYS),
                    "torre sq={} occ={:#x}",
                    sq,
                    occ
                );
                assert_eq!(
                    bishop_attacks(sq, occ),
                    slider_attacks_ref(sq, occ, &BISHOP_RAYS),
                    "alfil sq={} occ={:#x}",
                    sq,
                    occ
                );
            }
        }
    }
}
