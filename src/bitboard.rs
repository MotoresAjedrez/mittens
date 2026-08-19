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
pub const fn bit(sq: Square) -> Bitboard {
    1u64 << sq
}

#[inline(always)]
pub const fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

#[inline(always)]
pub const fn lsb(bb: Bitboard) -> Square {
    bb.trailing_zeros() as Square
}

#[inline(always)]
pub const fn msb(bb: Bitboard) -> Square {
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

/// Tablas de geometria pura (caballo, rey, ataques de peon y rayos): no
/// dependen de nada del tablero, asi que se calculan al COMPILAR en vez de
/// dentro de un OnceLock. Antes cada `knight_attacks`/`king_attacks`/
/// `pawn_attacks` y cada consulta de rayos pagaba una carga atomica Acquire
/// mas un salto para comprobar la inicializacion; ahora es un acceso directo
/// a un static. El contenido es identico por construccion (mismo codigo,
/// ejecutado por el evaluador de constantes) y el test
/// `tablas_const_coinciden_con_construccion` lo verifica entrada por entrada.
static TABLES_CONST: Tables = build_tables();

const fn dir_index(dir: Dir) -> usize {
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

const fn step(file: i32, rank: i32, dir: Dir) -> Option<(i32, i32)> {
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
    if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
        Some((nf, nr))
    } else {
        None
    }
}

/// Todas las direcciones, en el mismo orden que `ROOK_DIRS ++ BISHOP_DIRS`.
const TODAS_LAS_DIRS: [Dir; 8] = [
    Dir::N,
    Dir::S,
    Dir::E,
    Dir::W,
    Dir::NE,
    Dir::NW,
    Dir::SE,
    Dir::SW,
];

const fn build_tables() -> Tables {
    let mut knight = [0u64; 64];
    let mut king = [0u64; 64];
    let mut pawn_attacks = [[0u64; 64]; 2];
    let mut rays = [[0u64; 64]; 8];

    let knight_deltas: [(i32, i32); 8] = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];

    let mut sq = 0u8;
    while sq < 64 {
        let f = file_of(sq) as i32;
        let r = rank_of(sq) as i32;

        // Caballo
        let mut i = 0;
        while i < 8 {
            let (df, dr) = knight_deltas[i];
            let (nf, nr) = (f + df, r + dr);
            if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                knight[sq as usize] |= bit(make_square(nf as u8, nr as u8));
            }
            i += 1;
        }

        // Rey
        let mut df = -1i32;
        while df <= 1 {
            let mut dr = -1i32;
            while dr <= 1 {
                if !(df == 0 && dr == 0) {
                    let (nf, nr) = (f + df, r + dr);
                    if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                        king[sq as usize] |= bit(make_square(nf as u8, nr as u8));
                    }
                }
                dr += 1;
            }
            df += 1;
        }

        // Peones (ataques diagonales, no incluye avance)
        let mut color_idx = 0usize;
        while color_idx < 2 {
            let dr = if color_idx == 0 { 1i32 } else { -1i32 };
            let mut j = 0;
            while j < 2 {
                let df = if j == 0 { -1i32 } else { 1i32 };
                let (nf, nr) = (f + df, r + dr);
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    pawn_attacks[color_idx][sq as usize] |= bit(make_square(nf as u8, nr as u8));
                }
                j += 1;
            }
            color_idx += 1;
        }

        // Rayos para piezas deslizantes
        let mut d = 0usize;
        while d < 8 {
            let dir = TODAS_LAS_DIRS[d];
            let mut ray = 0u64;
            let (mut cf, mut cr) = (f, r);
            while let Some((nf, nr)) = step(cf, cr, dir) {
                ray |= bit(make_square(nf as u8, nr as u8));
                cf = nf;
                cr = nr;
            }
            rays[dir_index(dir)][sq as usize] = ray;
            d += 1;
        }
        sq += 1;
    }

    Tables {
        knight,
        king,
        pawn_attacks,
        rays,
    }
}

#[inline(always)]
fn tables() -> &'static Tables {
    &TABLES_CONST
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    TABLES_CONST.knight[sq as usize]
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    TABLES_CONST.king[sq as usize]
}

#[inline(always)]
pub fn pawn_attacks(color: crate::types::Color, sq: Square) -> Bitboard {
    TABLES_CONST.pawn_attacks[color as usize][sq as usize]
}

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_H: Bitboard = 0x8080_8080_8080_8080;

/// Ataques de peón de TODO un conjunto de peones a la vez, con dos
/// desplazamientos en vez de un lookup por peón. Un peón blanco en `sq`
/// ataca `sq+7` (columna-1) y `sq+9` (columna+1); enmascarar la columna a / h
/// antes de desplazar evita el envolvimiento de columna. Idéntico bit a bit a
/// unir `pawn_attacks(color, sq)` sobre cada peón (verificado en tests).
#[inline(always)]
pub fn pawn_attacks_set(color: crate::types::Color, pawns: Bitboard) -> Bitboard {
    match color {
        crate::types::Color::White => ((pawns & !FILE_A) << 7) | ((pawns & !FILE_H) << 9),
        crate::types::Color::Black => ((pawns & !FILE_A) >> 9) | ((pawns & !FILE_H) >> 7),
    }
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


// ---------------------------------------------------------------------------
// NUMEROS MAGICOS PRECALCULADOS
// ---------------------------------------------------------------------------
// `build_magics` los BUSCABA por prueba y error en cada arranque, con un RNG
// de semilla fija. Al ser la semilla fija, el resultado era SIEMPRE el mismo:
// se recalculaban ~800 ms de trabajo identico cada vez que arrancaba el motor
// (medido: 809 ms de los 840 ms de arranque total).
//
// En una computadora de escritorio pasa desapercibido, pero en un telefono
// lento son varios segundos antes de poder mover, y en una partida de 1 minuto
// eso es una fraccion enorme del reloj tirada.
//
// Aca estan los mismos numeros que la busqueda encontraba, ya resueltos. La
// construccion de la tabla de ataques sigue igual; lo unico que se elimina es
// la busqueda. El test `magicos_precalculados_siguen_siendo_validos` comprueba
// que cada uno indexa sin colisiones destructivas, asi que si alguno estuviera
// mal el fallo salta en los tests, no en una partida.
/// Numeros magicos de TORRE, precalculados (ver `build_magics`).
static MAGICOS_TORRE: [u64; 64] = [
    0x2080002080400010, 0x00C0002001401000, 0x2100110008402002, 0x0880080081041000,
    0x0200020020041008, 0x2300040008010012, 0x0C00283004008201, 0x0180010000407A80,
    0x0168800080400020, 0x0010400040201000, 0x1001002001001048, 0x1001002408100100,
    0x0801000408010012, 0x4001000209000400, 0x08A20004C8020001, 0x2002801145002280,
    0x0080860021004200, 0x001000C009402002, 0x00B0002004002800, 0x100A808010020800,
    0x8101010008000410, 0x0244008002000480, 0x0000040010810208, 0x2000020000448534,
    0x4104400480008033, 0x0000810100204000, 0x0440430900200010, 0x4600240900100100,
    0x0060080080040080, 0x0001000300080400, 0x0004084400011002, 0x0023040200008041,
    0x0580050043002080, 0x0400804002802008, 0x0001002001004010, 0x1000200901001000,
    0x4410800801800C00, 0xA012003806001004, 0x0020100104008802, 0x0004808402000041,
    0x0010400170898000, 0x0080500020004004, 0x1040408012020020, 0x8010040008004040,
    0x2001080100110004, 0x0000020004008080, 0x0021010810040002, 0x0800008C43020024,
    0x0000800021005100, 0x0070201040008080, 0x0000D04282006A00, 0x0010014400080240,
    0x0001080110050100, 0x0012000810240600, 0x0402000801040200, 0x028100108A004100,
    0x0050800300102045, 0x8208210040120882, 0x8010600101183441, 0x020B000910006045,
    0x0241001002480005, 0x0081000400880241, 0x0000009008024124, 0x0048122980410402,
];

/// Numeros magicos de ALFIL, precalculados (ver `build_magics`).
static MAGICOS_ALFIL: [u64; 64] = [
    0x0848020822040013, 0x8010A40085821200, 0x0008008430840822, 0x0808048108040000,
    0x1304042100008104, 0x5001012010204023, 0x81048801B8200420, 0x200A008084012000,
    0x0040102001042084, 0x840A505042428020, 0x0000700102202920, 0x44101C0C10800002,
    0x0040040422000000, 0x0180020802090202, 0x4020020811041202, 0x000104308C042000,
    0x4140661002424400, 0x0028012008010460, 0x0188062102002A00, 0x0014004840102008,
    0x0105000290400002, 0x8001022200410400, 0x104A041918013446, 0x008A000082008238,
    0x04A0060008100430, 0x0008220008820801, 0x2508041208005010, 0x4008080200202020,
    0x2441001013004000, 0x0030008060407000, 0x4008108000420800, 0x0012021050290100,
    0x0210080482200500, 0xCC01112048100480, 0x0020402806500440, 0x00048E0080580080,
    0x0040102020020080, 0x0028010440080807, 0x4601041108008800, 0x8040810E04104200,
    0x901210110400088A, 0xA003080212081050, 0x00C1004048401004, 0x900000A014400800,
    0x0008021040405401, 0x4020008206002090, 0x0004190424030100, 0x0424008A02026250,
    0x8004088250900040, 0x1C00430088A04200, 0x0001020094040001, 0x8040210020880061,
    0x2010040450442032, 0x0800840850044001, 0x0004040802140004, 0x0004080A04222020,
    0x8088802110022000, 0x1081A10416114400, 0x0205010A24060820, 0x0000000720411080,
    0x1008000208430400, 0x580C026028810840, 0x802020441020A110, 0x12C0022401020018,
];

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

    for (es_torre, (dirs, out)) in [(true, (&ROOK_RAYS, &mut rook)), (false, (&BISHOP_RAYS, &mut bishop))] {
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
            // Se toma el magico ya resuelto para esta casilla en vez de
            // buscarlo (ver la nota de MAGICOS_TORRE/MAGICOS_ALFIL arriba).
            // El bucle se conserva porque si algun magico precalculado no
            // sirviera, cae a la busqueda de siempre en vez de romperse.
            let mut precalculado = Some(if es_torre { MAGICOS_TORRE[sq as usize] } else { MAGICOS_ALFIL[sq as usize] });
            'busqueda: loop {
                let magic = match precalculado.take() {
                    Some(m) => m,
                    // Candidato disperso (AND de tres aleatorios) -- converge rápido.
                    None => xorshift(&mut rng) & xorshift(&mut rng) & xorshift(&mut rng),
                };
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
/// Igual que las tablas de geometria: calculada al compilar (antes era un
/// `OnceLock<Vec<..>>`, o sea una carga atomica MAS una indireccion al heap y
/// un chequeo de limites de slice en cada consulta). `pinned_pieces` la
/// consulta una vez por atacante deslizante alineado con el rey.
static BETWEEN: [Bitboard; 64 * 64] = build_between();

const fn build_between() -> [Bitboard; 64 * 64] {
    let mut t = [0u64; 64 * 64];
    let mut a = 0u8;
    while a < 64 {
        let mut d = 0usize;
        while d < 8 {
            let dir = TODAS_LAS_DIRS[d];
            let mut acc = 0u64;
            let (mut cf, mut cr) = (file_of(a) as i32, rank_of(a) as i32);
            while let Some((nf, nr)) = step(cf, cr, dir) {
                let b = make_square(nf as u8, nr as u8);
                t[a as usize * 64 + b as usize] = acc;
                acc |= bit(b);
                cf = nf;
                cr = nr;
            }
            d += 1;
        }
        a += 1;
    }
    t
}

#[inline(always)]
pub fn between(a: Square, b: Square) -> Bitboard {
    BETWEEN[a as usize * 64 + b as usize]
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
    // Solo vale la pena el lookup magico si el rival tiene piezas deslizantes
    // capaces de clavarlo en esa linea (si enemy_rook_like == 0, el resultado
    // del lookup & 0 es cero de todos modos -- se ahorra el trabajo de
    // antemano).
    if enemy_rook_like != 0 {
        let mut candidatos = rook_attacks(king_sq, occ_sin_propias) & enemy_rook_like;
        while candidatos != 0 {
            let atacante = pop_lsb(&mut candidatos);
            let interpuestas = between(king_sq, atacante) & own;
            if popcount(interpuestas) == 1 {
                pinned |= interpuestas;
            }
        }
    }
    if enemy_bishop_like != 0 {
        let mut candidatos = bishop_attacks(king_sq, occ_sin_propias) & enemy_bishop_like;
        while candidatos != 0 {
            let atacante = pop_lsb(&mut candidatos);
            let interpuestas = between(king_sq, atacante) & own;
            if popcount(interpuestas) == 1 {
                pinned |= interpuestas;
            }
        }
    }
    pinned
}

#[cfg(test)]
mod tablas_const_tests {
    use super::*;

    /// Reconstruye las tablas EN TIEMPO DE EJECUCION con el codigo original
    /// (bucles con iteradores) y compara entrada por entrada con las tablas
    /// que ahora produce el evaluador de constantes.
    #[test]
    fn tablas_const_coinciden_con_construccion() {
        for sq in 0..64u8 {
            let f = file_of(sq) as i32;
            let r = rank_of(sq) as i32;

            let mut knight = 0u64;
            for (df, dr) in [
                (1, 2),
                (2, 1),
                (2, -1),
                (1, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
            ] {
                let (nf, nr) = (f + df, r + dr);
                if (0..8).contains(&nf) && (0..8).contains(&nr) {
                    knight |= bit(make_square(nf as u8, nr as u8));
                }
            }
            assert_eq!(knight_attacks(sq), knight, "caballo en {}", sq);

            let mut king = 0u64;
            for df in -1..=1i32 {
                for dr in -1..=1i32 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let (nf, nr) = (f + df, r + dr);
                    if (0..8).contains(&nf) && (0..8).contains(&nr) {
                        king |= bit(make_square(nf as u8, nr as u8));
                    }
                }
            }
            assert_eq!(king_attacks(sq), king, "rey en {}", sq);

            for (color_idx, dr) in [(0usize, 1i32), (1usize, -1i32)] {
                let mut pa = 0u64;
                for df in [-1i32, 1i32] {
                    let (nf, nr) = (f + df, r + dr);
                    if (0..8).contains(&nf) && (0..8).contains(&nr) {
                        pa |= bit(make_square(nf as u8, nr as u8));
                    }
                }
                let color = if color_idx == 0 {
                    crate::types::Color::White
                } else {
                    crate::types::Color::Black
                };
                assert_eq!(pawn_attacks(color, sq), pa, "peon {:?} en {}", color, sq);
            }

            for &dir in ROOK_DIRS.iter().chain(BISHOP_DIRS.iter()) {
                let mut ray = 0u64;
                let (mut cf, mut cr) = (f, r);
                while let Some((nf, nr)) = step(cf, cr, dir) {
                    ray |= bit(make_square(nf as u8, nr as u8));
                    cf = nf;
                    cr = nr;
                }
                assert_eq!(
                    tables().rays[dir_index(dir)][sq as usize],
                    ray,
                    "rayo dir={} sq={}",
                    dir_index(dir),
                    sq
                );
            }
        }
    }

    #[test]
    fn between_const_coincide_con_construccion() {
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
        for a in 0..64u8 {
            for b in 0..64u8 {
                assert_eq!(between(a, b), t[a as usize * 64 + b as usize], "{a}-{b}");
            }
        }
    }
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

#[cfg(test)]
mod pawn_set_tests {
    use super::*;
    use crate::types::Color;

    #[test]
    fn pawn_attacks_set_equivale_al_bucle() {
        let mut rng: u64 = 0xABCD_1234_5678_9EF0;
        for _ in 0..20_000 {
            // Peones solo en las filas 2..7 (nunca en 1 ni en 8).
            let pawns = xorshift(&mut rng) & 0x00FF_FFFF_FFFF_FF00;
            for color in [Color::White, Color::Black] {
                let mut esperado = 0u64;
                let mut bb = pawns;
                while bb != 0 {
                    esperado |= pawn_attacks(color, pop_lsb(&mut bb));
                }
                assert_eq!(
                    pawn_attacks_set(color, pawns),
                    esperado,
                    "color={:?} pawns={:#x}",
                    color,
                    pawns
                );
            }
        }
    }
}

#[cfg(test)]
mod magicos_tests {
    use super::*;

    /// Los magicos precalculados tienen que ser VALIDOS: cada uno debe indexar
    /// todos los subconjuntos de su mascara sin colisiones destructivas.
    ///
    /// Importa que este test exista porque el fallo seria SILENCIOSO: si un
    /// magico estuviera mal, `build_magics` cae a buscar uno nuevo y el motor
    /// sigue jugando bien... pero perdiendo los ~800 ms de arranque que este
    /// cambio vino a ahorrar, sin que nadie se entere.
    #[test]
    fn magicos_precalculados_siguen_siendo_validos() {
        for (es_torre, dirs) in [(true, &ROOK_RAYS), (false, &BISHOP_RAYS)] {
            for sq in 0..64u8 {
                let mask = relevant_mask(sq, dirs);
                let bits = popcount(mask);
                let size = 1usize << bits;
                let shift = 64 - bits;
                let magic = if es_torre {
                    MAGICOS_TORRE[sq as usize]
                } else {
                    MAGICOS_ALFIL[sq as usize]
                };

                let mut tabla = vec![0u64; size];
                let mut usado = vec![false; size];
                let mut sub: Bitboard = 0;
                loop {
                    let referencia = slider_attacks_ref(sq, sub, dirs);
                    let idx = (sub.wrapping_mul(magic) >> shift) as usize;
                    if usado[idx] {
                        assert_eq!(
                            tabla[idx], referencia,
                            "colision destructiva: {} en la casilla {} con el magico 0x{:016X}",
                            if es_torre { "torre" } else { "alfil" },
                            sq,
                            magic
                        );
                    } else {
                        usado[idx] = true;
                        tabla[idx] = referencia;
                    }
                    sub = sub.wrapping_sub(mask) & mask;
                    if sub == 0 {
                        break;
                    }
                }
            }
        }
    }

    /// Los ataques que produce la tabla ya construida tienen que coincidir con
    /// la referencia lenta, para cualquier ocupacion.
    #[test]
    fn los_ataques_coinciden_con_la_referencia() {
        let mut estado: u64 = 0xDEAD_BEEF_1234_5678;
        for sq in 0..64u8 {
            for _ in 0..40 {
                let occ = xorshift(&mut estado) & xorshift(&mut estado);
                assert_eq!(
                    rook_attacks(sq, occ),
                    slider_attacks_ref(sq, occ, &ROOK_RAYS),
                    "torre en {sq} con ocupacion 0x{occ:016X}"
                );
                assert_eq!(
                    bishop_attacks(sq, occ),
                    slider_attacks_ref(sq, occ, &BISHOP_RAYS),
                    "alfil en {sq} con ocupacion 0x{occ:016X}"
                );
            }
        }
    }
}
