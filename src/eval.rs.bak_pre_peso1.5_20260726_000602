// Evaluación estática, portada de mi_motor.py (identidad Tal) -- mismos
// valores numéricos ya ajustados en Python, no reinventados. Devuelve el
// puntaje en centipeones desde el punto de vista del bando que mueve.

use crate::bitboard::{
    Bitboard, bishop_attacks, king_attacks, knight_attacks, pawn_attacks, popcount, queen_attacks,
    rook_attacks,
};
use crate::board::Board;
use crate::types::{Color, PieceType, Square, file_of, make_square, rank_of};
use std::sync::atomic::{AtomicU8, Ordering};

// Dos identidades de evaluacion, seleccionables en caliente (UCI "setoption
// name Personalidad" o variable de entorno MIMOTOR_PERSONALIDAD), que
// COEXISTEN -- no se reemplaza Tal, se agrega Universal como alternativa.
// Estado global de solo-lectura durante la busqueda (se fija antes de "go",
// nunca cambia a mitad de una busqueda concurrente) -- por eso un atomico
// simple alcanza, sin necesitar pasar el parametro por cada llamada interna.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Personalidad {
    Tal,
    Universal,
}

static PERSONALIDAD_ACTUAL: AtomicU8 = AtomicU8::new(0);

pub fn set_personalidad(p: Personalidad) {
    PERSONALIDAD_ACTUAL.store(
        if p == Personalidad::Universal { 1 } else { 0 },
        Ordering::Relaxed,
    );
}

pub fn personalidad_desde_texto(s: &str) -> Option<Personalidad> {
    match s.to_lowercase().as_str() {
        "tal" => Some(Personalidad::Tal),
        "universal" => Some(Personalidad::Universal),
        _ => None,
    }
}

fn personalidad_actual() -> Personalidad {
    if PERSONALIDAD_ACTUAL.load(Ordering::Relaxed) == 1 {
        Personalidad::Universal
    } else {
        Personalidad::Tal
    }
}

const VALOR: [i32; 6] = [100, 320, 330, 500, 900, 0]; // Pawn,Knight,Bishop,Rook,Queen,King
const ESCALA_MATERIAL_TAL: f64 = 0.9;
const ESCALA_MATERIAL_UNIVERSAL: f64 = 1.0;
const TEMPO: i32 = 12;

const PESO_MOV: [i32; 6] = [0, 4, 4, 3, 1, 0];
const PESO_ATQ: [i32; 6] = [0, 2, 2, 3, 5, 0];
// v12: bajado de 1.45 a 1.15. Diagnostico de divergencia contra Stockfish
// (profundidad 18) en las posiciones criticas de 2 derrotas reales contra
// simpleEval mostro a Tal sobreestimando la posicion propia por 100-380cp de
// forma sostenida -- comparado en la MISMA posicion, Universal (factor 1.0)
// daba un numero mucho mas cercano a Stockfish. 1.45 inflaba demasiado el
// termino de ataque al rey (hasta +225cp extra en zonas muy atacadas), dando
// confianza excesiva en complicaciones sin comprobar si de verdad progresan.
// 1.15 mantiene la identidad agresiva (sigue por encima del 1.0 neutral de
// Universal) sin el sesgo tan marcado. Verificado con torneo h2h antes de
// aceptarlo (ver resultados_tal_calibrado_h2h.txt).
const FACTOR_ATAQUE_TAL: f64 = 1.15;
const FACTOR_ATAQUE_UNIVERSAL: f64 = 1.0; // ataque al rey de peso normal, sin el bono no-lineal agresivo de Tal

const TABLA_SEGURIDAD: [i32; 62] = [
    0, 0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30, 35, 39, 44, 50, 56, 62, 68, 75, 82, 85, 89, 97,
    105, 113, 122, 131, 140, 150, 169, 180, 191, 202, 213, 225, 237, 248, 260, 272, 283, 295, 307,
    318, 330, 342, 354, 366, 377, 389, 401, 412, 424, 436, 448, 459, 471, 483, 494, 500,
];

// Piece-square tables, escritas visualmente (primera fila = 8a fila = rank 7 en indice 0..8).
#[rustfmt::skip]
const PEON_MG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const PEON_EG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     80,  80,  80,  80,  80,  80,  80,  80,
     50,  50,  50,  50,  50,  50,  50,  50,
     30,  30,  30,  30,  30,  30,  30,  30,
     15,  15,  15,  15,  15,  15,  15,  15,
      5,   5,   5,   5,   5,   5,   5,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const CABALLO: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
];
#[rustfmt::skip]
const ALFIL: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
];
#[rustfmt::skip]
const TORRE: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
];
#[rustfmt::skip]
const DAMA: [i32; 64] = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
];
#[rustfmt::skip]
const REY_MG: [i32; 64] = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
];
#[rustfmt::skip]
const REY_EG: [i32; 64] = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
];

fn pst_mg(pt: PieceType) -> &'static [i32; 64] {
    match pt {
        PieceType::Pawn => &PEON_MG,
        PieceType::Knight => &CABALLO,
        PieceType::Bishop => &ALFIL,
        PieceType::Rook => &TORRE,
        PieceType::Queen => &DAMA,
        PieceType::King => &REY_MG,
    }
}

fn pst_eg(pt: PieceType) -> &'static [i32; 64] {
    match pt {
        PieceType::Pawn => &PEON_EG,
        PieceType::Knight => &CABALLO,
        PieceType::Bishop => &ALFIL,
        PieceType::Rook => &TORRE,
        PieceType::Queen => &DAMA,
        PieceType::King => &REY_EG,
    }
}

fn piece_attacks(pt: PieceType, sq: Square, occ: Bitboard, color: Color) -> Bitboard {
    match pt {
        PieceType::Pawn => pawn_attacks(color, sq),
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occ),
        PieceType::Rook => rook_attacks(sq, occ),
        PieceType::Queen => queen_attacks(sq, occ),
        PieceType::King => king_attacks(sq),
    }
}

fn distancia_chebyshev(a: Square, b: Square) -> i32 {
    let (fa, ra) = (file_of(a) as i32, rank_of(a) as i32);
    let (fb, rb) = (file_of(b) as i32, rank_of(b) as i32);
    (fa - fb).abs().max((ra - rb).abs())
}

// Indice de avance hacia la coronacion (0 = fila de salida, 7 = coronando),
// independiente del color: para blancas es directamente rank_of, para negras
// es la fila espejada.
fn indice_avance(color: Color, sq: Square) -> usize {
    let r = rank_of(sq) as usize;
    if color == Color::White { r } else { 7 - r }
}

// Peon pasado: ningun peon rival en la misma columna ni en las adyacentes,
// por delante de esta casilla en direccion a la coronacion. Se calcula con
// mascaras de fila/columna en vez de generar el tablero completo -- barato,
// se llama una vez por peon en cada evaluacion.
fn es_pasado(color: Color, sq: Square, peones_rivales: Bitboard) -> bool {
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    let mut columnas: Bitboard = 0;
    for df in -1..=1i32 {
        let nf = f + df;
        if (0..8).contains(&nf) {
            columnas |= 0x0101010101010101u64 << nf;
        }
    }
    let filas_delante: Bitboard = if color == Color::White {
        if r == 7 { 0 } else { !0u64 << ((r + 1) * 8) }
    } else if r == 0 {
        0
    } else {
        !(!0u64 << (r * 8))
    };
    (peones_rivales & columnas & filas_delante) == 0
}

const PASO_BONUS: [i32; 8] = [0, 5, 10, 20, 35, 60, 100, 150];
const VENTAJA_DECISIVA: f64 = 500.0; // ~una torre de diferencia
const PESO_ACERCAMIENTO_REY: f64 = 4.0;

// Puesto avanzado (profilaxis, solo personalidad Universal): pieza menor
// propia en territorio rival que ningun peon rival puede llegar a atacar
// nunca (mismo patron de mascara que un peon pasado, pero mirando si HAY
// peones rivales adelante en columnas adyacentes, no si NO los hay). Premia
// restringir el territorio del rival, no solo maximizar la movilidad propia
// -- una pieza asi no se puede expulsar con un peon, limita permanentemente
// los planes del rival en esa zona del tablero.
fn en_territorio_rival(color: Color, sq: Square) -> bool {
    let r = rank_of(sq);
    if color == Color::White {
        r >= 4
    } else {
        r <= 3
    }
}

fn puesto_avanzado_seguro(color: Color, sq: Square, peones_rivales: Bitboard) -> bool {
    es_pasado(color, sq, peones_rivales) // misma mascara: sin peon rival por delante en columnas adyacentes
}

const BONUS_PUESTO: [i32; 6] = [0, 25, 12, 0, 0, 0]; // solo caballo/alfil

fn pareja_alfiles_bonus(color: Color, b: &Board) -> f64 {
    if popcount(b.pieces[color as usize][PieceType::Bishop as usize]) >= 2 {
        35.0
    } else {
        0.0
    }
}

fn king_zone(ksq: Square, color: Color) -> Bitboard {
    let mut z = king_attacks(ksq) | (1u64 << ksq);
    let f = file_of(ksq) as i32;
    let r = rank_of(ksq) as i32 + if color == Color::White { 2 } else { -2 };
    if (0..8).contains(&r) {
        for df in -1..=1i32 {
            let nf = f + df;
            if (0..8).contains(&nf) {
                z |= 1u64 << make_square(nf as u8, r as u8);
            }
        }
    }
    z
}

struct AtaqueRey {
    ataque_w: f64,
    ataque_b: f64,
}

fn puntuar_ataque_rey(
    b: &Board,
    factor_ataque: f64,
    unidades: [i32; 2],
    n_atacantes: [i32; 2],
) -> AtaqueRey {
    let rey_w = b.king_square(Color::White);
    let rey_b = b.king_square(Color::Black);
    let puntaje = |idx: usize, color: Color| -> f64 {
        let mut u = unidades[idx];
        if u <= 0 {
            return 0.0;
        }
        let tiene_dama = b.pieces[color as usize][PieceType::Queen as usize] != 0;
        if !tiene_dama {
            u /= 2;
        }
        let mut s = TABLA_SEGURIDAD[u.min(61) as usize] as f64 * factor_ataque;
        if n_atacantes[idx] < 2 {
            s *= 0.35;
        }
        let rey_rival = if color == Color::White { rey_b } else { rey_w };
        let frey = file_of(rey_rival);
        if frey <= 2 || frey >= 6 {
            s *= 1.15;
        }
        s
    };

    AtaqueRey {
        ataque_w: puntaje(1, Color::White),
        ataque_b: puntaje(0, Color::Black),
    }
}

fn calcular_ataque_rey(b: &Board, factor_ataque: f64) -> AtaqueRey {
    let rey_w = b.king_square(Color::White);
    let rey_b = b.king_square(Color::Black);
    let zona_w = king_zone(rey_w, Color::White);
    let zona_b = king_zone(rey_b, Color::Black);
    let mut unidades = [0i32; 2]; // [negro, blanco]
    let mut n_atacantes = [0i32; 2];

    for (color, idx, zona_rival) in [
        (Color::White, 1usize, zona_b),
        (Color::Black, 0usize, zona_w),
    ] {
        let piezas = b.pieces[color as usize];
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let mut bb = piezas[pt as usize];
            while bb != 0 {
                let sq = crate::bitboard::pop_lsb(&mut bb);
                let u = popcount(piece_attacks(pt, sq, b.occupied, color) & zona_rival) as i32;
                if pt == PieceType::Pawn {
                    unidades[idx] += u;
                } else if u > 0 {
                    unidades[idx] += PESO_ATQ[pt as usize] * u;
                    n_atacantes[idx] += 1;
                }
            }
        }
    }
    puntuar_ataque_rey(b, factor_ataque, unidades, n_atacantes)
}

// -------------------------------------------------------------------------
// Acumulador clasico incremental.
//
// Board se mantiene deliberadamente pequeno: generate_legal() hace muchas
// copias de Board solo para filtrar jaques. Por eso este estado acompana a la
// busqueda como sidecar, igual que NNUE, y no vive dentro de Board. Los
// campos son componentes *raw* desde la perspectiva blanco-negro; turno,
// personalidad y mezcla NNUE se aplican solamente al componer el score.
// -------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassicalAccumulator {
    // El orden conserva el evaluador anterior: [negras, blancas].
    material: [i32; 2],
    pst_mg: [i32; 2],
    pst_eg: [i32; 2],
    phase_units: i32,
    non_pawn_pieces: i32,
    // Estructura sin el peso dependiente de personalidad: valores negativos.
    pawn_structure_raw: [i32; 2],
    king_shield_raw: [i32; 2],
    passed_pawns: [Bitboard; 2],
    rook_activity_raw: [i32; 2],
    movilidad: [i32; 2],
    king_attack_units: [i32; 2],
    king_attackers: [i32; 2],
}

#[inline]
fn eval_idx(color: Color) -> usize {
    if color == Color::White { 1 } else { 0 }
}

#[inline]
fn pst_idx(color: Color, sq: Square) -> usize {
    if color == Color::White {
        (sq ^ 56) as usize
    } else {
        sq as usize
    }
}

#[inline]
fn phase_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Knight | PieceType::Bishop => 1,
        PieceType::Rook => 2,
        PieceType::Queen => 4,
        PieceType::Pawn | PieceType::King => 0,
    }
}

fn shield_raw(ksq: Square, color: Color, propios: Bitboard) -> i32 {
    let mut s = 0;
    let f = file_of(ksq) as i32;
    let r = rank_of(ksq) as i32;
    let dr = if color == Color::White { 1 } else { -1 };
    for df in -1..=1i32 {
        let nf = f + df;
        if (0..8).contains(&nf) {
            let r1 = r + dr;
            let r2 = r + 2 * dr;
            if (0..8).contains(&r1) && propios & (1u64 << make_square(nf as u8, r1 as u8)) != 0 {
                s += 12;
            } else if (0..8).contains(&r2)
                && propios & (1u64 << make_square(nf as u8, r2 as u8)) != 0
            {
                s += 6;
            }
        }
    }
    s
}

fn pawn_structure_raw(pawns: Bitboard) -> i32 {
    let mut score = 0;
    for f in 0..8u8 {
        let fm: Bitboard = 0x0101010101010101u64 << f;
        let adyacentes: Bitboard = (if f > 0 {
            0x0101010101010101u64 << (f - 1)
        } else {
            0
        }) | (if f < 7 {
            0x0101010101010101u64 << (f + 1)
        } else {
            0
        });
        let count = popcount(pawns & fm) as i32;
        if count > 0 {
            if count > 1 {
                score -= count - 1;
            }
            if pawns & adyacentes == 0 {
                score -= count;
            }
        }
    }
    score
}

fn passed_pawns(color: Color, propios: Bitboard, rivales: Bitboard) -> Bitboard {
    let mut resultado = 0;
    let mut bb = propios;
    while bb != 0 {
        let sq = crate::bitboard::pop_lsb(&mut bb);
        if es_pasado(color, sq, rivales) {
            resultado |= 1u64 << sq;
        }
    }
    resultado
}

fn rook_activity_raw(b: &Board, color: Color, propios: Bitboard, rivales: Bitboard) -> i32 {
    let mut score = 0;
    let mut bb = b.pieces[color as usize][PieceType::Rook as usize];
    while bb != 0 {
        let sq = crate::bitboard::pop_lsb(&mut bb);
        let r = rank_of(sq);
        let en_7ma = if color == Color::White {
            r == 6
        } else {
            r == 1
        };
        if en_7ma {
            score += 20;
        }
        let columna: Bitboard = 0x0101010101010101u64 << file_of(sq);
        let hay_propio = propios & columna != 0;
        let hay_rival = rivales & columna != 0;
        if !hay_propio && !hay_rival {
            score += 15;
        } else if !hay_propio {
            score += 8;
        }
    }
    score
}

impl ClassicalAccumulator {
    pub fn desde_tablero(b: &Board) -> ClassicalAccumulator {
        let mut acumulador = ClassicalAccumulator {
            material: [0; 2],
            pst_mg: [0; 2],
            pst_eg: [0; 2],
            phase_units: 0,
            non_pawn_pieces: 0,
            pawn_structure_raw: [0; 2],
            king_shield_raw: [0; 2],
            passed_pawns: [0; 2],
            rook_activity_raw: [0; 2],
            movilidad: [0; 2],
            king_attack_units: [0; 2],
            king_attackers: [0; 2],
        };
        for color in [Color::White, Color::Black] {
            for pt in crate::types::ALL_PIECE_TYPES {
                let mut bb = b.pieces[color as usize][pt as usize];
                while bb != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut bb);
                    acumulador.add_piece(color, pt, sq);
                }
            }
        }
        acumulador.recalcular_dependientes_de_peones(b);
        acumulador.recalcular_dinamica(b);
        acumulador
    }

    #[inline]
    fn add_piece(&mut self, color: Color, pt: PieceType, sq: Square) {
        let idx = eval_idx(color);
        self.material[idx] += VALOR[pt as usize];
        self.pst_mg[idx] += pst_mg(pt)[pst_idx(color, sq)];
        self.pst_eg[idx] += pst_eg(pt)[pst_idx(color, sq)];
        self.phase_units += phase_value(pt);
        if pt != PieceType::Pawn && pt != PieceType::King {
            self.non_pawn_pieces += 1;
        }
    }

    #[inline]
    fn remove_piece(&mut self, color: Color, pt: PieceType, sq: Square) {
        let idx = eval_idx(color);
        self.material[idx] -= VALOR[pt as usize];
        self.pst_mg[idx] -= pst_mg(pt)[pst_idx(color, sq)];
        self.pst_eg[idx] -= pst_eg(pt)[pst_idx(color, sq)];
        self.phase_units -= phase_value(pt);
        if pt != PieceType::Pawn && pt != PieceType::King {
            self.non_pawn_pieces -= 1;
        }
    }

    /// Estructura y peones pasados dependen exclusivamente de ambos
    /// bitboards de peones. Separarlos evita reescanearlos cuando una torre
    /// se mueve/captura sin que haya cambiado ningún peón.
    #[inline]
    fn recalcular_estructura_y_pasados(&mut self, b: &Board) {
        let pw = b.pieces[Color::White as usize][PieceType::Pawn as usize];
        let pb = b.pieces[Color::Black as usize][PieceType::Pawn as usize];
        self.pawn_structure_raw[1] = pawn_structure_raw(pw);
        self.pawn_structure_raw[0] = pawn_structure_raw(pb);
        self.passed_pawns[1] = passed_pawns(Color::White, pw, pb);
        self.passed_pawns[0] = passed_pawns(Color::Black, pb, pw);
    }

    /// El escudo solo depende del rey propio y de sus peones, no de torres
    /// ni del resto de la ocupación.
    #[inline]
    fn recalcular_escudos(&mut self, b: &Board) {
        let pw = b.pieces[Color::White as usize][PieceType::Pawn as usize];
        let pb = b.pieces[Color::Black as usize][PieceType::Pawn as usize];
        self.king_shield_raw[1] = shield_raw(b.king_square(Color::White), Color::White, pw);
        self.king_shield_raw[0] = shield_raw(b.king_square(Color::Black), Color::Black, pb);
    }

    /// La actividad de torres solo cambia si cambió una torre o un peón de
    /// su columna. Mantenerla aislada conserva exactamente el evaluador pero
    /// elimina los recálculos de peones pasados en movimientos de torre.
    #[inline]
    fn recalcular_actividad_torres(&mut self, b: &Board) {
        let pw = b.pieces[Color::White as usize][PieceType::Pawn as usize];
        let pb = b.pieces[Color::Black as usize][PieceType::Pawn as usize];
        self.rook_activity_raw[1] = rook_activity_raw(b, Color::White, pw, pb);
        self.rook_activity_raw[0] = rook_activity_raw(b, Color::Black, pb, pw);
    }

    fn recalcular_dependientes_de_peones(&mut self, b: &Board) {
        self.recalcular_estructura_y_pasados(b);
        self.recalcular_escudos(b);
        self.recalcular_actividad_torres(b);
    }

    /// Zona del rey que ataca `color`. Esta función se mantiene separada de
    /// la actualización para que el mismo contrato del evaluador completo se
    /// use tanto al construir como al aplicar deltas.
    #[inline]
    fn zona_rey_rival(b: &Board, color: Color) -> Bitboard {
        match color {
            Color::White => king_zone(b.king_square(Color::Black), Color::Black),
            Color::Black => king_zone(b.king_square(Color::White), Color::White),
        }
    }

    /// Contribución de una pieza a los dos términos que antes se recalculaban
    /// en cada nodo: movilidad y ataque a la zona del rey rival. Es idéntica
    /// a los dos bucles de `evaluar_clasica`/`calcular_ataque_rey`.
    #[inline]
    fn dinamica_pieza(
        pt: PieceType,
        sq: Square,
        ocupacion: Bitboard,
        color: Color,
        zona_rival: Bitboard,
    ) -> (i32, i32, i32) {
        if pt == PieceType::King {
            return (0, 0, 0);
        }
        let ataques = piece_attacks(pt, sq, ocupacion, color);
        let movilidad = PESO_MOV[pt as usize] * popcount(ataques) as i32;
        let en_zona = popcount(ataques & zona_rival) as i32;
        if pt == PieceType::Pawn {
            (movilidad, en_zona, 0)
        } else if en_zona > 0 {
            (movilidad, PESO_ATQ[pt as usize] * en_zona, 1)
        } else {
            (movilidad, 0, 0)
        }
    }

    #[inline]
    fn ajustar_dinamica_pieza(
        &mut self,
        color: Color,
        pt: PieceType,
        sq: Square,
        ocupacion: Bitboard,
        zona_rival: Bitboard,
        signo: i32,
    ) {
        let idx = eval_idx(color);
        let (movilidad, unidades, atacantes) =
            Self::dinamica_pieza(pt, sq, ocupacion, color, zona_rival);
        self.movilidad[idx] += signo * movilidad;
        self.king_attack_units[idx] += signo * unidades;
        self.king_attackers[idx] += signo * atacantes;
    }

    fn recalcular_dinamica(&mut self, b: &Board) {
        self.movilidad = [0; 2];
        self.king_attack_units = [0; 2];
        self.king_attackers = [0; 2];
        for color in [Color::White, Color::Black] {
            let zona_rival = Self::zona_rey_rival(b, color);
            for pt in [
                PieceType::Pawn,
                PieceType::Knight,
                PieceType::Bishop,
                PieceType::Rook,
                PieceType::Queen,
            ] {
                let mut bb = b.pieces[color as usize][pt as usize];
                while bb != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut bb);
                    self.ajustar_dinamica_pieza(color, pt, sq, b.occupied, zona_rival, 1);
                }
            }
        }
    }

    /// Actualiza exactamente los ataques afectados por una jugada. Peones y
    /// caballos dependen solo de su casilla, mientras que un cambio de
    /// ocupación puede abrir/cerrar rayos de alfiles, torres y damas muy lejos
    /// de la jugada. Para cada casilla de ocupación cambiada se marca la unión
    /// de los rayos antes/después; las damas se procesan una sola vez.
    fn actualizar_dinamica(&mut self, antes: &Board, despues: &Board, reyes_cambiaron: bool) {
        if reyes_cambiaron {
            self.recalcular_dinamica(despues);
            return;
        }

        for color in [Color::White, Color::Black] {
            let zona_rival = Self::zona_rey_rival(despues, color);
            for pt in [PieceType::Pawn, PieceType::Knight] {
                let before = antes.pieces[color as usize][pt as usize];
                let after = despues.pieces[color as usize][pt as usize];
                let mut salen = before & !after;
                while salen != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut salen);
                    self.ajustar_dinamica_pieza(color, pt, sq, antes.occupied, zona_rival, -1);
                }
                let mut entran = after & !before;
                while entran != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut entran);
                    self.ajustar_dinamica_pieza(color, pt, sq, despues.occupied, zona_rival, 1);
                }
            }
        }

        let mut sliders_antes_alfil = 0;
        let mut sliders_despues_alfil = 0;
        let mut sliders_antes_torre = 0;
        let mut sliders_despues_torre = 0;
        let mut afectados = 0;
        for color in [Color::White, Color::Black] {
            for pt in [PieceType::Bishop, PieceType::Queen] {
                let before = antes.pieces[color as usize][pt as usize];
                let after = despues.pieces[color as usize][pt as usize];
                sliders_antes_alfil |= before;
                sliders_despues_alfil |= after;
                afectados |= before ^ after;
            }
            for pt in [PieceType::Rook, PieceType::Queen] {
                let before = antes.pieces[color as usize][pt as usize];
                let after = despues.pieces[color as usize][pt as usize];
                sliders_antes_torre |= before;
                sliders_despues_torre |= after;
                afectados |= before ^ after;
            }
        }

        let mut ocupacion_cambiada = antes.occupied ^ despues.occupied;
        while ocupacion_cambiada != 0 {
            let sq = crate::bitboard::pop_lsb(&mut ocupacion_cambiada);
            afectados |= (bishop_attacks(sq, antes.occupied)
                | bishop_attacks(sq, despues.occupied))
                & (sliders_antes_alfil | sliders_despues_alfil);
            afectados |= (rook_attacks(sq, antes.occupied) | rook_attacks(sq, despues.occupied))
                & (sliders_antes_torre | sliders_despues_torre);
        }

        while afectados != 0 {
            let sq = crate::bitboard::pop_lsb(&mut afectados);
            if let Some((color, pt @ (PieceType::Bishop | PieceType::Rook | PieceType::Queen))) =
                antes.piece_at(sq)
            {
                self.ajustar_dinamica_pieza(
                    color,
                    pt,
                    sq,
                    antes.occupied,
                    Self::zona_rey_rival(antes, color),
                    -1,
                );
            }
            if let Some((color, pt @ (PieceType::Bishop | PieceType::Rook | PieceType::Queen))) =
                despues.piece_at(sq)
            {
                self.ajustar_dinamica_pieza(
                    color,
                    pt,
                    sq,
                    despues.occupied,
                    Self::zona_rey_rival(despues, color),
                    1,
                );
            }
        }
    }

    /// Actualiza deltas de piezas sin tocar Board. Recalcula solamente los
    /// subcomponentes que dependen de peones, rey o torres; el resto queda
    /// exacto incluso en captura, en-passant, promoción y enroque porque el
    /// diff de bitboards ve las piezas que salen y entran.
    pub fn despues_de_jugada(&self, antes: &Board, despues: &Board) -> ClassicalAccumulator {
        let mut siguiente = *self;
        let pawns_changed = antes.pieces[Color::White as usize][PieceType::Pawn as usize]
            != despues.pieces[Color::White as usize][PieceType::Pawn as usize]
            || antes.pieces[Color::Black as usize][PieceType::Pawn as usize]
                != despues.pieces[Color::Black as usize][PieceType::Pawn as usize];
        let kings_changed = antes.pieces[Color::White as usize][PieceType::King as usize]
            != despues.pieces[Color::White as usize][PieceType::King as usize]
            || antes.pieces[Color::Black as usize][PieceType::King as usize]
                != despues.pieces[Color::Black as usize][PieceType::King as usize];
        let rooks_changed = antes.pieces[Color::White as usize][PieceType::Rook as usize]
            != despues.pieces[Color::White as usize][PieceType::Rook as usize]
            || antes.pieces[Color::Black as usize][PieceType::Rook as usize]
                != despues.pieces[Color::Black as usize][PieceType::Rook as usize];

        for color in [Color::White, Color::Black] {
            for pt in crate::types::ALL_PIECE_TYPES {
                let before = antes.pieces[color as usize][pt as usize];
                let after = despues.pieces[color as usize][pt as usize];
                let mut salen = before & !after;
                while salen != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut salen);
                    siguiente.remove_piece(color, pt, sq);
                }
                let mut entran = after & !before;
                while entran != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut entran);
                    siguiente.add_piece(color, pt, sq);
                }
            }
        }
        if pawns_changed {
            siguiente.recalcular_estructura_y_pasados(despues);
        }
        if pawns_changed || kings_changed {
            siguiente.recalcular_escudos(despues);
        }
        if pawns_changed || rooks_changed {
            siguiente.recalcular_actividad_torres(despues);
        }
        if antes.occupied != despues.occupied || kings_changed {
            siguiente.actualizar_dinamica(antes, despues, kings_changed);
        }
        siguiente
    }
}

#[derive(Clone)]
pub struct EvalState {
    classical: ClassicalAccumulator,
    nnue: Option<crate::neural::NnueAccumulator>,
}

pub fn crear_eval_state(b: &Board) -> EvalState {
    EvalState {
        classical: ClassicalAccumulator::desde_tablero(b),
        nnue: crate::neural::crear_acumulador(b),
    }
}

impl EvalState {
    pub fn despues_de_jugada(&self, antes: &Board, despues: &Board) -> EvalState {
        EvalState {
            classical: self.classical.despues_de_jugada(antes, despues),
            nnue: self
                .nnue
                .as_ref()
                .map(|acumulador| acumulador.despues_de_jugada(antes, despues)),
        }
    }

    /// Propaga únicamente el acumulador clásico. Se usa dentro de
    /// quiescence experimental cuando el stand-pat no consume NNUE: evitar
    /// crear el delta de amenazas NNUE por cada captura es la parte que hace
    /// efectiva la optimización, no solo cambiar la suma final.
    pub fn despues_de_jugada_solo_clasica(&self, antes: &Board, despues: &Board) -> EvalState {
        EvalState {
            classical: self.classical.despues_de_jugada(antes, despues),
            nnue: None,
        }
    }
}

/// Parte clasica de la evaluacion. Con `cache=None` conserva el recorrido
/// completo original y funciona como oraculo de regresion para pruebas; con
/// cache usa los componentes que no dependen de ocupacion actualizados por
/// delta en el sidecar de la busqueda.
fn evaluar_clasica(b: &Board, cache: Option<&ClassicalAccumulator>) -> i32 {
    let pers = personalidad_actual();
    let es_universal = pers == Personalidad::Universal;
    let escala_material = if es_universal {
        ESCALA_MATERIAL_UNIVERSAL
    } else {
        ESCALA_MATERIAL_TAL
    };
    let factor_ataque = if es_universal {
        FACTOR_ATAQUE_UNIVERSAL
    } else {
        FACTOR_ATAQUE_TAL
    };
    // Universal castiga mas la estructura de peones rota (filosofia tecnica
    // clasica); Tal la castiga poco a proposito, para no desalentar sacrificios.
    let peso_estructura: i32 = if es_universal { 16 } else { 8 };
    // Universal se apoya mas en la tecnica de finales (rey activo, peones
    // pasados, torres activas) ya que esa es literalmente su identidad.
    let peso_final_extra: f64 = if es_universal { 1.3 } else { 1.0 };

    let occ_w = b.occupied_co[Color::White as usize];
    let occ_b = b.occupied_co[Color::Black as usize];

    let (mat, pst_mg_sum, pst_eg_sum, phase_units, non_pawn_cached, movilidad) = match cache {
        Some(acumulador) => (
            acumulador.material,
            acumulador.pst_mg,
            acumulador.pst_eg,
            acumulador.phase_units,
            acumulador.non_pawn_pieces,
            acumulador.movilidad,
        ),
        None => {
            let mut mat = [0; 2];
            let mut pst_mg_sum = [0; 2];
            let mut pst_eg_sum = [0; 2];
            let mut movilidad = [0; 2];
            for pt in crate::types::ALL_PIECE_TYPES {
                let vpt = VALOR[pt as usize];
                let m = pst_mg(pt);
                let e = pst_eg(pt);
                for (color, idx) in [(Color::White, 1usize), (Color::Black, 0usize)] {
                    let mut bb = b.pieces[color as usize][pt as usize];
                    while bb != 0 {
                        let sq = crate::bitboard::pop_lsb(&mut bb);
                        mat[idx] += vpt;
                        let pst_idx = pst_idx(color, sq);
                        pst_mg_sum[idx] += m[pst_idx];
                        pst_eg_sum[idx] += e[pst_idx];
                        let peso = PESO_MOV[pt as usize];
                        if peso != 0 {
                            movilidad[idx] +=
                                peso * popcount(piece_attacks(pt, sq, b.occupied, color)) as i32;
                        }
                    }
                }
            }
            (mat, pst_mg_sum, pst_eg_sum, 0, 0, movilidad)
        }
    };
    let fase = if cache.is_some() {
        phase_units.min(24) as f64
    } else {
        (popcount(
            b.pieces[0][PieceType::Knight as usize] | b.pieces[1][PieceType::Knight as usize],
        ) + popcount(
            b.pieces[0][PieceType::Bishop as usize] | b.pieces[1][PieceType::Bishop as usize],
        ) + 2 * popcount(
            b.pieces[0][PieceType::Rook as usize] | b.pieces[1][PieceType::Rook as usize],
        ) + 4 * popcount(
            b.pieces[0][PieceType::Queen as usize] | b.pieces[1][PieceType::Queen as usize],
        ))
        .min(24) as f64
    };
    let mgf = fase / 24.0;
    let egf = 1.0 - mgf;

    // Estructura de peones: doblados y aislados (penalización deliberadamente baja)
    let pw = b.pieces[Color::White as usize][PieceType::Pawn as usize];
    let pb = b.pieces[Color::Black as usize][PieceType::Pawn as usize];
    let estructura = match cache {
        Some(acumulador) => [
            acumulador.pawn_structure_raw[0] * peso_estructura,
            acumulador.pawn_structure_raw[1] * peso_estructura,
        ],
        None => [
            pawn_structure_raw(pb) * peso_estructura,
            pawn_structure_raw(pw) * peso_estructura,
        ],
    };

    // Escudo de peones frente al propio rey (solo pesa en medio juego).
    let rey_w = b.king_square(Color::White);
    let rey_b = b.king_square(Color::Black);
    let (shield_w, shield_b) = match cache {
        Some(acumulador) => (acumulador.king_shield_raw[1], acumulador.king_shield_raw[0]),
        None => (
            shield_raw(rey_w, Color::White, pw),
            shield_raw(rey_b, Color::Black, pb),
        ),
    };
    let escudo_w = shield_w as f64 * mgf;
    let escudo_b = shield_b as f64 * mgf;

    let ar = match cache {
        Some(acumulador) => puntuar_ataque_rey(
            b,
            factor_ataque,
            acumulador.king_attack_units,
            acumulador.king_attackers,
        ),
        None => calcular_ataque_rey(b, factor_ataque),
    };

    // Peones pasados: bono creciente por avance, reforzado por escolta del
    // rey propio (mas cerca de la casilla de coronacion que el rey rival) y
    // a plena fuerza en el final -- en medio juego pesa la mitad porque
    // todavia no es facil de convertir con piezas en el tablero.
    let mut pasados = [0.0f64; 2];
    for (color, idx, propios, rivales) in [
        (Color::White, 1usize, pw, pb),
        (Color::Black, 0usize, pb, pw),
    ] {
        let mut bb = match cache {
            Some(acumulador) => acumulador.passed_pawns[idx],
            None => passed_pawns(color, propios, rivales),
        };
        while bb != 0 {
            let sq = crate::bitboard::pop_lsb(&mut bb);
            let base = PASO_BONUS[indice_avance(color, sq)] as f64;
            let mut bono = base * (0.5 + 0.5 * egf);
            let casilla_coronacion = if color == Color::White {
                make_square(file_of(sq), 7)
            } else {
                make_square(file_of(sq), 0)
            };
            let (rey_propio, rey_rival) = if color == Color::White {
                (rey_w, rey_b)
            } else {
                (rey_b, rey_w)
            };
            let dist_propio = distancia_chebyshev(rey_propio, casilla_coronacion);
            let dist_rival = distancia_chebyshev(rey_rival, casilla_coronacion);
            bono += (dist_rival - dist_propio) as f64 * 3.0 * egf;
            pasados[idx] += bono;
        }
    }

    // Torres en la 7ma fila (2da del rival) y en columnas abiertas/semi-
    // abiertas: mucho mas activas ahi, sobre todo en finales de torres.
    let torres_activas = match cache {
        Some(acumulador) => [
            acumulador.rook_activity_raw[0] as f64,
            acumulador.rook_activity_raw[1] as f64,
        ],
        None => [
            rook_activity_raw(b, Color::Black, pb, pw) as f64,
            rook_activity_raw(b, Color::White, pw, pb) as f64,
        ],
    };

    // Actividad de rey en finales con ventaja decisiva: cuando el material
    // ya alcanza para ganar (mas de una torre de diferencia), el objetivo
    // deja de ser "no arriesgar" y pasa a "progresar" -- se premia que el
    // rey del bando que gana se acerque al rey rival (tecnica clasica de
    // conversion). Solo pesa en el final (egf) y solo del lado que gana.
    let dif_material = (mat[1] - mat[0]) as f64 * escala_material;
    let cierre_rey = if dif_material > VENTAJA_DECISIVA {
        (7 - distancia_chebyshev(rey_w, rey_b)) as f64 * PESO_ACERCAMIENTO_REY * egf
    } else if dif_material < -VENTAJA_DECISIVA {
        -((7 - distancia_chebyshev(rey_w, rey_b)) as f64 * PESO_ACERCAMIENTO_REY * egf)
    } else {
        0.0
    };

    // Bloque especifico de personalidad: Tal mantiene piezas de ataque
    // cuando hay iniciativa (identidad agresiva/sacrificial); Universal en
    // cambio busca pareja de alfiles y restringe al rival con puestos
    // avanzados seguros (profilaxis).
    let mut extra = 0.0f64;
    if !es_universal {
        if ar.ataque_w - ar.ataque_b > 60.0 {
            if b.pieces[Color::White as usize][PieceType::Queen as usize] & occ_w != 0 {
                extra += 30.0;
            }
            extra += 8.0
                * popcount(b.pieces[Color::White as usize][PieceType::Rook as usize] & occ_w).min(2)
                    as f64;
        } else if ar.ataque_b - ar.ataque_w > 60.0 {
            if b.pieces[Color::Black as usize][PieceType::Queen as usize] & occ_b != 0 {
                extra -= 30.0;
            }
            extra -= 8.0
                * popcount(b.pieces[Color::Black as usize][PieceType::Rook as usize] & occ_b).min(2)
                    as f64;
        }
    } else {
        extra += pareja_alfiles_bonus(Color::White, b) - pareja_alfiles_bonus(Color::Black, b);

        for (color, signo, rivales) in [(Color::White, 1.0f64, pb), (Color::Black, -1.0f64, pw)] {
            for pt in [PieceType::Knight, PieceType::Bishop] {
                let mut bb = b.pieces[color as usize][pt as usize];
                while bb != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut bb);
                    if en_territorio_rival(color, sq) && puesto_avanzado_seguro(color, sq, rivales)
                    {
                        extra += signo * BONUS_PUESTO[pt as usize] as f64;
                    }
                }
            }
        }
    }

    // Simplificar hacia un final ganado -- o EVITAR simplificar estando peor
    // -- aplica a las DOS personalidades por igual, no es un gusto de estilo
    // sino un principio basico: cambiar piezas (sobre todo damas) estando
    // material abajo apaga las complicaciones/chances de swindle y hace
    // trivial la tecnica de conversion del rival. Antes vivia solo en el
    // bloque de Universal; se generaliza aca porque es correcto para
    // cualquier personalidad, no una cuestion de estilo.
    let piezas_no_peon = match cache {
        Some(_) => non_pawn_cached as f64,
        None => crate::types::ALL_PIECE_TYPES
            .iter()
            .filter(|&&pt| pt != PieceType::Pawn && pt != PieceType::King)
            .map(|&pt| popcount(b.pieces[0][pt as usize] | b.pieces[1][pt as usize]))
            .sum::<u32>() as f64,
    };
    const UMBRAL_SIMPLIFICACION: f64 = 150.0;
    if dif_material > UMBRAL_SIMPLIFICACION {
        extra += (10.0 - piezas_no_peon) * 2.0 * (0.4 + 0.6 * egf);
    } else if dif_material < -UMBRAL_SIMPLIFICACION {
        extra -= (10.0 - piezas_no_peon) * 2.0 * (0.4 + 0.6 * egf);
    }

    let total = (mat[1] - mat[0]) as f64 * escala_material
        + (pst_mg_sum[1] - pst_mg_sum[0]) as f64 * mgf
        + (pst_eg_sum[1] - pst_eg_sum[0]) as f64 * egf
        + (movilidad[1] - movilidad[0]) as f64
        + (ar.ataque_w - ar.ataque_b)
        + (estructura[1] - estructura[0]) as f64
        + (escudo_w - escudo_b)
        + (pasados[1] - pasados[0]) * peso_final_extra
        + (torres_activas[1] - torres_activas[0]) * peso_final_extra
        + cierre_rey * peso_final_extra
        + extra;

    let total_i = total.round() as i32;
    let perspectiva = if b.turn == Color::White {
        total_i
    } else {
        -total_i
    };
    perspectiva + TEMPO
}

const PESO_RED: f64 = 0.5;

/// Ruta de referencia conservada para herramientas externas y tests. La
/// búsqueda usa `evaluate_with_state`, que evita reescanear los componentes
/// clásicos que no cambiaron.
#[allow(dead_code)]
pub fn evaluate_with_nnue(b: &Board, nnue: Option<&crate::neural::NnueAccumulator>) -> i32 {
    let clasica = evaluar_clasica(b, None);
    match nnue {
        Some(acumulador) => clasica + (PESO_RED * acumulador.evaluar() as f64).round() as i32,
        None => clasica,
    }
}

pub fn evaluate_with_state(b: &Board, state: &EvalState) -> i32 {
    let clasica = evaluar_clasica(b, Some(&state.classical));
    match state.nnue.as_ref() {
        Some(acumulador) => clasica + (PESO_RED * acumulador.evaluar() as f64).round() as i32,
        None => clasica,
    }
}

/// Evaluación clásica desde el acumulador incremental ya disponible.
/// La quiescence puede usar esta ruta sin recomputar componentes clásicos.
pub fn evaluate_classical_with_state(b: &Board, state: &EvalState) -> i32 {
    evaluar_clasica(b, Some(&state.classical))
}

#[cfg(test)]
mod incremental_tests {
    use super::*;
    use crate::movegen::generate_legal;
    use std::sync::Mutex;

    // La personalidad es un atómico global UCI. Serializamos estas pruebas
    // para verificar ambas identidades sin carreras con otro test futuro.
    static PERSONALIDAD_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn comprobar_hijo(fen: &str, uci: &str) {
        let antes = Board::from_fen(fen).unwrap();
        let estado = crear_eval_state(&antes);
        let mv = generate_legal(&antes)
            .into_iter()
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("jugada legal no encontrada: {uci}"));
        let despues = antes.make_move(&mv);
        let incremental = estado.despues_de_jugada(&antes, &despues);
        let solo_clasico = estado.despues_de_jugada_solo_clasica(&antes, &despues);

        assert_eq!(
            incremental.classical,
            ClassicalAccumulator::desde_tablero(&despues)
        );
        assert!(solo_clasico.nnue.is_none());
        for personalidad in [Personalidad::Tal, Personalidad::Universal] {
            set_personalidad(personalidad);
            assert_eq!(
                evaluate_with_state(&despues, &incremental),
                evaluate_with_nnue(&despues, None),
                "cache clasico difiere de referencia para {uci} con {:?}",
                personalidad
            );
            assert_eq!(
                evaluate_with_state(&despues, &solo_clasico),
                evaluate_with_nnue(&despues, None),
                "estado solo clasico difiere de referencia para {uci} con {:?}",
                personalidad
            );
        }
        set_personalidad(Personalidad::Tal);
    }

    #[test]
    fn acumulador_clasico_cubre_movimientos_especiales() {
        let _guard = PERSONALIDAD_TEST_LOCK.lock().unwrap();
        let casos = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "e2e4",
            ),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
                "e7e5",
            ),
            ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3"),
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1"),
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1c1"),
            ("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", "e8g8"),
            ("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", "e8c8"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1q"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1n"),
            ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"),
            ("4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1", "a2b1q"),
            // Cambios de ocupación que abren/cortan rayos lejanos: son los
            // casos importantes para la actualización local de sliders.
            ("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1", "a2a4"),
            ("r3k3/p7/8/8/8/8/8/4K3 b - - 0 1", "a7a5"),
            ("4k3/8/8/8/8/8/1P6/2B1K3 w - - 0 1", "b2b4"),
            ("2b1k3/1p6/8/8/8/8/8/4K3 b - - 0 1", "b7b5"),
            ("4k3/8/8/8/8/8/p7/R3K3 w - - 0 1", "a1a2"),
        ];
        for (fen, uci) in casos {
            comprobar_hijo(fen, uci);
        }
    }

    #[test]
    fn acumulador_clasico_conserva_padre_y_null_move() {
        let _guard = PERSONALIDAD_TEST_LOCK.lock().unwrap();
        let padre = Board::startpos();
        let estado_padre = crear_eval_state(&padre);
        let movimientos = generate_legal(&padre);
        for mv in movimientos.iter().take(2) {
            let hijo = padre.make_move(mv);
            let estado_hijo = estado_padre.despues_de_jugada(&padre, &hijo);
            assert_eq!(
                estado_hijo.classical,
                ClassicalAccumulator::desde_tablero(&hijo)
            );
            assert_eq!(
                estado_padre.classical,
                ClassicalAccumulator::desde_tablero(&padre)
            );
        }
        let null = padre.make_null_move();
        let estado_null = estado_padre.despues_de_jugada(&padre, &null);
        assert_eq!(estado_null.classical, estado_padre.classical);
        assert_eq!(
            evaluate_with_state(&null, &estado_null),
            evaluate_with_nnue(&null, None)
        );
    }

    #[test]
    fn acumulador_clasico_fuzz_determinista() {
        let _guard = PERSONALIDAD_TEST_LOCK.lock().unwrap();
        let mut semilla = 0x5EED_CAFE_D00Du64;
        for _partida in 0..32 {
            let mut tablero = Board::startpos();
            let mut estado = crear_eval_state(&tablero);
            for _ply in 0..128 {
                let legales = generate_legal(&tablero);
                if legales.is_empty() {
                    break;
                }
                semilla ^= semilla << 7;
                semilla ^= semilla >> 9;
                let mv = legales[(semilla as usize) % legales.len()];
                let siguiente = tablero.make_move(&mv);
                let estado_siguiente = estado.despues_de_jugada(&tablero, &siguiente);
                assert_eq!(
                    estado_siguiente.classical,
                    ClassicalAccumulator::desde_tablero(&siguiente),
                    "diferencia tras {}",
                    mv.to_uci()
                );
                assert_eq!(
                    evaluate_with_state(&siguiente, &estado_siguiente),
                    evaluate_with_nnue(&siguiente, None),
                    "score clasico difiere tras {}",
                    mv.to_uci()
                );
                tablero = siguiente;
                estado = estado_siguiente;
            }
        }
    }
}
