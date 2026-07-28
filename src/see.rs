// Static Exchange Evaluation (SEE).
//
// Simula capturas y recapturas sobre la casilla de destino usando bitboards
// locales. La version reparada trata al rey como la pieza mas valiosa, impide
// recapturas ilegales del rey sobre casillas defendidas y suma correctamente
// la ganancia de una promocion.

use crate::bitboard::{
    Bitboard, bishop_attacks, bit, king_attacks, knight_attacks, lsb, pawn_attacks, rook_attacks,
};
use crate::board::Board;
use crate::types::{Color, Move, MoveFlag, PieceType, Square, file_of, make_square, rank_of};

const VALOR: [i32; 6] = [100, 320, 330, 500, 900, 20_000];
const ORDEN_POR_VALOR: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
];

#[inline]
fn valor(pt: PieceType) -> i32 {
    VALOR[pt as usize]
}

#[inline]
fn promociona_en(color: Color, pt: PieceType, to_sq: Square) -> bool {
    pt == PieceType::Pawn
        && ((color == Color::White && rank_of(to_sq) == 7)
            || (color == Color::Black && rank_of(to_sq) == 0))
}

/// Valor de la pieza que queda en `to_sq` y bonificacion inmediata por
/// promocion. SEE supone dama para recapturas que coronan, la opcion material
/// maxima; la jugada inicial respeta la promocion indicada por UCI.
#[inline]
fn resultado_capturador(color: Color, pt: PieceType, to_sq: Square) -> (i32, i32) {
    if promociona_en(color, pt, to_sq) {
        (
            valor(PieceType::Queen),
            valor(PieceType::Queen) - valor(PieceType::Pawn),
        )
    } else {
        (valor(pt), 0)
    }
}

/// Atacantes de `color` sobre `sq`, calculados con los bitboards locales de
/// la secuencia SEE.
fn atacantes_a(color: Color, sq: Square, occupied: Bitboard, bb: &[[Bitboard; 6]; 2]) -> Bitboard {
    let idx = color as usize;
    (king_attacks(sq) & bb[idx][PieceType::King as usize])
        | (knight_attacks(sq) & bb[idx][PieceType::Knight as usize])
        | (bishop_attacks(sq, occupied)
            & (bb[idx][PieceType::Bishop as usize] | bb[idx][PieceType::Queen as usize]))
        | (rook_attacks(sq, occupied)
            & (bb[idx][PieceType::Rook as usize] | bb[idx][PieceType::Queen as usize]))
        | (pawn_attacks(color.opposite(), sq) & bb[idx][PieceType::Pawn as usize])
}

/// Comprueba la legalidad especial de una recaptura del rey. Un rey no puede
/// entrar en una casilla atacada. Las piezas clavadas no se filtran en SEE
/// clasico, pero el rey si debe tratarse con exactitud porque de otro modo una
/// captura defendida puede parecer materialmente perdedora cuando no lo es.
fn rey_puede_capturar(
    color: Color,
    from_sq: Square,
    to_sq: Square,
    occupied: Bitboard,
    bb: &[[Bitboard; 6]; 2],
) -> bool {
    let mut occ2 = occupied;
    let mut bb2 = *bb;
    occ2 &= !bit(from_sq);
    bb2[color as usize][PieceType::King as usize] &= !bit(from_sq);
    let ataques = atacantes_a(color.opposite(), to_sq, occ2, &bb2) & !bit(to_sq);
    ataques == 0
}

fn see_recurse(
    to_sq: Square,
    occupied: Bitboard,
    bb: &[[Bitboard; 6]; 2],
    color_en_turno: Color,
    valor_en_to: i32,
) -> i32 {
    let atacantes = atacantes_a(color_en_turno, to_sq, occupied, bb) & !bit(to_sq);
    for &pt in &ORDEN_POR_VALOR {
        let disponibles = atacantes & bb[color_en_turno as usize][pt as usize];
        if disponibles == 0 {
            continue;
        }
        let sq = lsb(disponibles);
        if pt == PieceType::King && !rey_puede_capturar(color_en_turno, sq, to_sq, occupied, bb) {
            continue;
        }

        let mut occ2 = occupied;
        let mut bb2 = *bb;
        occ2 &= !bit(sq);
        bb2[color_en_turno as usize][pt as usize] &= !bit(sq);
        let (nuevo_valor, bonus_promocion) = resultado_capturador(color_en_turno, pt, to_sq);
        let g = valor_en_to + bonus_promocion
            - see_recurse(to_sq, occ2, &bb2, color_en_turno.opposite(), nuevo_valor);
        return g.max(0);
    }
    0
}

pub fn see(b: &Board, mv: &Move) -> i32 {
    let to_sq = mv.to;
    let from_sq = mv.from;
    let es_al_paso = mv.flag == MoveFlag::EnPassant;

    let victima_tipo = if es_al_paso {
        Some(PieceType::Pawn)
    } else {
        b.piece_at(to_sq).map(|(_, pt)| pt)
    };
    let (color_atacante, atacante_tipo) = b.piece_at(from_sq).expect("see: no hay pieza en 'from'");

    let mut occupied = b.occupied;
    let mut bb = b.pieces;
    let mut gain0 = victima_tipo.map(valor).unwrap_or(0);

    bb[color_atacante as usize][atacante_tipo as usize] &= !bit(from_sq);
    occupied &= !bit(from_sq);

    if es_al_paso {
        let them = color_atacante.opposite();
        let victima_sq = make_square(file_of(to_sq), rank_of(from_sq));
        bb[them as usize][PieceType::Pawn as usize] &= !bit(victima_sq);
        occupied &= !bit(victima_sq);
    } else if let Some((color_victima, tipo_victima)) = b.piece_at(to_sq) {
        bb[color_victima as usize][tipo_victima as usize] &= !bit(to_sq);
    }

    let valor_en_to = if let Some(promo) = mv.promotion {
        gain0 += valor(promo) - valor(PieceType::Pawn);
        valor(promo)
    } else {
        valor(atacante_tipo)
    };

    let resto = see_recurse(to_sq, occupied, &bb, color_atacante.opposite(), valor_en_to);
    gain0 - resto
}

// Oraculo de fuerza bruta: prueba todos los atacantes disponibles. Comparte
// las reglas de legalidad del rey y promocion, pero no el atajo de elegir solo
// el atacante de menor valor.
fn oracle_recurse(
    to_sq: Square,
    occupied: Bitboard,
    bb: &[[Bitboard; 6]; 2],
    color_en_turno: Color,
    valor_en_to: i32,
) -> i32 {
    let atacantes = atacantes_a(color_en_turno, to_sq, occupied, bb) & !bit(to_sq);
    let mut mejor = 0;
    for &pt in &ORDEN_POR_VALOR {
        let mut candidatos = atacantes & bb[color_en_turno as usize][pt as usize];
        while candidatos != 0 {
            let sq = crate::bitboard::pop_lsb(&mut candidatos);
            if pt == PieceType::King && !rey_puede_capturar(color_en_turno, sq, to_sq, occupied, bb)
            {
                continue;
            }
            let mut occ2 = occupied;
            let mut bb2 = *bb;
            occ2 &= !bit(sq);
            bb2[color_en_turno as usize][pt as usize] &= !bit(sq);
            let (nuevo_valor, bonus_promocion) = resultado_capturador(color_en_turno, pt, to_sq);
            let g = valor_en_to + bonus_promocion
                - oracle_recurse(to_sq, occ2, &bb2, color_en_turno.opposite(), nuevo_valor);
            mejor = mejor.max(g);
        }
    }
    mejor
}

pub fn see_oracle(b: &Board, mv: &Move) -> i32 {
    let to_sq = mv.to;
    let from_sq = mv.from;
    let es_al_paso = mv.flag == MoveFlag::EnPassant;
    let victima_tipo = if es_al_paso {
        Some(PieceType::Pawn)
    } else {
        b.piece_at(to_sq).map(|(_, pt)| pt)
    };
    let (color_atacante, atacante_tipo) = b
        .piece_at(from_sq)
        .expect("see_oracle: no hay pieza en 'from'");

    let mut occupied = b.occupied;
    let mut bb = b.pieces;
    let mut gain0 = victima_tipo.map(valor).unwrap_or(0);
    bb[color_atacante as usize][atacante_tipo as usize] &= !bit(from_sq);
    occupied &= !bit(from_sq);

    if es_al_paso {
        let them = color_atacante.opposite();
        let victima_sq = make_square(file_of(to_sq), rank_of(from_sq));
        bb[them as usize][PieceType::Pawn as usize] &= !bit(victima_sq);
        occupied &= !bit(victima_sq);
    } else if let Some((color_victima, tipo_victima)) = b.piece_at(to_sq) {
        bb[color_victima as usize][tipo_victima as usize] &= !bit(to_sq);
    }

    let valor_en_to = if let Some(promo) = mv.promotion {
        gain0 += valor(promo) - valor(PieceType::Pawn);
        valor(promo)
    } else {
        valor(atacante_tipo)
    };
    let resto = oracle_recurse(to_sq, occupied, &bb, color_atacante.opposite(), valor_en_to);
    gain0 - resto
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal;

    fn move_uci(b: &Board, uci: &str) -> Move {
        generate_legal(b)
            .into_iter()
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("jugada legal no encontrada: {uci}"))
    }

    #[test]
    fn rey_no_recaptura_en_casilla_defendida() {
        let b = Board::from_fen("8/4k3/4p3/3K4/8/8/8/4R3 w - - 0 1").unwrap();
        let mv = move_uci(&b, "e1e6");
        assert_eq!(see(&b, &mv), 100);
    }

    #[test]
    fn captura_promocion_suma_la_pieza_nueva() {
        let b = Board::from_fen("k6r/6P1/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let mv = move_uci(&b, "g7h8q");
        assert_eq!(see(&b, &mv), 1_300);
    }
}
