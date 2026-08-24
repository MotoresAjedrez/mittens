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
    // Antes se copiaba el arreglo entero de bitboards (96 bytes) para quitarle
    // el rey de `color`... y despues se preguntaba por los atacantes de
    // `color.opposite()`. `atacantes_a` lee UNICAMENTE `bb[color_que_ataca]`,
    // asi que esa modificacion nunca se leia: la copia era trabajo muerto.
    // Lo unico que de verdad importa es sacar al rey de la OCUPACION (para
    // que no se tape a si mismo de un jaque deslizante), y eso sigue igual.
    let occ2 = occupied & !bit(from_sq);
    let ataques = atacantes_a(color.opposite(), to_sq, occ2, bb) & !bit(to_sq);
    ataques == 0
}

/// Atacantes de LOS DOS colores sobre `sq`. Misma geometria que
/// `atacantes_a` (que solo mira un color), reunida en un unico bitboard:
/// como los bitboards por tipo son disjuntos, `atacantes_ambos(..) &
/// bb[color][pt]` da EXACTAMENTE lo mismo que `atacantes_a(color, ..) &
/// bb[color][pt]`, que es la unica forma en que la secuencia SEE consulta el
/// conjunto.
#[inline]
fn atacantes_ambos(sq: Square, occupied: Bitboard, bb: &[[Bitboard; 6]; 2]) -> Bitboard {
    let b0 = &bb[Color::White as usize];
    let b1 = &bb[Color::Black as usize];
    let diagonales = b0[PieceType::Bishop as usize]
        | b0[PieceType::Queen as usize]
        | b1[PieceType::Bishop as usize]
        | b1[PieceType::Queen as usize];
    let ortogonales = b0[PieceType::Rook as usize]
        | b0[PieceType::Queen as usize]
        | b1[PieceType::Rook as usize]
        | b1[PieceType::Queen as usize];
    (king_attacks(sq) & (b0[PieceType::King as usize] | b1[PieceType::King as usize]))
        | (knight_attacks(sq) & (b0[PieceType::Knight as usize] | b1[PieceType::Knight as usize]))
        | (bishop_attacks(sq, occupied) & diagonales)
        | (rook_attacks(sq, occupied) & ortogonales)
        | (pawn_attacks(Color::Black, sq) & b0[PieceType::Pawn as usize])
        | (pawn_attacks(Color::White, sq) & b1[PieceType::Pawn as usize])
}

/// Pone al dia el conjunto de atacantes despues de sacar de `sq` a un
/// atacante de tipo `pt_quitado` (que ya NO esta ni en `occ2` ni en `bb`).
///
/// Quitar un bloqueo solo puede AGREGAR atacantes deslizantes que estaban
/// tapados detras de esa casilla; nunca puede quitar ninguno. Y solo por el
/// rayo por el que el atacante quitado miraba a `to_sq`:
///   - peon y alfil estaban en una DIAGONAL   -> puede aparecer alfil/dama;
///   - torre estaba en fila o columna         -> puede aparecer torre/dama;
///   - dama y rey pueden estar en cualquiera  -> hay que mirar las dos;
///   - caballo no esta en ninguna linea de dama -> no destapa nada.
/// Por eso el caso comun cuesta UN lookup magico en vez de los dos (mas los
/// tres baratos de rey/caballo/peon) que costaba recalcular todo el conjunto.
#[inline]
fn destapar_atacantes(
    to_sq: Square,
    occ2: Bitboard,
    atacantes: Bitboard,
    pt_quitado: PieceType,
    bb: &[[Bitboard; 6]; 2],
) -> Bitboard {
    let mut a = atacantes;
    let b0 = &bb[Color::White as usize];
    let b1 = &bb[Color::Black as usize];
    let mira_diagonal = matches!(
        pt_quitado,
        PieceType::Pawn | PieceType::Bishop | PieceType::Queen | PieceType::King
    );
    let mira_ortogonal = matches!(
        pt_quitado,
        PieceType::Rook | PieceType::Queen | PieceType::King
    );
    if mira_diagonal {
        let diagonales = b0[PieceType::Bishop as usize]
            | b0[PieceType::Queen as usize]
            | b1[PieceType::Bishop as usize]
            | b1[PieceType::Queen as usize];
        a |= bishop_attacks(to_sq, occ2) & diagonales & occ2;
    }
    if mira_ortogonal {
        let ortogonales = b0[PieceType::Rook as usize]
            | b0[PieceType::Queen as usize]
            | b1[PieceType::Rook as usize]
            | b1[PieceType::Queen as usize];
        a |= rook_attacks(to_sq, occ2) & ortogonales & occ2;
    }
    a
}

/// `bb` se recibe por referencia MUTABLE y se restaura antes de volver: el
/// atacante elegido se quita, se recursa y se vuelve a poner. Antes cada
/// nivel copiaba el arreglo completo (`[[Bitboard; 6]; 2]`, 96 bytes) solo
/// para apagar UN bit. Con el hacer/deshacer el estado que ve cada nivel es
/// exactamente el mismo (quitar y volver a poner el mismo bit con `&= !b` y
/// `|= b` es la operacion inversa exacta) y el llamador recupera su arreglo
/// intacto, asi que el valor devuelto es identico.
///
/// `atacantes` llega YA CALCULADO por el nivel de arriba (INCREMENTAL): antes
/// cada nivel lo recalculaba entero con `atacantes_a`, que son cinco consultas
/// de ataque, dos de ellas magicas. Medido con contadores en `bench 14`:
/// 1.371.598 llamadas a `see()` y 2.589.618 a `atacantes_a` (1,89 por SEE).
fn see_recurse(
    to_sq: Square,
    occupied: Bitboard,
    atacantes: Bitboard,
    bb: &mut [[Bitboard; 6]; 2],
    color_en_turno: Color,
    valor_en_to: i32,
) -> i32 {
    for &pt in &ORDEN_POR_VALOR {
        let disponibles = atacantes & bb[color_en_turno as usize][pt as usize];
        if disponibles == 0 {
            continue;
        }
        let sq = lsb(disponibles);
        if pt == PieceType::King && !rey_puede_capturar(color_en_turno, sq, to_sq, occupied, bb) {
            continue;
        }

        let occ2 = occupied & !bit(sq);
        let (nuevo_valor, bonus_promocion) = resultado_capturador(color_en_turno, pt, to_sq);
        bb[color_en_turno as usize][pt as usize] &= !bit(sq);
        let atacantes2 = destapar_atacantes(to_sq, occ2, atacantes & !bit(sq), pt, bb);
        let resto = see_recurse(
            to_sq,
            occ2,
            atacantes2,
            bb,
            color_en_turno.opposite(),
            nuevo_valor,
        );
        bb[color_en_turno as usize][pt as usize] |= bit(sq);
        let g = valor_en_to + bonus_promocion - resto;
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

    // Conjunto inicial de atacantes, ya con la ocupacion definitiva (sin la
    // pieza que se fue de `from` y, si fue al paso, sin el peon comido): a
    // partir de aca cada nivel lo hereda y solo le destapa lo que corresponda.
    let atacantes = atacantes_ambos(to_sq, occupied, &bb) & !bit(to_sq);
    let resto = see_recurse(
        to_sq,
        occupied,
        atacantes,
        &mut bb,
        color_atacante.opposite(),
        valor_en_to,
    );
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

    /// Version de REFERENCIA de la recursion: recalcula el conjunto de
    /// atacantes de cero en cada nivel, tal como se hacia antes de la
    /// version incremental. Sirve de oraculo exacto para el test de abajo.
    fn see_recurse_referencia(
        to_sq: Square,
        occupied: Bitboard,
        bb: &mut [[Bitboard; 6]; 2],
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
            if pt == PieceType::King
                && !rey_puede_capturar(color_en_turno, sq, to_sq, occupied, bb)
            {
                continue;
            }
            let occ2 = occupied & !bit(sq);
            let (nuevo_valor, bonus_promocion) = resultado_capturador(color_en_turno, pt, to_sq);
            bb[color_en_turno as usize][pt as usize] &= !bit(sq);
            let resto =
                see_recurse_referencia(to_sq, occ2, bb, color_en_turno.opposite(), nuevo_valor);
            bb[color_en_turno as usize][pt as usize] |= bit(sq);
            let g = valor_en_to + bonus_promocion - resto;
            return g.max(0);
        }
        0
    }

    fn see_referencia(b: &Board, mv: &Move) -> i32 {
        let to_sq = mv.to;
        let from_sq = mv.from;
        let es_al_paso = mv.flag == MoveFlag::EnPassant;
        let victima_tipo = if es_al_paso {
            Some(PieceType::Pawn)
        } else {
            b.piece_at(to_sq).map(|(_, pt)| pt)
        };
        let (color_atacante, atacante_tipo) =
            b.piece_at(from_sq).expect("see_referencia: no hay pieza en 'from'");
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
        let resto = see_recurse_referencia(
            to_sq,
            occupied,
            &mut bb,
            color_atacante.opposite(),
            valor_en_to,
        );
        gain0 - resto
    }

    fn comparar_see_en(b: &Board) {
        for mv in generate_legal(b) {
            let rapido = see(b, &mv);
            let lento = see_referencia(b, &mv);
            assert_eq!(
                rapido,
                lento,
                "SEE incremental dio {} y el de referencia {} para {} en {}",
                rapido,
                lento,
                mv.to_uci(),
                b.to_fen()
            );
        }
    }

    fn explorar_see(b: &Board, profundidad: u32) {
        comparar_see_en(b);
        if profundidad == 0 {
            return;
        }
        for mv in generate_legal(b) {
            explorar_see(&b.make_move(&mv), profundidad - 1);
        }
    }

    /// El conjunto de atacantes de la secuencia SEE se mantiene INCREMENTAL
    /// (cada nivel hereda el del anterior y solo destapa los rayos que abrio
    /// la pieza que se fue). Este test contrasta el resultado contra la
    /// version que recalculaba todo de cero, jugada por jugada, en un arbol
    /// de posiciones con clavadas, rayos-x, al paso, promociones y finales.
    /// Un solo rayo destapado de menos cambiaria en silencio el orden de
    /// jugadas y todas las podas por SEE.
    #[test]
    fn see_incremental_coincide_con_recalcular_todo() {
        let posiciones = [
            Board::startpos(),
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -")
                .expect("fen valido"),
            // Bateria de torres y damas en columna: rayos-x encadenados.
            Board::from_fen("3r1k2/3r4/3q4/3p4/3P4/3Q4/3R4/3R1K2 w - - 0 1").expect("fen valido"),
            // Al paso con torre detras en la quinta fila.
            Board::from_fen("8/8/8/R2pP2k/8/8/8/K7 w - d6 0 1").expect("fen valido"),
            // Promociones con captura y piezas menores amontonadas.
            Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
                .expect("fen valido"),
            // Final con reyes cerca: recapturas de rey y casillas defendidas.
            Board::from_fen("8/2p2pk1/1p1p2p1/p2Pp2p/P1P1P2P/1P3PP1/6K1/8 w - - 0 1")
                .expect("fen valido"),
            // Alfiles en fianchetto con damas detras: rayos diagonales largos.
            Board::from_fen("2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 11")
                .expect("fen valido"),
        ];
        for b in &posiciones {
            explorar_see(b, 2);
        }
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
