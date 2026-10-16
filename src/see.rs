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

/// `bb` se recibe por referencia MUTABLE y se restaura antes de volver: el
/// atacante elegido se quita, se recursa y se vuelve a poner. Antes cada
/// nivel copiaba el arreglo completo (`[[Bitboard; 6]; 2]`, 96 bytes) solo
/// para apagar UN bit. Con el hacer/deshacer el estado que ve cada nivel es
/// exactamente el mismo (quitar y volver a poner el mismo bit con `&= !b` y
/// `|= b` es la operacion inversa exacta) y el llamador recupera su arreglo
/// intacto, asi que el valor devuelto es identico.
#[cfg_attr(not(test), allow(dead_code))]
fn see_recurse(
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
        if pt == PieceType::King && !rey_puede_capturar(color_en_turno, sq, to_sq, occupied, bb) {
            continue;
        }

        let occ2 = occupied & !bit(sq);
        let (nuevo_valor, bonus_promocion) = resultado_capturador(color_en_turno, pt, to_sq);
        bb[color_en_turno as usize][pt as usize] &= !bit(sq);
        let resto = see_recurse(to_sq, occ2, bb, color_en_turno.opposite(), nuevo_valor);
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

    let resto = see_iterativo(to_sq, occupied, &mut bb, color_atacante.opposite(), valor_en_to);
    gain0 - resto
}

/// Evalua si la ganancia estatica de `mv` es mayor o igual a `threshold`.
/// Aprovecha cotas matematicas exactas para podar el calculo de rayos y
/// atacantes en la mayoria de las llamadas:
/// - Ganancia maxima posible es `gain0` (el rival solo puede restar).
/// - Si `gain0 < threshold`, es imposible alcanzarlo -> `false`.
/// - Si `gain0 - valor(atacante) >= threshold` (y no es rey), aun perdiendo la
///   pieza inmediatamente la ganancia no bajara de ahi -> `true`.
#[inline]
pub fn see_ge(b: &Board, mv: &Move, threshold: i32) -> bool {
    let to_sq = mv.to;
    let from_sq = mv.from;
    let es_al_paso = mv.flag == MoveFlag::EnPassant;

    let victima_tipo = if es_al_paso {
        Some(PieceType::Pawn)
    } else {
        b.piece_at(to_sq).map(|(_, pt)| pt)
    };
    let (_color_atacante, atacante_tipo) = b.piece_at(from_sq).expect("see_ge: no hay pieza en 'from'");

    let mut gain0 = victima_tipo.map(valor).unwrap_or(0);
    let valor_en_to = if let Some(promo) = mv.promotion {
        gain0 += valor(promo) - valor(PieceType::Pawn);
        valor(promo)
    } else {
        valor(atacante_tipo)
    };

    if gain0 < threshold {
        return false;
    }

    if atacante_tipo != PieceType::King && gain0 - valor_en_to >= threshold {
        return true;
    }

    see(b, mv) >= threshold
}

/// VELOCIDAD (exacto): misma funcion que `see_recurse`, sin recursion y sin
/// recalcular los rayos deslizantes en cada nivel.
///
/// `see_recurse` elige en cada nivel el PRIMER capturador disponible por
/// orden de valor (y si es el rey y no puede capturar legalmente, nadie) y
/// devuelve `max(0, termino - resto)`: no ramifica, asi que la secuencia de
/// capturas es determinista. Aqui se recorre hacia adelante guardando los
/// terminos (valor en la casilla + bonus de promocion) y se pliega hacia
/// atras con el mismo `max(0, t - r)`: resultado identico.
///
/// Los atacantes de cada nivel son los mismos que da `atacantes_a` con la
/// ocupacion vigente: los ataques de rey/caballo/peon se filtran por los
/// bitboards locales (que pierden cada pieza usada), y los rayos de alfil y
/// torre DESDE `to_sq` solo cambian si la pieza quitada estaba en ellos
/// (si no es primer bloqueador de ningun rayo, quitarla no cambia el
/// conjunto), asi que solo entonces se recalculan.
fn see_iterativo(
    to_sq: Square,
    mut occupied: Bitboard,
    bb: &mut [[Bitboard; 6]; 2],
    mut color: Color,
    mut valor_en_to: i32,
) -> i32 {
    const B: usize = PieceType::Bishop as usize;
    const R: usize = PieceType::Rook as usize;
    const Q: usize = PieceType::Queen as usize;
    // Como mucho 32 piezas pueden capturar en la casilla.
    let mut terminos = [0i32; 32];
    let mut n = 0usize;
    let mut rayos_d = bishop_attacks(to_sq, occupied);
    let mut rayos_o = rook_attacks(to_sq, occupied);
    let fijos_rey = king_attacks(to_sq);
    let fijos_caballo = knight_attacks(to_sq);
    let fijos_peones = [
        pawn_attacks(Color::Black, to_sq),
        pawn_attacks(Color::White, to_sq),
    ];
    let fuera = !bit(to_sq);
    loop {
        let idx = color as usize;
        let p = &bb[idx];
        let atacantes = ((fijos_rey & p[PieceType::King as usize])
            | (fijos_caballo & p[PieceType::Knight as usize])
            | (rayos_d & (p[B] | p[Q]))
            | (rayos_o & (p[R] | p[Q]))
            | (fijos_peones[idx] & p[PieceType::Pawn as usize]))
            & fuera;
        if atacantes == 0 {
            break;
        }
        let mut elegido = None;
        for &pt in &ORDEN_POR_VALOR {
            let disponibles = atacantes & bb[idx][pt as usize];
            if disponibles == 0 {
                continue;
            }
            let sq = lsb(disponibles);
            if pt == PieceType::King && !rey_puede_capturar(color, sq, to_sq, occupied, bb) {
                continue;
            }
            elegido = Some((pt, sq));
            break;
        }
        let Some((pt, sq)) = elegido else { break };
        let (nuevo_valor, bonus_promocion) = resultado_capturador(color, pt, to_sq);
        if n < terminos.len() {
            terminos[n] = valor_en_to + bonus_promocion;
            n += 1;
        } else {
            // Inalcanzable (32 piezas como maximo); por las dudas, la
            // referencia exacta.
            debug_assert!(false, "see_iterativo: mas de 32 capturas");
            break;
        }
        let b_sq = bit(sq);
        occupied &= !b_sq;
        bb[idx][pt as usize] &= !b_sq;
        if rayos_d & b_sq != 0 {
            rayos_d = bishop_attacks(to_sq, occupied);
        }
        if rayos_o & b_sq != 0 {
            rayos_o = rook_attacks(to_sq, occupied);
        }
        valor_en_to = nuevo_valor;
        color = color.opposite();
    }
    let mut resto = 0;
    for &t in terminos[..n].iter().rev() {
        resto = (t - resto).max(0);
    }
    resto
}

/// SEE original (recursiva), referencia exacta de `see` en los tests.
#[cfg(test)]
pub fn see_referencia(b: &Board, mv: &Move) -> i32 {
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
    let resto = see_recurse(to_sq, occupied, &mut bb, color_atacante.opposite(), valor_en_to);
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

    /// VELOCIDAD bit-exacta: `see` (iterativa) debe dar EXACTAMENTE lo
    /// mismo que la recursion original para TODAS las jugadas legales (no
    /// solo capturas: la busqueda tambien la llama para quiets en algunos
    /// sitios) de miles de posiciones de recorridos aleatorios con muchas
    /// capturas, promociones, al paso y reyes cerca de la accion.
    #[test]
    fn see_iterativa_igual_que_recursiva() {
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 11",
            "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 4 8",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/4k3/4p3/3K4/8/8/8/4R3 w - - 0 1",
        ];
        let mut semilla = 0x8C4F_2A6B_D15E_0937u64;
        let mut azar = move || {
            semilla ^= semilla << 13;
            semilla ^= semilla >> 7;
            semilla ^= semilla << 17;
            semilla
        };
        let (mut comparadas, mut capturas) = (0u32, 0u32);
        for fen in fens {
            for _ in 0..120 {
                let mut b = Board::from_fen(fen).unwrap();
                for _ in 0..50 {
                    let jugadas = generate_legal(&b);
                    if jugadas.is_empty() {
                        break;
                    }
                    for mv in &jugadas {
                        assert_eq!(see(&b, mv), see_referencia(&b, mv),
                            "SEE distinta para {} en {}", mv.to_uci(), b.to_fen());
                        comparadas += 1;
                        capturas += mv.is_capture() as u32;
                    }
                    // Preferir capturas para llegar a posiciones tacticas.
                    let caps: Vec<_> = jugadas.iter().filter(|m| m.is_capture()).collect();
                    let mv = if !caps.is_empty() && azar() % 3 != 0 {
                        *caps[(azar() % caps.len() as u64) as usize]
                    } else {
                        jugadas[(azar() % jugadas.len() as u64) as usize]
                    };
                    b = b.make_move(&mv);
                }
            }
        }
        assert!(comparadas > 200_000 && capturas > 20_000, "{comparadas} {capturas}");
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

    #[test]
    fn see_ge_coincide_con_see_en_recorridos_aleatorios() {
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 11",
            "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 4 8",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/4k3/4p3/3K4/8/8/8/4R3 w - - 0 1",
        ];
        let umbrales = [-1500, -500, -200, -50, -1, 0, 1, 50, 200, 500, 1500];
        for fen in fens {
            let b = Board::from_fen(fen).unwrap();
            for mv in generate_legal(&b) {
                let real = see(&b, &mv);
                for &u in &umbrales {
                    assert_eq!(
                        see_ge(&b, &mv, u),
                        real >= u,
                        "Divergencia see_ge({u}) para {} en {}: real={}",
                        mv.to_uci(),
                        b.to_fen(),
                        real
                    );
                }
            }
        }
    }
}
