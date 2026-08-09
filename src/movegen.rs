use crate::bitboard::{
    Bitboard, EMPTY, bishop_attacks, bit, king_attacks, knight_attacks, pawn_attacks, pop_lsb,
    queen_attacks, rook_attacks,
};
use crate::board::{Board, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};
use crate::types::{Color, Move, MoveFlag, PieceType, Square, file_of, make_square, rank_of};
use arrayvec::ArrayVec;

/// En ajedrez estándar no existen más de 218 jugadas legales. Reservamos 256
/// para cubrir también el conjunto pseudo-legal y evitar una asignación de
/// heap en cada nodo de búsqueda. `ArrayVec::new()` no inicializa elementos:
/// solo mantiene el contador y los movimientos realmente generados.
pub const MAX_MOVES: usize = 256;
pub type MoveList = ArrayVec<Move, MAX_MOVES>;

const PROMO_PIECES: [PieceType; 4] = [
    PieceType::Queen,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
];

/// Genera todas las jugadas pseudo-legales (no filtra jaques propios).
pub fn generate_pseudo_legal(b: &Board) -> MoveList {
    let mut moves = MoveList::new();
    let us = b.turn;
    let them = us.opposite();
    let own = b.occupied_co[us as usize];
    let enemy = b.occupied_co[them as usize];
    let occ = b.occupied;

    gen_pawn_moves(b, us, enemy, occ, &mut moves);
    gen_piece_moves(b, us, PieceType::Knight, own, occ, &mut moves, |sq, _| {
        knight_attacks(sq)
    });
    gen_piece_moves(
        b,
        us,
        PieceType::Bishop,
        own,
        occ,
        &mut moves,
        bishop_attacks,
    );
    gen_piece_moves(b, us, PieceType::Rook, own, occ, &mut moves, rook_attacks);
    gen_piece_moves(b, us, PieceType::Queen, own, occ, &mut moves, queen_attacks);
    gen_piece_moves(b, us, PieceType::King, own, occ, &mut moves, |sq, _| {
        king_attacks(sq)
    });
    gen_castling(b, us, &mut moves);

    moves
}

fn gen_piece_moves<F>(
    b: &Board,
    us: Color,
    pt: PieceType,
    own: Bitboard,
    occ: Bitboard,
    moves: &mut MoveList,
    attacks_fn: F,
) where
    F: Fn(Square, Bitboard) -> Bitboard,
{
    let mut pieces = b.pieces[us as usize][pt as usize];
    while pieces != 0 {
        let from = pop_lsb(&mut pieces);
        let mut targets = attacks_fn(from, occ) & !own;
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            let flag = if bit(to) & occ != 0 {
                MoveFlag::Capture
            } else {
                MoveFlag::Quiet
            };
            moves.push(Move::new(from, to, None, flag));
        }
    }
}

fn gen_pawn_moves(b: &Board, us: Color, enemy: Bitboard, occ: Bitboard, moves: &mut MoveList) {
    let mut pawns = b.pieces[us as usize][PieceType::Pawn as usize];
    let (push_dir, start_rank, promo_rank): (i32, u8, u8) = match us {
        Color::White => (1, 1, 7),
        Color::Black => (-1, 6, 0),
    };

    while pawns != 0 {
        let from = pop_lsb(&mut pawns);
        let f = file_of(from) as i32;
        let r = rank_of(from) as i32;

        // Avance simple
        let one_rank = r + push_dir;
        if (0..8).contains(&one_rank) {
            let to = make_square(f as u8, one_rank as u8);
            if bit(to) & occ == 0 {
                push_pawn_move(
                    moves,
                    from,
                    to,
                    promo_rank == one_rank as u8,
                    MoveFlag::Quiet,
                );

                // Avance doble
                if rank_of(from) == start_rank {
                    let two_rank = r + 2 * push_dir;
                    let to2 = make_square(f as u8, two_rank as u8);
                    if bit(to2) & occ == 0 {
                        moves.push(Move::new(from, to2, None, MoveFlag::DoublePush));
                    }
                }
            }
        }

        // Capturas (incluye al paso)
        let mut att = pawn_attacks(us, from) & enemy;
        while att != 0 {
            let to = pop_lsb(&mut att);
            let is_promo = rank_of(to) == promo_rank;
            push_pawn_move(moves, from, to, is_promo, MoveFlag::Capture);
        }
        if let Some(ep) = b.ep_square
            && pawn_attacks(us, from) & bit(ep) != 0
        {
            moves.push(Move::new(from, ep, None, MoveFlag::EnPassant));
        }
    }
}

fn push_pawn_move(moves: &mut MoveList, from: Square, to: Square, is_promo: bool, flag: MoveFlag) {
    if is_promo {
        for &p in PROMO_PIECES.iter() {
            moves.push(Move::new(from, to, Some(p), flag));
        }
    } else {
        moves.push(Move::new(from, to, None, flag));
    }
}

fn gen_castling(b: &Board, us: Color, moves: &mut MoveList) {
    // El enroque SIEMPRE es ilegal con el rey en jaque (regla FIDE): si el
    // rey esta en jaque, todos los chequeos de casillas atacadas de abajo son
    // trabajo desperdiciado -- las jugadas generadas se descartarian igual en
    // el filtro de legalidad. Saltar la generacion completa.
    if b.in_check(us) {
        return;
    }
    let occ = b.occupied;
    match us {
        Color::White => {
            if b.castling_rights & CASTLE_WK != 0
                && b.pieces[Color::White as usize][PieceType::King as usize] & bit(4) != 0
                && b.pieces[Color::White as usize][PieceType::Rook as usize] & bit(7) != 0
                && occ & (bit(5) | bit(6)) == EMPTY
                && !b.is_square_attacked_by(4, Color::Black)
                && !b.is_square_attacked_by(5, Color::Black)
                && !b.is_square_attacked_by(6, Color::Black)
            {
                moves.push(Move::new(4, 6, None, MoveFlag::CastleKing));
            }
            if b.castling_rights & CASTLE_WQ != 0
                && b.pieces[Color::White as usize][PieceType::King as usize] & bit(4) != 0
                && b.pieces[Color::White as usize][PieceType::Rook as usize] & bit(0) != 0
                && occ & (bit(1) | bit(2) | bit(3)) == EMPTY
                && !b.is_square_attacked_by(4, Color::Black)
                && !b.is_square_attacked_by(3, Color::Black)
                && !b.is_square_attacked_by(2, Color::Black)
            {
                moves.push(Move::new(4, 2, None, MoveFlag::CastleQueen));
            }
        }
        Color::Black => {
            if b.castling_rights & CASTLE_BK != 0
                && b.pieces[Color::Black as usize][PieceType::King as usize] & bit(60) != 0
                && b.pieces[Color::Black as usize][PieceType::Rook as usize] & bit(63) != 0
                && occ & (bit(61) | bit(62)) == EMPTY
                && !b.is_square_attacked_by(60, Color::White)
                && !b.is_square_attacked_by(61, Color::White)
                && !b.is_square_attacked_by(62, Color::White)
            {
                moves.push(Move::new(60, 62, None, MoveFlag::CastleKing));
            }
            if b.castling_rights & CASTLE_BQ != 0
                && b.pieces[Color::Black as usize][PieceType::King as usize] & bit(60) != 0
                && b.pieces[Color::Black as usize][PieceType::Rook as usize] & bit(56) != 0
                && occ & (bit(57) | bit(58) | bit(59)) == EMPTY
                && !b.is_square_attacked_by(60, Color::White)
                && !b.is_square_attacked_by(59, Color::White)
                && !b.is_square_attacked_by(58, Color::White)
            {
                moves.push(Move::new(60, 58, None, MoveFlag::CastleQueen));
            }
        }
    }
}

/// Solo capturas, EP y promociones (incluidas las de avance sin captura) en
/// pseudo-legal. Usado por quiescence: generar el conjunto completo de
/// jugadas (incluidas todas las silenciosas) y luego filtrar a capturas es
/// trabajo tirado -- la mayoria de objetivos de ataque de una pieza son
/// casillas vacias, y ninguna cuesta legalizar solo para descartarla despues.
pub fn generate_pseudo_legal_captures(b: &Board) -> MoveList {
    let mut moves = MoveList::new();
    let us = b.turn;
    let them = us.opposite();
    let enemy = b.occupied_co[them as usize];
    let occ = b.occupied;

    gen_pawn_captures(b, us, enemy, occ, &mut moves);
    gen_piece_captures(b, us, PieceType::Knight, enemy, occ, &mut moves, |sq, _| {
        knight_attacks(sq)
    });
    gen_piece_captures(b, us, PieceType::Bishop, enemy, occ, &mut moves, bishop_attacks);
    gen_piece_captures(b, us, PieceType::Rook, enemy, occ, &mut moves, rook_attacks);
    gen_piece_captures(b, us, PieceType::Queen, enemy, occ, &mut moves, queen_attacks);
    gen_piece_captures(b, us, PieceType::King, enemy, occ, &mut moves, |sq, _| {
        king_attacks(sq)
    });
    moves
}

fn gen_piece_captures<F>(
    b: &Board,
    us: Color,
    pt: PieceType,
    enemy: Bitboard,
    occ: Bitboard,
    moves: &mut MoveList,
    attacks_fn: F,
) where
    F: Fn(Square, Bitboard) -> Bitboard,
{
    let mut pieces = b.pieces[us as usize][pt as usize];
    while pieces != 0 {
        let from = pop_lsb(&mut pieces);
        let mut targets = attacks_fn(from, occ) & enemy;
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            moves.push(Move::new(from, to, None, MoveFlag::Capture));
        }
    }
}

fn gen_pawn_captures(b: &Board, us: Color, enemy: Bitboard, occ: Bitboard, moves: &mut MoveList) {
    let mut pawns = b.pieces[us as usize][PieceType::Pawn as usize];
    let (push_dir, promo_rank): (i32, u8) = match us {
        Color::White => (1, 7),
        Color::Black => (-1, 0),
    };

    while pawns != 0 {
        let from = pop_lsb(&mut pawns);
        let f = file_of(from) as i32;
        let r = rank_of(from) as i32;

        // Avance simple SOLO si corona (una promocion sin captura sigue
        // siendo "ruidosa" -- quiescence la incluye via promotion.is_some()).
        let one_rank = r + push_dir;
        if (0..8).contains(&one_rank) {
            let to = make_square(f as u8, one_rank as u8);
            if bit(to) & occ == 0 && promo_rank == one_rank as u8 {
                push_pawn_move(moves, from, to, true, MoveFlag::Quiet);
            }
        }

        // Capturas (incluye al paso) y promociones por captura.
        let mut att = pawn_attacks(us, from) & enemy;
        while att != 0 {
            let to = pop_lsb(&mut att);
            let is_promo = rank_of(to) == promo_rank;
            push_pawn_move(moves, from, to, is_promo, MoveFlag::Capture);
        }
        if let Some(ep) = b.ep_square
            && pawn_attacks(us, from) & bit(ep) != 0
        {
            moves.push(Move::new(from, ep, None, MoveFlag::EnPassant));
        }
    }
}

/// Equivalente a generate_legal(b) filtrado a capturas/promociones, pero sin
/// generar ni legalizar las jugadas silenciosas que se descartarian despues.
/// Usa el mismo atajo de clavadas que generate_legal para el camino comun.
pub fn generate_captures_legal(b: &Board) -> MoveList {
    use crate::bitboard::pinned_pieces;

    let us = b.turn;
    let them = us.opposite();
    let king_sq = b.king_square(us);

    let camino_lento = |mv: &Move| {
        let after = b.make_move(mv);
        !after.in_check(us)
    };

    if b.in_check(us) {
        // No deberia llamarse en jaque (quiescence usa generate_legal para
        // evasiones, que incluyen bloqueos silenciosos), pero se deja
        // correcto por si se reutiliza en otro contexto.
        let mut moves = generate_pseudo_legal_captures(b);
        moves.retain(|mv| camino_lento(mv));
        return moves;
    }

    let own = b.occupied_co[us as usize];
    let enemy_rook_like = b.pieces[them as usize][PieceType::Rook as usize]
        | b.pieces[them as usize][PieceType::Queen as usize];
    let enemy_bishop_like = b.pieces[them as usize][PieceType::Bishop as usize]
        | b.pieces[them as usize][PieceType::Queen as usize];
    let pinned = pinned_pieces(king_sq, own, enemy_rook_like, enemy_bishop_like, b.occupied);

    let mut moves = generate_pseudo_legal_captures(b);
    moves.retain(|mv| {
        if bit(mv.from) & pinned == 0 && mv.from != king_sq && mv.flag != MoveFlag::EnPassant {
            true
        } else {
            camino_lento(mv)
        }
    });
    moves
}

/// Filtra las jugadas pseudo-legales: descarta las que dejan al propio rey en jaque.
///
/// Camino rapido: para jugadas que no son de rey/al paso/enroque, en vez de
/// copiar el tablero completo (Board::make_move) y verificar jaque jugada por
/// jugada, se calcula UNA vez por nodo el conjunto de piezas propias
/// clavadas (`pinned_pieces`) -- una jugada de una pieza no clavada siempre
/// es legal; una pieza clavada solo es legal si se mueve dentro de la misma
/// linea de clavada (rey-atacante). Jugadas de rey, al paso y enroque siguen
/// el camino lento (make_move + in_check) por sus reglas especiales (al paso
/// puede exponer un jaque horizontal poco comun; el rey necesita saber si la
/// casilla destino esta atacada, no si queda clavado).
/// Equivalencia verificada exhaustivamente en tests (bitboard::pin_tests /
/// movegen fuzz) y con la suite de perft completa sin cambios de resultado.
pub fn generate_legal(b: &Board) -> MoveList {
    use crate::bitboard::pinned_pieces;

    let us = b.turn;
    let them = us.opposite();
    let king_sq = b.king_square(us);

    let camino_lento = |mv: &Move| {
        let after = b.make_move(mv);
        !after.in_check(us)
    };

    if b.in_check(us) {
        // Evasion de jaque (posibles jaques dobles, bloqueos, etc.): el
        // caso menos frecuente y mas delicado -- se deja el camino lento,
        // siempre correcto, sin ganancia medible de nps (pocas jugadas
        // pseudo-legales cuando el rey esta en jaque).
        let mut moves = generate_pseudo_legal(b);
        moves.retain(|mv| camino_lento(mv));
        return moves;
    }

    let own = b.occupied_co[us as usize];
    let enemy_rook_like = b.pieces[them as usize][PieceType::Rook as usize]
        | b.pieces[them as usize][PieceType::Queen as usize];
    let enemy_bishop_like = b.pieces[them as usize][PieceType::Bishop as usize]
        | b.pieces[them as usize][PieceType::Queen as usize];
    let pinned = pinned_pieces(king_sq, own, enemy_rook_like, enemy_bishop_like, b.occupied);

    let mut moves = generate_pseudo_legal(b);
    moves.retain(|mv| {
        if bit(mv.from) & pinned == 0
            && mv.from != king_sq
            && !matches!(
                mv.flag,
                MoveFlag::EnPassant | MoveFlag::CastleKing | MoveFlag::CastleQueen
            )
        {
            // Pieza no clavada, no es el rey, no es al paso/enroque:
            // siempre legal (el rey no esta en jaque en esta rama, y mover
            // una pieza no clavada no puede exponerlo).
            true
        } else {
            // Casos poco comunes: camino lento, siempre correcto.
            camino_lento(mv)
        }
    });
    moves
}

/// ¿`mv` es una jugada LEGAL en `b`? Validador directo, sin generar la lista
/// completa de jugadas. Usado por la busqueda por etapas (staged movegen):
/// permite probar la jugada de la TT sola, ANTES de generar nada, y si corta
/// beta no se genera ni ordena ninguna otra jugada.
///
/// La jugada puede venir de la TT (otra posicion con colision de zobrist, o
/// una entrada vieja), asi que se valida TODO: pieza propia en `from`,
/// consistencia del flag con el tablero, patron de movimiento de la pieza,
/// promocion coherente, y por ultimo que el propio rey no quede en jaque
/// (make_move + in_check, siempre correcto, tambien en jaque actual).
///
/// Enroque y al paso (raros como jugada de TT) van por el camino lento pero
/// trivialmente correcto: pertenencia a generate_legal.
/// Equivalencia exacta con `generate_legal(b).contains(mv)` verificada
/// exhaustivamente en tests (es_legal_coincide_con_generate_legal).
pub fn es_legal(b: &Board, mv: &Move) -> bool {
    use crate::bitboard::{bishop_attacks, king_attacks, knight_attacks, queen_attacks, rook_attacks};

    let us = b.turn;
    if mv.from > 63 || mv.to > 63 || mv.from == mv.to {
        return false;
    }
    // Casos con reglas especiales de legalidad: camino lento, siempre exacto.
    if matches!(
        mv.flag,
        MoveFlag::EnPassant | MoveFlag::CastleKing | MoveFlag::CastleQueen
    ) {
        return generate_legal(b).contains(mv);
    }
    // Pieza propia en el origen.
    let Some((c, pt)) = b.piece_at(mv.from) else {
        return false;
    };
    if c != us {
        return false;
    }
    // Consistencia del flag con el contenido del destino.
    match mv.flag {
        MoveFlag::Capture => match b.piece_at(mv.to) {
            Some((cv, ptv)) => {
                // Nunca capturar pieza propia ni al rey (una entrada TT
                // corrupta con "captura de rey" romperia make_move).
                if cv == us || ptv == PieceType::King {
                    return false;
                }
            }
            None => return false,
        },
        MoveFlag::Quiet | MoveFlag::DoublePush => {
            if bit(mv.to) & b.occupied != 0 {
                return false;
            }
        }
        _ => unreachable!(),
    }
    // Promocion coherente: solo peon a ultima fila, y a una de las 4 piezas
    // que la generacion realmente produce.
    let promo_rank: u8 = match us {
        Color::White => 7,
        Color::Black => 0,
    };
    match mv.promotion {
        Some(p) => {
            if pt != PieceType::Pawn
                || rank_of(mv.to) != promo_rank
                || !matches!(
                    p,
                    PieceType::Queen | PieceType::Rook | PieceType::Bishop | PieceType::Knight
                )
            {
                return false;
            }
        }
        None => {
            // La generacion SIEMPRE marca promocion en peon a ultima fila.
            if pt == PieceType::Pawn && rank_of(mv.to) == promo_rank {
                return false;
            }
        }
    }
    // DoublePush es exclusivo del peon: la generacion jamas lo pone en otra
    // pieza (una entrada TT corrupta si podria).
    if mv.flag == MoveFlag::DoublePush && pt != PieceType::Pawn {
        return false;
    }
    // Patron de movimiento de la pieza sobre la ocupacion actual.
    let patron_ok = match pt {
        PieceType::Knight => knight_attacks(mv.from) & bit(mv.to) != 0,
        PieceType::Bishop => bishop_attacks(mv.from, b.occupied) & bit(mv.to) != 0,
        PieceType::Rook => rook_attacks(mv.from, b.occupied) & bit(mv.to) != 0,
        PieceType::Queen => queen_attacks(mv.from, b.occupied) & bit(mv.to) != 0,
        PieceType::King => king_attacks(mv.from) & bit(mv.to) != 0,
        PieceType::Pawn => {
            let (push_dir, start_rank): (i32, u8) = match us {
                Color::White => (1, 1),
                Color::Black => (-1, 6),
            };
            let f = file_of(mv.from) as i32;
            let r = rank_of(mv.from) as i32;
            match mv.flag {
                MoveFlag::Capture => pawn_attacks(us, mv.from) & bit(mv.to) != 0,
                MoveFlag::Quiet => {
                    let one = r + push_dir;
                    (0..8).contains(&one) && mv.to == make_square(f as u8, one as u8)
                }
                MoveFlag::DoublePush => {
                    if rank_of(mv.from) != start_rank {
                        false
                    } else {
                        let mid = make_square(f as u8, (r + push_dir) as u8);
                        let dst = make_square(f as u8, (r + 2 * push_dir) as u8);
                        mv.to == dst && bit(mid) & b.occupied == 0
                    }
                }
                _ => false,
            }
        }
    };
    if !patron_ok {
        return false;
    }
    // Legalidad real: el propio rey no puede quedar en jaque.
    !b.make_move(mv).in_check(us)
}

/// Referencia lenta (ray-casting puro, sin atajo de clavadas) usada SOLO en
/// tests para verificar que generate_legal produce exactamente el mismo
/// conjunto de jugadas.
#[cfg(test)]
fn generate_legal_referencia(b: &Board) -> MoveList {
    let us = b.turn;
    generate_pseudo_legal(b)
        .into_iter()
        .filter(|mv| {
            let after = b.make_move(mv);
            !after.in_check(us)
        })
        .collect()
}

#[cfg(test)]
mod equivalencia_legal_tests {
    use super::*;
    use crate::perft::perft;

    fn mismo_conjunto(b: &Board) {
        assert_eq!(
            generate_legal(b),
            generate_legal_referencia(b),
            "orden de jugadas divergente: {}",
            b.to_fen()
        );
        let mut rapido: Vec<String> = generate_legal(b).iter().map(|m| m.to_uci()).collect();
        let mut lento: Vec<String> = generate_legal_referencia(b)
            .iter()
            .map(|m| m.to_uci())
            .collect();
        rapido.sort();
        lento.sort();
        assert_eq!(rapido, lento, "FEN divergente: {}", b.to_fen());
    }

    fn explorar(b: &Board, profundidad: u32) {
        mismo_conjunto(b);
        if profundidad == 0 {
            return;
        }
        for mv in generate_legal_referencia(b) {
            explorar(&b.make_move(&mv), profundidad - 1);
        }
    }

    #[test]
    fn generate_legal_coincide_con_referencia_posiciones_estandar() {
        let posiciones = [
            crate::board::Board::startpos(),
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -")
                .unwrap(),
            Board::from_fen("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2")
                .unwrap(),
            Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap(),
            Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
                .unwrap(),
        ];
        for b in &posiciones {
            explorar(b, 3);
        }
    }

    /// es_legal debe coincidir EXACTAMENTE con la pertenencia a
    /// generate_legal, para cualquier Move posible (incluidos los invalidos
    /// que podrian salir de una entrada TT vieja o con colision).
    fn es_legal_coincide(b: &Board) {
        let legales = generate_legal(b);
        // Todas las legales deben pasar.
        for mv in &legales {
            assert!(
                es_legal(b, mv),
                "es_legal rechaza jugada legal {} en {}",
                mv.to_uci(),
                b.to_fen()
            );
        }
        // Barrido de jugadas arbitrarias: ninguna no-legal debe pasar.
        let flags = [
            MoveFlag::Quiet,
            MoveFlag::Capture,
            MoveFlag::DoublePush,
            MoveFlag::EnPassant,
            MoveFlag::CastleKing,
            MoveFlag::CastleQueen,
        ];
        let promos = [None, Some(PieceType::Queen), Some(PieceType::Knight)];
        for from in 0..64u8 {
            for to in 0..64u8 {
                for &flag in &flags {
                    for &promo in &promos {
                        let mv = Move::new(from, to, promo, flag);
                        let esperado = legales.contains(&mv);
                        assert_eq!(
                            es_legal(b, &mv),
                            esperado,
                            "es_legal diverge en {} para {:?}",
                            b.to_fen(),
                            mv
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn es_legal_coincide_con_generate_legal() {
        let posiciones = [
            crate::board::Board::startpos(),
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -")
                .unwrap(),
            Board::from_fen("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2")
                .unwrap(),
            Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap(),
            Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
                .unwrap(),
            Board::from_fen("4k3/1P6/8/8/8/8/1p6/4K3 w - - 0 1").unwrap(),
        ];
        for b in &posiciones {
            es_legal_coincide(b);
            // Ademas, un nivel de profundidad para cubrir respuestas (jaques,
            // al paso recien habilitado, perdida de derechos de enroque...).
            for mv in generate_legal(b) {
                es_legal_coincide(&b.make_move(&mv));
            }
        }
    }

    fn mismo_conjunto_capturas(b: &Board) {
        let rapido: MoveList = generate_captures_legal(b);
        let referencia: MoveList = generate_legal(b)
            .into_iter()
            .filter(|m| m.is_capture() || m.promotion.is_some())
            .collect();
        let mut rapido_uci: Vec<String> = rapido.iter().map(|m| m.to_uci()).collect();
        let mut ref_uci: Vec<String> = referencia.iter().map(|m| m.to_uci()).collect();
        rapido_uci.sort();
        ref_uci.sort();
        assert_eq!(
            rapido_uci,
            ref_uci,
            "generate_captures_legal diverge en {}",
            b.to_fen()
        );
    }

    fn explorar_capturas(b: &Board, profundidad: u32) {
        mismo_conjunto_capturas(b);
        if profundidad == 0 {
            return;
        }
        for mv in generate_legal_referencia(b) {
            explorar_capturas(&b.make_move(&mv), profundidad - 1);
        }
    }

    #[test]
    fn generate_captures_legal_coincide_con_generate_legal_filtrado() {
        let posiciones = [
            crate::board::Board::startpos(),
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -")
                .unwrap(),
            Board::from_fen("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2")
                .unwrap(),
            Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap(),
            Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
                .unwrap(),
            // Posicion con promociones y capturas al paso combinadas.
            Board::from_fen("4k3/1P6/8/8/8/8/1p6/4K3 w - - 0 1").unwrap(),
        ];
        for b in &posiciones {
            explorar_capturas(b, 3);
        }
    }

    #[test]
    fn perft_no_cambia_con_atajo_de_clavadas() {
        // El atajo de clavadas es una optimizacion de generate_legal, no de
        // movegen en general -- si perft (que llama generate_legal en cada
        // nodo via perft/perft_divide) sigue dando los mismos numeros que
        // los certificados, la equivalencia esta confirmada tambien a
        // profundidad mayor con miles de posiciones intermedias reales.
        let b = Board::startpos();
        assert_eq!(perft(&b, 5), 4_865_609);
    }
}
