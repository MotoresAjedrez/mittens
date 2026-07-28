use crate::bitboard::{
    Bitboard, EMPTY, bishop_attacks, bit, king_attacks, knight_attacks, pawn_attacks, pop_lsb,
    rook_attacks,
};
use crate::types::{
    ALL_PIECE_TYPES, Color, Move, MoveFlag, PieceType, Square, file_of, make_square, rank_of,
    square_from_name, square_name,
};
use crate::zobrist::keys;

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

#[derive(Clone, Copy, Debug)]
pub struct Board {
    pub pieces: [[Bitboard; 6]; 2], // [color][piece_type]
    pub occupied_co: [Bitboard; 2],
    pub occupied: Bitboard,
    pub turn: Color,
    pub castling_rights: u8,
    pub ep_square: Option<Square>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    pub zobrist: u64,
}

impl Board {
    pub fn empty() -> Board {
        Board {
            pieces: [[EMPTY; 6]; 2],
            occupied_co: [EMPTY; 2],
            occupied: EMPTY,
            turn: Color::White,
            castling_rights: 0,
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            zobrist: 0,
        }
    }

    pub fn startpos() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<(Color, PieceType)> {
        let b = bit(sq);
        if self.occupied & b == 0 {
            return None;
        }
        let color = if self.occupied_co[Color::White as usize] & b != 0 {
            Color::White
        } else {
            Color::Black
        };
        for &pt in ALL_PIECE_TYPES.iter() {
            if self.pieces[color as usize][pt as usize] & b != 0 {
                return Some((color, pt));
            }
        }
        unreachable!("casilla ocupada sin tipo de pieza -- estado inconsistente")
    }

    fn recompute_derived(&mut self) {
        self.occupied_co[0] = self.pieces[0].iter().fold(0, |a, &b| a | b);
        self.occupied_co[1] = self.pieces[1].iter().fold(0, |a, &b| a | b);
        self.occupied = self.occupied_co[0] | self.occupied_co[1];
    }

    /// Devuelve el archivo de en-passant que debe entrar en la clave
    /// Zobrist. Siguiendo la convencion de Polyglot, solo se hashea cuando
    /// el bando al turno tiene al menos un peon que podria capturar en esa
    /// casilla. Esto evita distinguir posiciones con exactamente las mismas
    /// jugadas legales solo porque el FEN conserve una casilla EP inutil.
    #[inline]
    pub fn ep_hash_file(&self) -> Option<usize> {
        let ep = self.ep_square?;
        let peones = self.pieces[self.turn as usize][PieceType::Pawn as usize];
        let atacantes = pawn_attacks(self.turn.opposite(), ep) & peones;
        (atacantes != 0).then_some(file_of(ep) as usize)
    }

    #[inline]
    pub fn ep_is_capturable(&self) -> bool {
        self.ep_hash_file().is_some()
    }

    fn recompute_zobrist(&mut self) {
        let k = keys();
        let mut z = 0u64;
        for c in 0..2 {
            for p in 0..6 {
                let mut bb = self.pieces[c][p];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb);
                    z ^= k.piece_square[c][p][sq as usize];
                }
            }
        }
        z ^= k.castling[(self.castling_rights & 0xF) as usize];
        if let Some(file) = self.ep_hash_file() {
            z ^= k.en_passant_file[file];
        }
        if self.turn == Color::Black {
            z ^= k.side_to_move;
        }
        self.zobrist = z;
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("FEN incompleto: {}", fen));
        }
        let mut b = Board::empty();

        // 1. Colocación de piezas
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(format!("FEN con {} filas, se esperaban 8", ranks.len()));
        }
        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i as u8; // FEN empieza en la fila 8
            let mut file = 0u8;
            for ch in rank_str.chars() {
                if ch.is_ascii_digit() {
                    let skip = ch as u8 - b'0';
                    if !(1..=8).contains(&skip) || file + skip > 8 {
                        return Err(format!("fila FEN inválida: {}", rank_str));
                    }
                    file += skip;
                } else {
                    if file >= 8 {
                        return Err(format!("fila FEN demasiado larga: {}", rank_str));
                    }
                    let color = if ch.is_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let pt = PieceType::from_char(ch)
                        .ok_or_else(|| format!("carácter de pieza inválido: {}", ch))?;
                    let sq = make_square(file, rank);
                    b.pieces[color as usize][pt as usize] |= bit(sq);
                    file += 1;
                }
            }
            if file != 8 {
                return Err(format!("fila FEN incompleta: {}", rank_str));
            }
        }

        // 2. Turno
        b.turn = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("turno inválido: {}", other)),
        };

        // 3. Enroque
        let mut cr = 0u8;
        if parts[2] != "-" {
            for ch in parts[2].chars() {
                let derecho = match ch {
                    'K' => CASTLE_WK,
                    'Q' => CASTLE_WQ,
                    'k' => CASTLE_BK,
                    'q' => CASTLE_BQ,
                    _ => return Err(format!("derecho de enroque inválido: {}", ch)),
                };
                if cr & derecho != 0 {
                    return Err(format!("derecho de enroque repetido: {}", ch));
                }
                cr |= derecho;
            }
        }
        b.castling_rights = cr;

        // 4. Al paso
        b.ep_square = if parts[3] == "-" {
            None
        } else {
            let sq = square_from_name(parts[3])
                .ok_or_else(|| format!("casilla al paso inválida: {}", parts[3]))?;
            let rank = rank_of(sq);
            if rank != 2 && rank != 5 {
                return Err(format!("casilla al paso en rango inválido: {}", parts[3]));
            }
            if (b.turn == Color::White && rank != 5) || (b.turn == Color::Black && rank != 2) {
                return Err(format!(
                    "casilla al paso incompatible con el turno: {}",
                    parts[3]
                ));
            }
            Some(sq)
        };

        // 5-6. Contadores (opcionales en algunos FEN recortados)
        b.halfmove_clock = match parts.get(4) {
            Some(value) => value
                .parse()
                .map_err(|_| format!("reloj de medio movimiento inválido: {}", value))?,
            None => 0,
        };
        b.fullmove_number = match parts.get(5) {
            Some(value) => value
                .parse()
                .map_err(|_| format!("número de jugada inválido: {}", value))?,
            None => 1,
        };
        if b.fullmove_number == 0 {
            return Err("número de jugada debe ser al menos 1".to_string());
        }

        b.recompute_derived();

        // Validaciones estructurales para que king_square/in_check nunca
        // reciban un tablero imposible y terminen desplazando por una casilla
        // 64. Cada color debe tener exactamente un rey y no pueden tocarse.
        for color in [Color::White, Color::Black] {
            let reyes = b.pieces[color as usize][PieceType::King as usize].count_ones();
            if reyes != 1 {
                return Err(format!(
                    "se esperaba exactamente un rey {:?}, hay {}",
                    color, reyes
                ));
            }
        }
        let rey_w = crate::bitboard::lsb(b.pieces[Color::White as usize][PieceType::King as usize]);
        let rey_b = crate::bitboard::lsb(b.pieces[Color::Black as usize][PieceType::King as usize]);
        if king_attacks(rey_w) & bit(rey_b) != 0 {
            return Err("los reyes no pueden estar adyacentes".to_string());
        }
        let filas_extremas = 0xFF000000000000FFu64;
        if (b.pieces[0][PieceType::Pawn as usize] | b.pieces[1][PieceType::Pawn as usize])
            & filas_extremas
            != 0
        {
            return Err("peon en primera u octava fila".to_string());
        }
        if let Some(ep) = b.ep_square {
            if b.occupied & bit(ep) != 0 {
                return Err("la casilla al paso debe estar vacia".to_string());
            }
            let pawn_sq = if b.turn == Color::White {
                ep - 8
            } else {
                ep + 8
            };
            let last_mover = b.turn.opposite();
            if b.pieces[last_mover as usize][PieceType::Pawn as usize] & bit(pawn_sq) == 0 {
                return Err("casilla al paso sin peon que haya hecho doble avance".to_string());
            }
        }

        let tiene = |color, pt, sq| b.pieces[color as usize][pt as usize] & bit(sq) != 0;
        if !tiene(Color::White, PieceType::King, make_square(4, 0))
            && cr & (CASTLE_WK | CASTLE_WQ) != 0
        {
            return Err("enroque blanco sin rey en e1".to_string());
        }
        if !tiene(Color::Black, PieceType::King, make_square(4, 7))
            && cr & (CASTLE_BK | CASTLE_BQ) != 0
        {
            return Err("enroque negro sin rey en e8".to_string());
        }
        if cr & CASTLE_WK != 0 && !tiene(Color::White, PieceType::Rook, make_square(7, 0)) {
            return Err("enroque blanco corto sin torre en h1".to_string());
        }
        if cr & CASTLE_WQ != 0 && !tiene(Color::White, PieceType::Rook, make_square(0, 0)) {
            return Err("enroque blanco largo sin torre en a1".to_string());
        }
        if cr & CASTLE_BK != 0 && !tiene(Color::Black, PieceType::Rook, make_square(7, 7)) {
            return Err("enroque negro corto sin torre en h8".to_string());
        }
        if cr & CASTLE_BQ != 0 && !tiene(Color::Black, PieceType::Rook, make_square(0, 7)) {
            return Err("enroque negro largo sin torre en a8".to_string());
        }

        b.recompute_zobrist();
        Ok(b)
    }

    pub fn to_fen(self) -> String {
        let mut s = String::new();
        for i in 0..8 {
            let rank = 7 - i;
            let mut empty_count = 0u8;
            for file in 0..8 {
                let sq = make_square(file, rank);
                match self.piece_at(sq) {
                    None => empty_count += 1,
                    Some((color, pt)) => {
                        if empty_count > 0 {
                            s.push((b'0' + empty_count) as char);
                            empty_count = 0;
                        }
                        s.push(pt.to_char(color));
                    }
                }
            }
            if empty_count > 0 {
                s.push((b'0' + empty_count) as char);
            }
            if i != 7 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.turn == Color::White { 'w' } else { 'b' });
        s.push(' ');
        if self.castling_rights == 0 {
            s.push('-');
        } else {
            if self.castling_rights & CASTLE_WK != 0 {
                s.push('K');
            }
            if self.castling_rights & CASTLE_WQ != 0 {
                s.push('Q');
            }
            if self.castling_rights & CASTLE_BK != 0 {
                s.push('k');
            }
            if self.castling_rights & CASTLE_BQ != 0 {
                s.push('q');
            }
        }
        s.push(' ');
        match self.ep_square {
            Some(sq) => s.push_str(&square_name(sq)),
            None => s.push('-'),
        }
        s.push(' ');
        s.push_str(&self.halfmove_clock.to_string());
        s.push(' ');
        s.push_str(&self.fullmove_number.to_string());
        s
    }

    /// Ataques totales de `color` que cubren la casilla `sq` (sin importar si hay pieza ahí).
    pub fn attackers_to(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let mut attackers = 0u64;
        let white = self.pieces[Color::White as usize];
        let black = self.pieces[Color::Black as usize];

        attackers |= knight_attacks(sq)
            & (white[PieceType::Knight as usize] | black[PieceType::Knight as usize]);
        attackers |=
            king_attacks(sq) & (white[PieceType::King as usize] | black[PieceType::King as usize]);
        let bishops_queens = white[PieceType::Bishop as usize]
            | white[PieceType::Queen as usize]
            | black[PieceType::Bishop as usize]
            | black[PieceType::Queen as usize];
        attackers |= bishop_attacks(sq, occupied) & bishops_queens;
        let rooks_queens = white[PieceType::Rook as usize]
            | white[PieceType::Queen as usize]
            | black[PieceType::Rook as usize]
            | black[PieceType::Queen as usize];
        attackers |= rook_attacks(sq, occupied) & rooks_queens;

        // Peones: un peon blanco en X ataca sq si sq esta en pawn_attacks(black, sq) invertido --
        // mas simple: un atacante peon blanco esta en las casillas que UN PEON NEGRO en sq atacaria.
        attackers |= pawn_attacks(Color::Black, sq) & white[PieceType::Pawn as usize];
        attackers |= pawn_attacks(Color::White, sq) & black[PieceType::Pawn as usize];

        attackers
    }

    pub fn is_square_attacked_by(&self, sq: Square, by_color: Color) -> bool {
        self.attackers_to(sq, self.occupied) & self.occupied_co[by_color as usize] != 0
    }

    pub fn king_square(&self, color: Color) -> Square {
        crate::bitboard::lsb(self.pieces[color as usize][PieceType::King as usize])
    }

    pub fn in_check(&self, color: Color) -> bool {
        self.is_square_attacked_by(self.king_square(color), color.opposite())
    }

    fn remove_piece(&mut self, color: Color, pt: PieceType, sq: Square) {
        self.pieces[color as usize][pt as usize] &= !bit(sq);
        self.zobrist ^= keys().piece_square[color as usize][pt as usize][sq as usize];
    }

    fn add_piece(&mut self, color: Color, pt: PieceType, sq: Square) {
        self.pieces[color as usize][pt as usize] |= bit(sq);
        self.zobrist ^= keys().piece_square[color as usize][pt as usize][sq as usize];
    }

    /// Aplica una jugada (ya asumida pseudo-legal) y devuelve un NUEVO tablero.
    /// Copiar el tablero completo es barato (struct chico, sin heap) y evita
    /// toda la clase de bugs de un unmake_move mal implementado -- prioridad
    /// de esta sesión es correctitud (perft exacto) antes que velocidad.
    pub fn make_move(&self, mv: &Move) -> Board {
        let mut b = *self;
        let us = b.turn;
        let them = us.opposite();
        let k = keys();

        // Limpiar la clave de al paso anterior (se vuelve a poner si aplica)
        if let Some(file) = b.ep_hash_file() {
            b.zobrist ^= k.en_passant_file[file];
        }
        b.ep_square = None;

        let (_, moving_pt) = self
            .piece_at(mv.from)
            .expect("make_move: no hay pieza en 'from'");

        // Captura (normal o al paso)
        if mv.flag == MoveFlag::EnPassant {
            let captured_sq = make_square(file_of(mv.to), rank_of(mv.from));
            b.remove_piece(them, PieceType::Pawn, captured_sq);
            b.halfmove_clock = 0;
        } else if let Some((_, captured_pt)) = self.piece_at(mv.to) {
            if captured_pt == PieceType::King {
                // Nunca deberia pasar en una partida legal (generate_legal ya
                // filtra jugadas que dejan al propio rey en jaque) -- si esto
                // se dispara, la posicion de entrada es ilegal o hay un bug real.
                panic!(
                    "intento de capturar un REY -- posicion ilegal o bug real. FEN antes de la jugada: {}  jugada: {}",
                    self.to_fen(),
                    mv.to_uci()
                );
            }
            b.remove_piece(them, captured_pt, mv.to);
            b.halfmove_clock = 0;
        }

        b.remove_piece(us, moving_pt, mv.from);
        if let Some(promo) = mv.promotion {
            b.add_piece(us, promo, mv.to);
        } else {
            b.add_piece(us, moving_pt, mv.to);
        }

        if moving_pt == PieceType::Pawn || mv.is_capture() {
            b.halfmove_clock = 0;
        } else {
            b.halfmove_clock += 1;
        }

        // Enroque: mover también la torre
        if mv.flag == MoveFlag::CastleKing {
            let (rook_from, rook_to) = match us {
                Color::White => (make_square(7, 0), make_square(5, 0)),
                Color::Black => (make_square(7, 7), make_square(5, 7)),
            };
            b.remove_piece(us, PieceType::Rook, rook_from);
            b.add_piece(us, PieceType::Rook, rook_to);
        } else if mv.flag == MoveFlag::CastleQueen {
            let (rook_from, rook_to) = match us {
                Color::White => (make_square(0, 0), make_square(3, 0)),
                Color::Black => (make_square(0, 7), make_square(3, 7)),
            };
            b.remove_piece(us, PieceType::Rook, rook_from);
            b.add_piece(us, PieceType::Rook, rook_to);
        }

        // Doble avance de peón: fija la casilla al paso
        if mv.flag == MoveFlag::DoublePush {
            let ep_sq = make_square(file_of(mv.from), (rank_of(mv.from) + rank_of(mv.to)) / 2);
            b.ep_square = Some(ep_sq);
        }

        // Actualizar derechos de enroque
        let old_cr = b.castling_rights;
        let mut new_cr = old_cr;
        if moving_pt == PieceType::King {
            new_cr &= match us {
                Color::White => !(CASTLE_WK | CASTLE_WQ),
                Color::Black => !(CASTLE_BK | CASTLE_BQ),
            };
        }
        let touches = |sq: Square, cr: &mut u8| {
            if sq == make_square(0, 0) {
                *cr &= !CASTLE_WQ;
            } else if sq == make_square(7, 0) {
                *cr &= !CASTLE_WK;
            } else if sq == make_square(0, 7) {
                *cr &= !CASTLE_BQ;
            } else if sq == make_square(7, 7) {
                *cr &= !CASTLE_BK;
            }
        };
        touches(mv.from, &mut new_cr);
        touches(mv.to, &mut new_cr);
        if new_cr != old_cr {
            b.zobrist ^= k.castling[old_cr as usize];
            b.zobrist ^= k.castling[new_cr as usize];
            b.castling_rights = new_cr;
        }

        if us == Color::Black {
            b.fullmove_number += 1;
        }
        b.turn = them;
        b.zobrist ^= k.side_to_move;

        b.recompute_derived();
        if let Some(file) = b.ep_hash_file() {
            b.zobrist ^= k.en_passant_file[file];
        }
        b
    }

    pub fn make_null_move(&self) -> Board {
        let mut b = *self;
        let k = keys();
        if let Some(file) = b.ep_hash_file() {
            b.zobrist ^= k.en_passant_file[file];
        }
        b.ep_square = None;
        b.turn = b.turn.opposite();
        b.zobrist ^= k.side_to_move;
        b
    }
}

#[cfg(test)]
mod tests {
    use super::Board;

    #[test]
    fn rechaza_fila_fen_de_mas_de_ocho_casillas() {
        let fen = "4k3/8/8/8/8/8/8/9 w - - 0 1";
        assert!(Board::from_fen(fen).is_err());
    }

    #[test]
    fn rechaza_enroque_sin_torre() {
        let fen = "4k3/8/8/8/8/8/8/4K3 w K - 0 1";
        assert!(Board::from_fen(fen).is_err());
    }

    #[test]
    fn rechaza_casilla_al_paso_invalida() {
        let fen = "4k3/8/8/8/8/8/8/4K3 w - i9 0 1";
        assert!(Board::from_fen(fen).is_err());
    }
    #[test]
    fn zobrist_ignora_ep_si_nadie_puede_capturar() {
        let con_ep = Board::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1").unwrap();
        let sin_ep = Board::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1").unwrap();
        assert_eq!(con_ep.zobrist, sin_ep.zobrist);
        assert!(!con_ep.ep_is_capturable());
    }

    #[test]
    fn zobrist_incluye_ep_si_hay_captura_posible() {
        let con_ep = Board::from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1").unwrap();
        let sin_ep = Board::from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1").unwrap();
        assert_ne!(con_ep.zobrist, sin_ep.zobrist);
        assert!(con_ep.ep_is_capturable());
    }

    #[test]
    fn rechaza_fen_sin_reyes() {
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w - - 0 1").is_err());
    }

    #[test]
    fn rechaza_reyes_adyacentes() {
        assert!(Board::from_fen("8/8/8/8/8/8/4k3/4K3 w - - 0 1").is_err());
    }
}
