// Tipos básicos: color, tipo de pieza, casillas, jugadas.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

pub const ALL_PIECE_TYPES: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
];

impl PieceType {
    pub fn to_char(self, color: Color) -> char {
        let c = match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        };
        if color == Color::White {
            c.to_ascii_uppercase()
        } else {
            c
        }
    }

    pub fn from_char(c: char) -> Option<PieceType> {
        match c.to_ascii_lowercase() {
            'p' => Some(PieceType::Pawn),
            'n' => Some(PieceType::Knight),
            'b' => Some(PieceType::Bishop),
            'r' => Some(PieceType::Rook),
            'q' => Some(PieceType::Queen),
            'k' => Some(PieceType::King),
            _ => None,
        }
    }
}

// Casillas: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63 (little-endian rank-file).
pub type Square = u8;

#[inline(always)]
pub fn make_square(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

#[inline(always)]
pub fn file_of(sq: Square) -> u8 {
    sq % 8
}

#[inline(always)]
pub fn rank_of(sq: Square) -> u8 {
    sq / 8
}

pub fn square_name(sq: Square) -> String {
    let f = (b'a' + file_of(sq)) as char;
    let r = (b'1' + rank_of(sq)) as char;
    format!("{}{}", f, r)
}

pub fn square_from_name(s: &str) -> Option<Square> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let file = bytes[0].checked_sub(b'a')?;
    let rank = bytes[1].checked_sub(b'1')?;
    if file > 7 || rank > 7 {
        return None;
    }
    Some(make_square(file, rank))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveFlag {
    Quiet,
    Capture,
    DoublePush,
    EnPassant,
    CastleKing,
    CastleQueen,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceType>,
    pub flag: MoveFlag,
}

impl Move {
    pub fn new(from: Square, to: Square, promotion: Option<PieceType>, flag: MoveFlag) -> Move {
        Move {
            from,
            to,
            promotion,
            flag,
        }
    }

    pub fn is_capture(&self) -> bool {
        matches!(self.flag, MoveFlag::Capture | MoveFlag::EnPassant)
    }

    /// Notación UCI: e2e4, e7e8q, etc.
    pub fn to_uci(self) -> String {
        let mut s = format!("{}{}", square_name(self.from), square_name(self.to));
        if let Some(p) = self.promotion {
            s.push(p.to_char(Color::Black)); // minúscula siempre en UCI
        }
        s
    }
}
