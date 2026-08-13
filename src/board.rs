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

/// Derechos de enroque que SOBREVIVEN cuando la casilla es origen o destino
/// de una jugada. Solo las cuatro esquinas (torres) y e1/e8 (reyes) quitan
/// algo; el resto deja los cuatro derechos intactos. Los derechos del rey se
/// siguen quitando aparte en make_move (moving_pt == King), asi que aqui solo
/// hacen falta las esquinas.
const MASCARA_ENROQUE: [u8; 64] = {
    let mut t = [0xFu8; 64];
    t[0] = 0xF & !CASTLE_WQ; // a1
    t[7] = 0xF & !CASTLE_WK; // h1
    t[56] = 0xF & !CASTLE_BQ; // a8
    t[63] = 0xF & !CASTLE_BK; // h8
    t
};

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

    /// Sin ramas: en vez de recorrer los 6 bitboards de tipo hasta encontrar
    /// el que contiene la casilla (bucle con salto impredecible, promedio ~3
    /// vueltas), el indice del tipo se ARMA bit a bit con tres pruebas fijas.
    /// El orden de PieceType (0=Peon, 1=Caballo, 2=Alfil, 3=Torre, 4=Dama,
    /// 5=Rey) permite exactamente esta descomposicion binaria:
    ///   bit 0 (valor 1) = Caballo(1), Torre(3), Rey(5)
    ///   bit 1 (valor 2) = Alfil(2), Torre(3)
    ///   bit 2 (valor 4) = Dama(4), Rey(5)
    /// Sumando: 0=Peon, 1=Caballo, 2=Alfil, 1+2=3=Torre, 4=Dama, 1+4=5=Rey.
    /// Mismo resultado exacto que el bucle, sin saltos condicionales.
    /// `piece_at` es de las funciones mas llamadas del motor (make_move la usa
    /// dos veces por jugada, mas SEE, evaluacion y NNUE).
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<(Color, PieceType)> {
        let b = bit(sq);
        if self.occupied & b == 0 {
            return None;
        }
        let (color, ci) = if self.occupied_co[Color::White as usize] & b != 0 {
            (Color::White, Color::White as usize)
        } else {
            (Color::Black, Color::Black as usize)
        };
        let p = &self.pieces[ci];
        let b0 = ((p[PieceType::Knight as usize] | p[PieceType::Rook as usize]
            | p[PieceType::King as usize])
            & b
            != 0) as usize;
        let b1 = ((p[PieceType::Bishop as usize] | p[PieceType::Rook as usize]) & b != 0) as usize;
        let b2 = ((p[PieceType::Queen as usize] | p[PieceType::King as usize]) & b != 0) as usize;
        let idx = b0 | (b1 << 1) | (b2 << 2);
        debug_assert!(idx < 6, "casilla ocupada sin tipo de pieza valido");
        Some((color, ALL_PIECE_TYPES[idx]))
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

    pub(crate) fn recompute_zobrist(&mut self) {
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
        // Provisional: mas abajo, una vez colocadas las piezas, se sanean los
        // derechos imposibles y se vuelve a asignar el valor definitivo.
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

        // Saneo silencioso de los derechos de enroque (criterio Stockfish).
        //
        // Muchas GUIs, libros de aperturas y bases de datos emiten FENs con el
        // campo de enroque sin actualizar: dicen "KQkq" cuando el rey ya se
        // movio o cuando la torre correspondiente ya no esta en su esquina.
        // Rechazar esos FENs con error deja al motor inutilizable con esas
        // herramientas, y aceptarlos tal cual corrompe la generacion de
        // enroques y la clave zobrist. La salida es la de Stockfish: conservar
        // un derecho SOLO si el rey esta en su casilla original Y la torre
        // correspondiente en la suya; en caso contrario, descartar ese derecho
        // en silencio. El motor es de ajedrez estandar puro (no hay soporte de
        // Chess960/FRC en ninguna parte del codigo), asi que las casillas son
        // las clasicas: e1/e8 para los reyes, a1/h1/a8/h8 para las torres.
        //
        // El saneo va ANTES de recompute_zobrist(), de modo que el hash de la
        // posicion se calcula con los derechos ya saneados y coincide con el
        // que produce el mantenimiento incremental de make_move.
        let tiene = |color, pt, sq| b.pieces[color as usize][pt as usize] & bit(sq) != 0;
        if !tiene(Color::White, PieceType::King, make_square(4, 0)) {
            cr &= !(CASTLE_WK | CASTLE_WQ);
        }
        if !tiene(Color::Black, PieceType::King, make_square(4, 7)) {
            cr &= !(CASTLE_BK | CASTLE_BQ);
        }
        if !tiene(Color::White, PieceType::Rook, make_square(7, 0)) {
            cr &= !CASTLE_WK;
        }
        if !tiene(Color::White, PieceType::Rook, make_square(0, 0)) {
            cr &= !CASTLE_WQ;
        }
        if !tiene(Color::Black, PieceType::Rook, make_square(7, 7)) {
            cr &= !CASTLE_BK;
        }
        if !tiene(Color::Black, PieceType::Rook, make_square(0, 7)) {
            cr &= !CASTLE_BQ;
        }
        b.castling_rights = cr;

        // Invariante fundamental de una posicion legal: el bando que ACABA de
        // mover (el que NO tiene el turno) nunca puede dejar su propio rey en
        // jaque -- si lo hizo, la jugada anterior fue ilegal y la posicion es
        // imposible en una partida real. Ejemplo concreto de la clase de bug:
        //   "8/4k2R/3b4/5p2/5pr1/3N1K2/5P2/8 w - - 18 70"
        // tiene al rey negro en e7 bajo ataque directo de la torre blanca en
        // h7 siendo el turno de BLANCAS -- el ultimo movimiento de negras
        // dejo su propio rey en jaque.
        // Sin este chequeo, generate_legal/make_move (que asumen posiciones
        // legales) aceptan la jugada blanca "h7e7" -- para blancas no deja su
        // rey en jaque, asi que es "legal" -- y make_move intenta capturar el
        // REY rival, panic que en release (panic=abort) TUMBA el proceso a
        // media partida. Rechazar la posicion en el borde es la causa raiz.
        if b.in_check(b.turn.opposite()) {
            return Err(
                "el bando que acabo de mover dejo su propio rey en jaque (posicion ilegal)"
                    .to_string(),
            );
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

    /// Igual que `attackers_to(...) & occupied_co[by_color] != 0` pero mirando
    /// SOLO las piezas del color que ataca y cortando en cuanto encuentra un
    /// atacante. La version anterior construia siempre el conjunto completo de
    /// atacantes de AMBOS colores (uniones de alfil+dama y torre+dama de los
    /// dos bandos) para despues descartar la mitad. Resultado booleano
    /// identico. Se llama en cada verificacion de jaque, o sea muchisimo.
    pub fn is_square_attacked_by(&self, sq: Square, by_color: Color) -> bool {
        let p = &self.pieces[by_color as usize];
        if pawn_attacks(by_color.opposite(), sq) & p[PieceType::Pawn as usize] != 0 {
            return true;
        }
        if knight_attacks(sq) & p[PieceType::Knight as usize] != 0 {
            return true;
        }
        if king_attacks(sq) & p[PieceType::King as usize] != 0 {
            return true;
        }
        let alfil_dama = p[PieceType::Bishop as usize] | p[PieceType::Queen as usize];
        if alfil_dama != 0 && bishop_attacks(sq, self.occupied) & alfil_dama != 0 {
            return true;
        }
        let torre_dama = p[PieceType::Rook as usize] | p[PieceType::Queen as usize];
        if torre_dama != 0 && rook_attacks(sq, self.occupied) & torre_dama != 0 {
            return true;
        }
        false
    }

    /// Mapa completo de casillas atacadas por `by_color` en la posicion
    /// actual: la union de los ataques de TODAS sus piezas. Para cualquier
    /// casilla sq, `attack_map(c) & bit(sq) != 0` es exactamente equivalente
    /// a `is_square_attacked_by(sq, c)` (mismos generadores de ataque, misma
    /// ocupacion). Se usa para amortizar: cuando hay que consultar la amenaza
    /// de MUCHAS casillas en el mismo nodo (ordenamiento de jugadas
    /// silenciosas), construir el mapa una vez (~O(piezas) lookups) es mas
    /// barato que ~5 lookups por cada consulta individual.
    pub fn attack_map(&self, by_color: Color) -> Bitboard {
        let p = &self.pieces[by_color as usize];
        let mut map: Bitboard = EMPTY;
        // Peones: los ocho de golpe con dos desplazamientos en vez de un
        // lookup por peon (mismo resultado exacto).
        map |= crate::bitboard::pawn_attacks_set(by_color, p[PieceType::Pawn as usize]);
        let mut knights = p[PieceType::Knight as usize];
        while knights != 0 {
            map |= knight_attacks(pop_lsb(&mut knights));
        }
        map |= king_attacks(crate::bitboard::lsb(p[PieceType::King as usize]));
        let mut alfil_dama = p[PieceType::Bishop as usize] | p[PieceType::Queen as usize];
        while alfil_dama != 0 {
            map |= bishop_attacks(pop_lsb(&mut alfil_dama), self.occupied);
        }
        let mut torre_dama = p[PieceType::Rook as usize] | p[PieceType::Queen as usize];
        while torre_dama != 0 {
            map |= rook_attacks(pop_lsb(&mut torre_dama), self.occupied);
        }
        map
    }

    pub fn king_square(&self, color: Color) -> Square {
        crate::bitboard::lsb(self.pieces[color as usize][PieceType::King as usize])
    }

    pub fn in_check(&self, color: Color) -> bool {
        self.is_square_attacked_by(self.king_square(color), color.opposite())
    }

    /// Quita la pieza y mantiene AL DIA el estado derivado (occupied_co /
    /// occupied). Antes make_move recalculaba los tres bitboards derivados
    /// desde cero al final (recompute_derived: releer los 12 bitboards de tipo
    /// y unirlos) aunque la jugada tocara como mucho 4 casillas. Actualizarlos
    /// aqui es exacto -- una casilla solo cambia de ocupacion cuando pasa por
    /// remove_piece/add_piece -- y elimina ese barrido por jugada.
    #[inline(always)]
    fn remove_piece(&mut self, color: Color, pt: PieceType, sq: Square) {
        let b = bit(sq);
        self.pieces[color as usize][pt as usize] &= !b;
        self.occupied_co[color as usize] &= !b;
        self.occupied &= !b;
        self.zobrist ^= keys().piece_square[color as usize][pt as usize][sq as usize];
    }

    #[inline(always)]
    fn add_piece(&mut self, color: Color, pt: PieceType, sq: Square) {
        let b = bit(sq);
        self.pieces[color as usize][pt as usize] |= b;
        self.occupied_co[color as usize] |= b;
        self.occupied |= b;
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

        // Captura (normal o al paso). El halfmove_clock se resetea UNA sola
        // vez, en el bloque general de abajo (mv.is_capture() cubre tambien
        // EnPassant) -- asignarlo aca y de nuevo alla era una duplicacion.
        if mv.flag == MoveFlag::EnPassant {
            let captured_sq = make_square(file_of(mv.to), rank_of(mv.from));
            b.remove_piece(them, PieceType::Pawn, captured_sq);
        } else if let Some((_, captured_pt)) = self.piece_at(mv.to) {
            if captured_pt == PieceType::King {
                // Red de seguridad contra posiciones ilegales (NUNCA debe
                // dispararse en una partida real: from_fen ya rechaza FENs
                // donde el bando que acaba de mover dejo su rey en jaque, y
                // generate_legal filtra las jugadas que dejan el propio rey
                // en jaque). El panic! historico aqui era fatal en produccion:
                // el perfil release usa panic=abort, asi que una posicion
                // ilegal recibida del GUI tumbaba el proceso entero a media
                // partida. Ahora se registra el diagnostico, se retira el rey
                // (comportamiento de una captura normal) y se sigue, en vez
                // de abortar. debug_assert! mantiene la deteccion en tests
                // (cargo test, perfil debug) sin coste en release.
                eprintln!(
                    "MITTENS: posicion ilegal en make_move (captura de rey en {}). FEN antes de la jugada: {}  jugada: {}",
                    square_name(mv.to),
                    self.to_fen(),
                    mv.to_uci()
                );
                debug_assert!(false, "intento de capturar un REY -- posicion ilegal o bug real");
            }
            b.remove_piece(them, captured_pt, mv.to);
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
        // Tabla en vez de la cadena de 4 comparaciones por casilla: cada
        // casilla guarda los derechos que SOBREVIVEN si esa casilla es origen
        // o destino de la jugada (solo a1/h1/a8/h8 quitan algo). Dos lecturas
        // de tabla sustituyen hasta ocho comparaciones por jugada.
        new_cr &= MASCARA_ENROQUE[mv.from as usize] & MASCARA_ENROQUE[mv.to as usize];
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

        // Sin recompute_derived: occupied/occupied_co ya vienen al dia desde
        // remove_piece/add_piece.
        debug_assert_eq!(b.occupied_co[0], b.pieces[0].iter().fold(0, |a, &x| a | x));
        debug_assert_eq!(b.occupied_co[1], b.pieces[1].iter().fold(0, |a, &x| a | x));
        debug_assert_eq!(b.occupied, b.occupied_co[0] | b.occupied_co[1]);
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

    // --- Saneo silencioso de derechos de enroque (estilo Stockfish) -------
    //
    // Un derecho solo sobrevive si el rey esta en su casilla original Y la
    // torre correspondiente en la suya. Lo demas se descarta EN SILENCIO (no
    // es error) para no romper la interoperabilidad con GUIs y libros que
    // emiten el campo de enroque sin actualizar.

    use super::{CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};

    /// Comprueba que `fen` se acepta y que sus derechos quedan en `esperados`,
    /// y de paso que el zobrist recalculado usa ya los derechos saneados.
    fn saneo(fen: &str, esperados: u8) {
        let b = Board::from_fen(fen).unwrap_or_else(|e| panic!("FEN rechazado {}: {}", fen, e));
        assert_eq!(
            b.castling_rights, esperados,
            "derechos saneados inesperados en {}",
            fen
        );
        // El hash guardado debe coincidir con el recalculado desde cero, que
        // usa castling_rights: si el saneo ocurriera despues del hash, este
        // assert fallaria.
        let mut copia = b;
        copia.recompute_zobrist();
        assert_eq!(b.zobrist, copia.zobrist, "zobrist inconsistente en {}", fen);
    }

    #[test]
    fn sanea_enroque_sin_torre() {
        // Rey en e1 pero sin torre en h1: 'K' se descarta en silencio.
        saneo("4k3/8/8/8/8/8/8/4K3 w K - 0 1", 0);
        // Sin torre en a1: 'Q' se descarta.
        saneo("4k3/8/8/8/8/8/8/4K2R w KQ - 0 1", CASTLE_WK);
        // Torre negra movida de h8 a g8: 'k' cae, 'q' sobrevive.
        saneo("r3k1r1/8/8/8/8/8/8/4K3 w kq - 0 1", CASTLE_BQ);
    }

    #[test]
    fn sanea_enroque_con_rey_movido() {
        // Rey blanco en d1 con ambas torres en su sitio: caen 'K' y 'Q'.
        saneo("4k3/8/8/8/8/8/8/R2K3R w KQ - 0 1", 0);
        // Rey negro en e7: caen 'k' y 'q', los blancos se conservan.
        saneo("r6r/4k3/8/8/8/8/8/R3K2R w KQkq - 0 1", CASTLE_WK | CASTLE_WQ);
    }

    #[test]
    fn sanea_enroque_rey_y_torre_movidos() {
        // Rey blanco fuera de e1 y ninguna torre en su esquina; negras intactas.
        saneo("r3k2r/8/8/8/8/8/8/3K1R2 w KQkq - 0 1", CASTLE_BK | CASTLE_BQ);
        // Todo movido por ambos bandos: no queda ningun derecho.
        saneo("1r2k1r1/8/8/8/8/8/8/1R2K1R1 w KQkq - 0 1", 0);
    }

    #[test]
    fn sanea_solo_el_lado_afectado() {
        // Solo falta la torre de a1: cae 'Q' y sobreviven 'K', 'k' y 'q'.
        saneo(
            "r3k2r/8/8/8/8/8/8/4K2R w KQkq - 0 1",
            CASTLE_WK | CASTLE_BK | CASTLE_BQ,
        );
        // Solo falta la torre de h8: cae 'k' y sobreviven los demas.
        saneo(
            "r3k3/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            CASTLE_WK | CASTLE_WQ | CASTLE_BQ,
        );
    }

    /// Bateria de FENs bien formados y variados (incluidas posiciones con
    /// enroque legitimo por ambos lados) que NO deben verse alterados por el
    /// saneo: los derechos se conservan exactamente y el FEN re-emitido es
    /// identico al de entrada.
    #[test]
    fn fens_validos_no_cambian_en_absoluto() {
        let casos: [(&str, u8); 12] = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ,
            ),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ,
            ),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                CASTLE_BK | CASTLE_BQ,
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                CASTLE_WK | CASTLE_WQ,
            ),
            (
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
                0,
            ),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 0),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
            (
                "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2",
                CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ,
            ),
            (
                "rnbqkbnr/ppp1pppp/8/8/3pP3/5N2/PPPP1PPP/RNBQKB1R b KQkq e3 0 3",
                CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ,
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ,
            ),
            ("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 5 42", CASTLE_WK | CASTLE_BQ),
            ("4k2r/8/8/8/8/8/8/R3K3 w Qk - 0 1", CASTLE_WQ | CASTLE_BK),
        ];
        for (fen, derechos) in casos {
            let b = Board::from_fen(fen).unwrap_or_else(|e| panic!("FEN valido {}: {}", fen, e));
            assert_eq!(
                b.castling_rights, derechos,
                "el saneo altero un FEN valido: {}",
                fen
            );
            assert_eq!(b.to_fen(), fen, "round-trip de FEN alterado: {}", fen);
            let mut copia = b;
            copia.recompute_zobrist();
            assert_eq!(b.zobrist, copia.zobrist, "zobrist inconsistente en {}", fen);
        }
    }

    /// El saneo debe dejar la posicion indistinguible de la version del FEN ya
    /// corregida a mano: mismos derechos, mismo FEN emitido y mismo zobrist.
    #[test]
    fn fen_saneado_equivale_al_fen_ya_corregido() {
        let pares = [
            (
                "r3k2r/8/8/8/8/8/8/4K2R w KQkq - 0 1",
                "r3k2r/8/8/8/8/8/8/4K2R w Kkq - 0 1",
            ),
            (
                "1r2k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "1r2k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1",
            ),
            (
                "r6r/4k3/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "r6r/4k3/8/8/8/8/8/R3K2R w KQ - 0 1",
            ),
        ];
        for (sucio, limpio) in pares {
            let a = Board::from_fen(sucio).unwrap();
            let b = Board::from_fen(limpio).unwrap();
            assert_eq!(a.castling_rights, b.castling_rights, "derechos: {}", sucio);
            assert_eq!(a.to_fen(), b.to_fen(), "fen: {}", sucio);
            assert_eq!(a.zobrist, b.zobrist, "zobrist: {}", sucio);
            assert_eq!(a.to_fen(), limpio);
        }
    }

    /// El saneo no debe romper el mantenimiento incremental del hash: tras
    /// jugar sobre una posicion saneada, el zobrist incremental sigue
    /// coincidiendo con el recomputado.
    #[test]
    fn zobrist_incremental_consistente_tras_saneo() {
        let b = Board::from_fen("r3k2r/8/8/8/8/8/8/R2K3R w KQkq - 0 1").unwrap();
        assert_eq!(b.castling_rights, CASTLE_BK | CASTLE_BQ);
        let mut copia = b;
        copia.recompute_zobrist();
        assert_eq!(b.zobrist, copia.zobrist);

        for mv in crate::movegen::generate_legal(&b) {
            let hijo = b.make_move(&mv);
            let mut recomputado = hijo;
            recomputado.recompute_zobrist();
            assert_eq!(
                hijo.zobrist,
                recomputado.zobrist,
                "hash incremental != recomputado tras {}",
                mv.to_uci()
            );
        }
    }

    /// Los derechos malformados (caracter desconocido o repetido) siguen
    /// siendo un error: el saneo es para derechos imposibles, no para basura.
    #[test]
    fn rechaza_campo_de_enroque_malformado() {
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/4K3 w X - 0 1").is_err());
        assert!(Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KK - 0 1").is_err());
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
