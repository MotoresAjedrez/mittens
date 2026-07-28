use crate::board::Board;
use crate::movegen::generate_legal;

pub fn perft(b: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = generate_legal(b);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0u64;
    for mv in &moves {
        let next = b.make_move(mv);
        nodes += perft(&next, depth - 1);
    }
    nodes
}

/// perft "divide": nodos por cada jugada de raíz -- útil para encontrar en qué
/// rama exacta diverge un perft incorrecto comparando contra otro motor.
pub fn perft_divide(b: &Board, depth: u32) -> Vec<(String, u64)> {
    if depth == 0 {
        return Vec::new();
    }
    let moves = generate_legal(b);
    let mut out = Vec::new();
    for mv in &moves {
        let next = b.make_move(mv);
        let n = perft(&next, depth - 1);
        out.push((mv.to_uci(), n));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divide_profundidad_cero_no_desborda() {
        let b = Board::startpos();
        assert!(perft_divide(&b, 0).is_empty());
    }

    #[test]
    fn perft_inicio_hasta_tres() {
        let b = Board::startpos();
        assert_eq!(perft(&b, 1), 20);
        assert_eq!(perft(&b, 2), 400);
        assert_eq!(perft(&b, 3), 8_902);
    }
}
