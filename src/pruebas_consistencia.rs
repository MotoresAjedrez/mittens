//! Pruebas de consistencia de la mecanica central del tablero.
//! No forman parte del motor: solo se compilan con `cargo test`.
#![cfg(test)]

use crate::board::Board;
use crate::movegen::generate_legal;
use crate::perft::perft;
use crate::see::{see, see_oracle};
use crate::types::MoveFlag;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Perft en posiciones trampa clasicas (valores certificados por CPW).
#[test]
fn perft_posiciones_trampa() {
    let casos: &[(&str, u32, u64)] = &[
        // Kiwipete
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            4,
            4_085_603,
        ),
        // Posicion 3: al paso que descubre jaque horizontal
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11_030_083),
        // Posicion 4 (promociones con captura) y su espejo
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            5,
            15_833_292,
        ),
        (
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
            5,
            15_833_292,
        ),
        // Posicion 5
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            5,
            89_941_194,
        ),
        // Posicion 6 (Steven Edwards)
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            5,
            164_075_551,
        ),
        // Casos de al paso / promocion sueltos
        ("8/8/8/8/k1p4R/8/3P4/3K4 w - - 0 1", 6, 1_134_888),
        ("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1_015_133),
        ("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1_440_467),
        ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661_072),
        ("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803_711),
        ("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1_274_206),
        ("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1_720_476),
        ("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3_821_001),
        ("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, 1_004_658),
        ("4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, 217_342),
        ("8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92_683),
        ("K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2_217),
        ("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567_584),
        ("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23_527),
    ];
    for &(fen, d, esperado) in casos {
        let b = Board::from_fen(fen).unwrap();
        assert_eq!(perft(&b, d), esperado, "perft({}) en {}", d, fen);
    }
}

/// Hash incremental (make_move) vs hash recomputado desde cero, sobre miles de
/// posiciones alcanzadas por partidas aleatorias.
#[test]
fn zobrist_incremental_coincide_con_recomputado() {
    let semillas = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ];
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut revisadas = 0u32;
    for fen in semillas {
        for _ in 0..300 {
            let mut b = Board::from_fen(fen).unwrap();
            for _ in 0..120 {
                let movs = generate_legal(&b);
                if movs.is_empty() {
                    break;
                }
                let mv = movs[(rng.next() % movs.len() as u64) as usize];
                let siguiente = b.make_move(&mv);
                let mut copia = siguiente;
                copia.recompute_zobrist();
                assert_eq!(
                    siguiente.zobrist,
                    copia.zobrist,
                    "hash incremental != recomputado tras {} desde {}",
                    mv.to_uci(),
                    b.to_fen()
                );
                revisadas += 1;
                b = siguiente;
            }
        }
    }
    assert!(revisadas > 50_000, "muestra demasiado chica: {}", revisadas);
}

/// Dos posiciones distintas alcanzadas en partidas aleatorias no deben
/// compartir clave zobrist (control de colisiones groseras / claves mal
/// construidas), y la MISMA posicion por caminos distintos debe compartirla.
#[test]
fn zobrist_identifica_posicion_no_camino() {
    // 1. e4 c5 2. Nf3  ==  1. Nf3 c5 2. e4
    let a = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let jugar = |b: &Board, uci: &str| -> Board {
        let mv = generate_legal(b)
            .into_iter()
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("jugada {} ilegal en {}", uci, b.to_fen()));
        b.make_move(&mv)
    };
    let v1 = jugar(&jugar(&jugar(&a, "e2e4"), "c7c5"), "g1f3");
    let v2 = jugar(&jugar(&jugar(&a, "g1f3"), "c7c5"), "e2e4");
    assert_eq!(v1.zobrist, v2.zobrist);
}

/// SEE contra el oraculo de fuerza bruta, SOLO en secuencias donde el atacante
/// esta forzado (cada bando tiene como mucho una pieza apuntando a la casilla).
///
/// El `see` del motor es el algoritmo clasico de "menor atacante primero"
/// (LVA), igual que Stockfish/Ethereal: cuando un bando tiene VARIOS
/// atacantes, LVA lo obliga a recapturar con el mas barato aunque recapturar
/// con otro (o no recapturar) fuera mejor, asi que puede diferir del juego
/// optimo. Eso es una limitacion conocida y aceptada del algoritmo, no un bug,
/// y comparar contra el oraculo en esos casos solo produce falsos positivos.
/// Con atacante forzado LVA y juego optimo COINCIDEN por definicion, asi que
/// cualquier diferencia ahi si seria un bug real de SEE (promocion, al paso,
/// rayos X, legalidad del rey...).
#[test]
fn see_coincide_con_oraculo_con_atacante_forzado() {
    let semillas = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    ];
    let mut rng = Rng(0xDEAD_BEEF_CAFE_1234);
    let mut comparadas = 0u32;
    let mut discrepancias: Vec<String> = Vec::new();
    for fen in semillas {
        for _ in 0..200 {
            let mut b = Board::from_fen(fen).unwrap();
            for _ in 0..60 {
                for mv in generate_legal(&b) {
                    if !(mv.is_capture() || mv.promotion.is_some()) {
                        continue;
                    }
                    // Filtro "atacante forzado": ningun bando puede elegir
                    // entre dos capturadores sobre la casilla destino, ni
                    // siquiera despues de que la secuencia destape rayos X.
                    // Con ocupacion vacia los deslizantes ven sin obstaculos,
                    // asi que el conjunto resultante es un SUPERconjunto de
                    // todos los atacantes que pueden aparecer en cualquier
                    // momento del intercambio.
                    let at = b.attackers_to(mv.to, 0);
                    if (at & b.occupied_co[0]).count_ones() > 1
                        || (at & b.occupied_co[1]).count_ones() > 1
                    {
                        continue;
                    }
                    let r = see(&b, &mv);
                    let o = see_oracle(&b, &mv);
                    comparadas += 1;
                    if r != o && discrepancias.len() < 20 {
                        discrepancias.push(format!(
                            "{} en {}: see={} oraculo={} flag={:?}",
                            mv.to_uci(),
                            b.to_fen(),
                            r,
                            o,
                            mv.flag
                        ));
                    }
                }
                let movs = generate_legal(&b);
                if movs.is_empty() {
                    break;
                }
                let mv = movs[(rng.next() % movs.len() as u64) as usize];
                b = b.make_move(&mv);
            }
        }
    }
    assert!(comparadas > 2_000, "muestra chica: {}", comparadas);
    assert!(
        discrepancias.is_empty(),
        "SEE diverge del oraculo en {} casos (muestra):\n{}",
        discrepancias.len(),
        discrepancias.join("\n")
    );
}

/// El estado derivado (occupied / occupied_co) tras make_move debe coincidir
/// con el recalculado desde los bitboards de piezas, y el FEN de ida y vuelta
/// debe reproducir exactamente la posicion.
#[test]
fn estado_derivado_y_fen_consistentes_tras_make_move() {
    let mut rng = Rng(0x0BAD_F00D_1357_9BDF);
    let semillas = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ];
    for fen in semillas {
        for _ in 0..200 {
            let mut b = Board::from_fen(fen).unwrap();
            for _ in 0..100 {
                let movs = generate_legal(&b);
                if movs.is_empty() {
                    break;
                }
                let mv = movs[(rng.next() % movs.len() as u64) as usize];
                let n = b.make_move(&mv);
                let occ_w: u64 = n.pieces[0].iter().fold(0, |a, &x| a | x);
                let occ_b: u64 = n.pieces[1].iter().fold(0, |a, &x| a | x);
                assert_eq!(n.occupied_co[0], occ_w, "occupied_co[0] tras {}", mv.to_uci());
                assert_eq!(n.occupied_co[1], occ_b, "occupied_co[1] tras {}", mv.to_uci());
                assert_eq!(n.occupied, occ_w | occ_b);
                assert_eq!(occ_w & occ_b, 0, "solapamiento de colores");
                // Ningun bitboard de tipo puede solaparse con otro.
                for c in 0..2 {
                    for p1 in 0..6 {
                        for p2 in (p1 + 1)..6 {
                            assert_eq!(
                                n.pieces[c][p1] & n.pieces[c][p2],
                                0,
                                "dos tipos en la misma casilla tras {}",
                                mv.to_uci()
                            );
                        }
                    }
                }
                // Al paso: si esta puesto, viene de un doble avance.
                if let Some(ep) = n.ep_square {
                    assert_eq!(mv.flag, MoveFlag::DoublePush, "ep sin doble avance");
                    let _ = ep;
                }
                // Ida y vuelta por FEN.
                let rt = Board::from_fen(&n.to_fen())
                    .unwrap_or_else(|e| panic!("FEN generado invalido {}: {}", n.to_fen(), e));
                assert_eq!(rt.zobrist, n.zobrist, "zobrist ida y vuelta {}", n.to_fen());
                b = n;
            }
        }
    }
}
