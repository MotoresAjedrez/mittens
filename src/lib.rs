mod bitboard;
mod board;
mod bullet_net;
mod bullet_net_amenazas;
mod eval;
mod movegen;
mod neural;
mod perft;
mod polyglot;
mod polyglot_random;
mod pruebas_consistencia;
mod search;
mod see;
mod syzygy;
mod types;
mod zobrist;
// Puente FFI para embeber el motor (iOS/macOS): usa pipes de Unix
// (dup2/RawFd), no existe equivalente en Windows y no lo necesita el
// binario UCI de consola. Se compila solo en plataformas Unix-like.
#[cfg(unix)]
pub mod ffi;
// Puente JNI: solo en Android (necesita la dependencia `jni`).
#[cfg(target_os = "android")]
pub mod jni_bridge;

use board::Board;
use search::Searcher;
use std::env;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;
use types::{Move, MoveFlag, PieceType};

/// Factor por el que la evaluacion INTERNA supera la escala estandar de UCI,
/// en la que 100 cp = un peon. Medido contra Stockfish 18 sobre 16 posiciones
/// a 1.5s/posicion, filtrando al rango donde la eval sigue siendo lineal
/// (|eval| entre 30 y 700 cp): el ratio mediano mittens/stockfish es 1.56, y
/// sube hacia ~1.9 en posiciones ya muy ganadas. Se toma 1.6 porque cuadra el
/// rango normal, que es donde de verdad se leen las evaluaciones; las
/// ventajas decisivas seguiran reportandose algo altas.
///
/// El origen de la inflacion es que `evaluate_with_state` suma la clasica
/// COMPLETA y la red bullet COMPLETA (ambas evaluan la posicion entera), asi
/// que el material se cuenta dos veces. Eso NO es un bug de fuerza: medido,
/// el modo "red pura" busca ~1.4 plies menos hondo a 3s y un h2h previo lo
/// dio peor (55.0% contra 63.3%). Los margenes de poda se afinaron dentro de
/// esta escala y son consistentes con ella.
///
/// Por eso este ajuste es SOLO de presentacion: se aplica al imprimir y a
/// nada mas. La busqueda, los margenes y las ventanas de aspiracion siguen
/// trabajando en la escala interna intacta, asi que no puede mover un Elo.
/// Se desactiva con MIMOTOR_ESCALA_UCI=1.0.
fn escala_uci() -> f64 {
    static CACHE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("MIMOTOR_ESCALA_UCI")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && (0.1..=10.0).contains(v))
            .unwrap_or(1.6)
    })
}

/// Formatea el score para el campo "info score" del protocolo UCI: cerca del
/// limite de mate (search::MATE) se reporta como "mate N" (N = jugadas hasta
/// el mate, negativo si el que mueve va a ser matado), como exige el estandar
/// UCI -- antes siempre se imprimia "cp X" incluso con X en la zona de mate
/// (p.ej. 28987), que herramientas/GUIs no interpretan como mate inminente.
fn formatear_score_uci(score: i32) -> String {
    const UMBRAL: i32 = search::MATE - 1000;
    if score > UMBRAL {
        let plies = search::MATE - score;
        format!("mate {}", (plies + 1) / 2)
    } else if score < -UMBRAL {
        let plies = search::MATE + score;
        format!("mate -{}", (plies + 1) / 2)
    } else {
        // Normalizado a "100 = un peon" (ver escala_uci). La rama de mate no
        // se toca: "mate N" cuenta jugadas, no centipeones.
        format!("cp {}", (score as f64 / escala_uci()).round() as i32)
    }
}

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const POSITION3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const POSITION5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1";

/// Modo comodo para probar una posicion suelta sin protocolo UCI:
///   mimotor-tal-rust simple "FEN"
///   mimotor-tal-rust simple 3000 "FEN"   (movetime en ms, default 2000)
fn run_simple(fen: &str, movetime_ms: u64) {
    let b = match Board::from_fen(fen) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FEN invalido: {}", e);
            return;
        }
    };
    let mut s = Searcher::new(64);
    let (mv, sc, prof) = s.search_time(&b, Some(movetime_ms), 64, |_, _, _, _, _| {});
    let pv = s.extraer_pv(&b, 12);
    let pv_txt = pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ");
    println!("================ MiMotor Tal ================");
    println!(
        "Mejor jugada: {}",
        mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string())
    );
    println!("Evaluacion: {:+.2}", sc as f64 / 100.0);
    println!("Profundidad: {}", prof);
    println!("Nodos: {}", s.nodes);
    println!(
        "Linea principal: {}",
        if pv_txt.is_empty() {
            "(no disponible)"
        } else {
            &pv_txt
        }
    );
    println!("===============================================");
}

fn run_perft_suite() {
    let cases: Vec<(&str, &str, Vec<u64>)> = vec![
        (
            "posicion inicial",
            STARTPOS,
            vec![1, 20, 400, 8902, 197281, 4865609, 119060324],
        ),
        (
            "kiwipete",
            KIWIPETE,
            vec![1, 48, 2039, 97862, 4085603, 193690690],
        ),
        (
            "posicion 3",
            POSITION3,
            vec![1, 14, 191, 2812, 43238, 674624, 11030083],
        ),
        (
            "posicion 5",
            POSITION5,
            vec![1, 44, 1486, 62379, 2103487, 89941194],
        ),
    ];

    let mut todo_ok = true;
    for (nombre, fen, esperados) in &cases {
        let b = Board::from_fen(fen).expect("FEN invalido en suite de perft");
        println!("\n=== {} ===", nombre);
        for (depth, &esperado) in esperados.iter().enumerate() {
            let t0 = Instant::now();
            let n = perft::perft(&b, depth as u32);
            let dt = t0.elapsed();
            let ok = n == esperado;
            todo_ok &= ok;
            let nps = if dt.as_secs_f64() > 0.0 {
                n as f64 / dt.as_secs_f64()
            } else {
                0.0
            };
            println!(
                "  depth {}: {} (esperado {}) {}  [{:.2}s, {:.0} nps]",
                depth,
                n,
                esperado,
                if ok { "OK" } else { "*** FALLO ***" },
                dt.as_secs_f64(),
                nps
            );
            if !ok {
                break;
            }
        }
    }
    println!(
        "\n{}",
        if todo_ok {
            "TODOS LOS PERFT OK"
        } else {
            "HAY FALLOS DE PERFT -- NO AVANZAR A FASE 3"
        }
    );
}

fn run_divide(fen: &str, depth: u32) {
    let b = Board::from_fen(fen).expect("FEN invalido");
    let mut total = 0u64;
    for (uci, n) in perft::perft_divide(&b, depth) {
        println!("{}: {}", uci, n);
        total += n;
    }
    println!("total: {}", total);
}

fn run_smp_bench(movetime_ms: u64) {
    let fen = "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6";
    let b = Board::from_fen(fen).unwrap();
    println!(
        "Benchmark Lazy SMP -- posicion de medio juego, {}ms por hilo/config",
        movetime_ms
    );
    for n in [1usize, 2, 4, 6, 8] {
        let (tt, tt_mask) = search::construir_tt(64);
        let generacion = AtomicU8::new(0);
        let t0 = Instant::now();
        let (mv, sc, nodos, resultados) = search::buscar_lazy_smp(
            &b,
            Some(movetime_ms),
            64,
            n,
            &tt,
            &generacion,
            tt_mask,
            true,
            true,
            0,
            &[],
            Arc::new(AtomicBool::new(false)),
            None,
        );
        let dt = t0.elapsed();
        let nps = nodos as f64 / dt.as_secs_f64().max(0.0001);
        let profs: Vec<i32> = resultados.iter().map(|r| r.profundidad).collect();
        println!(
            "  {} hilo(s): jugada={} score={} | {} nodos TOTALES en {:.2}s = {:.0} nps combinados | profundidades por hilo: {:?}",
            n,
            mv.map(|m| m.to_uci()).unwrap_or_default(),
            sc,
            nodos,
            dt.as_secs_f64(),
            nps,
            profs
        );
    }
}

// Posiciones de referencia para "bench": apertura, un par de medio juegos
// (abierto y cerrado), un final tactico y uno de solo peones. Ninguna
// depende de NNUE (bench corre con evaluacion clasica, sin cargar pesos,
// para que el numero sea reproducible en cualquier maquina sin depender de
// un archivo externo) -- asi sirve como comparacion de velocidad pura de
// motor/hardware, que es para lo que testers como los de CCRL lo usan.
const BENCH_POSICIONES: [(&str, &str); 6] = [
    ("inicial", STARTPOS),
    (
        "medio juego abierto",
        "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
    ),
    (
        "medio juego cerrado",
        "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 11",
    ),
    (
        "tactico",
        "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 4 8",
    ),
    ("final de torres", "8/2p2pk1/1p1p2p1/p2Pp2p/P1P1P2P/1P3PP1/6K1/8 w - - 0 1"),
    ("final de peones", "8/8/1p1k4/p1p5/P1P5/1P1K4/8/8 w - - 0 1"),
];

fn run_bench(depth: i32) {
    let mut nodos_totales: u64 = 0;
    let mut ms_totales: u128 = 0;
    for (nombre, fen) in BENCH_POSICIONES {
        let b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::new(64);
        let t0 = Instant::now();
        let (mv, sc, nodes) = s.search_fixed_depth(&b, depth);
        let dt = t0.elapsed();
        let nps = nodes as f64 / dt.as_secs_f64().max(0.0001);
        nodos_totales += nodes;
        ms_totales += dt.as_millis();
        println!(
            "{}: profundidad {} -> {} (score {}) | {} nodos en {:.2}s = {:.0} nps",
            nombre,
            depth,
            mv.map(|m| m.to_uci()).unwrap_or_default(),
            sc,
            nodes,
            dt.as_secs_f64(),
            nps
        );
        if s.lmr_intentos > 0 {
            println!(
                "  LMR: {} intentos, {} re-busquedas a profundidad completa ({:.1}%)",
                s.lmr_intentos,
                s.lmr_reintentos,
                100.0 * s.lmr_reintentos as f64 / s.lmr_intentos as f64
            );
        }
    }
    // Bloque resumen en el formato convencional (Stockfish y similares) que
    // los testers automaticos (Cutechess, OpenBench, scripts de CCRL) suelen
    // parsear -- ademas de las lineas en espanol de arriba, no en vez de.
    let nps_total = nodos_totales as f64 / (ms_totales as f64 / 1000.0).max(0.0001);
    println!("===========================");
    println!("Total time (ms) : {}", ms_totales);
    println!("Nodes searched  : {}", nodos_totales);
    println!("Nodes/second    : {:.0}", nps_total);
}

fn run_lmr_diagnostico(depth: i32) {
    // Compara LMR encendido vs apagado en un lote de posiciones tacticas
    // conocidas, a PROFUNDIDAD FIJA (no por tiempo, para que la comparacion
    // sea limpia): si LMR encuentra el mismo score o mejor en todas, es
    // evidencia fuerte de que el mecanismo en si es correcto (no hay jugadas
    // tacticas que se esten perdiendo por la reduccion) y la perdida de ELO
    // medida en torneo es un tema de presupuesto de tiempo/profundidad
    // efectiva, no un bug de logica.
    let posiciones = [
        ("inicial", STARTPOS),
        (
            "medio juego",
            "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        ),
        (
            "tactica capturas",
            "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/1PN1PN2/PB3PPP/R2Q1RK1 w - - 0 10",
        ),
        (
            "bug cxb4",
            "r1b1k2r/ppqp1ppp/4p3/4n3/1b6/2PQBN2/P1P2PPP/R3KB1R w KQkq - 0 11",
        ),
        (
            "mate pastor",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        ),
        ("final torres", "8/5pk1/6p1/8/8/1R6/5PPP/6K1 w - - 0 1"),
        (
            "kiwipete-ish",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ),
    ];
    let mut peor_encontrado = false;
    for (nombre, fen) in posiciones {
        let b = Board::from_fen(fen).unwrap();
        let mut s_sin = Searcher::new(32);
        s_sin.modo_lmr = false;
        let (mv_sin, sc_sin, _) = s_sin.search_fixed_depth(&b, depth);

        let mut s_con = Searcher::new(32);
        s_con.modo_lmr = true;
        let (mv_con, sc_con, _) = s_con.search_fixed_depth(&b, depth);

        let diff = sc_con - sc_sin;
        let peor = diff < -20; // margen chico de ruido por desempates de orden
        peor_encontrado |= peor;
        println!(
            "{:20} sin-LMR: {} (score {})   con-LMR: {} (score {})   diff={:+}  {}",
            nombre,
            mv_sin.map(|m| m.to_uci()).unwrap_or_default(),
            sc_sin,
            mv_con.map(|m| m.to_uci()).unwrap_or_default(),
            sc_con,
            diff,
            if peor {
                "*** LMR PEOR ***"
            } else if mv_sin == mv_con {
                "(misma jugada)"
            } else {
                "(distinta jugada, score similar)"
            }
        );
    }
    println!(
        "\n{}",
        if peor_encontrado {
            "LMR encontro una jugada claramente peor en al menos una posicion -- revisar mas"
        } else {
            "LMR no perdio ninguna tactica en este lote a profundidad fija -- consistente con tradeoff de tiempo, no bug de logica"
        }
    );
}

fn run_singular_diagnostico(depth: i32) {
    // Mismo protocolo que run_lmr_diagnostico (comparacion a profundidad FIJA,
    // no por tiempo): si singular extensions es correcto, el score CON
    // extensiones nunca deberia ser peor que SIN ellas en este lote -- una
    // jugada "singular" que en realidad no lo era, o un bug de contabilidad
    // de profundidad/ventana, se notaria como una caida de score o (mas
    // grave) una explosion de nodos sin límite claro. Verde en TODO este
    // lote es la condicion pedida antes de considerar activar el default.
    let posiciones = [
        ("inicial", STARTPOS),
        (
            "medio juego",
            "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        ),
        (
            "tactica capturas",
            "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/1PN1PN2/PB3PPP/R2Q1RK1 w - - 0 10",
        ),
        (
            "bug cxb4",
            "r1b1k2r/ppqp1ppp/4p3/4n3/1b6/2PQBN2/P1P2PPP/R3KB1R w KQkq - 0 11",
        ),
        (
            "mate pastor",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        ),
        ("final torres", "8/5pk1/6p1/8/8/1R6/5PPP/6K1 w - - 0 1"),
        (
            "kiwipete-ish",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ),
        (
            "cacería de rey (partida real v11, jug.31)",
            "2r3k1/p1bQ1ppp/6b1/6N1/8/5P2/P4P1P/6K1 b - - 1 31",
        ),
    ];
    let mut peor_encontrado = false;
    let mut nodos_explotaron = false;
    for (nombre, fen) in posiciones {
        let b = Board::from_fen(fen).unwrap();
        let mut s_sin = Searcher::new(32);
        s_sin.modo_singular = false;
        let (mv_sin, sc_sin, nodos_sin) = s_sin.search_fixed_depth(&b, depth);

        let mut s_con = Searcher::new(32);
        s_con.modo_singular = true;
        let (mv_con, sc_con, nodos_con) = s_con.search_fixed_depth(&b, depth);

        let diff = sc_con - sc_sin;
        let peor = diff < -20; // margen chico de ruido por desempates de orden
        let ratio_nodos = nodos_con as f64 / nodos_sin.max(1) as f64;
        let exploto = ratio_nodos > 4.0; // sondeos de SE suman busqueda extra, pero no deberian multiplicar por mucho mas que eso
        peor_encontrado |= peor;
        nodos_explotaron |= exploto;
        println!(
            "{:30} sin-SE: {} (score {}, {} nodos)   con-SE: {} (score {}, {} nodos, x{:.1})   diff={:+}  {}{}",
            nombre,
            mv_sin.map(|m| m.to_uci()).unwrap_or_default(),
            sc_sin,
            nodos_sin,
            mv_con.map(|m| m.to_uci()).unwrap_or_default(),
            sc_con,
            nodos_con,
            ratio_nodos,
            diff,
            if peor {
                "*** SE PEOR ***"
            } else if mv_sin == mv_con {
                "(misma jugada)"
            } else {
                "(distinta jugada, score similar)"
            },
            if exploto {
                "  *** EXPLOSION DE NODOS ***"
            } else {
                ""
            }
        );
    }
    println!(
        "\n{}",
        if peor_encontrado || nodos_explotaron {
            "singular extensions FALLO el diagnostico -- NO activar por defecto (MIMOTOR_SINGULAR=1 sigue disponible solo para pruebas)"
        } else {
            "singular extensions paso el diagnostico: sin perdidas de score ni explosion de nodos en este lote"
        }
    );
}

fn run_mate_tests() {
    let casos = [
        (
            "Mate en 1: Ta8# (pasillo)",
            "6k1/8/6K1/8/8/8/8/R7 w - - 0 1",
            "a1a8",
        ),
        (
            "Mate en 1: Dh4# (mate del loco)",
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 1",
            "d8h4",
        ),
        (
            "Mate en 1: Dxf7# (mate pastor)",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "h5f7",
        ),
    ];
    let mut todo_ok = true;
    for (nombre, fen, esperada) in casos.iter() {
        let b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::new(16);
        let (mv, sc, nodes) = s.search_fixed_depth(&b, 4);
        let uci = mv.map(|m| m.to_uci()).unwrap_or_default();
        let ok = uci == *esperada;
        todo_ok &= ok;
        println!(
            "{} {} -> {} (esperada {}) score={} nodos={}",
            if ok { "OK  " } else { "FAIL" },
            nombre,
            uci,
            esperada,
            sc,
            nodes
        );
    }
    println!(
        "{}",
        if todo_ok {
            "TODOS LOS MATES OK"
        } else {
            "HAY FALLOS EN LA SUITE DE MATES"
        }
    );
}

fn run_prueba_apertura() {
    // 1.e4 d5 2.exd5 Qxd5 -- el motor deberia enrocar en unas pocas jugadas
    // propias en vez de salir a cazar peones con la dama (bug historico de v1).
    let mut b = Board::from_fen(STARTPOS).unwrap();
    let mut s = Searcher::new(64);
    let jugadas_iniciales = ["e2e4", "d7d5", "e4d5", "d8d5"];
    for uci in jugadas_iniciales {
        b = aplicar_uci(&b, uci);
    }
    println!("Posicion tras 1.e4 d5 2.exd5 Qxd5, buscando 10 jugadas propias del motor...");
    let mut enrocó = false;
    for i in 0..10 {
        let (mv, sc, nodes) = s.search_fixed_depth(&b, 5);
        let mv = match mv {
            Some(m) => m,
            None => break,
        };
        println!(
            "  jugada propia #{}: {} (score {}, {} nodos)",
            i + 1,
            mv.to_uci(),
            sc,
            nodes
        );
        if mv.flag == MoveFlag::CastleKing || mv.flag == MoveFlag::CastleQueen {
            enrocó = true;
        }
        b = b.make_move(&mv);
        if b.turn == types::Color::Black {
            // saltar la respuesta rival con una jugada legal cualquiera (la primera)
            let respuestas = movegen::generate_legal(&b);
            if let Some(r) = respuestas.first() {
                b = b.make_move(r);
            }
        }
        if enrocó {
            break;
        }
    }
    println!(
        "{}",
        if enrocó {
            "OK: el motor enrocó"
        } else {
            "el motor NO enrocó en 10 jugadas"
        }
    );
}

fn aplicar_uci(b: &Board, uci: &str) -> Board {
    let mv = parse_uci_move(b, uci).expect("jugada UCI invalida");
    b.make_move(&mv)
}

fn parse_uci_move(b: &Board, uci: &str) -> Option<Move> {
    let moves = movegen::generate_legal(b);
    let bytes = uci.as_bytes();
    if bytes.len() != 4 && bytes.len() != 5 {
        return None;
    }
    let from = types::square_from_name(&uci[0..2])?;
    let to = types::square_from_name(&uci[2..4])?;
    let promo = if bytes.len() == 5 {
        PieceType::from_char(bytes[4] as char)
    } else {
        None
    };
    if bytes.len() == 5 && promo.is_none() {
        return None;
    }
    moves
        .into_iter()
        .find(|m| m.from == from && m.to == to && m.promotion == promo)
}

/// Aplica la lista "moves ..." del comando UCI "position" de forma ESTRICTA:
/// TODAS las jugadas deben ser legales en la posicion actual del motor. Si
/// cualquiera falla, la posicion completa se rechaza (devuelve false SIN
/// tocar `board` ni `game_history`) y el llamador restaura la ultima
/// posicion valida (ver el handler de "position" en uci_loop, que hace
/// snapshot de board + game_history antes de parsear).
///
/// POR QUE: antes cada jugada ilegal se saltaba en silencio y la busqueda
/// seguia con un tablero interno DIVERGENTE del juego real -- el motor podia
/// emitir jugadas perfectamente legales en su mundo pero ILEGALES en la
/// partida real (clase de bug del incidente 'bestmove h5c5' con la dama en
/// c5: la dama estaba en h5 en el tablero interno porque una jugada previa
/// del stream se habia descartado en silencio). Un stream inconsistente es
/// un error grave del GUI/driver; se reporta y se conserva la ultima
/// posicion valida en vez de seguir jugando desde un tablero inventado.
fn aplicar_moves_position(
    board: &mut Board,
    game_history: &mut Vec<u64>,
    moves_uci: &[&str],
) -> bool {
    let mut b2 = *board;
    let mut hist: Vec<u64> = Vec::with_capacity(moves_uci.len());
    for uci in moves_uci {
        match parse_uci_move(&b2, uci) {
            Some(mv) => {
                hist.push(b2.zobrist);
                b2 = b2.make_move(&mv);
            }
            None => {
                println!(
                    "info string error de position: jugada ilegal o inaplicable en este tablero: {}",
                    uci
                );
                return false;
            }
        }
    }
    *board = b2;
    *game_history = hist;
    true
}

/// Construye una jugada SIN validar legalidad real de movimiento (solo
/// determina el flag -- Captura/AlPaso/Silenciosa -- a partir del estado
/// del tablero). Necesario para los casos sinteticos de test de SEE, donde
/// el "atacante inicial" puede estar bloqueado en la realidad (igual que
/// hace el test_see.py de Python, que tampoco valida legalidad del primer
/// movimiento -- SEE en si mismo confia en que el llamador ya eligio un
/// atacante valido).
fn jugada_sintetica(b: &Board, uci: &str) -> Move {
    let from = types::square_from_name(&uci[0..2]).unwrap();
    let to = types::square_from_name(&uci[2..4]).unwrap();
    let es_al_paso =
        b.piece_at(from).map(|(_, pt)| pt) == Some(PieceType::Pawn) && Some(to) == b.ep_square;
    let flag = if es_al_paso {
        MoveFlag::EnPassant
    } else if b.piece_at(to).is_some() {
        MoveFlag::Capture
    } else {
        MoveFlag::Quiet
    };
    Move::new(from, to, None, flag)
}

fn run_cxb4_bug() {
    let fen = "r1b1k2r/ppqp1ppp/4p3/4n3/1b6/2PQBN2/P1P2PPP/R3KB1R w KQkq - 0 11";
    let b = Board::from_fen(fen).unwrap();
    let mut s = Searcher::new(64);
    let (mv, sc, nodes) = s.search_fixed_depth(&b, 6);
    let uci = mv.map(|m| m.to_uci()).unwrap_or_default();
    println!(
        "FEN bug cxb4: jugada elegida = {} (score {}, {} nodos)",
        uci, sc, nodes
    );
    println!(
        "{}",
        if uci == "c3b4" {
            "eligio cxb4 (igual que v3 sin proteccion)"
        } else {
            "NO eligio cxb4"
        }
    );
}

fn run_endgame_tests() {
    // Finales basicos, resultado teorico conocido sin ambiguedad:
    //  1. K+D vs K: cualquier jugada razonable debe ser decisiva (score muy
    //     alto) y NUNCA ahogar al rey rival (blunder clasico de motores
    //     debiles en este final "trivial").
    //  2. K+P vs K con la oposicion ya tomada por el bando fuerte: el motor
    //     debe reconocerlo como claramente ganado (score alto), no como
    //     tablas -- el caso de libro mas basico de finales de peones.
    let mut todo_ok = true;

    {
        let fen = "8/8/7k/8/3Q4/8/8/K7 w - - 0 1";
        let b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::new(16);
        let (mv, sc, _) = s.search_fixed_depth(&b, 6);
        let mv = mv.unwrap();
        let next = b.make_move(&mv);
        let ahoga = movegen::generate_legal(&next).is_empty() && !next.in_check(next.turn);
        let ok = !ahoga && sc > 500;
        todo_ok &= ok;
        println!(
            "{} K+D vs K: jugada={} score={} {}",
            if ok { "OK  " } else { "FAIL" },
            mv.to_uci(),
            sc,
            if ahoga {
                "*** AHOGA AL REY RIVAL ***"
            } else {
                "(sin ahogar, score decisivo)"
            }
        );
    }

    {
        // Rey negro demasiado lejos para alcanzar al peon (sin ambiguedad
        // de oposicion): debe ser claramente ganado con tecnica trivial.
        let fen = "6k1/8/4K3/4P3/8/8/8/8 w - - 0 1";
        let b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::new(16);
        let (mv, sc, _) = s.search_fixed_depth(&b, 8);
        let ok = sc > 400; // claramente ganado, no "tablas" ni "un poco mejor"
        todo_ok &= ok;
        println!(
            "{} K+P vs K (rey rival fuera de alcance): jugada={} score={} {}",
            if ok { "OK  " } else { "FAIL" },
            mv.map(|m| m.to_uci()).unwrap_or_default(),
            sc,
            if ok {
                "(reconocido como ganado)"
            } else {
                "*** NO lo ve claramente ganado ***"
            }
        );
    }

    println!(
        "{}",
        if todo_ok {
            "TODOS LOS CASOS DE FINALES OK"
        } else {
            "HAY FALLOS EN LOS CASOS DE FINALES"
        }
    );
}

fn run_repetition_tests() {
    // Metodologia: buscar SIN contexto de repeticion para obtener la jugada
    // "natural" y la posicion que resulta de jugarla; despues sembrar
    // game_history con ESA posicion exacta (simulando que ya ocurrio una
    // vez antes en la partida) y volver a buscar. Esto prueba el mecanismo
    // de forma directa y verificable sin depender de armar a mano una
    // secuencia real de jaques repetibles.
    let mut ok = true;

    println!("=== Test A: GANANDO (K+D+T vs K), debe EVITAR repetir ===");
    // halfmove_clock=10 (no 0): la ventana de repeticion usa hc como
    // "cuantas jugadas reversibles hacia atras puedo mirar" -- con hc=0
    // es matematicamente imposible que exista una ocurrencia previa (el
    // contador se acaba de reiniciar), asi que hace falta margen real.
    let fen_a = "4k3/8/4K3/8/8/8/8/3QR3 w - - 10 6";
    let ba = Board::from_fen(fen_a).unwrap();
    let mut s1 = Searcher::new(16);
    let (mv1, sc1, _) = s1.search_fixed_depth(&ba, 6);
    let mv1 = mv1.expect("deberia haber jugada legal");
    let p2 = ba.make_move(&mv1);
    println!("  sin historial: jugada={} score={}", mv1.to_uci(), sc1);

    let mut s2 = Searcher::new(16);
    s2.set_game_history(vec![p2.zobrist]);
    let (mv2, sc2, _) = s2.search_fixed_depth(&ba, 6);
    let mv2 = mv2.expect("deberia haber jugada legal");
    println!(
        "  con esa posicion ya \"vista\": jugada={} score={}",
        mv2.to_uci(),
        sc2
    );
    // Debe seguir siendo claramente ganador (no cerca de 0) -- si eligio la
    // MISMA jugada que antes, su resultado NO debe coincidir con p2 (busco
    // otra continuacion), o el score debe seguir siendo muy alto.
    let evito = mv2 != mv1 || sc2.abs() > 500;
    ok &= evito;
    println!(
        "  {}",
        if evito {
            "OK: sigue jugando para ganar, no repite a lo tonto"
        } else {
            "FALLO: parece haber aceptado repetir estando ganando"
        }
    );

    println!("\n=== Test B: PERDIENDO (K vs K+D+T), debe BUSCAR repetir ===");
    let fen_b = "8/8/4k3/8/8/4K3/8/3qr3 w - - 10 6";
    let bb = Board::from_fen(fen_b).unwrap();
    let mut s3 = Searcher::new(16);
    let (mv3, sc3, _) = s3.search_fixed_depth(&bb, 6);
    let mv3 = mv3.expect("deberia haber jugada legal");
    let p2b = bb.make_move(&mv3);
    println!(
        "  sin historial: jugada={} score={} (posicion perdida de verdad)",
        mv3.to_uci(),
        sc3
    );

    let mut s4 = Searcher::new(16);
    s4.set_game_history(vec![p2b.zobrist]);
    let (mv4, sc4, _) = s4.search_fixed_depth(&bb, 6);
    let mv4 = mv4.expect("deberia haber jugada legal");
    println!(
        "  con esa posicion ya \"vista\": jugada={} score={}",
        mv4.to_uci(),
        sc4
    );
    // Ahora debe preferir la tabla en vez de seguir perdiendo (sc3, que
    // deberia ser muy negativo o mate en contra). El "contempt" dinamico
    // (agregado en esta misma sesion) puntua la repeticion a +/-200 cuando
    // el eval esta claramente decidido, no exactamente 0 -- por eso el
    // margen es <= 200 (no < 200): el valor esperado de verdad es 200.
    let busca_tablas = sc4 > sc3 + 200 && sc4.abs() <= 200;
    ok &= busca_tablas;
    println!(
        "  {}",
        if busca_tablas {
            "OK: prefiere la repeticion (tablas) en vez de seguir perdiendo"
        } else {
            "FALLO: no aprovecho la repeticion disponible"
        }
    );

    println!(
        "\n{}",
        if ok {
            "TODOS LOS TESTS DE REPETICION OK"
        } else {
            "HAY FALLOS EN LA DETECCION DE REPETICION"
        }
    );
}

fn run_see_tests() {
    let mut ok = true;
    let casos: Vec<(&str, &str, &str, i32)> = vec![
        (
            "Caso 1 (captura libre)",
            "k7/8/8/7p/8/8/8/K6R w - - 0 1",
            "h1h5",
            100,
        ),
        (
            "Caso 2 (1v1, mal cambio)",
            "k2r4/8/8/3p4/8/8/8/K2R4 w - - 0 1",
            "d1d5",
            100 - 500,
        ),
        (
            "Caso 3 (2 atacantes vs 1 defensor)",
            "k2r4/8/8/3p4/8/8/3R4/K2R4 w - - 0 1",
            "d1d5",
            100,
        ),
        (
            "Caso 4 (cxd5 con recaptura de peon)",
            "k7/8/4p3/3n4/2P5/1N6/8/K7 w - - 0 1",
            "c4d5",
            220,
        ),
        (
            "Caso 5 (al paso, sin recaptura)",
            "k7/8/8/3pP3/8/8/8/K7 w - d6 0 1",
            "e5d6",
            100,
        ),
        (
            "Caso 5b (al paso con recaptura)",
            "k7/2p5/8/3pP3/8/8/8/K7 w - d6 0 1",
            "e5d6",
            0,
        ),
        (
            "Caso 6 (FEN del bug, cxb4)",
            "r1b1k2r/ppqp1ppp/4p3/4n3/1b6/2PQBN2/P1P2PPP/R3KB1R w KQkq - 0 11",
            "c3b4",
            330,
        ),
    ];
    for (nombre, fen, uci, esperado) in &casos {
        let b = Board::from_fen(fen).unwrap();
        let mv = jugada_sintetica(&b, uci);
        let r = see::see(&b, &mv);
        let pass = r == *esperado;
        ok &= pass;
        println!(
            "{} {}: see={} esperado={}",
            if pass { "OK  " } else { "FALLO" },
            nombre,
            r,
            esperado
        );
    }
    println!(
        "{}",
        if ok {
            "TODOS LOS CASOS DE SEE OK"
        } else {
            "HAY FALLOS EN SEE"
        }
    );

    // Oraculo de fuerza bruta: posiciones al azar (self-play con jugadas
    // aleatorias desde la posicion inicial), comparando see() contra un
    // minimax real sobre TODAS las jugadas de captura posibles.
    println!("\nVerificando contra oraculo de fuerza bruta...");
    let mut rng_state: u64 = 0xC0FFEE1234567u64;
    let mut next_rand = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    let mut total_capturas = 0u64;
    let mut discrepancias = 0u64;
    for _partida in 0..5000 {
        let mut b = Board::startpos();
        let plies = 4 + (next_rand() % 20) as u32;
        let mut valida = true;
        for _ in 0..plies {
            let moves = movegen::generate_legal(&b);
            if moves.is_empty() {
                valida = false;
                break;
            }
            let mv = moves[(next_rand() as usize) % moves.len()];
            b = b.make_move(&mv);
        }
        if !valida {
            continue;
        }
        let capturas: Vec<Move> = movegen::generate_legal(&b)
            .into_iter()
            .filter(|m| m.is_capture())
            .collect();
        for mv in capturas {
            let rapido = see::see(&b, &mv);
            let oraculo = see::see_oracle(&b, &mv);
            total_capturas += 1;
            if rapido != oraculo {
                discrepancias += 1;
                println!(
                    "  DISCREPANCIA: fen='{}' jugada={} see={} oraculo={}",
                    b.to_fen(),
                    mv.to_uci(),
                    rapido,
                    oraculo
                );
            }
        }
    }
    println!(
        "{} capturas comparadas, {} discrepancias -- {}",
        total_capturas,
        discrepancias,
        if discrepancias == 0 {
            "SEE COINCIDE CON EL ORACULO"
        } else {
            "HAY DISCREPANCIAS, revisar"
        }
    );
}

/// Handle de una busqueda corriendo en su propio hilo (spawneada al recibir
/// "go") + la bandera compartida para pedirle que pare ("stop"). El hilo
/// devuelve el Searcher de un solo hilo si lo tomo prestado (para
/// recuperarlo y seguir usandolo en la siguiente jugada), o None si la
/// busqueda fue Lazy SMP (esos Searchers son desechables, no hay uno
/// persistente que recuperar -- solo la TT compartida, que vive aparte).
struct BusquedaActiva {
    handle: std::thread::JoinHandle<Option<Searcher>>,
    stop_flag: Arc<AtomicBool>,
}

/// Si hay una busqueda en curso, le pide que pare y espera a que termine
/// (casi instantaneo: el hilo de busqueda revisa la bandera desde el primer nodo y luego cada 256 nodos)
/// para recuperar el Searcher de un solo hilo, si corresponde. Se llama
/// defensivamente antes de procesar cualquier comando UCI que no sea
/// "isready" -- un GUI que cumple el protocolo siempre manda "stop" antes de
/// "position"/"go"/"ucinewgame" nuevos, pero esto evita un panic si alguno
/// no lo hace.
fn detener_y_recuperar(activa: &mut Option<BusquedaActiva>, searcher_slot: &mut Option<Searcher>) {
    if let Some(a) = activa.take() {
        a.stop_flag.store(true, Ordering::Relaxed);
        if let Ok(Some(s)) = a.handle.join() {
            *searcher_slot = Some(s);
        }
    }
}

/// UCI permite espacios tanto en el nombre como en el valor de una opcion.
/// En particular, BookPath, NNPath y SyzygyPath deben poder apuntar a rutas
/// normales de macOS que contengan espacios.
fn parse_setoption(partes: &[&str]) -> Option<(String, Option<String>)> {
    let inicio_nombre = partes.iter().position(|&p| p == "name")? + 1;
    if inicio_nombre >= partes.len() {
        return None;
    }
    let inicio_valor = partes
        .iter()
        .enumerate()
        .skip(inicio_nombre)
        .find_map(|(i, &p)| (p == "value").then_some(i));
    let fin_nombre = inicio_valor.unwrap_or(partes.len());
    let nombre = partes[inicio_nombre..fin_nombre].join(" ");
    if nombre.is_empty() {
        return None;
    }
    let valor = inicio_valor.and_then(|i| {
        let valor = partes[i + 1..].join(" ");
        (!valor.is_empty()).then_some(valor)
    });
    Some((nombre, valor))
}

fn calcular_movetime_reloj(mio_i: i64, inc_i: i64, movestogo: i64, move_overhead_ms: u64) -> u64 {
    let mio = mio_i.max(0) as u64;
    let inc = inc_i.max(0) as u64;
    let disponible = mio.saturating_sub(move_overhead_ms);
    const UMBRAL_CORRESPONDENCIA_MS: u64 = 40 * 60 * 1000;
    let techo = if mio > UMBRAL_CORRESPONDENCIA_MS {
        3_500
    } else {
        20_000
    };
    let base = disponible / movestogo.max(1) as u64;
    let objetivo = base.saturating_add(inc.saturating_mul(8) / 10);
    objetivo.max(1).min(techo).min(disponible.max(1))
}

/// Techo DURO del reloj para la jugada actual ("maximo" de timeman.cpp).
/// El presupuesto objetivo (`calcular_movetime_reloj`) puede estirarse por
/// los factores reactivos de la busqueda, pero NUNCA mas alla de esto.
///
/// DOS limites, y manda el mas chico:
///  - 80% del tiempo realmente disponible en el reloj (ya descontado el
///    Move Overhead): la red de seguridad absoluta.
///  - 4x el presupuesto objetivo: sin esto, una sola iteracion profunda
///    arrancada justo antes del corte blando podia comerse casi todo el
///    reloj de golpe (se midio una partida perdida por tiempo asi).
/// Perder por tiempo es peor que cualquier Elo que se gane.
fn calcular_movetime_maximo(mio_i: i64, move_overhead_ms: u64, objetivo_ms: u64) -> u64 {
    let mio = mio_i.max(0) as u64;
    let disponible = mio.saturating_sub(move_overhead_ms);
    let por_reloj = disponible.saturating_mul(80) / 100;
    let por_objetivo = objetivo_ms.saturating_mul(4);
    por_reloj.min(por_objetivo).max(1)
}

pub fn uci_loop() {
    let stdin = io::stdin();
    let mut board = Board::from_fen(STARTPOS).unwrap();
    let mut tt_mb: usize = 64;
    let mut move_overhead_ms: u64 = 75;
    // Lazy SMP y el hilo principal deben compartir una sola TT. Antes se
    // construía aquí una TT propia y, unas líneas después, otra para SMP;
    // con Hash=512 eso duplicaba la reserva de memoria desde el arranque.
    let modo_lmr_inicial = std::env::var("MIMOTOR_LMR").as_deref() != Ok("0");
    let mut game_history: Vec<u64> = Vec::new();
    // MIMOTOR_HILOS sigue funcionando como default de conveniencia para
    // pruebas locales, pero un tester UCI (CCRL y similares) SOLO configura
    // motores por "setoption", nunca por variables de entorno -- por eso
    // "Threads" tambien es una opcion UCI real (ver abajo) que sobreescribe
    // este valor inicial.
    let mut n_hilos: usize = std::env::var("MIMOTOR_HILOS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .clamp(1, 16);
    // Por defecto la quiescence conserva NNUE. Esta opción permite medir la
    // variante ultrabullet sin depender de variables de entorno ocultas.
    let mut qsearch_nnue = true;
    // La NNUE viene embebida en el binario y se activa por defecto: sin esto
    // un tester UCI que no configure NNUEPath jugaria con evaluacion clasica
    // pura, que es mucho mas debil.
    match neural::cargar_embebida() {
        Ok(checksum) => {
            neural::set_activa(true);
            eprintln!("info string NNUE embebida activa checksum {:016x}", checksum);
        }
        Err(e) => eprintln!("info string no se pudo cargar la NNUE embebida: {}", e),
    }
    let mut nnue_classical_depth = 0i32;
    // Para `go movetime <=25` con un único hilo, evita crear un hilo nativo
    // que solo vive unos milisegundos. Es experimental y apagado por defecto;
    // el modo normal conserva stop asíncrono sin ninguna variación.
    let mut sync_ultrabullet = false;
    let mut multipv: usize = 1;
    if let Ok(p) = std::env::var("MIMOTOR_PERSONALIDAD")
        && let Some(pers) = eval::personalidad_desde_texto(&p)
    {
        eval::set_personalidad(pers);
    }
    if let Ok(path) = std::env::var("MIMOTOR_SYZYGY_PATH") {
        match syzygy::init(&path) {
            Ok(max) => eprintln!("info string tablas Syzygy cargadas ({} piezas max)", max),
            Err(e) => eprintln!("info string error cargando tablas Syzygy: {}", e),
        }
    }
    if let Ok(path) = std::env::var("MIMOTOR_BOOK_PATH") {
        match polyglot::init(&path) {
            Ok(n) => eprintln!("info string libro de aperturas cargado ({} entradas)", n),
            Err(e) => eprintln!("info string error cargando libro de aperturas: {}", e),
        }
    }
    let usar_libro_inicial: bool = std::env::var("MIMOTOR_SIN_LIBRO").as_deref() != Ok("1");
    polyglot::set_activo(usar_libro_inicial);
    // El camino de un hilo usa una TT local sin Mutex. Solo Lazy SMP necesita
    // una TT compartida entre hilos; reservarla tambien en Threads=1 añade un
    // lock/unlock en cada probe sin aportar sincronizacion util.
    let (mut smp_tt, mut smp_tt_mask) = if n_hilos > 1 {
        search::construir_tt(tt_mb)
    } else {
        (Arc::new(Vec::new()), 0)
    };
    // Generacion de aging de la TT compartida: persiste entre jugadas JUNTO a
    // smp_tt. El bug original: cada Searcher nuevo de buscar_lazy_smp
    // arrancaba con tt_generation 0 y toda busqueda SMP quedaba en
    // generacion 1, dejando el aging muerto aunque la TT compartida si
    // persista. Avanza una vez por "go" dentro de buscar_lazy_smp; se
    // resetea cuando la TT se reconstruye (ucinewgame / cambio de Hash o de
    // evaluacion).
    let smp_generacion = Arc::new(AtomicU8::new(0));
    let mut searcher_slot: Option<Searcher> = Some(if n_hilos > 1 {
        Searcher::new_con_tt_compartida(Arc::clone(&smp_tt), smp_tt_mask, modo_lmr_inicial)
    } else {
        Searcher::new(tt_mb)
    });
    searcher_slot
        .as_mut()
        .expect("searcher inicial")
        .set_qsearch_nnue(qsearch_nnue);
    searcher_slot
        .as_mut()
        .expect("searcher inicial")
        .set_nnue_classical_depth(nnue_classical_depth);
    let mut activa: Option<BusquedaActiva> = None;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let partes: Vec<&str> = line.split_whitespace().collect();

        // "stop" se maneja aparte. Todo lo demas primero se asegura de que no
        // haya una busqueda activa antes de tocar board/searcher.
        if partes[0] == "stop" {
            detener_y_recuperar(&mut activa, &mut searcher_slot);
            continue;
        }
        // "isready" NO debe interrumpir la busqueda. El texto del protocolo
        // UCI es explicito: "this command ... can be sent also when the engine
        // is calculating in which case the engine should also immediately
        // answer with readyok WITHOUT stopping the search".
        //
        // BUG CORREGIDO: antes esta rama llamaba a detener_y_recuperar, o sea
        // que cualquier GUI/arbitro que mande "isready" mientras el motor
        // piensa (varios lo hacen como chequeo de vida, y esta permitido)
        // ABORTABA la busqueda en curso. El hilo de busqueda contestaba
        // "bestmove" en el acto, con la profundidad que hubiera alcanzado
        // hasta ese milisegundo -- en la practica, jugar de inmediato durante
        // toda la partida. No se nota probando con python-chess ni con
        // cutechess (no mandan isready mientras se piensa), pero si con otros
        // arbitros, y es candidato serio a explicar un resultado de torneo
        // externo muy por debajo de la fuerza medida en casa.
        if partes[0] == "isready" {
            println!("readyok");
            io::stdout().flush().ok();
            continue;
        }
        detener_y_recuperar(&mut activa, &mut searcher_slot);

        match partes[0] {
            "uci" => {
                println!("id name Mittens");
                println!("id author Tavito y Claude");
                println!("option name Hash type spin default 64 min 1 max 1024");
                println!("option name Clear Hash type button");
                println!("option name Move Overhead type spin default 75 min 0 max 5000");
                println!(
                    "option name Threads type spin default {} min 1 max 16",
                    n_hilos
                );
                println!("option name Personalidad type combo default tal var tal var universal");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name BookPath type string default <empty>");
                println!("option name OwnBook type check default true");
                println!("option name UseNNUE type check default true");
                println!("option name NNUEPath type string default <empty>");
                println!("option name UseNN type check default false");
                println!("option name NNPath type string default <empty>");
                println!("option name QSearchNNUE type check default true");
                println!("option name NNUEClassicalDepth type spin default 0 min 0 max 4");
                println!("option name SyncUltraBullet type check default false");
                println!("option name MultiPV type spin default 1 min 1 max 10");
                println!("uciok");
                io::stdout().flush().ok();
            }
            "setoption" => {
                let mut reiniciar_tt = false;
                if let Some((nombre, valor)) = parse_setoption(&partes) {
                    let valor = valor.as_deref();
                    if nombre.eq_ignore_ascii_case("personalidad") {
                        if let Some(valor) = valor {
                            if let Some(pers) = eval::personalidad_desde_texto(valor) {
                                eval::set_personalidad(pers);
                                reiniciar_tt = true;
                            } else {
                                println!("info string valor de Personalidad invalido: {}", valor);
                            }
                        }
                    } else if nombre.eq_ignore_ascii_case("hash") {
                        if let Some(mb) = valor.and_then(|v| v.parse::<usize>().ok()) {
                            tt_mb = mb.clamp(1, 1024);
                            reiniciar_tt = true;
                        }
                    } else if nombre.eq_ignore_ascii_case("clear hash") {
                        reiniciar_tt = true;
                    } else if nombre.eq_ignore_ascii_case("move overhead") {
                        if let Some(ms) = valor.and_then(|v| v.parse::<u64>().ok()) {
                            move_overhead_ms = ms.min(5000);
                        }
                    } else if nombre.eq_ignore_ascii_case("threads") {
                        if let Some(n) = valor.and_then(|v| v.parse::<usize>().ok()) {
                            n_hilos = n.clamp(1, 16);
                            // Cambiar entre local y Lazy SMP requiere cambiar
                            // de backend de TT; descartar el contenido evita
                            // mezclar una tabla de otro modo de busqueda.
                            reiniciar_tt = true;
                        }
                    } else if nombre.eq_ignore_ascii_case("syzygypath") {
                        if let Some(path) = valor {
                            match syzygy::init(path) {
                                Ok(max) => println!(
                                    "info string tablas Syzygy cargadas ({} piezas max)",
                                    max
                                ),
                                Err(e) => {
                                    println!("info string error cargando tablas Syzygy: {}", e)
                                }
                            }
                        }
                    } else if nombre.eq_ignore_ascii_case("bookpath") {
                        if let Some(path) = valor {
                            match polyglot::init(path) {
                                Ok(n) => println!(
                                    "info string libro de aperturas cargado ({} entradas)",
                                    n
                                ),
                                Err(e) => {
                                    println!("info string error cargando libro de aperturas: {}", e)
                                }
                            }
                        }
                    } else if nombre.eq_ignore_ascii_case("ownbook") {
                        if let Some(v) = valor {
                            polyglot::set_activo(v.eq_ignore_ascii_case("true"));
                        }
                    } else if nombre.eq_ignore_ascii_case("qsearchnnue") {
                        if let Some(v) = valor {
                            qsearch_nnue = v.eq_ignore_ascii_case("true");
                            // Los valores de quiescence llegan a bounds de
                            // padres guardados en TT: no mezclar ambos modos.
                            reiniciar_tt = true;
                        }
                    } else if nombre.eq_ignore_ascii_case("nnueclassicaldepth") {
                        if let Some(v) = valor.and_then(|v| v.parse::<i32>().ok()) {
                            nnue_classical_depth = v.clamp(0, 4);
                            reiniciar_tt = true;
                        }
                    } else if nombre.eq_ignore_ascii_case("syncultrabullet") {
                        if let Some(v) = valor {
                            sync_ultrabullet = v.eq_ignore_ascii_case("true");
                        }
                    } else if nombre.eq_ignore_ascii_case("multipv") {
                        if let Some(n) = valor.and_then(|v| v.parse::<usize>().ok()) {
                            multipv = n.clamp(1, 10);
                        }
                    } else if nombre.eq_ignore_ascii_case("nnuepath")
                        || nombre.eq_ignore_ascii_case("nnpath")
                    {
                        if let Some(path) = valor {
                            match neural::cargar_detallado(path) {
                                Ok(checksum) => {
                                    println!(
                                        "info string NNUE cargada desde {} checksum {:016x}",
                                        path, checksum
                                    );
                                    reiniciar_tt = true;
                                }
                                Err(e) => println!(
                                    "info string error cargando NNUE desde {}: {} (se conserva la red anterior)",
                                    path, e
                                ),
                            }
                        }
                    } else if (nombre.eq_ignore_ascii_case("usennue")
                        || nombre.eq_ignore_ascii_case("usenn"))
                        && let Some(v) = valor
                    {
                        let activar = v.eq_ignore_ascii_case("true");
                        if activar && !neural::hay_red_cargada() {
                            println!(
                                "info string UseNNUE pendiente: se activara al cargar NNUEPath"
                            );
                        }
                        // set_activa conserva la solicitud aunque los pesos
                        // lleguen despues, evitando depender del orden UCI.
                        neural::set_activa(activar);
                        reiniciar_tt = true;
                    }
                }
                if reiniciar_tt {
                    // Los scores de TT dependen de la evaluacion activa. Al
                    // cambiar personalidad/NNUE, o al pedir Clear Hash/Hash,
                    // se reconstruyen ambas tablas para no mezclar resultados.
                    // Liberar la tabla anterior antes de reservar la nueva:
                    // evita un pico de dos tablas completas durante Hash.
                    drop(searcher_slot.take());
                    let old_tt = std::mem::replace(&mut smp_tt, Arc::new(Vec::new()));
                    drop(old_tt);
                    // TT nueva => contador de aging desde cero.
                    smp_generacion.store(0, Ordering::Relaxed);
                    let modo_lmr = std::env::var("MIMOTOR_LMR").as_deref() != Ok("0");
                    if n_hilos > 1 {
                        let (nueva_tt, nueva_mask) = search::construir_tt(tt_mb);
                        searcher_slot = Some(Searcher::new_con_tt_compartida(
                            Arc::clone(&nueva_tt),
                            nueva_mask,
                            modo_lmr,
                        ));
                        smp_tt = nueva_tt;
                        smp_tt_mask = nueva_mask;
                    } else {
                        searcher_slot = Some(Searcher::new(tt_mb));
                        smp_tt = Arc::new(Vec::new());
                        smp_tt_mask = 0;
                    }
                    searcher_slot
                        .as_mut()
                        .expect("searcher reconstruido")
                        .set_qsearch_nnue(qsearch_nnue);
                    searcher_slot
                        .as_mut()
                        .expect("searcher reconstruido")
                        .set_nnue_classical_depth(nnue_classical_depth);
                }
                io::stdout().flush().ok();
            }
            "ucinewgame" => {
                board = Board::from_fen(STARTPOS).unwrap();
                game_history.clear();
                drop(searcher_slot.take());
                let old_tt = std::mem::replace(&mut smp_tt, Arc::new(Vec::new()));
                drop(old_tt);
                // TT nueva => contador de aging desde cero.
                smp_generacion.store(0, Ordering::Relaxed);
                let modo_lmr = std::env::var("MIMOTOR_LMR").as_deref() != Ok("0");
                if n_hilos > 1 {
                    let (nueva_tt, nueva_mask) = search::construir_tt(tt_mb);
                    searcher_slot = Some(Searcher::new_con_tt_compartida(
                        Arc::clone(&nueva_tt),
                        nueva_mask,
                        modo_lmr,
                    ));
                    smp_tt = nueva_tt;
                    smp_tt_mask = nueva_mask;
                } else {
                    searcher_slot = Some(Searcher::new(tt_mb));
                    smp_tt = Arc::new(Vec::new());
                    smp_tt_mask = 0;
                }
                searcher_slot
                    .as_mut()
                    .expect("searcher de nueva partida")
                    .set_qsearch_nnue(qsearch_nnue);
                searcher_slot
                    .as_mut()
                    .expect("searcher de nueva partida")
                    .set_nnue_classical_depth(nnue_classical_depth);
            }
            "position" => {
                // Snapshot del estado PREVIO (tablero + historial de repeticion):
                // si el stream de jugadas resulta inconsistente, se restaura este
                // par completo como "ultima posicion valida". Sin el snapshot, un
                // rechazo dejaria el tablero NUEVO (startpos/fen sin moves) con el
                // historial de la PARTIDA ANTERIOR -- un estado mixto que romperia
                // la deteccion de repeticion en el siguiente "go".
                let board_previo = board;
                let hist_previo = game_history.clone();
                let mut idx = 1;
                if partes.get(1) == Some(&"startpos") {
                    board = Board::from_fen(STARTPOS).unwrap();
                    idx = 2;
                } else if partes.get(1) == Some(&"fen") {
                    let moves_pos = partes
                        .iter()
                        .position(|&p| p == "moves")
                        .unwrap_or(partes.len());
                    let fen = partes[2..moves_pos].join(" ");
                    board = match Board::from_fen(&fen) {
                        Ok(b) => b,
                        Err(e) => {
                            println!("info string error de FEN: {}", e);
                            continue;
                        }
                    };
                    idx = moves_pos;
                } else {
                    println!("info string error de position: se esperaba startpos o fen");
                    continue;
                }
                // Historial de claves Zobrist de la partida real (para deteccion
                // de repeticion) -- clave de CADA posicion ancestro, sin incluir
                // la posicion final/actual (esa la maneja negamax directamente).
                // ESTRICTO: si alguna jugada del stream es ilegal, se rechaza la
                // posicion completa y se RESTAURA la ultima posicion valida
                // (tablero + historial; aplicar_moves_position imprime el error)
                // -- nunca divergir en silencio ni dejar estado mixto.
                if partes.get(idx) == Some(&"moves") {
                    let moves_uci: Vec<&str> = partes[idx + 1..].to_vec();
                    if !aplicar_moves_position(&mut board, &mut game_history, &moves_uci) {
                        board = board_previo;
                        game_history = hist_previo;
                        continue;
                    }
                } else {
                    game_history.clear();
                }
            }
            "go" => {
                let infinito = partes.contains(&"infinite");
                if let Some(i) = partes.iter().position(|&p| p == "depth") {
                    let depth: i32 = partes.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(6);
                    let mut s = searcher_slot.take().unwrap();
                    s.set_game_history(game_history.clone());
                    let stop_flag = Arc::new(AtomicBool::new(false));
                    s.set_external_stop(Some(Arc::clone(&stop_flag)));
                    let board_copy = board;
                    let handle = std::thread::spawn(move || {
                        // Sin movetime (None): search_time no aplica deadline y
                        // se detiene sola al completar `depth` (mismo bucle
                        // for d in 1..=max_depth que search_fixed_depth), pero
                        // a diferencia de esa reutiliza el reporte "info depth
                        // ... nodes ... time ... nps ..." por iteracion que ya
                        // usan movetime/infinite -- antes "go depth N" solo
                        // imprimia "info score cp X" suelto, sin depth/nodes/
                        // time/nps, una desviacion real del protocolo UCI que
                        // rompe el parseo de GUIs y testers (Cutechess, Arena,
                        // herramientas de CCRL).
                        let (mv, _sc, _) =
                            s.search_time(&board_copy, None, depth, |depth, score, nodes, ms, pv| {
                                let nps = nodes.saturating_mul(1000) / ms.max(1);
                                let pv_txt =
                                    pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ");
                                println!(
                                    "info depth {} score {} nodes {} time {} nps {} pv {}",
                                    depth, formatear_score_uci(score), nodes, ms, nps, pv_txt
                                );
                                io::stdout().flush().ok();
                            });
                        println!(
                            "bestmove {}",
                            mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string())
                        );
                        io::stdout().flush().ok();
                        Some(s)
                    });
                    activa = Some(BusquedaActiva { handle, stop_flag });
                    continue;
                }

                // movetime explicito, o wtime/btime(+winc/binc/movestogo), o
                // "go infinite" (sin limite propio, corta solo con "stop").
                let mut movetime: Option<u64> = None;
                // Solo se pone cuando el `go` trae reloj (wtime/btime); con
                // `go movetime` o `go depth` no hay dos presupuestos.
                let mut movetime_maximo: Option<u64> = None;
                if let Some(i) = partes.iter().position(|&p| p == "movetime") {
                    movetime = partes.get(i + 1).and_then(|s| s.parse().ok());
                } else if !infinito {
                    if let Some(i) = partes.iter().position(|&p| p == "wtime") {
                        let wtime: i64 = partes
                            .get(i + 1)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(10000);
                        let btime_i = partes.iter().position(|&p| p == "btime");
                        let btime: i64 = btime_i
                            .and_then(|j| partes.get(j + 1))
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(10000);
                        let winc: i64 = partes
                            .iter()
                            .position(|&p| p == "winc")
                            .and_then(|j| partes.get(j + 1))
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                        let binc: i64 = partes
                            .iter()
                            .position(|&p| p == "binc")
                            .and_then(|j| partes.get(j + 1))
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                        let movestogo: i64 = partes
                            .iter()
                            .position(|&p| p == "movestogo")
                            .and_then(|j| partes.get(j + 1))
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(30);
                        let (mio_i, inc_i) = if board.turn == types::Color::White {
                            (wtime, winc)
                        } else {
                            (btime, binc)
                        };
                        // Reparto con margen real: nunca inventa un minimo
                        // mayor al reloj y siempre reserva Move Overhead.
                        movetime = Some(calcular_movetime_reloj(
                            mio_i,
                            inc_i,
                            movestogo,
                            move_overhead_ms,
                        ));
                        // Techo duro para el time management de dos
                        // presupuestos (ver Searcher::tiempo_maximo_ms).
                        movetime_maximo = Some(calcular_movetime_maximo(
                            mio_i,
                            move_overhead_ms,
                            movetime.unwrap_or(0),
                        ));
                    } else {
                        movetime = Some(2000); // "go" sin ningun parametro de tiempo: default razonable
                    }
                }

                // Publica el techo duro del reloj para ESTA busqueda (lo
                // leen todos los caminos, incluidos los hilos de Lazy SMP).
                // Se fija siempre, tambien a None, para que un `go movetime`
                // no herede el techo del `go` anterior.
                search::fijar_tiempo_maximo(movetime_maximo);

                // "searchmoves e2e4 d2d4 ...": restringe la busqueda en la
                // RAIZ a solo estas jugadas -- pensado para repartir el
                // arbol entre varias maquinas (cada una recibe un
                // subconjunto disjunto de jugadas raiz).
                let searchmoves_filtro: Option<Vec<Move>> = partes
                    .iter()
                    .position(|&p| p == "searchmoves")
                    .map(|i| {
                        partes[i + 1..]
                            .iter()
                            .filter_map(|s| parse_uci_move(&board, s))
                            .collect()
                    });

                // JUGADA FORZADA (single reply): si en la raiz hay una sola
                // jugada legal, no hay nada que decidir -- pensar aunque sea
                // un milisegundo es tiempo tirado. Se responde de inmediato y
                // el reloj queda entero para las jugadas que si importan. En
                // partidas rapidas con jaques forzados esto ahorra segundos.
                // Solo aplica a busquedas con reloj (no "infinite", no
                // "searchmoves", no MultiPV: ahi el usuario quiere analisis).
                if !infinito
                    && movetime.is_some()
                    && multipv <= 1
                    && searchmoves_filtro.is_none()
                {
                    let legales = movegen::generate_legal(&board);
                    if legales.len() == 1 {
                        println!("info depth 1 score cp 0 nodes 1 time 0 pv {}", legales[0].to_uci());
                        println!("bestmove {}", legales[0].to_uci());
                        io::stdout().flush().ok();
                        continue;
                    }
                }

                let stop_flag = Arc::new(AtomicBool::new(false));
                let board_copy = board;
                let hist_copy = game_history.clone();

                // MultiPV: Mittens no tiene un algoritmo de multi-linea real
                // (compartir el ply de raiz entre N lineas simultaneas). En
                // su lugar se simula con N busquedas SECUENCIALES completas,
                // cada una excluyendo (via root_moves_filtro) las jugadas de
                // raiz ya encontradas en las anteriores -- el mismo truco que
                // "searchmoves" ya permitia hacer desde afuera, ahora nativo
                // y repartiendo el movetime total entre las N pasadas. Es
                // sincrono (no soporta "stop" a mitad de camino ni "ponder")
                // porque su uso previsto es analisis con movetime fijo, no
                // partidas en vivo -- el camino normal (MultiPV=1, el
                // default) no se toca y sigue siendo totalmente asincrono.
                if multipv > 1 {
                    let modo_lmr = searcher_slot.as_ref().unwrap().modo_lmr;
                    let universo: Vec<Move> = match &searchmoves_filtro {
                        Some(f) => f.clone(),
                        None => movegen::generate_legal(&board_copy).to_vec(),
                    };
                    let tiempo_por_linea = movetime.map(|t| (t / multipv as u64).max(1));
                    let mut excluidas: Vec<Move> = Vec::new();
                    // (score, profundidad, nodos, pv)
                    let mut lineas: Vec<(i32, i32, u64, Vec<Move>)> = Vec::new();
                    for _ in 0..multipv {
                        let candidatas: Vec<Move> = universo
                            .iter()
                            .copied()
                            .filter(|m| !excluidas.contains(m))
                            .collect();
                        if candidatas.is_empty() {
                            break;
                        }
                        let (mv, sc, nodos, prof, pv) = if n_hilos > 1 {
                            let (mv, sc, nodos, resultados) = search::buscar_lazy_smp(
                                &board_copy,
                                tiempo_por_linea,
                                64,
                                n_hilos,
                                &smp_tt,
                                &smp_generacion,
                                smp_tt_mask,
                                modo_lmr,
                                qsearch_nnue,
                                nnue_classical_depth,
                                &hist_copy,
                                Arc::new(AtomicBool::new(false)),
                                Some(candidatas),
                            );
                            let prof = resultados.iter().map(|r| r.profundidad).max().unwrap_or(0);
                            // La TT compartida ya tiene el resultado: se
                            // reconstruye la PV envolviendola en un Searcher
                            // liviano solo para leerla (no busca nada).
                            let lector = Searcher::new_con_tt_compartida(
                                Arc::clone(&smp_tt),
                                smp_tt_mask,
                                modo_lmr,
                            );
                            let pv = lector.extraer_pv(&board_copy, 10);
                            (mv, sc, nodos, prof, pv)
                        } else {
                            let mut s = searcher_slot.take().expect("searcher multipv");
                            s.set_game_history(hist_copy.clone());
                            s.root_moves_filtro = Some(candidatas);
                            let (mv, sc, prof) =
                                s.search_time(&board_copy, tiempo_por_linea, 64, |_, _, _, _, _| {});
                            let nodos = s.nodes;
                            let pv = s.extraer_pv(&board_copy, 10);
                            searcher_slot = Some(s);
                            (mv, sc, nodos, prof, pv)
                        };
                        let Some(mv) = mv else { break };
                        excluidas.push(mv);
                        let pv = if pv.first() == Some(&mv) { pv } else { vec![mv] };
                        lineas.push((sc, prof, nodos, pv));
                    }
                    lineas.sort_by(|a, b| b.0.cmp(&a.0));
                    for (i, (sc, prof, nodos, pv)) in lineas.iter().enumerate() {
                        let pv_txt = pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ");
                        println!(
                            "info depth {} multipv {} score {} nodes {} pv {}",
                            prof,
                            i + 1,
                            formatear_score_uci(*sc),
                            nodos,
                            pv_txt
                        );
                    }
                    io::stdout().flush().ok();
                    let bestmove = lineas.first().and_then(|(_, _, _, pv)| pv.first().copied());
                    println!(
                        "bestmove {}",
                        bestmove.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string())
                    );
                    io::stdout().flush().ok();
                    continue;
                }

                // A relojes extremadamente cortos el overhead de crear y
                // despertar el hilo UCI pesa tanto como una fracción de la
                // búsqueda. Como `go movetime` ya tiene deadline duro y este
                // camino es exclusivo de <=25ms/Threads=1, responder de modo
                // síncrono gana tiempo real sin dejar una búsqueda infinita
                // imposible de detener.
                if n_hilos == 1
                    && sync_ultrabullet
                    && !partes.contains(&"ponder")
                    && movetime.is_some_and(|ms| ms <= 25)
                {
                    let mut s = searcher_slot.take().expect("searcher ultrabullet");
                    s.set_game_history(hist_copy);
                    s.set_external_stop(Some(Arc::clone(&stop_flag)));
                    s.root_moves_filtro = searchmoves_filtro.clone();
                    let (mv, _sc, _) =
                        s.search_time(&board_copy, movetime, 64, |depth, score, nodes, ms, pv| {
                            let pv_txt =
                                pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ");
                            println!(
                                "info depth {} score {} nodes {} time {} pv {}",
                                depth, formatear_score_uci(score), nodes, ms, pv_txt
                            );
                            io::stdout().flush().ok();
                        });
                    println!(
                        "bestmove {}",
                        mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string())
                    );
                    io::stdout().flush().ok();
                    searcher_slot = Some(s);
                    continue;
                }

                if n_hilos > 1 {
                    let modo_lmr = searcher_slot.as_ref().unwrap().modo_lmr;
                    let qsearch_nnue = searcher_slot.as_ref().unwrap().qsearch_nnue;
                    let nnue_classical_depth = searcher_slot.as_ref().unwrap().nnue_classical_depth;
                    let tt = Arc::clone(&smp_tt);
                    let mask = smp_tt_mask;
                    let generacion = Arc::clone(&smp_generacion);
                    let flag = Arc::clone(&stop_flag);
                    let filtro = searchmoves_filtro.clone();
                    let handle = std::thread::spawn(move || {
                        let t0 = std::time::Instant::now();
                        let (mv, sc, nodos, resultados) = search::buscar_lazy_smp(
                            &board_copy,
                            movetime,
                            64,
                            n_hilos,
                            &tt,
                            &generacion,
                            mask,
                            modo_lmr,
                            qsearch_nnue,
                            nnue_classical_depth,
                            &hist_copy,
                            flag,
                            filtro,
                        );
                        let profundidad =
                            resultados.iter().map(|r| r.profundidad).max().unwrap_or(0);
                        let ms = t0.elapsed().as_millis().max(1) as u64;
                        let nps = nodos.saturating_mul(1000) / ms;
                        // La TT compartida ya tiene el resultado de todos los
                        // hilos: se reconstruye la PV envolviendola en un
                        // Searcher liviano solo para leerla (no busca nada),
                        // igual que en el camino MultiPV. Se usa el mismo
                        // tablero raiz para todos los hilos, asi que la
                        // caminata de la TT es coherente con el hilo 0.
                        let lector =
                            Searcher::new_con_tt_compartida(Arc::clone(&tt), mask, modo_lmr);
                        let pv = lector.extraer_pv(&board_copy, profundidad.max(1) as usize);
                        let pv_txt = pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ");
                        println!(
                            "info depth {} score {} nodes {} time {} nps {} pv {}",
                            profundidad, formatear_score_uci(sc), nodos, ms, nps, pv_txt
                        );
                        println!(
                            "bestmove {}",
                            mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string())
                        );
                        io::stdout().flush().ok();
                        None
                    });
                    activa = Some(BusquedaActiva { handle, stop_flag });
                } else {
                    let mut s = searcher_slot.take().unwrap();
                    s.set_game_history(hist_copy);
                    s.set_external_stop(Some(Arc::clone(&stop_flag)));
                    s.root_moves_filtro = searchmoves_filtro.clone();
                    let handle = std::thread::spawn(move || {
                        let (mv, _sc, _) =
                            s.search_time(&board_copy, movetime, 64, |depth, score, nodes, ms, pv| {
                                let pv_txt =
                                    pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ");
                                println!(
                                    "info depth {} score {} nodes {} time {} pv {}",
                                    depth, formatear_score_uci(score), nodes, ms, pv_txt
                                );
                                io::stdout().flush().ok();
                            });
                        println!(
                            "bestmove {}",
                            mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string())
                        );
                        io::stdout().flush().ok();
                        Some(s)
                    });
                    activa = Some(BusquedaActiva { handle, stop_flag });
                }
            }
            "bench" => {
                // Igual que el "bench" de linea de comandos (`mimotor-tal-rust
                // bench [depth]`), pero disponible tambien como comando UCI
                // en vivo -- convencion de Stockfish y la mayoria de motores,
                // que herramientas de testeo (Cutechess, OpenBench, scripts
                // de CCRL) a veces mandan directo por stdin en vez de
                // relanzar el proceso con un argumento.
                detener_y_recuperar(&mut activa, &mut searcher_slot);
                let depth: i32 = partes.get(1).and_then(|s| s.parse().ok()).unwrap_or(13);
                run_bench(depth);
            }
            "quit" => {
                detener_y_recuperar(&mut activa, &mut searcher_slot);
                break;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{calcular_movetime_reloj, parse_setoption, parse_uci_move};
    use crate::board::Board;

    #[test]
    fn setoption_conserva_ruta_con_espacios() {
        let partes = [
            "setoption",
            "name",
            "NNPath",
            "value",
            "/Users/Tavito/Mi",
            "Motor/red.bin",
        ];
        assert_eq!(
            parse_setoption(&partes),
            Some((
                "NNPath".to_string(),
                Some("/Users/Tavito/Mi Motor/red.bin".to_string())
            ))
        );
    }
    #[test]
    fn reloj_nunca_supera_lo_disponible() {
        assert_eq!(calcular_movetime_reloj(20, 0, 30, 75), 1);
        assert!(calcular_movetime_reloj(1_000, 0, 1, 75) <= 925);
        assert!(calcular_movetime_reloj(10_000, 100, 30, 75) <= 9_925);
    }

    #[test]
    fn correspondencia_conserva_techo() {
        assert_eq!(calcular_movetime_reloj(86_400_000, 0, 30, 75), 3_500);
    }

    #[test]
    fn parse_uci_rechaza_sufijos_y_promociones_invalidas() {
        let b = Board::startpos();
        assert!(parse_uci_move(&b, "e2e4junk").is_none());
        assert!(parse_uci_move(&b, "e2e4x").is_none());
        assert!(parse_uci_move(&b, "e2e4").is_some());
    }
}

/// Punto de entrada de la linea de comandos. Vive en la LIBRERIA, no en el
/// binario: `src/main.rs` es solo un envoltorio de una linea que llama aca.
///
/// Historia: hasta esta version, `src/main.rs` y `src/lib.rs` eran dos copias
/// completas del mismo codigo (~1800 lineas cada una) porque la libreria no
/// exponia un `rlib` que el binario pudiera enlazar. Cada arreglo habia que
/// aplicarlo DOS veces y ya habian divergido de hecho (la copia de lib.rs se
/// habia quedado sin MultiPV). Ahora hay una sola copia.
pub fn run_cli() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "perft" => {
                run_perft_suite();
                return;
            }
            "divide" if args.len() > 3 => {
                let depth: u32 = args[2].parse().expect("profundidad invalida");
                let fen = args[3..].join(" ");
                run_divide(&fen, depth);
                return;
            }
            "bench" => {
                let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
                run_bench(depth);
                return;
            }
            "matetest" => {
                run_mate_tests();
                return;
            }
            "aperturatest" => {
                run_prueba_apertura();
                return;
            }
            "cxb4test" => {
                run_cxb4_bug();
                return;
            }
            "seetest" => {
                run_see_tests();
                return;
            }
            "endgametest" => {
                run_endgame_tests();
                return;
            }
            "repetitiontest" => {
                run_repetition_tests();
                return;
            }
            "lmrdiag" => {
                let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
                run_lmr_diagnostico(depth);
                return;
            }
            "singulartest" => {
                let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
                run_singular_diagnostico(depth);
                return;
            }
            "simple" if args.len() >= 3 => {
                let (movetime, fen_inicio) = match args.get(2).and_then(|s| s.parse::<u64>().ok()) {
                    Some(ms) if args.len() >= 4 => (ms, 3usize),
                    _ => (2000u64, 2usize),
                };
                let fen = args[fen_inicio..].join(" ");
                run_simple(&fen, movetime);
                return;
            }
            "smpbench" => {
                let movetime: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
                run_smp_bench(movetime);
                return;
            }
            _ => {}
        }
    }
    uci_loop();
}
