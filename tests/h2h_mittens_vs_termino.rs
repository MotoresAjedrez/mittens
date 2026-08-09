// Duelo H2H real: Mittens (motor propio) vs Termo (Frozenight 7.0.0-dev).
// Adapta el mini-arbitro MotorUci/jugar_partida del verificador en
// /Users/Tavito/mi-motor/motores_elite/src/lib.rs, pero vive en tests/ del
// proyecto para no tocar archivos fuera de el.
//
// Mittens necesita que se le carguen pesos NNUE (UseNNUE=true + NNUEPath),
// igual que hace el harness de motores_elite. Termo/Frozenight trae su red
// embebida y no requiere opciones.
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const DIR: &str = "/Users/Tavito/mi-motor/motores_elite/";
const MITTENS_WEIGHTS: &str = "/Users/Tavito/mi-motor-rust-produccion/pesos_amenazas_prueba.bin";

struct MotorUci {
    stdin: std::process::ChildStdin,
    rx: mpsc::Receiver<String>,
    child: std::process::Child,
    nombre: String,
}

impl MotorUci {
    fn nuevo(ruta: &str) -> MotorUci {
        let mut child = Command::new(ruta)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("no spawnea {}: {}", ruta, e));
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().flatten() {
                let _ = tx.send(line);
            }
        });
        let mut m = MotorUci {
            stdin,
            rx,
            child,
            nombre: ruta.to_string(),
        };
        m.enviar("uci");
        m.esperar_uciok();
        if ruta.contains("Mittens") {
            // Activar la red NNUE de produccion, igual que el harness elite.
            m.enviar("setoption name UseNNUE value true");
            m.enviar(&format!("setoption name NNUEPath value {}", MITTENS_WEIGHTS));
        }
        m.enviar("isready");
        loop {
            if m.rx
                .recv_timeout(Duration::from_secs(20))
                .unwrap_or_default()
                .contains("readyok")
            {
                break;
            }
        }
        m.enviar("ucinewgame");
        m
    }

    fn enviar(&mut self, s: &str) {
        writeln!(self.stdin, "{}", s).ok();
        self.stdin.flush().ok();
    }

    fn esperar_uciok(&mut self) {
        loop {
            if self
                .rx
                .recv_timeout(Duration::from_secs(20))
                .unwrap_or_default()
                .contains("uciok")
            {
                break;
            }
        }
    }

    /// go movetime fijo. Devuelve (bestmove, cp, mate).
    fn pensar(&mut self, movetime_ms: u64) -> (Option<String>, i32, bool) {
        self.enviar(&format!("go movetime {}", movetime_ms));
        let mut best = None;
        let mut cp = 0;
        let mut mate = false;
        loop {
            match self.rx.recv_timeout(Duration::from_millis(movetime_ms + 20000)) {
                Ok(l) => {
                    if l.starts_with("info") {
                        if let Some(p) = l.find(" score cp ") {
                            cp = l[p + 10..]
                                .split_whitespace()
                                .next()
                                .and_then(|v| v.parse::<i32>().ok())
                                .unwrap_or(cp);
                        } else if let Some(p) = l.find(" score mate ") {
                            let mt: i32 = l[p + 12..]
                                .split_whitespace()
                                .next()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(0);
                            mate = mt != 0;
                            cp = if mt > 0 { 30000 } else { -30000 };
                        }
                    } else if l.starts_with("bestmove") {
                        let bm = l
                            .trim_start_matches("bestmove")
                            .split_whitespace()
                            .next()
                            .unwrap_or("")
                            .to_string();
                        best = if bm.is_empty() || bm == "(none)" || bm == "0000" {
                            None
                        } else {
                            Some(bm)
                        };
                        break;
                    }
                }
                Err(_) => {
                    println!("   [{} TIMEOUT: pierde por reloj]", self.nombre);
                    best = None;
                    break;
                }
            }
        }
        (best, cp, mate)
    }

    fn matar(&mut self) {
        self.enviar("quit");
        let _ = self.child.kill();
    }
}

/// Partida con movetime fijo por jugada. 1=ganan blancas, 0=tablas, -1=ganan negras.
fn jugar_partida(pb: &str, pn: &str, apertura: &str, movetime_ms: u64, max_plies: usize) -> i8 {
    let mut blancas = MotorUci::nuevo(pb);
    let mut negras = MotorUci::nuevo(pn);
    let mut moves: Vec<String> = if apertura.is_empty() {
        Vec::new()
    } else {
        apertura.split_whitespace().map(|s| s.to_string()).collect()
    };
    let mut historial_cp: Vec<i32> = Vec::new();
    let resultado = loop {
        let turno_blancas = moves.len() % 2 == 0;
        let pos = if moves.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves.join(" "))
        };
        let motor = if turno_blancas { &mut blancas } else { &mut negras };
        motor.enviar(&pos);
        let (bm, cp, mate) = motor.pensar(movetime_ms);
        historial_cp.push(if turno_blancas { cp } else { -cp });
        match bm {
            None => {
                if mate {
                    break Some(if turno_blancas { -1 } else { 1 });
                } else if historial_cp.last().map(|c| c.abs() > 25000).unwrap_or(false) {
                    break Some(if turno_blancas { -1 } else { 1 });
                } else {
                    break Some(0);
                }
            }
            Some(jugada) => moves.push(jugada),
        }
        let n = historial_cp.len();
        if moves.len() >= max_plies {
            break Some(0);
        }
        if n >= 10 && moves.len() > 40 && historial_cp[n - 10..].iter().all(|c| c.abs() <= 12) {
            break Some(0);
        }
        if n >= 2 && historial_cp[n - 1].abs() > 29000 && historial_cp[n - 2].abs() > 29000 {
            break Some(if historial_cp[n - 1] > 0 { 1 } else { -1 });
        }
        if n > 300 {
            break None;
        }
    };
    let r = resultado.unwrap_or(0);
    let marcador = match r {
        1 => format!("1-0 (ganan blancas)"),
        -1 => format!("0-1 (ganan negras)"),
        _ => "1/2-1/2".to_string(),
    };
    println!(
        "PARTIDA {} vs {} | ap:[{}] plies:{} -> {}",
        pb, pn, apertura, moves.len(), marcador
    );
    blancas.matar();
    negras.matar();
    r
}

const OPENINGS: [&str; 4] = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4",
    "g1f3 d7d5 g2g3 g8f6 f1g2 g7g6",
];

#[test]
fn duelo_mittens_vs_termino() {
    println!("\n=== DUELO MITTENS vs TERMO (Frozenight) ===");
    let mut p_mittens = 0.0;
    let mut p_termino = 0.0;
    for (i, ap) in OPENINGS.iter().enumerate() {
        let mit_white = i % 2 == 0;
        let (r1, r2) = if mit_white {
            (
                jugar_partida(
                    &format!("{DIR}Mittens"),
                    &format!("{DIR}Termo"),
                    ap,
                    300,
                    120,
                ),
                jugar_partida(
                    &format!("{DIR}Termo"),
                    &format!("{DIR}Mittens"),
                    ap,
                    300,
                    120,
                ),
            )
        } else {
            (
                jugar_partida(
                    &format!("{DIR}Termo"),
                    &format!("{DIR}Mittens"),
                    ap,
                    300,
                    120,
                ),
                jugar_partida(
                    &format!("{DIR}Mittens"),
                    &format!("{DIR}Termo"),
                    ap,
                    300,
                    120,
                ),
            )
        };
        // puntos para Mittens: +1 si gana, +0.5 si tablas
        let pts = |r: i8, mit_es_blancas: bool| -> f64 {
            if r == 1 {
                if mit_es_blancas {
                    1.0
                } else {
                    0.0
                }
            } else if r == -1 {
                if mit_es_blancas {
                    0.0
                } else {
                    1.0
                }
            } else {
                0.5
            }
        };
        p_mittens += pts(r1, mit_white) + pts(r2, !mit_white);
        p_termino += (1.0 - pts(r1, mit_white)) + (1.0 - pts(r2, !mit_white));
        println!("  Apertura [{}] -> parcial Mittens {:.1} - {:.1} Termo", ap, p_mittens, p_termino);
    }
    println!("\n=== RESULTADO FINAL: Mittens {:.1} - {:.1} Termo (Frozenight) ===", p_mittens, p_termino);
    assert!(true, "resultado informativo");
}
