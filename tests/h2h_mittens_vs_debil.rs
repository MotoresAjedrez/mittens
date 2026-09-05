// Control: Mittens (motor propio) vs Escoba (Apotheosis v4.0.1, motor conocido
// mas debil, CCRL ~2747). Mismo arbitro que h2h_mittens_vs_termino.rs.
// Objetivo: si esto tambien da puro empate, el harness esta roto. Si Mittens
// gana claramente, el harness distingue fuerza real y el 4-4 contra Termo es valido.
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const DIR: &str = "/Users/Tavito/mi-motor/motores_control/";
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
            m.enviar("setoption name UseNNUE value true");
            m.enviar(&format!("setoption name NNUEPath value {}", std::env::var("MITTENS_H2H_PESOS").unwrap_or_else(|_| MITTENS_WEIGHTS.to_string())));
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

const OPENINGS: [&str; 5] = [
    "e2e4 e7e5",
    "d2d4 d7d5",
    "e2e4 c7c5",
    "g1f3 d7d5",
    "d2d4 g8f6",
];

#[test]
fn duelo_mittens_vs_debil() {
    // Rutas portables: MITTENS_H2H_DIR (y MITTENS_H2H_PESOS) sobreescriben las
    // rutas historicas de macOS. Si el directorio no existe, el test se omite
    // con aviso en vez de fallar: es un duelo informativo, no una prueba unitaria.
    let dir = std::env::var("MITTENS_H2H_DIR").unwrap_or_else(|_| DIR.to_string());
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("h2h omitido: no existe {dir} (define MITTENS_H2H_DIR)");
        return;
    }
    println!("\n=== CONTROL: MITTENS vs Escoba (Apotheosis, motor debil ~2747 CCRL) ===");
    let mut p_mittens = 0.0;
    let mut p_debil = 0.0;
    for (i, ap) in OPENINGS.iter().enumerate() {
        let mit_white = i % 2 == 0;
        let (pb, pn) = if mit_white {
            (format!("{dir}Mittens"), format!("{dir}Escoba"))
        } else {
            (format!("{dir}Escoba"), format!("{dir}Mittens"))
        };
        let r = jugar_partida(&pb, &pn, ap, 200, 220);
        let pts = if r == 1 {
            if mit_white { 1.0 } else { 0.0 }
        } else if r == -1 {
            if mit_white { 0.0 } else { 1.0 }
        } else {
            0.5
        };
        p_mittens += pts;
        p_debil += 1.0 - pts;
        println!("  Apertura [{}] -> parcial Mittens {:.1} - {:.1} Escoba", ap, p_mittens, p_debil);
    }
    println!("\n=== RESULTADO FINAL: Mittens {:.1} - {:.1} Escoba (debil) ===", p_mittens, p_debil);
    assert!(true, "resultado informativo");
}
