// Sonda UCI para los binarios de /Users/Tavito/mi-motor/motores_elite.
// Mide: id name, tiempo real transcurrido ante `go movetime`, y ultima linea info (depth).
// No modifica nada fuera del proyecto: solo lanza procesos y lee su stdout.
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const DIR: &str = "/Users/Tavito/mi-motor/motores_elite/";

struct Resultado {
    motor: String,
    id: String,
    go: String,
    elapsed_ms: u128,
    ultima_info: String,
}

fn sondear(motor: &str, go_cmd: &str) -> Option<Resultado> {
    let ruta = format!("{DIR}{motor}");
    let mut child = Command::new(&ruta)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let stdout = child.stdout.take()?;
    let mut reader = BufReader::new(stdout);

    let (tx, rx) = mpsc::channel::<String>();
    let h = thread::spawn(move || {
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if tx.send(line.trim_end().to_string()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let mut enviar = |s: &str| {
        let _ = stdin.write_all(s.as_bytes());
        let _ = stdin.flush();
    };

    let mut id_name = String::new();
    enviar("uci\n");
    loop {
        match rx.recv_timeout(Duration::from_secs(15)) {
            Ok(l) => {
                if let Some(v) = l.strip_prefix("id name ") {
                    id_name = v.to_string();
                }
                if l == "uciok" {
                    break;
                }
            }
            Err(_) => {
                let _ = child.kill();
                let _ = h.join();
                return None;
            }
        }
    }
    enviar("isready\n");
    loop {
        match rx.recv_timeout(Duration::from_secs(15)) {
            Ok(l) => {
                if l == "readyok" {
                    break;
                }
            }
            Err(_) => {
                let _ = child.kill();
                let _ = h.join();
                return None;
            }
        }
    }
    enviar("position startpos\n");
    enviar(&format!("{go_cmd}\n"));
    let t0 = Instant::now();
    let mut ultima_info = String::new();
    let timeout_ms = if go_cmd.contains("movetime") {
        // extraemos el valor movetime si viene
        let v = go_cmd
            .split_whitespace()
            .nth(1)
            .and_then(|x| x.parse::<u64>().ok())
            .unwrap_or(2000);
        v + 20000
    } else {
        60000
    };
    loop {
        match rx.recv_timeout(Duration::from_millis(timeout_ms)) {
            Ok(l) => {
                if l.starts_with("bestmove") {
                    break;
                }
                if l.starts_with("info") {
                    ultima_info = l;
                }
            }
            Err(_) => {
                let _ = child.kill();
                let _ = h.join();
                return None;
            }
        }
    }
    let elapsed_ms = t0.elapsed().as_millis();
    enviar("quit\n");
    let _ = child.wait();
    let _ = h.join();
    Some(Resultado {
        motor: motor.to_string(),
        id: id_name,
        go: go_cmd.to_string(),
        elapsed_ms,
        ultima_info,
    })
}

fn resumir(r: &Resultado) {
    let info = if r.ultima_info.is_empty() {
        "(sin info)".to_string()
    } else {
        // extraemos depth, seldepth, nodes, nps, time de la ultima info
        let mut parts = Vec::new();
        for tok in ["depth", "seldepth", "nodes", "nps", "time"] {
            if let Some(idx) = r.ultima_info.find(tok) {
                let rest = &r.ultima_info[idx + tok.len()..];
                let val: String = rest
                    .trim_start()
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                parts.push(format!("{tok}={val}"));
            }
        }
        parts.join(" ")
    };
    println!(
        "[{}] id={} | go={} | tiempo_real={}ms | {}",
        r.motor, r.id, r.go, r.elapsed_ms, info
    );
}

#[test]
fn sonda_movetime_2000() {
    let motores = [
        "Bufanda", "Termo", "Jabon", "Cesped", "Escoba", "Musgo", "Mittens",
    ];
    for m in motores {
        match sondear(m, "go movetime 2000") {
            Some(r) => resumir(&r),
            None => println!("[{}] FALLO al sondear", m),
        }
    }
}

#[test]
fn sonda_reloj_real_bufanda_termo() {
    for m in ["Bufanda", "Termo", "Jabon"] {
        match sondear(m, "go wtime 2000 btime 2000 winc 200 binc 200") {
            Some(r) => resumir(&r),
            None => println!("[{}] FALLO al sondear con reloj", m),
        }
    }
}
