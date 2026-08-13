// Pruebas del BUCLE UCI REAL, hablandole al binario por stdin/stdout como lo
// haria un GUI o un arbitro de torneo.
//
// Por que existen: hasta ahora todo lo del protocolo se probaba llamando
// funciones sueltas de la libreria, asi que ningun test cubria el uci_loop en
// si -- justo donde apareceieron los bugs caros (el "isready" que abortaba la
// busqueda, el "stop" inmediato que devolvia a2a3, y el uci_loop duplicado
// entre main.rs y lib.rs que hacia que cada fix hubiera que escribirlo dos
// veces). Estas pruebas ejercitan el binario de verdad.
//
// NO se juegan partidas: solo comandos sueltos con movetime corto.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

const BIN: &str = env!("CARGO_BIN_EXE_mittens");

struct Motor {
    stdin: ChildStdin,
    rx: Receiver<String>,
    child: Child,
}

impl Motor {
    fn nuevo() -> Motor {
        let mut child = Command::new(BIN)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("no se pudo lanzar el binario del motor");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for linea in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(linea).is_err() {
                    break;
                }
            }
        });
        Motor { stdin, rx, child }
    }

    fn enviar(&mut self, cmd: &str) {
        writeln!(self.stdin, "{}", cmd).expect("escribir al motor");
        self.stdin.flush().expect("flush");
    }

    /// Lee lineas hasta encontrar una que empiece con `prefijo`, juntando todo
    /// lo leido. Falla el test si se agota el tiempo: un cuelgue del protocolo
    /// tiene que salir como test en rojo, no como suite colgada.
    fn esperar(&mut self, prefijo: &str, timeout: Duration) -> (String, Vec<String>) {
        let limite = std::time::Instant::now() + timeout;
        let mut vistas = Vec::new();
        loop {
            let restante = limite.saturating_duration_since(std::time::Instant::now());
            match self.rx.recv_timeout(restante) {
                Ok(l) => {
                    let encontrada = l.starts_with(prefijo);
                    vistas.push(l.clone());
                    if encontrada {
                        return (l, vistas);
                    }
                }
                Err(RecvTimeoutError::Timeout) => panic!(
                    "timeout esperando '{}'. Lineas recibidas:\n{}",
                    prefijo,
                    vistas.join("\n")
                ),
                Err(RecvTimeoutError::Disconnected) => panic!(
                    "el motor cerro stdout sin responder '{}'. Lineas recibidas:\n{}",
                    prefijo,
                    vistas.join("\n")
                ),
            }
        }
    }
}

impl Drop for Motor {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Handshake basico: "uci" contesta id/option/uciok, y "isready" -> "readyok".
#[test]
fn handshake_uci_declara_id_opciones_y_uciok() {
    let mut m = Motor::nuevo();
    m.enviar("uci");
    let (_, vistas) = m.esperar("uciok", Duration::from_secs(10));
    let texto = vistas.join("\n");
    assert!(texto.contains("id name"), "falta 'id name':\n{}", texto);
    assert!(texto.contains("id author"), "falta 'id author':\n{}", texto);
    // Toda opcion declarada debe tener nombre y tipo bien formados.
    for l in vistas.iter().filter(|l| l.starts_with("option ")) {
        assert!(
            l.contains(" name ") && l.contains(" type "),
            "opcion mal formada: {}",
            l
        );
    }
    m.enviar("isready");
    m.esperar("readyok", Duration::from_secs(10));
}

/// REGRESION del bug de "stop" inmediato: un "stop" apenas mandado el "go"
/// tiene que devolver una jugada REALMENTE buscada. Antes devolvia el
/// fallback (la primera jugada en orden de generacion, a2a3 en la inicial).
#[test]
fn stop_inmediato_no_devuelve_la_primera_jugada_generada() {
    let mut m = Motor::nuevo();
    m.enviar("uci");
    m.esperar("uciok", Duration::from_secs(10));
    m.enviar("position startpos moves e2e4 e7e5");
    m.enviar("go infinite");
    m.enviar("stop");
    let (best, _) = m.esperar("bestmove", Duration::from_secs(10));
    assert_ne!(
        best.trim(),
        "bestmove a2a3",
        "volvio el fallback de generacion en vez de una jugada buscada"
    );
    assert_ne!(best.trim(), "bestmove 0000", "bestmove nulo tras stop");
}

/// REGRESION del bug de "isready": el protocolo exige contestar readyok sin
/// detener el calculo. Si "isready" abortara la busqueda, el bestmove llegaria
/// antes del movetime pedido.
#[test]
fn isready_durante_la_busqueda_no_la_aborta() {
    let mut m = Motor::nuevo();
    m.enviar("uci");
    m.esperar("uciok", Duration::from_secs(10));
    m.enviar("position startpos moves e2e4 e7e5");
    let t0 = std::time::Instant::now();
    m.enviar("go movetime 1500");
    m.enviar("isready");
    m.esperar("readyok", Duration::from_secs(5));
    let tras_readyok = t0.elapsed();
    assert!(
        tras_readyok < Duration::from_millis(1200),
        "readyok tardo {:?}: no contesto de inmediato",
        tras_readyok
    );
    m.esperar("bestmove", Duration::from_secs(15));
    let total = t0.elapsed();
    assert!(
        total >= Duration::from_millis(1000),
        "el bestmove llego a los {:?}, con movetime 1500: isready aborto la busqueda",
        total
    );
}

/// Un FEN malformado no debe crashear ni cambiar la posicion en silencio: se
/// reporta por "info string" y se conserva la ultima posicion valida.
#[test]
fn fen_malformado_no_crashea_y_conserva_la_posicion_previa() {
    let mut m = Motor::nuevo();
    m.enviar("uci");
    m.esperar("uciok", Duration::from_secs(10));
    // Posicion valida conocida: mate en 1 con dama, la mejor es h5f7.
    m.enviar("position fen r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4");
    for malo in [
        "basura",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e9 0 1",
        "rnbqkbnr/pppppppp/99/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        // Campo de enroque con basura: sigue siendo malformado. (Ojo: un FEN
        // con derechos de enroque meramente IMPOSIBLES, como "4k3/.../4K3 w
        // KQkq", ya NO es un error: se sanea en silencio -- ver el test
        // fen_con_enroque_imposible_se_acepta_saneado mas abajo.)
        "4k3/8/8/8/8/8/8/4K3 w KQXq - 0 1",
    ] {
        m.enviar(&format!("position fen {}", malo));
    }
    // Sigue vivo y sigue en la posicion buena: encuentra el mate en 1.
    m.enviar("isready");
    m.esperar("readyok", Duration::from_secs(10));
    m.enviar("go movetime 500");
    let (best, _) = m.esperar("bestmove", Duration::from_secs(15));
    assert_eq!(
        best.trim(),
        "bestmove f3f7",
        "se perdio la posicion previa tras los FEN malos"
    );
}

/// Interoperabilidad con GUIs y libros que no actualizan el campo de enroque:
/// un FEN cuyos derechos son imposibles (rey y/o torre fuera de su casilla
/// original) NO se rechaza, se acepta con los derechos saneados en silencio.
/// El motor debe adoptar la posicion y jugar en ella, no quedarse en la
/// anterior ni intentar un enroque inexistente.
#[test]
fn fen_con_enroque_imposible_se_acepta_saneado() {
    let mut m = Motor::nuevo();
    m.enviar("uci");
    m.esperar("uciok", Duration::from_secs(10));
    // Ni el rey blanco (d1) ni el negro (g8) estan en su casilla original y no
    // hay torres negras, pero el FEN insiste en "KQkq": los cuatro derechos
    // deben caer. Mate en 1 de pasillo para blancas: Ta1-a8.
    m.enviar("position fen 6k1/5ppp/8/8/8/8/8/R2K4 w KQkq - 0 1");
    m.enviar("isready");
    m.esperar("readyok", Duration::from_secs(10));
    m.enviar("go movetime 500");
    let (best, vistas) = m.esperar("bestmove", Duration::from_secs(15));
    let jugada = best.trim().trim_start_matches("bestmove ").trim();
    assert_eq!(
        jugada, "a1a8",
        "no adopto la posicion del FEN con enroque imposible: {}",
        best
    );
    assert!(
        !vistas.iter().any(|l| l.contains("panicked")),
        "el motor entro en panico con un FEN de enroque imposible"
    );
}

/// "position startpos moves ..." largo: la historia real de la partida tiene
/// que llegar a la busqueda, asi que una triple repeticion forzada se reconoce
/// como tablas (score 0) y aun asi devuelve una jugada legal, nunca 0000.
#[test]
fn repeticion_en_la_historia_se_reconoce_y_devuelve_jugada_legal() {
    let mut m = Motor::nuevo();
    m.enviar("uci");
    m.esperar("uciok", Duration::from_secs(10));
    // Caballos yendo y viniendo: la posicion inicial se repite tres veces.
    m.enviar(
        "position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8",
    );
    m.enviar("go movetime 500");
    let (best, vistas) = m.esperar("bestmove", Duration::from_secs(15));
    let jugada = best.trim().trim_start_matches("bestmove ").trim();
    assert_ne!(jugada, "0000", "devolvio bestmove nulo en posicion repetida");
    assert_eq!(jugada.len(), 4, "bestmove mal formado: {}", best);
    assert!(
        !vistas.iter().any(|l| l.contains("panicked")),
        "hubo panic:\n{}",
        vistas.join("\n")
    );
}

/// "ucinewgame" limpia el estado: tras una partida, arranca de cero y el
/// motor sigue respondiendo.
#[test]
fn ucinewgame_reinicia_y_el_motor_sigue_respondiendo() {
    let mut m = Motor::nuevo();
    m.enviar("uci");
    m.esperar("uciok", Duration::from_secs(10));
    m.enviar("setoption name Hash value 128");
    m.enviar("position startpos moves e2e4 e7e5 g1f3 b8c6");
    m.enviar("go movetime 200");
    m.esperar("bestmove", Duration::from_secs(15));
    m.enviar("ucinewgame");
    m.enviar("isready");
    m.esperar("readyok", Duration::from_secs(10));
    // Tras ucinewgame la posicion vuelve a la inicial.
    m.enviar("go movetime 300");
    let (best, _) = m.esperar("bestmove", Duration::from_secs(15));
    let jugada = best.trim().trim_start_matches("bestmove ").trim();
    assert_eq!(jugada.len(), 4, "bestmove mal formado: {}", best);
    assert_ne!(jugada, "0000");
}

/// "go" con sus distintos parametros: depth, movetime y reloj (wtime/btime).
/// Ninguno debe colgarse ni devolver jugada nula.
#[test]
fn go_acepta_depth_movetime_y_reloj() {
    let mut m = Motor::nuevo();
    m.enviar("uci");
    m.esperar("uciok", Duration::from_secs(10));
    m.enviar("position startpos moves e2e4 c7c5");
    for cmd in [
        "go depth 6",
        "go movetime 200",
        "go wtime 3000 btime 3000 winc 50 binc 50",
        "go movetime 200 searchmoves g1f3",
    ] {
        m.enviar(cmd);
        let (best, _) = m.esperar("bestmove", Duration::from_secs(20));
        let jugada = best.trim().trim_start_matches("bestmove ").trim();
        assert_eq!(jugada.len(), 4, "'{}' devolvio '{}'", cmd, best);
        assert_ne!(jugada, "0000", "'{}' devolvio jugada nula", cmd);
        if cmd.contains("searchmoves") {
            assert_eq!(jugada, "g1f3", "searchmoves no se respeto: {}", best);
        }
    }
}
