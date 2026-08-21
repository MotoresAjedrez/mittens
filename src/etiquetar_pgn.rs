//! Etiqueta posiciones extraidas de partidas PGN reales (ej. Lichess Elite
//! Database) con el SCORE de busqueda de Mittens y el RESULTADO REAL de la
//! partida (ya viene en el PGN, no hay que jugarla).
//!
//! A diferencia de `datagen::run_datagen` (que JUEGA la partida y por lo
//! tanto conoce el resultado recien al final), aca el resultado ya esta
//! resuelto de entrada: lo unico que falta es el score, que se consigue
//! corriendo el motor sobre cada posicion muestreada.
//!
//! No usa ningun crate de PGN externo: parsea SAN a mano contra
//! `movegen::generate_legal`, reutilizando los tipos de Mittens de punta a
//! punta (nada de sincronizar dos representaciones de tablero distintas).

use crate::board::Board;
use crate::datagen::empaquetar;
use crate::movegen::generate_legal;
use crate::search::Searcher;
use crate::types::{Color, PieceType, Square, square_from_name};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

/// Un token de movetext ya separado de comentarios/NAGs/numeros de jugada.
struct TokenSan<'a> {
    texto: &'a str,
}

/// Intenta encontrar la unica jugada legal que coincide con un token SAN.
/// Devuelve None si el token no es un movimiento (numero de jugada, NAG,
/// resultado) o si no matchea ninguna/mas de una jugada (PGN corrupto).
fn resolver_san(b: &Board, token: &str) -> Option<crate::types::Move> {
    let t = token.trim_end_matches(['+', '#', '!', '?']);
    if t.is_empty() || t == "1-0" || t == "0-1" || t == "1/2-1/2" || t == "*" {
        return None;
    }

    let moves = generate_legal(b);

    // Enroque: SAN no usa casillas, hay que matchear por flag.
    if t == "O-O" || t == "0-0" {
        return moves
            .into_iter()
            .find(|m| matches!(m.flag, crate::types::MoveFlag::CastleKing));
    }
    if t == "O-O-O" || t == "0-0-0" {
        return moves
            .into_iter()
            .find(|m| matches!(m.flag, crate::types::MoveFlag::CastleQueen));
    }

    let bytes = t.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let (pieza, resto) = match bytes[0] {
        b'K' => (Some(PieceType::King), &t[1..]),
        b'Q' => (Some(PieceType::Queen), &t[1..]),
        b'R' => (Some(PieceType::Rook), &t[1..]),
        b'B' => (Some(PieceType::Bishop), &t[1..]),
        b'N' => (Some(PieceType::Knight), &t[1..]),
        _ => (None, t), // peon: sin letra de pieza
    };

    // Promocion: "=Q" al final (si existe).
    let (resto, promo) = if let Some(pos) = resto.find('=') {
        let pc = resto.as_bytes().get(pos + 1).copied();
        let pt = pc.and_then(|c| PieceType::from_char(c as char));
        (&resto[..pos], pt)
    } else {
        (resto, None)
    };

    // Ultimos 2 caracteres no-'x' son la casilla destino.
    let sin_captura: String = resto.chars().filter(|&c| c != 'x').collect();
    if sin_captura.len() < 2 {
        return None;
    }
    let destino_str = &sin_captura[sin_captura.len() - 2..];
    let to = square_from_name(destino_str)?;
    let desambiguacion = &sin_captura[..sin_captura.len() - 2];

    let piece_wanted = pieza.unwrap_or(PieceType::Pawn);

    let mut candidatos: Vec<crate::types::Move> = moves
        .into_iter()
        .filter(|m| {
            if m.to != to || m.promotion != promo {
                return false;
            }
            match b.piece_at(m.from) {
                Some((_, pt)) => pt == piece_wanted,
                None => false,
            }
        })
        .collect();

    if candidatos.len() > 1 && !desambiguacion.is_empty() {
        candidatos.retain(|m| {
            let from_str = crate::types::square_name(m.from);
            desambiguacion.chars().all(|c| from_str.contains(c))
        });
    }

    if candidatos.len() == 1 {
        Some(candidatos[0])
    } else {
        None
    }
}

struct Cabecera {
    resultado_blancas: Option<f32>,
}

fn parsear_resultado(valor: &str) -> Option<f32> {
    match valor {
        "1-0" => Some(1.0),
        "0-1" => Some(0.0),
        "1/2-1/2" => Some(0.5),
        _ => None,
    }
}

/// Separa el movetext en tokens, descartando numeros de jugada ("12.",
/// "12..."), comentarios entre llaves, y variantes entre parentesis (no las
/// seguimos: solo la linea principal).
fn tokenizar_movetext(mut texto: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut prof_parentesis = 0i32;
    let mut en_comentario = false;
    let bytes = texto.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if en_comentario {
            if c == '}' {
                en_comentario = false;
            }
            i += 1;
            continue;
        }
        match c {
            '{' => {
                en_comentario = true;
            }
            '(' => {
                prof_parentesis += 1;
            }
            ')' => {
                prof_parentesis -= 1;
            }
            c if c.is_whitespace() => {
                if prof_parentesis == 0 && !buf.is_empty() {
                    out.push(std::mem::take(&mut buf));
                }
            }
            _ => {
                if prof_parentesis == 0 {
                    buf.push(c);
                }
            }
        }
        i += 1;
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    let _ = &mut texto;
    out.into_iter()
        .filter(|t| {
            // Descarta "N." / "N..." / NAGs "$3" / resultados sueltos.
            let sin_puntos = t.trim_end_matches('.');
            !sin_puntos.is_empty()
                && !sin_puntos.chars().all(|c| c.is_ascii_digit())
                && !t.starts_with('$')
        })
        .collect()
}

/// `mittens etiquetar_pgn --entrada F.pgn --salida F.data [--cada-n-plies N]
/// [--nodos N] [--desde-ply N] [--elo-min E]`
pub fn run_etiquetar_pgn(args: &[String]) {
    let mut entrada = String::new();
    let mut salida = String::new();
    let mut cada_n_plies: usize = 8;
    let mut nodos: u64 = 8000;
    let mut desde_ply: usize = 16; // se salta la apertura de libro
    let mut elo_min: i32 = 0;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--entrada" => {
                entrada = args[i + 1].clone();
                i += 2;
            }
            "--salida" => {
                salida = args[i + 1].clone();
                i += 2;
            }
            "--cada-n-plies" => {
                cada_n_plies = args[i + 1].parse().unwrap_or(8);
                i += 2;
            }
            "--nodos" => {
                nodos = args[i + 1].parse().unwrap_or(8000);
                i += 2;
            }
            "--desde-ply" => {
                desde_ply = args[i + 1].parse().unwrap_or(16);
                i += 2;
            }
            "--elo-min" => {
                elo_min = args[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    if entrada.is_empty() || salida.is_empty() {
        eprintln!(
            "uso: mittens etiquetar_pgn --entrada F.pgn --salida F.data \
             [--cada-n-plies 8] [--nodos 8000] [--desde-ply 16] [--elo-min 0]"
        );
        std::process::exit(1);
    }

    let archivo = File::open(&entrada).unwrap_or_else(|e| {
        eprintln!("no se pudo abrir {}: {}", entrada, e);
        std::process::exit(1);
    });
    let lector = BufReader::new(archivo);

    let mut s = Searcher::new(16);
    s.nodes_limit = Some(nodos);

    let mut out = File::create(&salida).unwrap_or_else(|e| {
        eprintln!("no se pudo crear {}: {}", salida, e);
        std::process::exit(1);
    });
    let mut buf_salida: Vec<u8> = Vec::with_capacity(1 << 20);

    let mut partidas = 0u64;
    let mut posiciones = 0u64;
    let mut descartadas_bajo_elo = 0u64;
    let inicio = Instant::now();

    // Estado del juego que se esta leyendo.
    let mut resultado: Option<f32> = None;
    let mut white_elo: i32 = 0;
    let mut black_elo: i32 = 0;
    let mut movetext = String::new();
    let mut en_headers = false;
    let mut vio_algun_header = false;

    let procesar_partida = |movetext: &str,
                             resultado_blancas: f32,
                             s: &mut Searcher,
                             buf_salida: &mut Vec<u8>|
     -> u64 {
        // Sin esto la TT arrastra entradas de partidas anteriores: el mismo
        // patron de contaminacion ya documentado en el proyecto (motor
        // "fresco" por medicion). Con miles de partidas seguidas, entradas
        // viejas pueden disparar comportamiento degenerado de busqueda en
        // alguna posicion puntual (visto en la practica: partida 952 de una
        // tanda larga colgaba la busqueda pese al limite de nodos; sola,
        // sin TT contaminada, tardaba milisegundos).
        s.clear_hash();
        let tokens = tokenizar_movetext(movetext);
        let mut b = Board::startpos();
        let mut ply = 0usize;
        let mut generadas = 0u64;
        for tok in &tokens {
            let Some(mv) = resolver_san(&b, tok) else {
                break; // PGN inconsistente: cortamos la partida aca, no adivinamos
            };
            ply += 1;
            let toca_muestrear = ply >= desde_ply && (ply - desde_ply) % cada_n_plies == 0;
            if toca_muestrear && !b.in_check(b.turn) {
                s.set_game_history(Vec::new());
                let (_, score, _) = s.search_fixed_depth(&b, 30);
                if score.abs() < crate::search::MATE - 1000 {
                    let r = if b.turn == Color::White {
                        resultado_blancas
                    } else {
                        1.0 - resultado_blancas
                    };
                    empaquetar(&b, score, r, buf_salida);
                    generadas += 1;
                }
            }
            b = b.make_move(&mv);
        }
        generadas
    };

    for linea in lector.lines() {
        let linea = match linea {
            Ok(l) => l,
            Err(_) => continue, // bytes invalidos ocasionales en dumps grandes: se saltan
        };
        let l = linea.trim_end();

        if l.starts_with('[') {
            if !en_headers && vio_algun_header {
                // Empieza una cabecera nueva sin blank-line: cierre defensivo,
                // no deberia pasar en un PGN bien formado pero no confiamos.
            }
            en_headers = true;
            vio_algun_header = true;
            if let Some(v) = extraer_valor_header(l, "Result") {
                resultado = parsear_resultado(&v);
            } else if let Some(v) = extraer_valor_header(l, "WhiteElo") {
                white_elo = v.parse().unwrap_or(0);
            } else if let Some(v) = extraer_valor_header(l, "BlackElo") {
                black_elo = v.parse().unwrap_or(0);
            }
            continue;
        }

        if l.is_empty() {
            if en_headers {
                // Fin del bloque de headers, arranca el movetext en las
                // proximas lineas no vacias.
                en_headers = false;
            } else if !movetext.is_empty() {
                // Fin de la partida (movetext terminado).
                if let Some(res) = resultado {
                    if white_elo.min(black_elo) >= elo_min {
                        posiciones += procesar_partida(&movetext, res, &mut s, &mut buf_salida);
                        partidas += 1;
                    } else {
                        descartadas_bajo_elo += 1;
                    }
                }
                movetext.clear();
                resultado = None;
                white_elo = 0;
                black_elo = 0;

                if buf_salida.len() > (1 << 22) {
                    out.write_all(&buf_salida).expect("error escribiendo salida");
                    buf_salida.clear();
                }
                if partidas % 2000 == 0 {
                    let seg = inicio.elapsed().as_secs_f64().max(0.001);
                    println!(
                        "  {} partidas, {} posiciones, {:.1} pos/seg (descartadas por elo: {})",
                        partidas,
                        posiciones,
                        posiciones as f64 / seg,
                        descartadas_bajo_elo
                    );
                }
            }
            continue;
        }

        if !en_headers {
            if !movetext.is_empty() {
                movetext.push(' ');
            }
            movetext.push_str(l);
        }
    }

    // Ultima partida si el archivo no termina con linea en blanco.
    if let Some(res) = resultado
        && !movetext.is_empty()
        && white_elo.min(black_elo) >= elo_min
    {
        posiciones += procesar_partida(&movetext, res, &mut s, &mut buf_salida);
        partidas += 1;
    }

    if !buf_salida.is_empty() {
        out.write_all(&buf_salida).expect("error escribiendo salida final");
    }

    let seg = inicio.elapsed().as_secs_f64();
    println!(
        "listo: {} partidas, {} posiciones etiquetadas en {:.1}s ({:.1} pos/seg) -> {}",
        partidas,
        posiciones,
        seg,
        posiciones as f64 / seg.max(0.001),
        salida
    );
}

fn extraer_valor_header<'a>(linea: &'a str, nombre: &str) -> Option<String> {
    let prefijo = format!("[{} \"", nombre);
    if let Some(resto) = linea.strip_prefix(&prefijo) {
        resto.rfind('"').map(|fin| resto[..fin].to_string())
    } else {
        None
    }
}

#[allow(dead_code)]
fn _silenciar_warning(_: Square) {}
