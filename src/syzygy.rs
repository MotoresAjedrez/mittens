// Sondeo de tablas de finales Syzygy (3-4-5 piezas) via el crate shakmaty-syzygy
// (probing puro en Rust, no reinventa el formato de archivo). El motor sigue
// usando su propio Board/Move en todos lados; la conversion a shakmaty::Chess
// ocurre SOLO en el borde de este modulo (via FEN, ya que Board::to_fen ya
// existe), y solo se llama cuando la cantidad de piezas ya esta dentro de lo
// cubierto por las tablas -- en la inmensa mayoria de nodos de busqueda esto
// nunca se ejecuta.

use crate::board::Board;
use crate::movegen::generate_legal;
use crate::types::{Move, PieceType, square_from_name};
use shakmaty::{CastlingMode, Chess};
use shakmaty_syzygy::{Tablebase, Wdl};
use std::str::FromStr;
use std::sync::OnceLock;

static TABLES: OnceLock<Tablebase<Chess>> = OnceLock::new();

/// Puntaje para una victoria de tabla: por debajo de MATE (para no confundirse
/// con un mate real encontrado por la busqueda) pero muy por encima de
/// cualquier evaluacion estatica normal, para que alfa-beta siempre la
/// prefiera sobre cualquier linea sin tabla.
pub const TB_WIN: i32 = crate::search::MATE - 2000;

/// Carga las tablas desde un directorio (WDL + DTZ). Devuelve cuantas piezas
/// como maximo quedaron cubiertas, o Err si el directorio no tiene tablas
/// validas. Se llama una sola vez al arrancar el motor.
pub fn init(path: &str) -> Result<usize, String> {
    if TABLES.get().is_some() {
        return Err(
            "las tablas Syzygy ya estan cargadas; reinicia el motor para cambiar SyzygyPath"
                .to_string(),
        );
    }
    let mut tables = Tablebase::new();
    let n = tables.add_directory(path).map_err(|e| e.to_string())?;
    if n == 0 {
        return Err("no se encontraron archivos de tabla en el directorio".to_string());
    }
    let max = tables.max_pieces();
    TABLES
        .set(tables)
        .map_err(|_| "no se pudieron instalar las tablas Syzygy".to_string())?;
    Ok(max)
}

pub fn disponible() -> bool {
    TABLES.get().is_some()
}

fn max_piezas() -> usize {
    TABLES.get().map(|t| t.max_pieces()).unwrap_or(0)
}

fn a_shakmaty(b: &Board) -> Option<Chess> {
    let fen = shakmaty::fen::Fen::from_str(&b.to_fen()).ok()?;
    let pos: Chess = fen.into_position(CastlingMode::Standard).ok()?;
    Some(pos)
}

fn cantidad_piezas(b: &Board) -> u32 {
    b.occupied.count_ones()
}

/// True si la posicion tiene pocas piezas como para estar cubierta por las
/// tablas cargadas -- chequeo barato (solo popcount) para descartar la
/// enorme mayoria de nodos sin construir nada de shakmaty.
pub fn en_rango(b: &Board) -> bool {
    disponible() && (cantidad_piezas(b) as usize) <= max_piezas()
}

/// WDL exacto (ganada/tablas/perdida) desde la perspectiva del que mueve, ya
/// convertido a puntaje tipo centipeon. Devuelve None si la posicion no esta
/// cubierta o hay algun problema de conversion (se sigue con evaluacion
/// normal en ese caso, nunca es un error fatal).
pub fn probe_wdl(b: &Board) -> Option<i32> {
    if !en_rango(b) {
        return None;
    }
    let tables = TABLES.get()?;
    let pos = a_shakmaty(b)?;
    let wdl = tables.probe_wdl_after_zeroing(&pos).ok()?;
    Some(match wdl {
        Wdl::Loss => -TB_WIN,
        Wdl::BlessedLoss => -1, // perdida bajo juego perfecto pero tablas por regla de 50 -- tratar como levemente peor que tablas
        Wdl::Draw => 0,
        Wdl::CursedWin => 1, // analogo simetrico de BlessedLoss
        Wdl::Win => TB_WIN,
    })
}

fn uci_a_jugada(b: &Board, uci: &str) -> Option<Move> {
    let moves = generate_legal(b);
    let bytes = uci.as_bytes();
    if bytes.len() < 4 {
        return None;
    }
    let from = square_from_name(&uci[0..2])?;
    let to = square_from_name(&uci[2..4])?;
    let promo = if bytes.len() >= 5 {
        PieceType::from_char(bytes[4] as char)
    } else {
        None
    };
    moves
        .into_iter()
        .find(|m| m.from == from && m.to == to && m.promotion == promo)
}

/// Jugada recomendada por la tabla en la raiz, via DTZ (distance-to-zero):
/// progresa de verdad hacia el resultado optimo, no solo "no empeora". Se usa
/// SOLO en la raiz -- reemplaza directamente la jugada que hubiera elegido la
/// busqueda normal cuando la posicion ya esta cubierta por las tablas.
pub fn mejor_jugada_raiz(b: &Board) -> Option<(Move, i32)> {
    if !en_rango(b) {
        return None;
    }
    let tables = TABLES.get()?;
    let pos = a_shakmaty(b)?;
    let (mv, dtz) = tables.best_move(&pos).ok()??;
    let uci = mv.to_uci(CastlingMode::Standard).to_string();
    let my_mv = uci_a_jugada(b, &uci)?;
    // El DTZ que devuelve best_move es el de la posicion RESULTANTE despues
    // de la jugada, desde la perspectiva del rival (que pasa a mover) -- no
    // la nuestra antes de mover. Rival en numeros negativos (perdiendo) es
    // buena noticia para nosotros, y viceversa: signo invertido respecto a
    // "nuestra" perspectiva.
    let dtz_val = dtz.ignore_rounding().0;
    let score = if dtz_val < 0 {
        TB_WIN
    } else if dtz_val > 0 {
        -TB_WIN
    } else {
        0
    };
    Some((my_mv, score))
}
