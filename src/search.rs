// Negamax + poda alfa-beta + iterative deepening + quiescence + TT.
// Primera version jugable de la Fase 3: SEE, null-move, killers/history y
// LMR quedan para una siguiente pasada si el tiempo alcanza (documentado
// en el reporte final de la sesion).

use crate::board::Board;
use crate::eval::{
    EvalState, crear_eval_state, evaluate_classical_with_state, evaluate_with_state,
};
use crate::movegen::{MAX_MOVES, generate_captures_legal, generate_legal};
use crate::types::{Move, MoveFlag, PieceType};
use arrayvec::ArrayVec;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

pub const INFINITO: i32 = 30_000;
pub const MATE: i32 = 29_000;
const MAX_PLY: u32 = 64;
// Tamano del array de path para deteccion de repeticion. Cubre partidas
// largas (historial real) mas la rama de busqueda mas profunda. 512 es
// holgado: una partida rara vez pasa de ~300 medios-movimientos.
const MAX_PATH: usize = 512;

// Contempt dinamico: en vez de puntuar toda tabla (repeticion/regla de 50)
// como exactamente 0, se mide la evaluacion estatica de la posicion desde la
// perspectiva de quien mueve. Si esta claramente ganando (por encima del
// umbral), la tabla se puntua PEOR que 0 -- para que la busqueda la evite
// activamente cuando hay alternativas de progreso, en vez de solo reconocerla
// una vez que ya esta ahi. Si esta claramente perdiendo, se puntua MEJOR que
// 0 -- para que la busque activamente como recurso defensivo. Es una funcion
// de la posicion (no depende del historial de jugadas), asi que es seguro
// guardarla en la TT igual que cualquier otro puntaje.
const CONTEMPT_UMBRAL: i32 = 500;
const CONTEMPT_PENALIZACION: i32 = 200;

fn draw_score(
    b: &Board,
    eval_state: &EvalState,
    nnue: Option<&crate::neural::NnueAccumulator>,
) -> i32 {
    // Para decidir el signo del contempt en tablas (repeticion/regla de 50)
    // solo importa si la posicion esta CLARAMENTE ganada o perdida (fuera
    // del umbral de 500cp). La evaluacion clasica es mucho mas barata que la
    // NNUE completa, asi que se usa primero; solo si el resultado clasico
    // cae en la zona ambigua cerca del umbral se confirma con la NNUE.
    let se_clasico = evaluate_classical_with_state(b, eval_state);
    if se_clasico > CONTEMPT_UMBRAL {
        -CONTEMPT_PENALIZACION
    } else if se_clasico < -CONTEMPT_UMBRAL {
        CONTEMPT_PENALIZACION
    } else {
        let se = evaluate_with_state(b, eval_state, nnue);
        if se > CONTEMPT_UMBRAL {
            -CONTEMPT_PENALIZACION
        } else if se < -CONTEMPT_UMBRAL {
            CONTEMPT_PENALIZACION
        } else {
            0
        }
    }
}

fn solo_peones_y_rey(b: &Board, color: crate::types::Color) -> bool {
    let idx = color as usize;
    (b.pieces[idx][crate::types::PieceType::Knight as usize]
        | b.pieces[idx][crate::types::PieceType::Bishop as usize]
        | b.pieces[idx][crate::types::PieceType::Rook as usize]
        | b.pieces[idx][crate::types::PieceType::Queen as usize])
        == 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TTFlag {
    Exact,
    Alpha,
    Beta,
}

#[derive(Clone, Copy)]
pub struct TTEntry {
    key: u64,
    depth: i32,
    // Los scores de mate se guardan normalizados respecto de la raiz. Al
    // recuperar la entrada se convierten de nuevo usando el ply actual.
    score: i32,
    flag: TTFlag,
    best: Option<Move>,
    // Generacion de la busqueda que escribio esta entrada (aging): se
    // incrementa al inicio de cada busqueda y permite que tt_store prefiera
    // reemplazar entradas de busquedas ANTERIORES aunque sean mas profundas.
    generation: u8,
}

#[inline]
fn score_to_tt(score: i32, ply: u32) -> i32 {
    if score >= MATE - 1000 {
        score + ply as i32
    } else if score <= -MATE + 1000 {
        score - ply as i32
    } else {
        score
    }
}

#[inline]
fn score_from_tt(score: i32, ply: u32) -> i32 {
    if score >= MATE - 1000 {
        score - ply as i32
    } else if score <= -MATE + 1000 {
        score + ply as i32
    } else {
        score
    }
}

/// Reserva interna adicional al `Move Overhead` de UCI.
///
/// En ultrabullet, 5 ms fijos sacrifican una fracción enorme del presupuesto.
/// Se mantiene una reserva conservadora de 3 ms para absorber planificación
/// del sistema y la salida UCI; la ganancia de profundidad debe venir de
/// rendimiento real, no de gastar el reloj de forma insegura.
fn margen_interno_tiempo(movetime_ms: u64) -> u64 {
    match movetime_ms {
        0..=2 => 0,
        3..=10 => 2,
        11..=25 => 3,
        26..=100 => 4,
        _ => 5,
    }
}

#[derive(Debug)]
pub struct TimeUp;

const MAX_KILLER_PLY: usize = 100; // margen sobre MAX_PLY para cubrir extensiones de jaque

// Centinela para eval_stack: "no hay eval estatica en este ply" (nodo en
// jaque). Ninguna evaluacion real puede valer i32::MIN.
const EVAL_INVALIDA: i32 = i32::MIN;

// Tabla LMR precalculada: reduccion base en plies para cada par
// (profundidad, numero de jugada en el orden). Formula logaritmica clasica
// (Ethereal/Stockfish): crece suave con ambas -- las jugadas muy tardias a
// buena profundidad se reducen varios plies, las primeras casi nada.
// Sobre esta base se aplican ajustes contextuales (PV, historia, improving).
fn tabla_lmr() -> &'static [[i32; 64]; 64] {
    static TABLA: OnceLock<[[i32; 64]; 64]> = OnceLock::new();
    TABLA.get_or_init(|| {
        let mut t = [[0i32; 64]; 64];
        for d in 1..64usize {
            for m in 1..64usize {
                t[d][m] = (0.75 + (d as f64).ln() * (m as f64).ln() / 2.25) as i32;
            }
        }
        t
    })
}

// ---------------------------------------------------------------------------
// Correction history: ajusta la eval estatica segun el error historico entre
// ella y el score real de la busqueda en posiciones "parecidas" (misma
// estructura de peones / mismo material no-peon / misma jugada previa).
// Tres tablas: pawn corrhist, non-pawn corrhist (una por color de piezas) y
// continuation corrhist (jugada rival previa). Valores en centipeones.
// ---------------------------------------------------------------------------
const CORR_SIZE: usize = 16384;
const CORR_MASK: usize = CORR_SIZE - 1;
const CORR_MAX: i32 = 128;

/// Hash Zobrist solo de los peones (estructura de peones).
fn hash_peones(b: &Board) -> u64 {
    let k = crate::zobrist::keys();
    let mut h = 0u64;
    for color in 0..2usize {
        let mut bb = b.pieces[color][PieceType::Pawn as usize];
        while bb != 0 {
            let sq = crate::bitboard::pop_lsb(&mut bb);
            h ^= k.piece_square[color][PieceType::Pawn as usize][sq as usize];
        }
    }
    h
}

/// Hash Zobrist de las piezas que NO son peones de un color dado.
fn hash_no_peones(b: &Board, color: usize) -> u64 {
    let k = crate::zobrist::keys();
    let mut h = 0u64;
    for pt in 1..6usize {
        let mut bb = b.pieces[color][pt];
        while bb != 0 {
            let sq = crate::bitboard::pop_lsb(&mut bb);
            h ^= k.piece_square[color][pt][sq as usize];
        }
    }
    h
}

/// Actualizacion con "gravedad" (Stockfish): el bonus crece con la
/// profundidad y con el error, y se amortigua a medida que la entrada se
/// aleja de cero -- acota la tabla sin resets bruscos.
fn corrhist_update(entry: &mut i32, diff: i32, depth: i32) {
    let bonus = (diff * depth / 8).clamp(-CORR_MAX, CORR_MAX);
    *entry += bonus - *entry * bonus.abs() / 512;
    *entry = (*entry).clamp(-CORR_MAX, CORR_MAX);
}

// 6 tipos de pieza x 64 casilleros, dos veces (jugada rival + jugada propia).
const CONT_HIST_SIZE: usize = 6 * 64 * 6 * 64;

#[inline]
fn cont_idx(prev_pt: usize, prev_to: usize, pt: usize, to: usize) -> usize {
    ((prev_pt * 64 + prev_to) * 6 + pt) * 64 + to
}

/// Si el rival ya ataca `sq` ANTES de jugar el movimiento (tablero en la
/// posicion actual). Usado para elegir la tabla de historial: una jugada
/// silenciosa hacia una casilla amenazada tiene un significado tactico
/// distinto de una hacia una casilla segura.
#[inline]
fn casilla_amenazada(b: &Board, sq: crate::types::Square) -> bool {
    b.is_square_attacked_by(sq, b.turn.opposite())
}

// TT compartida entre hilos (Lazy SMP): LOCKLESS de verdad, sin Mutex.
// Motivacion (analisis comparativo contra Reckless, motor top-3 CCRL): su TT
// son clusters lockless de pocos bytes por entrada; la nuestra era un
// Vec<Mutex<Option<TTEntry>>> -- un candado por casillero mas una entrada
// bastante mas grande que 8 bytes, lo que da menos entradas por MB, peor
// aprovechamiento de cache, y contencion real de Lazy SMP en cada sondeo.
//
// Reemplazo: cada entrada se empaqueta en un UNICO u64 (AtomicU64), leido y
// escrito con una sola instruccion atomica de hardware -- sin bloqueos, sin
// lecturas a medias posibles (un u64 se lee/escribe siempre entero en un
// procesador de 64 bits). Layout de los 64 bits:
//
//   bits  0..18  (18): jugada  -- from(6) | to(6) | flag(3) | promocion(3)
//   bits 18..34  (16): score, como i16 (ya normalizado a mate-por-ply)
//   bits 34..41  ( 7): profundidad (0..127, de sobra: nunca llegamos ahi)
//   bits 41..43  ( 2): flag de TT (Exact/Alpha/Beta)
//   bits 43..48  ( 5): generacion de la busqueda que escribio la entrada
//                       (aging: 0..31, se incrementa por busqueda)
//   bit  48      ( 1): "ocupado" -- distingue un casillero vacio de una
//                       entrada real con todos los demas bits en cero
//   bits 49..64  (15): verificacion de clave (parte alta del zobrist, en
//                       vez de guardar los 64 bits completos de la clave)
//
// Una colision de 15 bits de verificacion (1 en 32768) puede aceptar por
// error la entrada de OTRA posicion que cayo en el mismo casillero -- igual
// de "peligroso" que cualquier TT normal ante colisiones de indice, motivo
// por el cual toda esta info se usa solo para PODAR/orientar la busqueda,
// nunca como fuente de verdad de las reglas del juego.
type SharedTT = Vec<AtomicU64>;
type LocalTT = Vec<Option<TTEntry>>;

const TT_OCUPADO: u64 = 1 << 48;

fn tt_empaquetar_move(mv: Option<Move>) -> u64 {
    match mv {
        // from=to=0 (a1a1) nunca es una jugada real (origen=destino), asi
        // que 0 es un sentinel seguro para "sin jugada" sin ambiguedad.
        None => 0,
        Some(m) => {
            let from = m.from as u64 & 0x3F;
            let to = m.to as u64 & 0x3F;
            let flag = (m.flag as u64) & 0x7;
            let promo = match m.promotion {
                None => 0u64,
                Some(pt) => (pt as u64 + 1) & 0x7,
            };
            from | (to << 6) | (flag << 12) | (promo << 15)
        }
    }
}

fn tt_moveflag_desde_u64(v: u64) -> MoveFlag {
    match v {
        0 => MoveFlag::Quiet,
        1 => MoveFlag::Capture,
        2 => MoveFlag::DoublePush,
        3 => MoveFlag::EnPassant,
        4 => MoveFlag::CastleKing,
        _ => MoveFlag::CastleQueen,
    }
}

fn tt_piecetype_desde_u64(v: u64) -> PieceType {
    match v {
        0 => PieceType::Pawn,
        1 => PieceType::Knight,
        2 => PieceType::Bishop,
        3 => PieceType::Rook,
        4 => PieceType::Queen,
        _ => PieceType::King,
    }
}

fn tt_desempaquetar_move(paquete: u64) -> Option<Move> {
    if paquete == 0 {
        return None;
    }
    let from = (paquete & 0x3F) as u8;
    let to = ((paquete >> 6) & 0x3F) as u8;
    let flag = tt_moveflag_desde_u64((paquete >> 12) & 0x7);
    let promo_raw = (paquete >> 15) & 0x7;
    let promotion = if promo_raw == 0 {
        None
    } else {
        Some(tt_piecetype_desde_u64(promo_raw - 1))
    };
    Some(Move { from, to, promotion, flag })
}

fn tt_flag_desde_u64(v: u64) -> TTFlag {
    match v {
        0 => TTFlag::Exact,
        1 => TTFlag::Alpha,
        _ => TTFlag::Beta,
    }
}

fn tt_empaquetar(entry: &TTEntry, key: u64, generation: u8) -> u64 {
    let mv = tt_empaquetar_move(entry.best) & 0x3FFFF; // 18 bits
    let score = (entry.score as i16 as u16) as u64; // 16 bits, con signo preservado
    let depth = (entry.depth.clamp(0, 127) as u64) & 0x7F; // 7 bits
    let flag = (entry.flag as u64) & 0x3; // 2 bits
    let generacion = (generation as u64) & 0x1F; // 5 bits
    let verif = (key >> (64 - 15)) & 0x7FFF; // 15 bits altos de la clave real
    mv | (score << 18) | (depth << 34) | (flag << 41) | (generacion << 43) | TT_OCUPADO | (verif << 49)
}

fn tt_desempaquetar(paquete: u64, key: u64) -> Option<TTEntry> {
    if paquete & TT_OCUPADO == 0 {
        return None;
    }
    let verif_guardado = (paquete >> 49) & 0x7FFF;
    let verif_real = (key >> (64 - 15)) & 0x7FFF;
    if verif_guardado != verif_real {
        return None; // colision de indice -- otra posicion, se descarta
    }
    let mv = tt_desempaquetar_move(paquete & 0x3FFFF);
    let score = ((paquete >> 18) & 0xFFFF) as u16 as i16 as i32;
    let depth = ((paquete >> 34) & 0x7F) as i32;
    let flag = tt_flag_desde_u64((paquete >> 41) & 0x3);
    let generation = ((paquete >> 43) & 0x1F) as u8;
    Some(TTEntry { key, depth, score, flag, best: mv, generation })
}

/// Emite una instruccion de prefetch (hint al hardware) para la linea de
/// cache que contiene `p`. Es una optimizacion PURA de velocidad: no lee ni
/// escribe memoria logicamente, solo avisa a la CPU que traiga la linea
/// antes de que `tt_probe` la toque. No puede cambiar ningun resultado de
/// busqueda (mismos nodos, mismo score, mismo bestmove).
///
/// Arquitecturas cubiertas con API ESTABLE de Rust (rustc estable,
/// edition 2024), SIN std::intrinsics (que exigiria nightly):
/// - aarch64 (Apple Silicon M5, Android ARM64): inline assembly `prfm
///   pldl1keep` (estable desde Rust 1.59). La intrinsics nativa
///   `core::arch::aarch64::_prefetch` SIGUE SIN ESTABILIZAR en este
///   toolchain (requiere el feature nightly `stdarch_aarch64_prefetch`,
///   tracking issue #117217), asi que aqui se usa la forma estable.
/// - x86_64 (Windows, emulador Android x86_64): `_mm_prefetch`
///   (PREFETCHT0), intrinsics estable.
/// - Cualquier otra (p.ej. armv7): no-op seguro -- ni core::arch ni asm!
///   ofrecen un prefetch estable para ARM de 32 bits y no vale la pena
///   forzarlo.
#[inline(always)]
fn prefetch_ptr(p: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PLDL1KEEP: prefetch para lectura, mantener la linea en L1.
        // `asm!` es estable en Rust para aarch64 (desde 1.59); evita la
        // intrinsics `_prefetch` que sigue tras el feature nightly.
        core::arch::asm!("prfm pldl1keep, [{}]", in(reg) p);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // _MM_HINT_T0 = cachear en todos los niveles (equivalente a
        // pldl1keep de ARM). Es el mismo hint que usa Stockfish.
        core::arch::x86_64::_mm_prefetch(p.cast(), core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p; // no-op seguro: sin instruccion de prefetch estable disponible
}

/// Un solo hilo no necesita sincronización para su tabla de transposición.
/// Lazy SMP conserva el backend compartido con `Mutex` por casillero.
enum TablaTransposicion {
    Local(LocalTT),
    Compartida(Arc<SharedTT>),
}

fn capacidad_tt(tt_mb: usize, slot_size: usize) -> usize {
    let bytes = tt_mb.saturating_mul(1024 * 1024);
    let objetivo = (bytes / slot_size.max(1)).max(1);
    let mut n_entries = objetivo.next_power_of_two();
    if n_entries > objetivo {
        n_entries >>= 1;
    }
    n_entries.max(1)
}

pub fn construir_tt(tt_mb: usize) -> (Arc<SharedTT>, usize) {
    // Cada casillero es un AtomicU64 de 8 bytes exactos -- "Hash 64" da
    // 64MiB/8 = 8M entradas (redondeado a potencia de 2), varias veces mas
    // que con el TTEntry+Mutex anterior (que ocupaba bastante mas de 8
    // bytes por casillero).
    let slot_size = std::mem::size_of::<AtomicU64>();
    let n_entries = capacidad_tt(tt_mb, slot_size);
    let tt: SharedTT = (0..n_entries).map(|_| AtomicU64::new(0)).collect();
    (Arc::new(tt), n_entries - 1)
}

fn construir_tt_local(tt_mb: usize) -> (LocalTT, usize) {
    let n_entries = capacidad_tt(tt_mb, std::mem::size_of::<Option<TTEntry>>());
    (vec![None; n_entries], n_entries - 1)
}

pub fn limpiar_tt(tt: &SharedTT) {
    for slot in tt {
        slot.store(0, Ordering::Relaxed);
    }
}

pub struct Searcher {
    tt: TablaTransposicion,
    tt_mask: usize,
    pub nodes: u64,
    deadline: Option<Instant>,
    stop: bool,
    // Generacion de la busqueda actual para aging de la TT (0..31): se
    // incrementa al inicio de cada busqueda y las entradas de la generacion
    // anterior se prefieren para reemplazo en tt_store.
    tt_generation: u8,
    // killers son validos solo dentro de esta busqueda (por ply del arbol
    // actual); history SI persiste entre jugadas de la partida, igual que la TT.
    killers: Vec<[Option<Move>; 2]>,
    history: Box<[[i32; 64]; 64]>, // [from][to] -- arreglo plano, mas rapido que un HashMap aqui
    // Historial separado para jugadas cuya casilla DESTINO esta amenazada por
    // el rival en el momento de jugarlas (idea tomada de Quanticade Cronus,
    // top-10 CCRL: indexa su historial por amenazas en vez de una sola tabla
    // plana). Una jugada silenciosa que entra a una casilla atacada tiene un
    // "significado" distinto de una que no -- mezclarlas en la misma tabla
    // difumina la señal. Se consulta ANTES de jugar el movimiento (con el
    // tablero todavia en la posicion actual), asi que refleja si la jugada
    // "camina hacia el peligro", no si termino resultando bien o mal.
    history_amenaza: Box<[[i32; 64]; 64]>,
    // Continuation history ("counter-move history"): a diferencia de history
    // (que solo sabe "esta jugada [from][to] corto mucho, en general"), esta
    // tabla sabe "esta jugada [pieza][to] corto mucho DESPUES de que el
    // rival jugara [pieza][to]" -- captura respuestas tacticas especificas a
    // una jugada rival concreta (p.ej. recapturas, bloqueos de jaque) que el
    // history plano no distingue del resto. Indexada
    // [pieza_rival][casillero_rival][pieza_propia][casillero_propio],
    // aplanada en un Vec para evitar arrays anidados de tamano fijo.
    cont_history: Vec<i32>,
    // Continuation history de 2 plies ("follow-up"): igual que cont_history
    // pero indexada por la jugada PROPIA de 2 plies atras en vez de la
    // jugada rival inmediata -- captura patrones de plan (p.ej. "despues de
    // avanzar este peon, esta jugada de caballo corta mucho").
    cont_history_2: Vec<i32>,
    // Counter moves: la mejor respuesta silenciosa conocida contra cada
    // jugada rival (pieza, destino). Se prueba en el ordenamiento despues
    // de los killers.
    counter_moves: Vec<Option<Move>>,
    pub modo_lmr: bool,
    // Desactivable solo para comparacion A/B en pruebas (MIMOTOR_NO_ASPIRATION=1)
    // -- en juego real siempre queda activado, la tecnica en si es segura por
    // construccion (ensancha hasta ventana completa si hace falta).
    pub modo_aspiration: bool,
    // Singular extensions: activadas por defecto desde el h2h de 250
    // partidas a 20ms (55.0% para singular -- justo en el umbral, no un
    // margen amplio como LMP/history malus, pero cumple el criterio).
    // MIMOTOR_SINGULAR=0 las desactiva si hiciera falta comparar de nuevo.
    pub modo_singular: bool,
    // La quiescence puede omitir NNUE de forma experimental. El resto del
    // árbol conserva la mezcla completa; esta bandera solo existe para medir
    // si el coste del horizonte a relojes ultracortos devuelve Elo real.
    pub qsearch_nnue: bool,
    // Acumulador NNUE del hilo. NO viaja dentro de EvalState (copiarlo por
    // nodo hijo costaba 2-4 KB de memcpy por nodo): es un buffer mutable
    // propio de este Searcher que se actualiza in-place al bajar a un hijo
    // (`siguiente_estado_*`) y se deshace al volver (`salir_hijo`). Cada
    // Searcher pertenece a un solo hilo, asi que no hay contencion.
    nnue: Option<crate::neural::NnueAccumulator>,
    // Profundidad máxima en la que los hijos usan solo ClassicalAccumulator.
    // Cero conserva el comportamiento previo bit a bit. Es una puerta de
    // rendimiento experimental: evita construir deltas NNUE que ningún nodo
    // superficial llegará a consultar antes de entrar a quiescence clásica.
    pub nnue_classical_depth: i32,
    // Historial de repeticion: claves Zobrist de la PARTIDA REAL (persiste
    // entre llamadas a go, la maneja el loop UCI) + las de la linea actual
    // de busqueda (crece/decrece durante la recursion, como el "self.hist"
    // de Python). No se usa la TT para esto porque una entrada de TT no
    // sabe CUANTAS veces se visito esa posicion en esta partida especifica.
    game_history: Vec<u64>,
    // path: historial de zobrist (partida + rama de busqueda actual) para
    // deteccion de repeticion. Array fijo para evitar allocacion/indireccion
    // de heap por nodo. Tamano generoso (MAX_PATH) para cubrir partidas
    // largas + profundidad de busqueda; path_push nunca desborda (guarda).
    path: [u64; MAX_PATH],
    path_len: usize,
    pub lmr_intentos: u64,
    pub lmr_reintentos: u64,
    // Hindsight reductions: para el hijo alcanzado mediante una busqueda
    // reducida guardamos la evaluacion estatica del padre y la reduccion
    // aplicada. Los vectores estan indexados por ply y son locales al hilo.
    hindsight_parent_eval: Vec<i32>,
    hindsight_reduction: Vec<i32>,
    // Eval estatica cruda por ply (EVAL_INVALIDA si el nodo estaba en jaque).
    // Base de la heuristica "improving": comparar la eval de este nodo con
    // la de 2 plies atras en la MISMA linea (mismo bando a mover).
    eval_stack: Vec<i32>,
    // Correction history (ver constantes CORR_* arriba): [bando][hash].
    corr_pawn: Vec<i32>,
    corr_nonpawn: [Vec<i32>; 2],
    // Continuation corrhist: (pieza_rival, destino_rival, bando) aplanado.
    corr_cont: Vec<i32>,
    // Lazy SMP: si esta activo, este hilo intercambia las 2 primeras
    // jugadas del orden en la RAIZ (una vez, al armar el orden inicial) para
    // no explorar exactamente la misma linea primero que los demas hilos.
    pub variante_orden_raiz: bool,
    // UCI "searchmoves": si Some, la RAIZ solo explora estas jugadas (util
    // para repartir el arbol entre varias maquinas -- cada una busca un
    // subconjunto disjunto de jugadas raiz y se compara el mejor resultado
    // de cada lado). None = comportamiento normal, explora todas.
    pub root_moves_filtro: Option<Vec<Move>>,
    // Lazy SMP: variacion de PARAMETROS de busqueda entre hilos (no solo
    // orden de jugadas). Cada hilo helper explora el arbol con una
    // reduccion de null-move ligeramente distinta (R=2, 3 o 1 segun el
    // indice del hilo modulo 3) -- hilos con R mas chico podan menos y
    // llegan menos hondo pero mas exhaustivo; con R mas grande podan mas
    // agresivo y llegan mas hondo pero mas arriesgado. Al compartir la
    // misma TT, las lineas que un hilo descarta por error las puede
    // encontrar otro con distinta agresividad -- variacion real de
    // busqueda, no solo de que jugada se mira primero.
    pub null_move_r_extra: i32,
    // Bandera compartida con el hilo principal del loop UCI: permite que el
    // comando "stop" interrumpa una busqueda en curso (que corre en su
    // propio hilo -- ver uci_loop en main.rs) sin depender solo del deadline
    // de tiempo. Necesario para "go infinite" y para cumplir el protocolo
    // UCI que exigen los testers de listas de rating como CCRL.
    external_stop: Option<Arc<AtomicBool>>,
}

fn valor_pieza(pt: crate::types::PieceType) -> i32 {
    match pt {
        crate::types::PieceType::Pawn => 100,
        crate::types::PieceType::Knight => 320,
        crate::types::PieceType::Bishop => 330,
        crate::types::PieceType::Rook => 500,
        crate::types::PieceType::Queen => 900,
        crate::types::PieceType::King => 20000,
    }
}

impl Searcher {
    pub fn new(tt_mb: usize) -> Searcher {
        let (tt, tt_mask) = construir_tt_local(tt_mb);
        Searcher {
            tt: TablaTransposicion::Local(tt),
            tt_mask,
            nodes: 0,
            deadline: None,
            stop: false,
            tt_generation: 0,
            killers: vec![[None, None]; MAX_KILLER_PLY],
            history: Box::new([[0i32; 64]; 64]),
            history_amenaza: Box::new([[0i32; 64]; 64]),
            cont_history: vec![0i32; CONT_HIST_SIZE],
            cont_history_2: vec![0i32; CONT_HIST_SIZE],
            counter_moves: vec![None; 6 * 64],
            // Activado por defecto: el torneo h2h de esta sesion confirmo
            // +80 ELO (61.3% en 40 partidas) con la reescritura PVS -- ver
            // resultados_lmr_h2h.txt en ~/mi-motor. MIMOTOR_LMR=0 lo desactiva
            // explicitamente para pruebas comparativas.
            modo_lmr: std::env::var("MIMOTOR_LMR").as_deref() != Ok("0"),
            modo_aspiration: std::env::var("MIMOTOR_NO_ASPIRATION").as_deref() != Ok("1"),
            modo_singular: std::env::var("MIMOTOR_SINGULAR").as_deref() != Ok("0"),
            qsearch_nnue: true,
            nnue: None,
            nnue_classical_depth: 0,
            game_history: Vec::new(),
            path: [0u64; MAX_PATH],
            path_len: 0,
            lmr_intentos: 0,
            lmr_reintentos: 0,
            hindsight_parent_eval: vec![0; MAX_KILLER_PLY],
            hindsight_reduction: vec![0; MAX_KILLER_PLY],
            eval_stack: vec![EVAL_INVALIDA; MAX_KILLER_PLY],
            corr_pawn: vec![0; 2 * CORR_SIZE],
            corr_nonpawn: [vec![0; 2 * CORR_SIZE], vec![0; 2 * CORR_SIZE]],
            corr_cont: vec![0; 6 * 64 * 2],
            variante_orden_raiz: false,
            root_moves_filtro: None,
            null_move_r_extra: 0,
            external_stop: None,
        }
    }

    /// Crea un Searcher que comparte la TT (Arc clonado, mismo mask) de otro
    /// -- para Lazy SMP, donde varios hilos buscan sobre la misma tabla.
    /// Killers/history/game_history quedan LOCALES de este hilo (no tiene
    /// sentido compartirlos, cada hilo ordena sus propias jugadas).
    pub fn new_con_tt_compartida(tt: Arc<SharedTT>, tt_mask: usize, modo_lmr: bool) -> Searcher {
        Searcher {
            tt: TablaTransposicion::Compartida(tt),
            tt_mask,
            nodes: 0,
            deadline: None,
            stop: false,
            killers: vec![[None, None]; MAX_KILLER_PLY],
            history: Box::new([[0i32; 64]; 64]),
            history_amenaza: Box::new([[0i32; 64]; 64]),
            cont_history: vec![0i32; CONT_HIST_SIZE],
            cont_history_2: vec![0i32; CONT_HIST_SIZE],
            counter_moves: vec![None; 6 * 64],
            tt_generation: 0,
            modo_lmr,
            modo_aspiration: std::env::var("MIMOTOR_NO_ASPIRATION").as_deref() != Ok("1"),
            modo_singular: std::env::var("MIMOTOR_SINGULAR").as_deref() != Ok("0"),
            qsearch_nnue: true,
            nnue: None,
            nnue_classical_depth: 0,
            game_history: Vec::new(),
            path: [0u64; MAX_PATH],
            path_len: 0,
            lmr_intentos: 0,
            lmr_reintentos: 0,
            hindsight_parent_eval: vec![0; MAX_KILLER_PLY],
            hindsight_reduction: vec![0; MAX_KILLER_PLY],
            eval_stack: vec![EVAL_INVALIDA; MAX_KILLER_PLY],
            corr_pawn: vec![0; 2 * CORR_SIZE],
            corr_nonpawn: [vec![0; 2 * CORR_SIZE], vec![0; 2 * CORR_SIZE]],
            corr_cont: vec![0; 6 * 64 * 2],
            variante_orden_raiz: false,
            root_moves_filtro: None,
            null_move_r_extra: 0,
            external_stop: None,
        }
    }

    /// Fija (o quita) la bandera compartida de "stop" externo -- se llama
    /// antes de lanzar la busqueda en su propio hilo desde uci_loop.
    pub fn set_external_stop(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.external_stop = flag;
    }

    pub fn set_qsearch_nnue(&mut self, active: bool) {
        self.qsearch_nnue = active;
    }

    pub fn set_nnue_classical_depth(&mut self, depth: i32) {
        self.nnue_classical_depth = depth.clamp(0, 4);
    }

    /// Siembra la generacion de aging de la TT (camino Lazy SMP). Todos los
    /// hilos de una misma llamada a `buscar_lazy_smp` usan la MISMA
    /// generacion compartida, que avanza UNA vez por llamada -- sin esto cada
    /// Searcher arrancaba en 0 y `search_time` la subia a 1 siempre, dejando
    /// TODA busqueda SMP en generacion 1 y matando el aging de la TT
    /// compartida (que si persiste entre jugadas). El camino single-thread
    /// (Searcher persistente) no la necesita: `search_time` ya la incrementa
    /// en cada "go".
    pub fn set_tt_generacion(&mut self, generacion: u8) {
        self.tt_generation = generacion & 0x1F;
    }

    /// Reinicia el acumulador NNUE del hilo a la posicion `b`. Se llama en
    /// la raiz de cada iteracion de profundizacion: cuesta O(piezas) una vez
    /// por iteracion y garantiza que un `TimeUp` que desenrolle la recursion
    /// sin pasar por los `salir_hijo` no deje el buffer desincronizado.
    #[inline]
    fn reiniciar_nnue(&mut self, b: &Board) {
        self.nnue = crate::neural::crear_acumulador(b);
    }

    /// Acumulador a consultar para `eval_state`. Devuelve None si este nodo
    /// tiene la NNUE apagada: en ese caso el buffer del Searcher quedo
    /// congelado en un ancestro y no corresponde al tablero actual.
    #[inline]
    fn nnue_de(&self, eval_state: &EvalState) -> Option<&crate::neural::NnueAccumulator> {
        if eval_state.nnue_activo() {
            self.nnue.as_ref()
        } else {
            None
        }
    }

    #[inline]
    fn evaluar_completo(&self, b: &Board, eval_state: &EvalState) -> i32 {
        evaluate_with_state(b, eval_state, self.nnue_de(eval_state))
    }

    #[inline]
    fn evaluar_quiescence(&self, b: &Board, eval_state: &EvalState) -> i32 {
        if self.qsearch_nnue {
            self.evaluar_completo(b, eval_state)
        } else {
            evaluate_classical_with_state(b, eval_state)
        }
    }

    /// Baja al hijo: construye su EvalState y, si el hijo conserva la NNUE,
    /// APLICA el delta in-place sobre el acumulador del Searcher. Todo
    /// llamador DEBE emparejarlo con `salir_hijo` (mismos argumentos) antes
    /// de volver o de romper el bucle.
    #[inline]
    fn entrar_hijo(
        &mut self,
        hijo: EvalState,
        antes: &Board,
        despues: &Board,
    ) -> EvalState {
        // Si el hijo apaga la NNUE, no se toca el acumulador: nadie en ese
        // subarbol lo va a leer (la bandera se hereda apagada), y asi el
        // undo tampoco tiene nada que hacer.
        if hijo.nnue_activo() {
            if let Some(acum) = self.nnue.as_mut() {
                acum.aplicar_jugada(antes, despues);
            }
        }
        hijo
    }

    /// Deshace exactamente lo que hizo `entrar_hijo`, invirtiendo los
    /// argumentos (la actualizacion incremental es simetrica: aplicarla con
    /// antes/despues intercambiados es su operacion inversa).
    #[inline]
    fn salir_hijo(&mut self, hijo: &EvalState, antes: &Board, despues: &Board) {
        if hijo.nnue_activo() {
            if let Some(acum) = self.nnue.as_mut() {
                acum.aplicar_jugada(despues, antes);
            }
        }
    }

    #[inline]
    fn siguiente_estado_quiescence(
        &mut self,
        eval_state: &EvalState,
        antes: &Board,
        despues: &Board,
    ) -> EvalState {
        let hijo = if self.qsearch_nnue {
            eval_state.despues_de_jugada(antes, despues)
        } else {
            eval_state.despues_de_jugada_solo_clasica(antes, despues)
        };
        self.entrar_hijo(hijo, antes, despues)
    }

    #[inline]
    fn siguiente_estado_busqueda(
        &mut self,
        eval_state: &EvalState,
        antes: &Board,
        despues: &Board,
        profundidad_hijo: i32,
    ) -> EvalState {
        // No soltar NNUE si la jugada deja al rival en jaque: negamax le
        // concede una extensión, por lo que ya no es realmente un nodo
        // superficial. Esto conserva la táctica forzada en el borde.
        let hijo = if self.nnue_classical_depth > 0
            && profundidad_hijo <= self.nnue_classical_depth
            && !despues.in_check(despues.turn)
        {
            eval_state.despues_de_jugada_solo_clasica(antes, despues)
        } else {
            eval_state.despues_de_jugada(antes, despues)
        };
        self.entrar_hijo(hijo, antes, despues)
    }

    pub fn clear_hash(&mut self) {
        match &mut self.tt {
            TablaTransposicion::Local(tt) => {
                for slot in tt {
                    *slot = None;
                }
            }
            TablaTransposicion::Compartida(tt) => limpiar_tt(tt),
        }
    }

    /// Decae (no resetea de golpe) las tablas de history/continuation
    /// history al arrancar cada "go" real de la partida. Sin esto, la
    /// tabla solo se inicializa una vez (Searcher::new) y ACUMULA sin
    /// limite durante toda la partida -- estadisticas de la apertura
    /// (jugada 5) pueden seguir sesgando el ordenamiento en el medio
    /// juego o el final (jugada 40+), donde el tipo de posicion es
    /// completamente distinto. Dividir a la mitad (no resetear a cero)
    /// preserva la señal relativa de jugadas que siguen funcionando bien
    /// mientras deja que estadisticas viejas pesen cada vez menos.
    fn decaer_history(&mut self) {
        for fila in self.history.iter_mut() {
            for v in fila.iter_mut() {
                *v /= 2;
            }
        }
        for fila in self.history_amenaza.iter_mut() {
            for v in fila.iter_mut() {
                *v /= 2;
            }
        }
        for v in self.cont_history.iter_mut() {
            *v /= 2;
        }
        for v in self.cont_history_2.iter_mut() {
            *v /= 2;
        }
        for v in self.corr_pawn.iter_mut() {
            *v /= 2;
        }
        for tabla in self.corr_nonpawn.iter_mut() {
            for v in tabla.iter_mut() {
                *v /= 2;
            }
        }
        for v in self.corr_cont.iter_mut() {
            *v /= 2;
        }
    }

    /// Eval estatica corregida por correction history: promedia las tres
    /// senales (peones, no-peones de ambos colores, jugada previa) y las
    /// suma a la eval cruda. Se usa SOLO para decisiones de poda/reduccion
    /// (RFP, razoring, futility), nunca se guarda en la TT.
    fn eval_corregida(&self, b: &Board, static_eval: i32, prev: Option<(usize, usize)>) -> i32 {
        let stm = b.turn as usize;
        let base = stm * CORR_SIZE;
        let pawn = self.corr_pawn[base + (hash_peones(b) as usize & CORR_MASK)];
        let npw = self.corr_nonpawn[0]
            [base + (hash_no_peones(b, 0) as usize & CORR_MASK)];
        let npb = self.corr_nonpawn[1]
            [base + (hash_no_peones(b, 1) as usize & CORR_MASK)];
        let cont = match prev {
            Some((pt, to)) => self.corr_cont[(pt * 64 + to) * 2 + stm],
            None => 0,
        };
        static_eval + (pawn + (npw + npb) / 2 + cont / 2) / 2
    }

    /// Registra el error entre el score real de la busqueda y la eval
    /// estatica cruda en las tablas de correction history correspondientes.
    fn corrhist_registrar(
        &mut self,
        b: &Board,
        static_eval: i32,
        score_real: i32,
        depth: i32,
        prev: Option<(usize, usize)>,
    ) {
        let diff = score_real - static_eval;
        let stm = b.turn as usize;
        let base = stm * CORR_SIZE;
        let idx_pawn = base + (hash_peones(b) as usize & CORR_MASK);
        corrhist_update(&mut self.corr_pawn[idx_pawn], diff, depth);
        let idx_npw = base + (hash_no_peones(b, 0) as usize & CORR_MASK);
        corrhist_update(&mut self.corr_nonpawn[0][idx_npw], diff, depth);
        let idx_npb = base + (hash_no_peones(b, 1) as usize & CORR_MASK);
        corrhist_update(&mut self.corr_nonpawn[1][idx_npb], diff, depth);
        if let Some((pt, to)) = prev {
            let idx = (pt * 64 + to) * 2 + stm;
            corrhist_update(&mut self.corr_cont[idx], diff, depth);
        }
    }

    fn registrar_corte(
        &mut self,
        b: &Board,
        mv: Move,
        ply: u32,
        depth: i32,
        prev: Option<(usize, usize)>,
        prev2: Option<(usize, usize)>,
        pt_mv: usize,
    ) {
        if mv.is_capture() {
            return; // MVV-LVA/SEE ya ordenan las capturas primero, no necesitan refuerzo
        }
        let p = ply as usize;
        if p < MAX_KILLER_PLY {
            let k = &mut self.killers[p];
            if k[0] != Some(mv) {
                k[1] = k[0];
                k[0] = Some(mv);
            }
        }
        if casilla_amenazada(b, mv.to) {
            self.history_amenaza[mv.from as usize][mv.to as usize] += depth * depth;
        } else {
            self.history[mv.from as usize][mv.to as usize] += depth * depth;
        }
        if let Some((prev_pt, prev_to)) = prev {
            let idx = cont_idx(prev_pt, prev_to, pt_mv, mv.to as usize);
            self.cont_history[idx] += depth * depth;
            self.counter_moves[prev_pt * 64 + prev_to] = Some(mv);
        }
        if let Some((p2_pt, p2_to)) = prev2 {
            let idx = cont_idx(p2_pt, p2_to, pt_mv, mv.to as usize);
            self.cont_history_2[idx] += depth * depth;
        }
    }

    /// Fija el historial de claves Zobrist de la PARTIDA REAL hasta la
    /// posicion actual (lo arma el loop UCI a partir de "position ...
    /// moves ..."). Se llama antes de cada busqueda para que la deteccion
    /// de repeticion vea jugadas ya ocurridas en la partida, no solo las
    /// que aparezcan dentro del arbol de esta busqueda.
    pub fn set_game_history(&mut self, hist: Vec<u64>) {
        self.game_history = hist;
    }

    /// Reconstruye la linea principal (PV) caminando la TT desde `b`,
    /// siguiendo la mejor jugada guardada en cada posicion. Se corta por
    /// `max_len`, por no encontrar entrada en la TT, o por repeticion de
    /// zobrist (posible en ciclos/tablas) -- nunca deberia colgarse.
    /// Uso: solo para mostrar informacion (modo "simple", UCI "info pv"),
    /// no participa de la busqueda en si.
    pub fn extraer_pv(&self, b: &Board, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::with_capacity(max_len);
        let mut vistos = Vec::with_capacity(max_len);
        let mut actual = *b;
        for _ in 0..max_len {
            if vistos.contains(&actual.zobrist) {
                break;
            }
            vistos.push(actual.zobrist);
            let mv = match self.tt_probe(actual.zobrist).and_then(|e| e.best) {
                Some(mv) => mv,
                None => break,
            };
            if !generate_legal(&actual).contains(&mv) {
                break;
            }
            pv.push(mv);
            actual = actual.make_move(&mv);
        }
        pv
    }

    fn tt_index(&self, key: u64) -> usize {
        (key as usize) & self.tt_mask
    }

    /// Prefetch de la linea de cache de la TT para la clave `key`. Calcula
    /// la MISMA direccion que usara `tt_probe` (indice = key & tt_mask,
    /// sobre Local o Compartida) y emite el hint antes de que la lectura
    /// real ocurra, para que la linea ya este en cache cuando `tt_probe` la
    /// toque. Es solo un hint: no altera la memoria ni los resultados.
    #[inline(always)]
    fn tt_prefetch(&self, key: u64) {
        let idx = self.tt_index(key);
        let ptr: *const u8 = match &self.tt {
            TablaTransposicion::Local(tt) => tt.as_ptr().wrapping_add(idx).cast(),
            TablaTransposicion::Compartida(tt) => tt.as_ptr().wrapping_add(idx).cast(),
        };
        prefetch_ptr(ptr);
    }

    fn tt_probe(&self, key: u64) -> Option<TTEntry> {
        let idx = self.tt_index(key);
        match &self.tt {
            TablaTransposicion::Local(tt) => match tt[idx] {
                Some(e) if e.key == key => Some(e),
                _ => None,
            },
            // Lectura atomica de un solo u64 -- nunca puede leerse "a
            // medias" (un procesador de 64 bits lee/escribe un u64 alineado
            // en una sola operacion), asi que no hace falta ningun candado
            // para que la lectura sea segura entre hilos.
            TablaTransposicion::Compartida(tt) => {
                tt_desempaquetar(tt[idx].load(Ordering::Relaxed), key)
            }
        }
    }

    // Reemplazo por profundidad, pero una colision de OTRA clave siempre debe
    // poder ocupar el casillero. A igual profundidad se prefiere una entrada
    // Exact sobre una cota Alpha/Beta. Con AGING, una entrada de una busqueda
    // ANTERIOR (generacion distinta) se reemplaza de inmediato aunque sea mas
    // profunda: su informacion ya no corresponde a la busqueda en curso y
    // conservarla solo satura la TT con posiciones que ya no se visitaran.
    fn tt_store(
        &mut self,
        key: u64,
        depth: i32,
        score: i32,
        ply: u32,
        flag: TTFlag,
        best: Option<Move>,
    ) {
        // Copiar la generacion antes del closure: el closure se usa mientras
        // `self.tt` esta prestado mutablemente, asi que no puede capturar
        // `self` por referencia.
        let generacion = self.tt_generation;
        let reemplazar = |slot: Option<TTEntry>| match slot {
            None => true,
            Some(existing) if existing.key != key => true,
            Some(existing) if existing.generation != generacion => true,
            Some(existing) => {
                depth > existing.depth
                    || (depth == existing.depth
                        && flag == TTFlag::Exact
                        && existing.flag != TTFlag::Exact)
            }
        };
        let entry = TTEntry {
            key,
            depth,
            score: score_to_tt(score, ply),
            flag,
            best,
            generation: generacion,
        };
        let idx = self.tt_index(key);
        match &mut self.tt {
            TablaTransposicion::Local(tt) => {
                if reemplazar(tt[idx]) {
                    tt[idx] = Some(entry);
                }
            }
            TablaTransposicion::Compartida(tt) => {
                // Leer-decidir-escribir sin candado: una carrera aca en el
                // peor caso hace una decision de reemplazo subotima (dos
                // hilos escriben "a la vez" y uno pisa al otro), nunca un
                // dato incorrecto -- la lectura siempre verifica la clave
                // completa al leer, asi que un casillero mal reemplazado
                // simplemente se descarta despues como si fuera una
                // colision de otra posicion, igual que cualquier TT normal.
                let actual = tt_desempaquetar(tt[idx].load(Ordering::Relaxed), key);
                if reemplazar(actual) {
                    tt[idx].store(tt_empaquetar(&entry, key, generacion), Ordering::Relaxed);
                }
            }
        }
    }

    fn check_time(&mut self) -> Result<(), TimeUp> {
        self.nodes += 1;
        if !self.stop && (self.nodes == 1 || self.nodes & 255 == 0) {
            if let Some(dl) = self.deadline
                && Instant::now() >= dl
            {
                self.stop = true;
            }
            if !self.stop
                && let Some(flag) = &self.external_stop
                && flag.load(Ordering::Relaxed)
            {
                self.stop = true;
            }
        }
        if self.stop { Err(TimeUp) } else { Ok(()) }
    }

    fn order_moves(&self, b: &Board, moves: &mut [Move], tt_move: Option<Move>) {
        self.order_moves_ply(b, moves, tt_move, MAX_KILLER_PLY as u32, None, None);
    }

    #[inline]
    fn clave_orden_movimiento(
        &self,
        b: &Board,
        mv: &Move,
        tt_move: Option<Move>,
        ply: u32,
        prev: Option<(usize, usize)>,
        prev2: Option<(usize, usize)>,
        see_precalculado: Option<i32>,
    ) -> i32 {
        if Some(*mv) == tt_move {
            return -1_000_000;
        }
        if mv.is_capture() {
            let see = see_precalculado.unwrap_or_else(|| crate::see::see(b, mv));
            if see >= 0 {
                return -(10_000 + see);
            }
            return 1000 - see;
        }
        if mv.promotion.is_some() {
            return -5000;
        }
        let p = ply as usize;
        if p < MAX_KILLER_PLY {
            let killers = self.killers[p];
            if killers[0] == Some(*mv) {
                return -3000;
            }
            if killers[1] == Some(*mv) {
                return -2900;
            }
        }
        // Counter move: la mejor respuesta silenciosa conocida contra la
        // jugada rival que llevo a esta posicion -- se prueba justo despues
        // de los killers, antes de caer al history/continuation general.
        if let Some((prev_pt, prev_to)) = prev {
            if self.counter_moves[prev_pt * 64 + prev_to] == Some(*mv) {
                return -2800;
            }
        }
        let h = if casilla_amenazada(b, mv.to) {
            self.history_amenaza[mv.from as usize][mv.to as usize]
        } else {
            self.history[mv.from as usize][mv.to as usize]
        };
        let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
        let ch = match prev {
            Some((prev_pt, prev_to)) => {
                self.cont_history[cont_idx(prev_pt, prev_to, pt_mv, mv.to as usize)]
            }
            None => 0,
        };
        let ch2 = match prev2 {
            Some((p2_pt, p2_to)) => {
                self.cont_history_2[cont_idx(p2_pt, p2_to, pt_mv, mv.to as usize)]
            }
            None => 0,
        };
        -(h + ch + ch2)
    }

    /// Igual que order_moves pero ademas usa killers/history (por ply) para
    /// ordenar las jugadas silenciosas -- capturas/TT siguen mandando.
    /// `prev` es (pieza, casillero_destino) de la jugada rival que llevo a
    /// esta posicion (None si no se conoce, p.ej. en la raiz) -- alimenta la
    /// continuation history para las jugadas silenciosas. `prev2` es la
    /// jugada PROPIA de 2 plies atras -- alimenta el follow-up history.
    fn order_moves_ply(
        &self,
        b: &Board,
        moves: &mut [Move],
        tt_move: Option<Move>,
        ply: u32,
        prev: Option<(usize, usize)>,
        prev2: Option<(usize, usize)>,
    ) {
        // Cachea SEE una vez por captura. El ordenamiento especializado de
        // la biblioteca estándar gana al insertion-sort manual en M5, aun
        // contando su almacenamiento temporal; conservarlo da mejor NPS.
        moves.sort_by_cached_key(|mv| {
            self.clave_orden_movimiento(b, mv, tt_move, ply, prev, prev2, None)
        });
    }

    fn quiescence(
        &mut self,
        b: &Board,
        eval_state: &EvalState,
        mut alpha: i32,
        beta: i32,
        ply: u32,
    ) -> Result<i32, TimeUp> {
        self.check_time()?;
        // Quiescence también puede cruzar una secuencia de 50 plies sin
        // captura ni peón. No usar el stand-pat allí: por regla es tablas.
        if b.halfmove_clock >= 100 {
            return Ok(draw_score(b, eval_state, self.nnue_de(eval_state)));
        }
        let en_jaque = b.in_check(b.turn);

        // En jaque no existe "stand pat": quedarse quieto es ilegal. Se deben
        // buscar TODAS las evasiones legales, incluidas jugadas silenciosas.
        if en_jaque {
            let mut evasiones = generate_legal(b);
            if evasiones.is_empty() {
                return Ok(-MATE + ply as i32);
            }
            self.order_moves_ply(b, &mut evasiones, None, ply, None, None);
            if ply >= MAX_PLY {
                // Tope defensivo contra secuencias patologicas de jaques. Aun
                // detectamos mate arriba; para posiciones no terminales usamos
                // la evaluacion estatica en vez de desbordar la pila.
                return Ok(self.evaluar_quiescence(b, eval_state));
            }
            let mut best = -INFINITO;
            for mv in evasiones {
                let next = b.make_move(&mv);
                let next_eval = self.siguiente_estado_quiescence(eval_state, b, &next);
                // Ligar el resultado ANTES de deshacer, y recien despues
                // aplicar el `?`: asi ningun retorno temprano se salta el
                // undo del acumulador NNUE.
                let res = self.quiescence(&next, &next_eval, -beta, -alpha, ply + 1);
                self.salir_hijo(&next_eval, b, &next);
                let sc = -res?;
                best = best.max(sc);
                alpha = alpha.max(sc);
                if alpha >= beta {
                    break;
                }
            }
            return Ok(best);
        }

        let stand_pat = self.evaluar_quiescence(b, eval_state);
        if ply >= MAX_PLY {
            return Ok(stand_pat);
        }
        if stand_pat >= beta {
            return Ok(beta);
        }
        alpha = alpha.max(stand_pat);

        // generate_captures_legal ya filtra a capturas/promociones sin
        // generar ni legalizar las jugadas silenciosas (que aqui se
        // descartarian de todos modos) -- evita el trabajo mas caro de
        // legalizar jugadas que nunca se van a buscar. Si NO hay capturas
        // legales, con eso solo no se puede distinguir "posicion tranquila
        // normal" de "ahogado real": para ese caso, y SOLO ese, se paga el
        // generate_legal completo para detectar el ahogado correctamente.
        let capturas = generate_captures_legal(b);
        if capturas.is_empty() && generate_legal(b).is_empty() {
            return Ok(0); // ahogado
        }
        // Igual que la lista principal: quiescence vive en las hojas y no
        // debe asignar un Vec por nodo solo para conservar el SEE calculado.
        let mut moves: ArrayVec<(Move, Option<i32>), MAX_MOVES> = ArrayVec::new();
        for mv in capturas {
            let see = mv.is_capture().then(|| crate::see::see(b, &mv));
            moves.push((mv, see));
        }
        // En quiescence la poda SEE se aplica despues del ordenamiento. Llevar
        // el resultado junto a la jugada evita calcular el mismo SEE dos veces.
        moves.sort_by_key(|(mv, see)| self.clave_orden_movimiento(b, mv, None, ply, None, None, *see));

        let mut best = stand_pat;
        for (mv, see) in moves {
            let next = b.make_move(&mv);
            let da_jaque = next.in_check(next.turn);

            // Nunca podar promociones ni jaques por SEE/delta: una captura
            // materialmente mala puede ser mate o forzar una secuencia tactica.
            if !da_jaque && mv.promotion.is_none() && see.unwrap_or(0) < -50 {
                continue;
            }
            let victim = if mv.flag == MoveFlag::EnPassant {
                100
            } else {
                b.piece_at(mv.to)
                    .map(|(_, pt)| valor_pieza(pt))
                    .unwrap_or(0)
            };
            let promo_gain = mv.promotion.map(|pt| valor_pieza(pt) - 100).unwrap_or(0);
            if !da_jaque && stand_pat + victim + promo_gain + 250 <= alpha {
                continue;
            }

            let next_eval = self.siguiente_estado_quiescence(eval_state, b, &next);
            let res = self.quiescence(&next, &next_eval, -beta, -alpha, ply + 1);
            self.salir_hijo(&next_eval, b, &next);
            let sc = -res?;
            best = best.max(sc);
            alpha = alpha.max(sc);
            if alpha >= beta {
                break;
            }
        }
        Ok(best)
    }

    // `en_sondeo_se`: true si este nodo ya es descendiente de una busqueda
    // de VERIFICACION de singular extensions. Critico: sin este freno, cada
    // nodo dentro de esa verificacion podria a su vez lanzar su PROPIA
    // verificacion (y esos, la suya), multiplicando el trabajo en cadena en
    // vez de sumarlo -- confirmado en la practica (una posicion tardo mas de
    // 9 minutos en profundidad fija 9 antes de este freno). Una vez que un
    // nodo entra en modo verificacion, TODOS sus descendientes lo heredan
    // (se propaga, no se resetea a cada paso) y ninguno intenta su propia
    // singular extension.
    #[allow(clippy::collapsible_if, clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        b: &Board,
        eval_state: &EvalState,
        mut depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: u32,
        prev: Option<(usize, usize)>,
        prev2: Option<(usize, usize)>,
        en_sondeo_se: bool,
    ) -> Result<i32, TimeUp> {
        // Prefetch anticipado del probe de TT de este nodo: la lectura real
        // ocurre varias decenas de instrucciones mas abajo (check_time,
        // deteccion de repeticion, probe de syzygy, extensiones, hindsight),
        // tiempo de sobra para traer la linea de cache mientras tanto. Es
        // un hint puro al hardware: no cambia ningun resultado de busqueda.
        self.tt_prefetch(b.zobrist);
        self.check_time()?;

        if b.halfmove_clock >= 100 {
            return Ok(draw_score(b, eval_state, self.nnue_de(eval_state)));
        }

        // Repeticion: si esta posicion ya aparecio entre los ancestros
        // (partida real + linea de busqueda actual) dentro de la ventana de
        // jugadas reversibles (halfmove_clock), tratarla como tablas -- asi
        // el motor las evita activamente cuando esta mejor y las busca
        // activamente cuando esta peor, en vez de solo "no perder por
        // descuido". No hace falta esperar la 3ra ocurrencia real: ver la
        // 2da dentro del arbol ya significa que la repeticion esta
        // disponible como opcion, que es lo que le interesa a la busqueda.
        let hc = b.halfmove_clock as usize;
        if hc > 0 {
            let start = self.path_len.saturating_sub(hc);
            if self.path[start..self.path_len].contains(&b.zobrist) {
                return Ok(draw_score(b, eval_state, self.nnue_de(eval_state)));
            }
        }

        // Tabla de finales: si la posicion ya esta cubierta (pocas piezas),
        // el WDL es un resultado EXACTO -- ganada/tablas/perdida bajo juego
        // perfecto, tratado como "despues de una jugada que reinicia la
        // regla de 50" (probe_wdl_after_zeroing), la forma segura de usarlo
        // dentro del arbol de busqueda. No se prueba en la raiz (eso lo
        // maneja search_time/search_fixed_depth aparte, via DTZ, para elegir
        // la jugada que progresa de verdad, no solo el resultado).
        if ply > 0
            && let Some(wdl) = crate::syzygy::probe_wdl(b)
        {
            return Ok(wdl);
        }

        let en_jaque = b.in_check(b.turn);
        if en_jaque && ply < 40 {
            depth += 1; // extension de jaque
        }

        // Hindsight, RFP, futility y LMR pueden pedir la misma evaluacion
        // estatica del MISMO nodo. La personalidad y el acumulador quedan
        // fijos durante una busqueda, asi que memoizarla localmente es
        // exactamente equivalente y evita repetir una mezcla NNUE+clasica.
        let mut static_eval_cache: Option<i32> = None;

        // Hindsight reductions, adaptado al LMR entero de MiMotor. Si una
        // jugada reducida deja una posicion peor de lo que sugeria la eval
        // del padre, recuperamos el ply perdido. Si la posicion mejora con
        // claridad, aceptamos un ply menos. Solo actua sobre hijos que
        // realmente llegaron mediante LMR; no cambia nodos PV normales.
        let p = ply as usize;
        if !en_jaque && p > 0 && p < MAX_KILLER_PLY && self.hindsight_reduction[p] > 0 {
            let eval_actual =
                *static_eval_cache.get_or_insert_with(|| self.evaluar_completo(b, eval_state));
            let eval_delta = eval_actual + self.hindsight_parent_eval[p - 1];
            if eval_delta < 0 {
                depth += 1;
            } else if depth >= 2 && eval_delta > 57 {
                depth -= 1;
            }
        }

        if depth <= 0 || ply >= MAX_PLY {
            return self.quiescence(b, eval_state, alpha, beta, ply);
        }

        let alpha_orig = alpha;
        let key = b.zobrist;
        let mut tt_move = None;
        let mut tt_entry_full: Option<TTEntry> = None;
        if let Some(mut entry) = self.tt_probe(key) {
            entry.score = score_from_tt(entry.score, ply);
            tt_move = entry.best;
            tt_entry_full = Some(entry);
            if entry.depth >= depth {
                match entry.flag {
                    TTFlag::Exact => return Ok(entry.score),
                    TTFlag::Beta if entry.score >= beta => return Ok(entry.score),
                    TTFlag::Alpha if entry.score <= alpha => return Ok(entry.score),
                    _ => {}
                }
            }
        }

        // Poda de futilidad inversa (reverse futility / static null move): si
        // la evaluacion estatica ya supera a beta por un margen que crece con
        // la profundidad, es muy improbable que la busqueda real encuentre
        // algo peor -- se poda sin generar jugadas. Solo a poca profundidad
        // (el margen se vuelve prohibitivo mas alla) y lejos de puntajes de
        // mate (la eval estatica no es fiable para distinguirlos).
        // Un nodo es PV cuando la ventana no es nula (beta > alfa+1). Se
        // define aca (antes de RFP) porque tanto razoring como la poda LMP
        // mas abajo la necesitan.
        let es_pv = beta - alpha > 1;

        // Mate distance pruning: un mate no puede tardar menos de `ply` plies
        // en aparecer desde la raiz (los plies ya caminados), asi que ninguna
        // ventana mas alla de ese rango es alcanzable en esta rama. Ajustar
        // la ventana a ese rango y cortar si queda vacia. 100% seguro: no
        // cambia el resultado de ningun nodo, solo poda ramas sin esperanza.
        if !es_pv {
            let mated_in = -MATE + ply as i32; // peor mate posible en este ply
            let mate_in = MATE - ply as i32 - 1; // mejor mate posible en este ply
            if alpha < mated_in {
                alpha = mated_in;
            }
            if beta > mate_in {
                beta = mate_in;
            }
            if alpha >= beta {
                return Ok(alpha);
            }
        }

        // Eval estatica por ply para la heuristica "improving": si la eval
        // estatica de este nodo es mejor que la de 2 plies atras EN LA
        // MISMA linea (mismo bando a mover), la posicion viene mejorando y
        // se puede podar mas agresivo (RFP/razoring/futility/LMP/LMR); si
        // no mejora, se poda mas conservador. En jaque no hay eval estatica
        // fiable: se guarda el centinela y se hereda "no mejora" (mas
        // conservador por seguridad).
        let p = ply as usize;
        if p < MAX_KILLER_PLY {
            self.eval_stack[p] = if en_jaque {
                EVAL_INVALIDA
            } else {
                *static_eval_cache.get_or_insert_with(|| self.evaluar_completo(b, eval_state))
            };
        }
        let improving = !en_jaque && p >= 2 && p < MAX_KILLER_PLY && {
            let anterior = self.eval_stack[p - 2];
            anterior != EVAL_INVALIDA && self.eval_stack[p] > anterior
        };

        const RFP_PROF_MAX: i32 = 8;
        const RFP_MARGEN_POR_PLY: i32 = 120;
        if !en_jaque && !es_pv && depth <= RFP_PROF_MAX && beta.abs() < MATE - 1000 {
            let raw =
                *static_eval_cache.get_or_insert_with(|| self.evaluar_completo(b, eval_state));
            let static_eval = self.eval_corregida(b, raw, prev);
            // improving: con mejora el margen se achica (poda mas agresivo);
            // sin mejora se agranda (mas conservador, poda menos).
            let margen_ply = if improving { RFP_MARGEN_POR_PLY * 3 / 5 } else { RFP_MARGEN_POR_PLY };
            if static_eval - margen_ply * depth >= beta {
                return Ok(static_eval - margen_ply * depth);
            }
        }

        // Razoring: la contraparte de RFP pero contra ALFA en vez de BETA.
        // A profundidad muy baja y fuera de la linea principal, si la eval
        // estatica esta tan por debajo de alfa que ni con un margen
        // generoso la alcanza, se verifica con quiescence antes de
        // recortar. h2h a 20ms/250 partidas: 53.8% -- AMBIGUO (no llego al
        // umbral de 55% que se usa normalmente), desplegado igual por
        // decision explicita del usuario, no por criterio tecnico.
        const RAZOR_PROF_MAX: i32 = 2;
        const RAZOR_MARGEN_BASE: i32 = 200;
        const RAZOR_MARGEN_POR_PLY: i32 = 180;
        if !en_jaque && !es_pv && depth <= RAZOR_PROF_MAX && alpha.abs() < MATE - 1000 {
            let raw =
                *static_eval_cache.get_or_insert_with(|| self.evaluar_completo(b, eval_state));
            let static_eval = self.eval_corregida(b, raw, prev);
            // improving: con mejora el margen se achica; sin mejora se agranda
            // (mas conservador, se razorea menos).
            let margen_base = if improving { RAZOR_MARGEN_BASE * 3 / 5 } else { RAZOR_MARGEN_BASE };
            let margen = margen_base + RAZOR_MARGEN_POR_PLY * depth;
            if static_eval + margen <= alpha {
                let sc = self.quiescence(b, eval_state, alpha, alpha + 1, ply)?;
                if sc <= alpha {
                    return Ok(sc);
                }
            }
        }

        // Null-move pruning: si "pasar el turno" y aun asi el rival no supera
        // beta, la posicion ya es tan buena que se poda sin generar jugadas
        // reales. Desactivado en jaque, en finales de solo peones (riesgo de
        // zugzwang) y cerca de puntajes de mate (poco fiable ahi).
        // NMP "guiado por evaluacion" (concepto tomado del analisis de
        // Obsidian ~3636 CCRL y Quanticade Cronus ~3624 CCRL; reimplementado
        // desde cero en Rust, no es codigo portado).
        //
        // Diferencia de fondo con la version anterior de MiMotor: antes se
        // intentaba el sondeo nulo en TODO nodo con depth>=3 fuera de jaque,
        // sin mirar la evaluacion estatica. Los dos motores de referencia
        // exigen ANTES que la evaluacion ya este por encima de beta: si la
        // posicion ni siquiera "parece" ganadora, el sondeo nulo casi nunca
        // corta y solo se gastan nodos. Esa condicion es una RESTRICCION
        // (poda menos veces, nunca mas), y es justamente lo que permite que
        // la reduccion R sea mucho mayor sin volverse insegura.
        //
        //   Obsidian:   R = min((eval-beta)/147, 4) + depth/3 + 4 + ttMoveNoisy
        //   Quanticade: R = depth/3 + 6, acotado a depth
        //   MiMotor previo: R = 2 (+1 en depth>=6, +1 en depth>=12) -> tope 4
        //
        // Aca se adopta la MISMA FORMA (base + termino por profundidad +
        // termino proporcional a cuanto la eval supera beta) pero con
        // constantes bastante mas conservadoras que las de ambos motores,
        // porque nuestra eval no es NNUE de su calidad:
        //   R = 3 + depth/4 + min((eval-beta)/200, 2)
        // acotado a [2, depth-1] para no caer nunca directo en quiescence.
        //
        // Ademas el retorno pasa a ser fail-soft (se devuelve el puntaje real
        // del sondeo, no `beta`): da cotas mas informativas a la TT y al
        // padre. Se sigue devolviendo `beta` si el sondeo trae un puntaje de
        // mate, porque un mate "descubierto" pasando el turno no es fiable.
        const NULL_MOVE_R_BASE: i32 = 3;
        const NULL_MOVE_DEPTH_DIV: i32 = 4;
        const NULL_MOVE_EVAL_DIV: i32 = 200;
        const NULL_MOVE_EVAL_MAX: i32 = 2;
        const NULL_MOVE_PROF_MIN: i32 = 3;
        if !en_jaque
            && !es_pv
            && depth >= NULL_MOVE_PROF_MIN
            && beta < MATE - 1000
            && alpha > -(MATE - 1000)
            && !solo_peones_y_rey(b, b.turn)
        {
            let raw =
                *static_eval_cache.get_or_insert_with(|| self.evaluar_completo(b, eval_state));
            let eval_nmp = self.eval_corregida(b, raw, prev);
            if eval_nmp >= beta {
                let r_eval = ((eval_nmp - beta) / NULL_MOVE_EVAL_DIV).min(NULL_MOVE_EVAL_MAX);
                let r_adaptativo = NULL_MOVE_R_BASE + depth / NULL_MOVE_DEPTH_DIV + r_eval;
                let r = (r_adaptativo + self.null_move_r_extra).clamp(2, (depth - 1).max(2));
                let r = r.min(depth - 1).max(1);
                let next = b.make_null_move();
                let next_eval = self.siguiente_estado_busqueda(eval_state, b, &next, depth - 1 - r);
                // El hijo NO llega por LMR: limpiar el estado hindsight del
                // ply hijo para que no lea la reduccion de otro subarbol.
                let child_ply = (ply + 1) as usize;
                if child_ply < MAX_KILLER_PLY {
                    self.hindsight_reduction[child_ply] = 0;
                }
                let res_null = self.negamax(
                    &next,
                    &next_eval,
                    depth - 1 - r,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    None,
                    None,
                    en_sondeo_se,
                );
                self.salir_hijo(&next_eval, b, &next);
                let sc_null = -res_null?;
                if sc_null >= beta {
                    return Ok(if sc_null >= MATE - 1000 { beta } else { sc_null });
                }
            }
        }

        // Probcut (generalizacion de multicut): si una captura ya prometedora
        // por SEE supera un umbral bien por encima de beta incluso con una
        // busqueda reducida (primero quiescence barata, despues negamax
        // reducido para confirmar), es muy probable que la busqueda completa
        // tambien corte por beta -- se corta directo sin explorar el resto
        // del arbol de este nodo. Conservador: solo fuera de PV, a
        // profundidad suficiente para que la busqueda reducida siga siendo
        // significativa, y lejos de puntajes de mate.
        const PROBCUT_PROF_MIN: i32 = 5;
        const PROBCUT_MARGEN: i32 = 150;
        if !en_jaque && !es_pv && depth >= PROBCUT_PROF_MIN && beta.abs() < MATE - 1000 {
            let probcut_beta = beta + PROBCUT_MARGEN;
            // Cachea SEE una vez por captura: el ordenamiento y el filtro
            // see<0 del loop reusan el mismo valor en vez de recalcularlo.
            let mut capturas: ArrayVec<(Move, i32), MAX_MOVES> = ArrayVec::new();
            for mv in generate_captures_legal(b) {
                let see = crate::see::see(b, &mv);
                capturas.push((mv, see));
            }
            capturas.sort_by_key(|(_, see)| -*see);
            let sdepth = (depth - 4).max(1);
            let mut probcut_corto = false;
            let mut probcut_score = 0;
            for (mv, see) in &capturas {
                if *see < 0 {
                    continue;
                }
                let next = b.make_move(mv);
                if next.in_check(next.turn) {
                    continue;
                }
                let next_eval = self.siguiente_estado_busqueda(eval_state, b, &next, sdepth);
                let res_rapido =
                    self.quiescence(&next, &next_eval, -probcut_beta, -probcut_beta + 1, ply + 1);
                let sc_rapido = match res_rapido {
                    Ok(v) => -v,
                    Err(e) => {
                        self.salir_hijo(&next_eval, b, &next);
                        return Err(e);
                    }
                };
                if sc_rapido < probcut_beta {
                    // `continue` tambien sale del hijo: hay que deshacer.
                    self.salir_hijo(&next_eval, b, &next);
                    continue;
                }
                let pt_mv2 = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                // El hijo NO llega por LMR: limpiar el estado hindsight del
                // ply hijo para que no lea la reduccion de otro subarbol.
                let child_ply = (ply + 1) as usize;
                if child_ply < MAX_KILLER_PLY {
                    self.hindsight_reduction[child_ply] = 0;
                }
                let res_confirmado = self.negamax(
                    &next,
                    &next_eval,
                    sdepth,
                    -probcut_beta,
                    -probcut_beta + 1,
                    ply + 1,
                    Some((pt_mv2, mv.to as usize)),
                    prev,
                    en_sondeo_se,
                );
                self.salir_hijo(&next_eval, b, &next);
                let sc_confirmado = -res_confirmado?;
                if sc_confirmado >= probcut_beta {
                    probcut_corto = true;
                    probcut_score = sc_confirmado;
                    break;
                }
            }
            if probcut_corto {
                return Ok(probcut_score);
            }
        }

        // Internal Iterative Reduction (IIR): si no hay jugada de la TT en
        // este nodo (nunca se completo una busqueda aqui a esta profundidad
        // o mayor), no hay ninguna pista de cual jugada probar primero -- el
        // orden de jugadas sera peor y la busqueda completa a profundidad
        // real es menos eficiente. Se reduce 1 ply antes de generar/ordenar
        // jugadas: la busqueda reducida suele completar una entrada de TT
        // (con su propia mejor jugada) que despues SI ordena bien la
        // busqueda real. No aplica en jaque (la extension de jaque ya
        // gestiona la profundidad ahi) ni a profundidad baja (el ahorro no
        // compensa el costo de una pasada extra).
        const IIR_PROF_MIN: i32 = 4;
        if !en_jaque && tt_move.is_none() && depth >= IIR_PROF_MIN {
            depth -= 1;
        }

        // Cachea SEE una vez por captura (mismo patron que quiescence): el
        // ordenamiento y el SEE-prune del loop reusan el mismo valor en vez
        // de calcular see() dos veces por la misma jugada.
        let mut moves: ArrayVec<(Move, Option<i32>), MAX_MOVES> = ArrayVec::new();
        for mv in generate_legal(b) {
            let see = mv.is_capture().then(|| crate::see::see(b, &mv));
            moves.push((mv, see));
        }
        if moves.is_empty() {
            return Ok(if en_jaque { -MATE + ply as i32 } else { 0 });
        }
        moves.sort_by_key(|(mv, see)| {
            self.clave_orden_movimiento(b, mv, tt_move, ply, prev, prev2, *see)
        });
        if self.path_len < MAX_PATH { self.path[self.path_len] = b.zobrist; self.path_len += 1; }

        // Singular extensions: si la jugada de la TT es tan claramente
        // superior a TODAS las demas que ninguna otra logra siquiera
        // acercarse a su puntaje (verificado con una busqueda reducida,
        // ventana nula, sobre el RESTO de las jugadas), esa jugada es
        // "singular" -- la unica opcion real en la posicion -- y merece 1
        // ply extra de profundidad real en vez de recortarse igual que
        // cualquier otra. Apagado por defecto (modo_singular), ver comentario
        // en la definicion del campo.
        // v12: encontrados DOS desvios del algoritmo estandar que explican
        // la explosion medida en v11 (no era la tecnica, era la condicion de
        // activacion demasiado permisiva):
        //  1) Aceptaba entradas TT con flag Exact ademas de Beta. Beta
        //     (fail-high real, cota inferior) es la UNICA que tiene sentido
        //     para esta prueba -- es la que dice "esta jugada ya demostro
        //     ser >= beta". Exact son nodos PV normales (alpha<score<beta),
        //     mucho mas frecuentes que los Beta, y probarlos multiplicaba la
        //     cantidad de nodos que disparaban la sonda por todo el arbol.
        //  2) No excluia jaque: en posiciones con jaque el numero de
        //     respuestas legales suele ser bajo (casi cualquier jugada
        //     "parece" singular) y la extension de jaque ya existente puede
        //     encadenarse con la sonda de verificacion, multiplicando el
        //     costo sin aportar nada (la extension de jaque ya cubre ese caso).
        const SE_PROF_MIN: i32 = 8;
        let mut jugada_singular: Option<Move> = None;
        if self.modo_singular && !en_sondeo_se && !en_jaque && ply > 0 && depth >= SE_PROF_MIN {
            if let (Some(entry), Some(tmv)) = (tt_entry_full, tt_move) {
                if entry.depth >= depth - 3
                    && entry.flag == TTFlag::Beta
                    && entry.score.abs() < MATE - 1000
                    && moves.iter().any(|(m, _)| *m == tmv)
                {
                    let margen = 2 * depth;
                    let sbeta = entry.score - margen;
                    let sdepth = (depth - 1) / 2;
                    let mut mejor_otra = -INFINITO;
                    let mut se_timed_out = false;
                    for (mv, _see_se) in &moves {
                        if *mv == tmv {
                            continue;
                        }
                        let next = b.make_move(mv);
                        let next_eval =
                            self.siguiente_estado_busqueda(eval_state, b, &next, sdepth);
                        // El hijo NO llega por LMR: limpiar el estado hindsight
                        // del ply hijo para que no lea la reduccion de otro
                        // subarbol.
                        let child_ply = (ply + 1) as usize;
                        if child_ply < MAX_KILLER_PLY {
                            self.hindsight_reduction[child_ply] = 0;
                        }
                        let res_se = self.negamax(
                            &next,
                            &next_eval,
                            sdepth,
                            -sbeta,
                            -sbeta + 1,
                            ply + 1,
                            None,
                            None,
                            true,
                        );
                        // Deshacer ANTES del match: los dos brazos pueden
                        // romper el bucle.
                        self.salir_hijo(&next_eval, b, &next);
                        match res_se {
                            Ok(v) => {
                                let sc = -v;
                                if sc > mejor_otra {
                                    mejor_otra = sc;
                                }
                                if mejor_otra >= sbeta {
                                    break; // otra jugada ya alcanza la ventana: no es singular
                                }
                            }
                            Err(_) => {
                                se_timed_out = true;
                                break;
                            }
                        }
                    }
                    if !se_timed_out && mejor_otra < sbeta {
                        jugada_singular = Some(tmv);
                    }
                }
            }
        }

        // Mas conservador que en Python (que reducia desde la jugada #3 a
        // partir de profundidad 3): con SEE+killers el motor en Rust ya
        // llega mucho mas hondo que Python en el mismo segundo de reloj
        // (nps 100-500x mayor), asi que "ganar una ply mas" vale bastante
        // menos y el riesgo de descartar una jugada buena en la
        // verificacion reducida pesa mas. Medido: la version "Python-like"
        // (desde jugada 3, prof>=3, reduccion de hasta 2 ply) le costo
        // ~320 ELO en el torneo de referencia (18 partidas) pese a haber
        // dado bien en un mini-torneo de 4 partidas -- reducir desde mas
        // tarde en el orden, a mas profundidad, y nunca mas de 1 ply.
        // CANDIDATO cand_lmr_move2 (h2h ambiguo, 51.00% +/- 2.40%, desplegado
        // por decision explicita del usuario pese a no llegar al 55%): se
        // baja el umbral de 5 a 2. Con idx >= 2 las dos primeras jugadas del
        // orden (la de la TT y la segunda mejor) siguen buscandose a
        // profundidad completa, y la reduccion empieza en la TERCERA jugada
        // del orden. El monto NO se toca: la tabla logaritmica ya es
        // auto-limitante en las jugadas recien incluidas (m = idx+1 = 3..5 da
        // 1 ply hasta prof 6-8, 2 plies hasta prof ~24 y 3 plies solo mas
        // alla). Siguen aplicandose los ajustes contextuales (-1 en PV, -1
        // con historia positiva, +1 sin improving) y el clamp a [1, depth-2].
        const LMR_MOVES_SIN_REDUCIR: usize = 2;
        const LMR_PROF_MIN: i32 = 3;

        // Futility pruning (frontera): cerca de las hojas, si la evaluacion
        // estatica del nodo mas un margen que crece con la profundidad
        // sigue sin alcanzar alfa, una jugada silenciosa individual
        // (no captura, no promocion, no jaque propio) casi nunca va a
        // remontar eso -- se descarta sin buscarla. Distinto de la poda de
        // futilidad inversa (que corta el NODO completo contra beta): esta
        // poda jugadas UNA POR UNA contra alfa, y solo si ya hay al menos
        // una jugada evaluada (nunca deja el nodo sin ninguna busqueda).
        const FUT_PROF_MAX: i32 = 4;
        const FUT_MARGEN_BASE: i32 = 150;
        const FUT_MARGEN_POR_PLY: i32 = 100;
        let mut fut_eval: Option<i32> = None;

        let mut best_score = -INFINITO;
        let mut best_move = None;
        // Jugadas silenciosas realmente BUSCADAS en este nodo (no las podadas
        // por LMP/futilidad). Si una jugada posterior causa un corte beta,
        // estas se probaron y fallaron: reciben un "malus" de history para que
        // el orden aprenda a probarlas mas tarde. Cota fija en la pila.
        let mut quiets_buscados: [(u8, u8, bool); 64] = [(0, 0, false); 64];
        let mut n_quiets_buscados = 0usize;
        for (idx, (mv, see_pre)) in moves.iter().enumerate() {
            // LMR: candidatas a reducir son jugadas silenciosas, tarde en el
            // orden (ya viene de mejor a peor), sin jaque propio ni jaque
            // que dan -- justo donde el orden ya filtra la mayoria de
            // jugadas malas sin gastar profundidad completa.
            let es_reducible = self.modo_lmr
                && !en_jaque
                && idx >= LMR_MOVES_SIN_REDUCIR
                && depth >= LMR_PROF_MIN
                && !mv.is_capture()
                && mv.promotion.is_none();

            if !en_jaque
                && depth <= FUT_PROF_MAX
                && idx > 0
                && best_move.is_some()
                && !mv.is_capture()
                && mv.promotion.is_none()
                && beta.abs() < MATE - 1000
            {
                let ev = *fut_eval.get_or_insert_with(|| {
                    let raw =
                        *static_eval_cache.get_or_insert_with(|| self.evaluar_completo(b, eval_state));
                    self.eval_corregida(b, raw, prev)
                });
                // improving: con mejora el margen de futilidad se achica -- se
                // descartan mas jugadas silenciosas tardias; sin mejora el
                // margen crece y se poda menos (mas conservador).
                let margen_ply = if improving {
                    FUT_MARGEN_POR_PLY * 3 / 5
                } else {
                    FUT_MARGEN_POR_PLY
                };
                if ev + FUT_MARGEN_BASE + margen_ply * depth <= alpha {
                    let next_probe = b.make_move(mv);
                    if !next_probe.in_check(next_probe.turn) {
                        continue;
                    }
                }
            }

            // Late Move Pruning (poda por conteo de jugadas): en nodos NO-PV,
            // a poca profundidad y lejos de puntajes de mate, una vez probadas
            // suficientes jugadas silenciosas (el umbral crece con la
            // profundidad: 3 + depth^2) el resto casi nunca mejora alfa -- se
            // saltan sin buscarlas. A diferencia de la futilidad (que compara
            // la eval estatica contra alfa), esta depende solo del conteo, y
            // muerde sobre todo cerca de las hojas -- donde vive la mayoria de
            // los nodos-- recortando el arbol para gastar ese presupuesto en
            // mas profundidad. Conservador: solo hasta profundidad 6, nunca en
            // jaque, y nunca salta una jugada que da jaque. Validado h2h vs el
            // motor desplegado: 56.9%/160 partidas a 300ms.
            const LMP_PROF_MAX: i32 = 6;
            // improving: el umbral de conteo se divide por (2 - improving):
            // sin mejora se poda a partir de aprox. la mitad de jugadas.
            let lmp_umbral = ((3 + depth * depth) / (2 - improving as i32)) as usize;
            if !es_pv
                && !en_jaque
                && depth <= LMP_PROF_MAX
                && best_move.is_some()
                && !mv.is_capture()
                && mv.promotion.is_none()
                && beta.abs() < MATE - 1000
                && idx >= lmp_umbral
            {
                let next_probe = b.make_move(mv);
                if !next_probe.in_check(next_probe.turn) {
                    continue;
                }
            }

            // SEE pruning en el loop principal (no solo en quiescence): en
            // nodos NO-PV y poca profundidad, una captura con SEE muy
            // negativo rara vez compensa aunque se reduzca por LMR -- se
            // descarta directo sin bajar al hijo.
            const SEE_PRUNE_PROF_MAX: i32 = 7;
            // Margen LINEAL (estandar estilo Stockfish, ~-85*depth/3): el
            // margen cuadratico anterior (-20*depth^2) daba -980 a depth 7,
            // casi nunca podaba. Lineal: -85*7/3 ~= -198 a depth 7 -- poda
            // capturas claramente perdedoras sin tocar las dudosas.
            const SEE_PRUNE_MARGEN_POR_PLY: i32 = -85;
            if !es_pv
                && !en_jaque
                && depth <= SEE_PRUNE_PROF_MAX
                && best_move.is_some()
                && mv.is_capture()
                && mv.promotion.is_none()
                && beta.abs() < MATE - 1000
                // see_pre ya quedo cacheado durante el ordenamiento (la
                // condicion mv.is_capture() garantiza que es Some).
                && see_pre.unwrap() < SEE_PRUNE_MARGEN_POR_PLY * depth / 3
            {
                let next_probe = b.make_move(mv);
                if !next_probe.in_check(next_probe.turn) {
                    continue;
                }
            }

            // Registrar esta quiet como "buscada" ANTES de buscarla, para
            // poder penalizarla si otra jugada posterior causa el corte.
            if !mv.is_capture() && mv.promotion.is_none() && n_quiets_buscados < 64 {
                quiets_buscados[n_quiets_buscados] =
                    (mv.from, mv.to, casilla_amenazada(b, mv.to));
                n_quiets_buscados += 1;
            }

            let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
            let next = b.make_move(mv);
            // Prefetch de la TT del hijo: `next.zobrist` se probeara al
            // entrar a la llamada recursiva de negamax de abajo, justo
            // despues del trabajo de siguiente_estado_busqueda (delta NNUE)
            // que se solapa con esta latencia. Hint puro al hardware.
            self.tt_prefetch(next.zobrist);
            let child_prev = Some((pt_mv, mv.to as usize));
            let child_prev2 = prev;
            let ext = if jugada_singular == Some(*mv) { 1 } else { 0 };
            // Para LMR usamos la profundidad de la posible re-búsqueda
            // completa, no la reducida: si falla alto no debe heredar una
            // evaluación clásica donde aún se requiere la NNUE.
            let next_eval = self.siguiente_estado_busqueda(eval_state, b, &next, depth - 1 + ext);
            // Se calcula el resultado como Result SIN aplicar `?` todavia:
            // el undo del acumulador NNUE tiene que correr en todos los
            // caminos, incluido el corte por tiempo.
            let sc_res: Result<i32, TimeUp> = if es_reducible && !next.in_check(next.turn) {
                self.lmr_intentos += 1;
                // Reduccion tablada + ajustes contextuales (PV / historia /
                // improving / capturas con SEE malo). Nunca menos de 1 ply
                // (piso historico del motor) ni mas de depth-1.
                let mut r = tabla_lmr()[(depth as usize).min(63)][(idx + 1).min(63)].max(1);
                if es_pv {
                    r -= 1;
                }
                if !improving {
                    r += 1;
                }
                if !mv.is_capture() {
                    let h = self.history[mv.from as usize][mv.to as usize];
                    let ch = match prev {
                        Some((p_pt, p_to)) => {
                            self.cont_history[cont_idx(p_pt, p_to, pt_mv, mv.to as usize)]
                        }
                        None => 0,
                    };
                    if h + ch > 0 {
                        r -= 1;
                    }
                }
                let r = r.clamp(1, (depth - 2).max(1));
                let child_ply = (ply + 1) as usize;
                if child_ply < MAX_KILLER_PLY {
                    self.hindsight_parent_eval[ply as usize] = *fut_eval.get_or_insert_with(|| {
                        *static_eval_cache.get_or_insert_with(|| self.evaluar_completo(b, eval_state))
                    });
                    self.hindsight_reduction[child_ply] = r;
                }
                // PVS real: el sondeo reducido usa ventana NULA (-alpha-1,-alpha)
                // -- solo pregunta "esto es mejor que lo que ya tengo?", no
                // cuanto mejor. Es un bound, no un valor exacto: si supera
                // alfa, no se confia en el numero, se re-busca a profundidad
                // Y ventana completas para obtener el valor real.
                match self.negamax(
                    &next,
                    &next_eval,
                    depth - 1 + ext - r,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    child_prev,
                    child_prev2,
                    en_sondeo_se,
                ) {
                    Err(e) => Err(e),
                    Ok(v) => {
                        let sondeo = -v;
                        if sondeo > alpha {
                            self.lmr_reintentos += 1;
                            if child_ply < MAX_KILLER_PLY {
                                self.hindsight_reduction[child_ply] = 0;
                            }
                            self.negamax(
                                &next,
                                &next_eval,
                                depth - 1 + ext,
                                -beta,
                                -alpha,
                                ply + 1,
                                child_prev,
                                child_prev2,
                                en_sondeo_se,
                            )
                            .map(|v2| -v2)
                        } else {
                            Ok(sondeo)
                        }
                    }
                }
            } else {
                let child_ply = (ply + 1) as usize;
                if child_ply < MAX_KILLER_PLY {
                    self.hindsight_reduction[child_ply] = 0;
                }
                // PVS: la primera jugada recibe la ventana completa. Las
                // siguientes se sondean con ventana nula; solo se repite la
                // búsqueda completa si realmente supera alpha y aún no es
                // un cutoff beta. Esto conserva el resultado de alfa-beta y
                // reduce nodos en posiciones con buen ordenamiento.
                if idx == 0 {
                    self.negamax(
                        &next,
                        &next_eval,
                        depth - 1 + ext,
                        -beta,
                        -alpha,
                        ply + 1,
                        child_prev,
                        child_prev2,
                        en_sondeo_se,
                    )
                    .map(|v| -v)
                } else {
                    match self.negamax(
                        &next,
                        &next_eval,
                        depth - 1 + ext,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        child_prev,
                        child_prev2,
                        en_sondeo_se,
                    ) {
                        Err(e) => Err(e),
                        Ok(v) => {
                            let sondeo = -v;
                            if sondeo > alpha && sondeo < beta {
                                self.negamax(
                                    &next,
                                    &next_eval,
                                    depth - 1 + ext,
                                    -beta,
                                    -alpha,
                                    ply + 1,
                                    child_prev,
                                    child_prev2,
                                    en_sondeo_se,
                                )
                                .map(|v2| -v2)
                            } else {
                                Ok(sondeo)
                            }
                        }
                    }
                }
            };
            self.salir_hijo(&next_eval, b, &next);
            let sc = sc_res?;

            if sc > best_score {
                best_score = sc;
                best_move = Some(*mv);
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                self.registrar_corte(b, *mv, ply, depth, prev, prev2, pt_mv);
                // History malus: las quiets buscadas antes de esta (que no
                // cortaron) se penalizan con la misma magnitud del bonus que
                // recibe la que si corto. Asi el ordenamiento distingue quiets
                // buenas de malas en vez de solo premiar las buenas. Solo la
                // tabla plana [from][to] (cont_history necesitaria el tipo de
                // pieza de cada una); el envejecimiento /2 por "go" lo acota.
                // El malus va a la MISMA tabla (amenaza o normal) que uso el
                // bonus de esa quiet en su momento -- simetria con el bonus.
                if !mv.is_capture() && mv.promotion.is_none() {
                    let malus = depth * depth;
                    for &(qf, qt, amenazada) in &quiets_buscados[..n_quiets_buscados] {
                        if (qf, qt) != (mv.from, mv.to) {
                            if amenazada {
                                self.history_amenaza[qf as usize][qt as usize] -= malus;
                            } else {
                                self.history[qf as usize][qt as usize] -= malus;
                            }
                        }
                    }
                }
                break;
            }
        }
        self.path_len = self.path_len.saturating_sub(1);

        let flag = if best_score <= alpha_orig {
            TTFlag::Alpha
        } else if best_score >= beta {
            TTFlag::Beta
        } else {
            TTFlag::Exact
        };
        self.tt_store(key, depth, best_score, ply, flag, best_move);

        // Correction history: registrar el error entre el score REAL de la
        // busqueda y la eval estatica cruda. Solo cuando la cota es coherente
        // con el lado del error (un fail-high solo informa si el score quedo
        // POR ENCIMA de la eval; un fail-low, solo si quedo por debajo) y
        // lejos de mates, donde la eval estatica no es comparable.
        if !en_jaque && best_score.abs() < MATE - 1000 && best_move.is_some() {
            let se = *static_eval_cache
                .get_or_insert_with(|| self.evaluar_completo(b, eval_state));
            let coherente = match flag {
                TTFlag::Exact => true,
                TTFlag::Beta => best_score > se,
                TTFlag::Alpha => best_score < se,
            };
            if coherente {
                self.corrhist_registrar(b, se, best_score, depth, prev);
            }
        }

        Ok(best_score)
    }

    /// Detecta si la posicion raiz YA es tablas reclamables ANTES de buscar
    /// (BUG 2): repeticion (la posicion actual ya aparecio entre los ancestros
    /// de la partida real dentro de la ventana de halfmove, mismo criterio que
    /// negamax) o regla de 50 jugadas. Requiere que `path` ya este poblado
    /// desde `game_history` (lo hacen search_fixed_depth y search_time justo
    /// antes de llamar).
    fn posicion_raiz_tablas(&self, b: &Board) -> bool {
        if b.halfmove_clock >= 100 {
            return true;
        }
        let hc = b.halfmove_clock as usize;
        if hc > 0 {
            let start = self.path_len.saturating_sub(hc);
            return self.path[start..self.path_len].contains(&b.zobrist);
        }
        false
    }

    /// Busqueda con profundidad fija (para benchmarks/tests, sin limite de tiempo).
    pub fn search_fixed_depth(&mut self, b: &Board, depth: i32) -> (Option<Move>, i32, u64) {
        self.nodes = 0;
        self.deadline = None;
        self.stop = false;
        // Nueva generacion: las entradas de la busqueda anterior quedan
        // "viejas" y seran las primeras candidatas a reemplazo (aging).
        self.tt_generation = (self.tt_generation + 1) & 0x1F;
        self.killers = vec![[None, None]; MAX_KILLER_PLY];
        // Truncamiento del historial de la partida (BUG 1): conservar los
        // ULTIMOS MAX_PATH elementos (los mas recientes), no los primeros --
        // en partidas largas (>512 jugadas) los recientes son los unicos
        // relevantes para la deteccion de repeticion. Los indices de path
        // son relativos al segmento copiado, asi que path_push y la ventana
        // de halfmove de negamax siguen funcionando igual.
        { let n = self.game_history.len().min(MAX_PATH); let ini = self.game_history.len() - n; self.path[..n].copy_from_slice(&self.game_history[ini..]); self.path_len = n; }
        let root_eval = crear_eval_state(b);
        // El acumulador NNUE del hilo se ancla a la raiz. Dentro del arbol
        // se muta in-place y se deshace al volver; ademas se re-ancla en
        // cada iteracion de profundizacion (ver reiniciar_nnue mas abajo)
        // para que un corte por tiempo que desenrolle la recursion no pueda
        // dejarlo desincronizado en la iteracion siguiente.
        self.reiniciar_nnue(b);
        // BUG 2 (fix): si la posicion raiz YA es tablas reclamables
        // (repeticion o regla de 50), reportarla de inmediato con el score
        // de tablas en vez de dejar que la profundizacion iterativa devuelva
        // un score/PV no-cero incorrecto. Mismo criterio que negamax.
        if self.posicion_raiz_tablas(b) {
            return (None, draw_score(b, &root_eval, self.nnue_de(&root_eval)), 0);
        }
        let mut mejor_mv = None;
        let mut mejor_sc: i32 = -INFINITO;
        for d in 1..=depth {
            self.reiniciar_nnue(b);
            let mut moves = generate_legal(b);
            if moves.is_empty() {
                break;
            }
            self.order_moves_ply(b, &mut moves, mejor_mv, 0, None, None);

            const VENTANA_INICIAL: i32 = 50;
            let (mut vent_alpha, mut vent_beta) = if self.modo_aspiration
                && d >= 2
                && mejor_sc.abs() < MATE - 1000
                && mejor_sc > -INFINITO
            {
                (mejor_sc - VENTANA_INICIAL, mejor_sc + VENTANA_INICIAL)
            } else {
                (-INFINITO, INFINITO)
            };
            let mut actual_mv;
            let mut actual_sc;
            let mut ancho = VENTANA_INICIAL;
            loop {
                let mut alpha = vent_alpha;
                actual_mv = moves[0];
                actual_sc = -INFINITO;
                if self.path_len < MAX_PATH { self.path[self.path_len] = b.zobrist; self.path_len += 1; }
                let mut interrumpido = false;
                for (idx, mv) in moves.iter().enumerate() {
                    let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                    let next = b.make_move(mv);
                    // Aplicar la misma puerta clásica que usan los hijos
                    // internos. Sin esto, las iteraciones d=1/d=2 todavía
                    // construyen un delta NNUE completo en cada hijo de raíz
                    // aunque esos nodos ya van a evaluarse en modo clásico.
                    let next_eval = self.siguiente_estado_busqueda(&root_eval, b, &next, d - 1);
                    let sondeo_alpha = if idx == 0 { -vent_beta } else { -alpha - 1 };
                    let sondeo_beta = -alpha;
                    let res_sondeo = self.negamax(
                        &next,
                        &next_eval,
                        d - 1,
                        sondeo_alpha,
                        sondeo_beta,
                        1,
                        Some((pt_mv, mv.to as usize)),
                        None,
                        false,
                    );
                    let sondeo = match res_sondeo {
                        Ok(v) => -v,
                        Err(_) => {
                            self.salir_hijo(&next_eval, b, &next);
                            interrumpido = true;
                            break;
                        }
                    };
                    let sc = if idx > 0 && sondeo > alpha && sondeo < vent_beta {
                        let res_full = self.negamax(
                            &next,
                            &next_eval,
                            d - 1,
                            -vent_beta,
                            -alpha,
                            1,
                            Some((pt_mv, mv.to as usize)),
                            None,
                            false,
                        );
                        match res_full {
                            Ok(v) => -v,
                            Err(_) => {
                                self.salir_hijo(&next_eval, b, &next);
                                interrumpido = true;
                                break;
                            }
                        }
                    } else {
                        sondeo
                    };
                    self.salir_hijo(&next_eval, b, &next);
                    if sc > actual_sc {
                        actual_sc = sc;
                        actual_mv = *mv;
                    }
                    if sc > alpha {
                        alpha = sc;
                    }
                    if alpha >= vent_beta {
                        break;
                    }
                }
                self.path_len = self.path_len.saturating_sub(1);
                if interrumpido {
                    return (mejor_mv.or(Some(actual_mv)), mejor_sc, self.nodes);
                }
                if actual_sc <= vent_alpha && vent_alpha > -INFINITO {
                    ancho = ancho.saturating_mul(2);
                    vent_alpha = mejor_sc.saturating_sub(ancho).max(-INFINITO);
                    continue;
                }
                if actual_sc >= vent_beta && vent_beta < INFINITO {
                    ancho = ancho.saturating_mul(2);
                    vent_beta = mejor_sc.saturating_add(ancho).min(INFINITO);
                    continue;
                }
                break;
            }
            mejor_mv = Some(actual_mv);
            mejor_sc = actual_sc;
        }
        (mejor_mv, mejor_sc, self.nodes)
    }

    /// Busqueda con presupuesto de tiempo (para UCI "go movetime").
    /// `movetime_ms = None` significa busqueda SIN limite de tiempo propio
    /// (modo "go infinite"): solo termina por `max_depth`, por encontrar un
    /// mate, o porque el hilo UCI activa `external_stop` al recibir "stop".
    pub fn search_time(
        &mut self,
        b: &Board,
        movetime_ms: Option<u64>,
        max_depth: i32,
        mut on_info: impl FnMut(i32, i32, u64, u64),
    ) -> (Option<Move>, i32, i32) {
        self.nodes = 0;
        self.stop = false;
        // Nueva generacion para aging de la TT (ver tt_generation en Searcher).
        self.tt_generation = (self.tt_generation + 1) & 0x1F;
        self.killers = vec![[None, None]; MAX_KILLER_PLY];
        self.decaer_history();
        // Truncamiento del historial de la partida (BUG 1): conservar los
        // ULTIMOS MAX_PATH elementos (los mas recientes), no los primeros --
        // solo ellos importan para la deteccion de repeticion en partidas
        // largas. Los indices de path son relativos al segmento copiado.
        { let n = self.game_history.len().min(MAX_PATH); let ini = self.game_history.len() - n; self.path[..n].copy_from_slice(&self.game_history[ini..]); self.path_len = n; }
        let root_eval = crear_eval_state(b);
        // El acumulador NNUE del hilo se ancla a la raiz. Dentro del arbol
        // se muta in-place y se deshace al volver; ademas se re-ancla en
        // cada iteracion de profundizacion (ver reiniciar_nnue mas abajo)
        // para que un corte por tiempo que desenrolle la recursion no pueda
        // dejarlo desincronizado en la iteracion siguiente.
        self.reiniciar_nnue(b);
        // BUG 2 (fix): si la posicion raiz YA es tablas reclamables
        // (repeticion o regla de 50), reportarla de inmediato con el score
        // de tablas en vez de dejar que la busqueda devuelva un score/PV
        // incorrecto. Va ANTES del libro y de la tabla de finales: un
        // reclamo de tablas por repeticion es un hecho de la partida, no
        // una decision de apertura/final.
        if self.posicion_raiz_tablas(b) {
            return (None, draw_score(b, &root_eval, self.nnue_de(&root_eval)), 0);
        }

        // Libro de aperturas: se consulta para CUALQUIER turno (blancas o
        // negras -- la clave Polyglot ya codifica de quien es el turno), no
        // solo cuando el motor abre la partida. Si hay un root_moves_filtro
        // activo (UCI "searchmoves", p.ej. un reintento excluyendo una
        // jugada marcada como blunder), la jugada del libro SOLO se usa si
        // esta dentro del filtro -- si no, se ignora el libro y se cae a la
        // busqueda normal (que ya respeta el filtro). Sin este chequeo, el
        // filtro quedaba bypaseado en cualquier posicion de libro: excluir
        // una jugada no serviria de nada si el libro la sigue proponiendo.
        let filtro_permite = |mv: &Move| match &self.root_moves_filtro {
            Some(filtro) => filtro.contains(mv),
            None => true,
        };
        if let Some(mv) = crate::polyglot::probe(b) {
            if filtro_permite(&mv) {
                on_info(1, 0, 0, 0);
                return (Some(mv), 0, 1);
            }
        }

        // Tabla de finales en la raiz: DTZ da la jugada que progresa de
        // verdad hacia el resultado optimo (no solo "no perder"), asi que
        // reemplaza directamente lo que hubiera elegido la busqueda normal
        // -- sin esto, alfa-beta con WDL exacto en las hojas puede quedar
        // indiferente entre varias jugadas que dan el mismo resultado
        // (todas "ganadas"), incluida una que no progresa nunca). Mismo
        // chequeo del filtro que el libro, por la misma razon.
        if let Some((mv, sc)) = crate::syzygy::mejor_jugada_raiz(b) {
            if filtro_permite(&mv) {
                on_info(1, sc, 0, 0);
                return (Some(mv), sc, 1);
            }
        }

        let inicio = Instant::now();
        self.deadline = movetime_ms.map(|ms| {
            let budget = ms.saturating_sub(margen_interno_tiempo(ms));
            inicio + std::time::Duration::from_millis(budget)
        });

        // Siempre conservar una jugada legal de emergencia. Asi un reloj de
        // pocos milisegundos no termina en bestmove 0000 si no completa depth 1.
        let fallback = match &self.root_moves_filtro {
            Some(filtro) => generate_legal(b).into_iter().find(|m| filtro.contains(m)),
            None => generate_legal(b).into_iter().next(),
        };
        let mut mejor_mv: Option<Move> = fallback;
        let mut mejor_sc: i32 = self.evaluar_completo(b, &root_eval);
        let mut mejor_prof = 0;
        // Time management avanzado: cuantas iteraciones SEGUIDAS la mejor
        // jugada de la raiz no cambio. Un PV estable es senal de que la
        // posicion ya esta resuelta -- se puede cortar el reloj mas
        // temprano. Si la mejor jugada acaba de cambiar (0), la posicion es
        // mas dudosa y conviene dejarle mas margen para una iteracion mas.
        let mut pv_estable: u32 = 0;

        for d in 1..=max_depth {
            self.reiniciar_nnue(b);
            let mut moves = generate_legal(b);
            if let Some(filtro) = &self.root_moves_filtro {
                moves.retain(|m| filtro.contains(m));
            }
            if moves.is_empty() {
                break;
            }
            self.order_moves_ply(b, &mut moves, mejor_mv, 0, None, None);
            if self.variante_orden_raiz && moves.len() >= 2 {
                moves.swap(0, 1);
            }

            // Aspiration windows: a partir de la 2da profundidad ya hay un
            // puntaje de referencia (el de la iteracion anterior), asi que en
            // vez de arrancar con ventana completa (-inf,+inf) se arranca
            // angosta alrededor de ese valor -- casi siempre alcanza y poda
            // mucho mas en las subramas, y si falla (la posicion cambio mas
            // de lo esperado) se ensancha y se repite. Nunca cambia la
            // jugada final elegida, solo cuanto cuesta encontrarla.
            const VENTANA_INICIAL: i32 = 50;
            let (mut vent_alpha, mut vent_beta) =
                if self.modo_aspiration && d >= 2 && mejor_sc.abs() < MATE - 1000 {
                    (mejor_sc - VENTANA_INICIAL, mejor_sc + VENTANA_INICIAL)
                } else {
                    (-INFINITO, INFINITO)
                };

            let mut actual_mv;
            let mut actual_sc;
            let mut timed_out = false;
            let mut ancho = VENTANA_INICIAL;

            loop {
                let mut alpha = vent_alpha;
                actual_mv = moves[0];
                actual_sc = -INFINITO;
                if self.path_len < MAX_PATH { self.path[self.path_len] = b.zobrist; self.path_len += 1; }
                for (idx, mv) in moves.iter().enumerate() {
                    let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                    let next = b.make_move(mv);
                    // Mantener el mismo contrato que negamax: si el hijo
                    // queda dentro de la zona clásica, no construir un delta
                    // NNUE que no se llegará a consultar.
                    let next_eval = self.siguiente_estado_busqueda(&root_eval, b, &next, d - 1);
                    let sondeo_alpha = if idx == 0 { -vent_beta } else { -alpha - 1 };
                    let sondeo_beta = -alpha;
                    let res_sondeo = self.negamax(
                        &next,
                        &next_eval,
                        d - 1,
                        sondeo_alpha,
                        sondeo_beta,
                        1,
                        Some((pt_mv, mv.to as usize)),
                        None,
                        false,
                    );
                    let sondeo = match res_sondeo {
                        Ok(v) => -v,
                        Err(_) => {
                            self.salir_hijo(&next_eval, b, &next);
                            timed_out = true;
                            break;
                        }
                    };
                    let sc = if idx > 0 && sondeo > alpha && sondeo < vent_beta {
                        let res_full = self.negamax(
                            &next,
                            &next_eval,
                            d - 1,
                            -vent_beta,
                            -alpha,
                            1,
                            Some((pt_mv, mv.to as usize)),
                            None,
                            false,
                        );
                        match res_full {
                            Ok(v) => -v,
                            Err(_) => {
                                self.salir_hijo(&next_eval, b, &next);
                                timed_out = true;
                                break;
                            }
                        }
                    } else {
                        sondeo
                    };
                    self.salir_hijo(&next_eval, b, &next);
                    if sc > actual_sc {
                        actual_sc = sc;
                        actual_mv = *mv;
                    }
                    if sc > alpha {
                        alpha = sc;
                    }
                    if alpha >= vent_beta {
                        break; // fail-high contra la ventana: cortar y reintentar mas ancho
                    }
                }
                self.path_len = self.path_len.saturating_sub(1);
                if timed_out {
                    break;
                }
                // Ensanchado exponencial (duplica cada reintento) con techo
                // en ventana completa -- garantiza terminar y converge rapido
                // incluso si la primera estimacion estaba muy lejos.
                if actual_sc <= vent_alpha && vent_alpha > -INFINITO {
                    ancho = ancho.saturating_mul(2);
                    vent_alpha = mejor_sc.saturating_sub(ancho).max(-INFINITO);
                    continue;
                }
                if actual_sc >= vent_beta && vent_beta < INFINITO {
                    ancho = ancho.saturating_mul(2);
                    vent_beta = mejor_sc.saturating_add(ancho).min(INFINITO);
                    continue;
                }
                break; // adentro de la ventana (o ya en ventana completa): valor confiable
            }
            if timed_out {
                break;
            }
            if mejor_mv == Some(actual_mv) {
                pv_estable += 1;
            } else {
                pv_estable = 0;
            }
            mejor_mv = Some(actual_mv);
            mejor_sc = actual_sc;
            mejor_prof = d;
            on_info(d, mejor_sc, self.nodes, inicio.elapsed().as_millis() as u64);

            if mejor_sc.abs() >= MATE - 1000 {
                break;
            }
            // En ultrabullet el deadline duro ya reserva margen; el corte
            // blando impedía completar la siguiente iteración aun cuando
            // quedaban milisegundos útiles. Para tiempos normales se
            // conserva el comportamiento previo, pero ahora la fraccion es
            // ADAPTATIVA segun la estabilidad del PV en vez de un 70% fijo:
            // PV estable hace varias iteraciones -> se puede cortar antes.
            // PV recien cambio -> se le da mas margen para una iteracion
            // adicional que confirme la jugada nueva.
            let fraccion_corte: u64 = if pv_estable >= 4 {
                55
            } else if pv_estable >= 2 {
                65
            } else if pv_estable == 0 {
                85
            } else {
                70
            };
            if let Some(ms) = movetime_ms
                && ms > 25
                && inicio.elapsed().as_millis() as u64 > ms.saturating_mul(fraccion_corte) / 100
            {
                break;
            }
        }
        // La raiz nunca pasa por negamax (el loop de arriba la maneja
        // aparte), asi que sin esto la TT no tiene entrada para ella y
        // extraer_pv() no puede ni arrancar a caminarla. Guardarla aca no
        // afecta la busqueda en si (pasa DESPUES del loop).
        if let Some(mv) = mejor_mv {
            self.tt_store(b.zobrist, mejor_prof, mejor_sc, 0, TTFlag::Exact, Some(mv));
        }
        (mejor_mv, mejor_sc, mejor_prof)
    }
}

// ============================================================
//  Lazy SMP: varios hilos nativos buscando la misma posicion raiz en
//  paralelo, compartiendo la TT (con locks por casillero). Cada hilo tiene
//  su propio killers/history (no compartidos, no hay beneficio claro y
//  complica el codigo sin necesidad). El resultado final es el del hilo
//  que llego mas profundo (o, empatados, el de score mas decisivo).
// ============================================================

pub struct ResultadoHilo {
    pub mv: Option<Move>,
    pub score: i32,
    pub profundidad: i32,
    pub nodos: u64,
}

/// Busca `b` con `n_hilos` hilos nativos compartiendo TT, con el mismo
/// presupuesto de reloj que una busqueda de un solo hilo (el paralelismo es
/// para ver MAS nodos en el mismo tiempo, no para tardar mas). Variacion
/// entre hilos: los hilos de indice impar arrancan con las dos primeras
/// jugadas del orden intercambiadas, para que no todos exploren exactamente
/// la misma linea primero -- ademas de la variacion natural que ya aporta
/// el timing real de acceso a la TT compartida entre hilos genuinamente
/// concurrentes (el mecanismo clasico detras de Lazy SMP).
/// `tt` se pasa ya construida (y se espera que el LLAMADOR la guarde y
/// reutilice entre jugadas de la misma partida, igual que la TT de un
/// Searcher normal persiste entre llamadas a "go" -- si se reconstruyera
/// de cero en cada jugada, Lazy SMP perderia la continuidad de la TT entre
/// plies, una desventaja injusta frente a la version de un solo hilo.
#[allow(clippy::too_many_arguments)]
pub fn buscar_lazy_smp(
    b: &Board,
    movetime_ms: Option<u64>,
    max_depth: i32,
    n_hilos: usize,
    tt: &Arc<SharedTT>,
    generacion_compartida: &AtomicU8,
    tt_mask: usize,
    modo_lmr: bool,
    qsearch_nnue: bool,
    nnue_classical_depth: i32,
    game_history: &[u64],
    external_stop: Arc<AtomicBool>,
    root_moves_filtro: Option<Vec<Move>>,
) -> (Option<Move>, i32, u64, Vec<ResultadoHilo>) {
    // Aging compartido de la TT (ver set_tt_generacion): el contador vive en
    // el llamador (junto a smp_tt, que persiste entre jugadas) y avanza UNA
    // vez por llamada. Todos los hilos de esta llamada se siembran con el
    // MISMO valor, asi que las entradas que escriben llevan una generacion
    // DISTINTA a la de jugadas anteriores y la regla de reemplazo de tt_store
    // puede expulsarlas de inmediato (aging vivo en Lazy SMP).
    let generacion = generacion_compartida.fetch_add(1, Ordering::Relaxed) & 0x1F;
    if n_hilos <= 1 {
        let mut s = Searcher::new_con_tt_compartida(Arc::clone(tt), tt_mask, modo_lmr);
        s.set_tt_generacion(generacion);
        s.set_qsearch_nnue(qsearch_nnue);
        s.set_nnue_classical_depth(nnue_classical_depth);
        s.set_external_stop(Some(external_stop));
        s.set_game_history(game_history.to_vec());
        s.root_moves_filtro = root_moves_filtro;
        let (mv, sc, prof) = s.search_time(b, movetime_ms, max_depth, |_, _, _, _| {});
        let nodos = s.nodes;
        return (
            mv,
            sc,
            nodos,
            vec![ResultadoHilo {
                mv,
                score: sc,
                profundidad: prof,
                nodos,
            }],
        );
    }

    let board_copy = *b;

    let handles: Vec<_> = (0..n_hilos)
        .map(|i| {
            let external_stop = Arc::clone(&external_stop);
            let tt = Arc::clone(tt);
            let game_history = game_history.to_vec();
            let root_moves_filtro = root_moves_filtro.clone();
            std::thread::spawn(move || {
                let mut s = Searcher::new_con_tt_compartida(tt, tt_mask, modo_lmr);
                s.set_tt_generacion(generacion);
                s.set_qsearch_nnue(qsearch_nnue);
                s.set_nnue_classical_depth(nnue_classical_depth);
                s.root_moves_filtro = root_moves_filtro;
                s.variante_orden_raiz = i % 2 == 1;
                s.null_move_r_extra = match i % 3 {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                s.set_external_stop(Some(external_stop));
                s.set_game_history(game_history);
                let (mv, sc, prof) =
                    s.search_time(&board_copy, movetime_ms, max_depth, |_, _, _, _| {});
                ResultadoHilo {
                    mv,
                    score: sc,
                    profundidad: prof,
                    nodos: s.nodes,
                }
            })
        })
        .collect();

    let resultados: Vec<ResultadoHilo> = handles
        .into_iter()
        .map(|h| h.join().expect("hilo de busqueda con panic"))
        .collect();

    let nodos_totales: u64 = resultados.iter().map(|r| r.nodos).sum();
    // v12: NO usar score.abs() para desempatar entre hilos con la misma
    // profundidad. Todos buscan la MISMA posicion raiz con el MISMO bando a
    // mover, asi que un score mas alto es sencillamente mejor -- no hace
    // falta "decision" alguna. Con abs(), un hilo que por suerte de orden de
    // jugadas NO vio una refutacion real (score optimista, ej. +400) le
    // ganaba a otro hilo que SI la encontro (score correcto pero cauteloso,
    // ej. -50), porque |400| > |-50| -- eligiendo la evaluacion equivocada
    // con mas confianza en vez de la correcta. Score crudo (sin abs) elige
    // siempre la mejor evaluacion real entre los hilos empatados en profundidad.
    let mejor = resultados
        .iter()
        .max_by_key(|r| (r.profundidad, r.score))
        .expect("al menos un hilo");

    (mejor.mv, mejor.score, nodos_totales, resultados)
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn score_mate_tt_roundtrip_en_distintos_plies() {
        for ply in [0, 1, 7, 31] {
            let gana = MATE - 12;
            let pierde = -MATE + 9;
            assert_eq!(score_from_tt(score_to_tt(gana, ply), ply), gana);
            assert_eq!(score_from_tt(score_to_tt(pierde, ply), ply), pierde);
            assert_eq!(score_from_tt(score_to_tt(123, ply), ply), 123);
        }
    }

    #[test]
    fn tt_colision_de_otra_clave_se_reemplaza() {
        let mut s = Searcher::new(1);
        let k1 = 0x10u64;
        let k2 = k1.wrapping_add((s.tt_mask as u64) + 1);
        s.tt_store(k1, 12, 50, 0, TTFlag::Exact, None);
        s.tt_store(k2, 1, 20, 0, TTFlag::Alpha, None);
        assert!(s.tt_probe(k1).is_none());
        assert_eq!(s.tt_probe(k2).map(|e| e.depth), Some(1));
    }

    // La TT COMPARTIDA (Lazy SMP) es un camino de codigo distinto al Local
    // (empaquetado lockless en un solo u64 via AtomicU64, ver tt_empaquetar/
    // tt_desempaquetar) -- este test lo ejercita especificamente, el de
    // arriba solo prueba la TT Local con Searcher::new().
    #[test]
    fn tt_compartida_lockless_roundtrip_y_colision() {
        let (tt, mask) = construir_tt(1);
        let mut s = Searcher::new_con_tt_compartida(Arc::clone(&tt), mask, true);

        // Roundtrip con jugada simple, sin promocion.
        let mv1 = Move { from: 8, to: 16, promotion: None, flag: MoveFlag::Quiet };
        s.tt_store(0xABCD, 10, 55, 0, TTFlag::Exact, Some(mv1));
        let e1 = s.tt_probe(0xABCD).expect("deberia encontrarse (misma clave)");
        assert_eq!(e1.depth, 10);
        assert_eq!(e1.score, 55);
        assert_eq!(e1.flag, TTFlag::Exact);
        assert_eq!(e1.best, Some(mv1));

        // Roundtrip con score negativo y jugada CON promocion (probando los
        // 3 bits de promocion del empaquetado, no solo el caso None).
        let mv2 = Move { from: 48, to: 56, promotion: Some(PieceType::Queen), flag: MoveFlag::Capture };
        s.tt_store(0x9999, 3, -1200, 0, TTFlag::Beta, Some(mv2));
        let e2 = s.tt_probe(0x9999).expect("deberia encontrarse (misma clave)");
        assert_eq!(e2.score, -1200);
        assert_eq!(e2.flag, TTFlag::Beta);
        assert_eq!(e2.best, Some(mv2));

        // Colision de INDICE (mismo casillero) con clave real distinta EN
        // LOS BITS DE VERIFICACION (los 15 bits altos) -- a diferencia del
        // test de la TT Local (que compara la clave de 64 bits completa),
        // aca hay que armar k2 a proposito con bits altos distintos: sumar
        // un numero chico (como hacia el test viejo) puede no tocar nunca
        // los bits 49..64 y el "choque" no se notaria, sin ser un bug real
        // -- en la practica las claves zobrist son pseudoaleatorias en
        // todos sus bits, este caso degenerado no ocurre.
        let k1 = 0x1234u64;
        let k2 = (k1 & mask as u64) | (0xABCDu64 << 49);
        s.tt_store(k1, 12, 50, 0, TTFlag::Exact, None);
        s.tt_store(k2, 1, 20, 0, TTFlag::Alpha, None);
        assert!(s.tt_probe(k1).is_none());
        assert_eq!(s.tt_probe(k2).map(|e| e.depth), Some(1));
    }

    #[test]
    fn quiescence_detecta_mate_en_jaque() {
        let b = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut s = Searcher::new(1);
        let eval_state = crear_eval_state(&b);
        let score = s
            .quiescence(&b, &eval_state, -INFINITO, INFINITO, 3)
            .unwrap();
        assert_eq!(score, -MATE + 3);
    }

    #[test]
    fn quiescence_respeta_regla_de_cincuenta() {
        let b = Board::from_fen("4k3/8/8/8/8/8/4Q3/4K3 w - - 100 1").unwrap();
        let mut s = Searcher::new(1);
        let eval_state = crear_eval_state(&b);
        assert_eq!(
            s.quiescence(&b, &eval_state, -INFINITO, INFINITO, 0)
                .unwrap(),
            draw_score(&b, &eval_state, None)
        );
    }

    #[test]
    fn reloj_ultracorto_usa_margen_adaptativo_seguro() {
        assert_eq!(margen_interno_tiempo(2), 0);
        assert_eq!(margen_interno_tiempo(15), 3);
        assert_eq!(margen_interno_tiempo(20), 3);
        assert_eq!(margen_interno_tiempo(50), 4);
        assert_eq!(margen_interno_tiempo(200), 5);
        assert_eq!(margen_interno_tiempo(600), 5);
        for ms in 1..=1_000 {
            assert!(margen_interno_tiempo(ms) < ms);
        }
    }

    #[test]
    fn bench_nps_depth12() {
        let b = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut s = Searcher::new(64);
        let start = std::time::Instant::now();
        let (_, score, nodes) = s.search_fixed_depth(&b, 12);
        let elapsed = start.elapsed();
        let nps = if elapsed.as_secs_f64() > 0.0 {
            (nodes as f64 / elapsed.as_secs_f64()) as u64
        } else {
            0
        };
        println!(
            "BENCH: depth=12 nodes={} time={:.3}s nps={} score={}",
            nodes,
            elapsed.as_secs_f64(),
            nps,
            score
        );
        assert!(nodes > 0, "Debe visitar algun nodo");
    }

    // BUG confirmado de aging en Lazy SMP: cada llamada a buscar_lazy_smp
    // creaba Searchers NUEVOS con tt_generation 0 y search_time la subia UNA
    // sola vez a 1. Como la TT compartida persiste entre jugadas pero el
    // contador no, TODA busqueda SMP quedaba en generacion 1 y el aging
    // jamas expulsaba entradas de jugadas anteriores. El fix pasa un contador
    // COMPARTIDO (AtomicU8 que vive junto a smp_tt en el llamador) que avanza
    // una vez por llamada y siembra todos los hilos con el mismo valor: dos
    // llamadas sucesivas deben escribir entradas con generaciones DISTINTAS.
    #[test]
    fn smp_tt_generacion_compartida_avanza_entre_llamadas() {
        let (tt, mask) = construir_tt(8);
        let generacion = AtomicU8::new(0);
        let b = Board::from_fen(
            "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        )
        .unwrap();
        let stop = Arc::new(AtomicBool::new(false));

        // Generaciones realmente escritas en la TT compartida (bits 43..48
        // del paquete crudo, el mismo layout de tt_empaquetar).
        let generaciones_presentes = |tt: &SharedTT| -> Vec<u8> {
            let mut gens: Vec<u8> = tt
                .iter()
                .filter_map(|slot| {
                    let raw = slot.load(Ordering::Relaxed);
                    if raw & TT_OCUPADO == 0 {
                        None
                    } else {
                        Some(((raw >> 43) & 0x1F) as u8)
                    }
                })
                .collect();
            gens.sort_unstable();
            gens.dedup();
            gens
        };

        // Llamada 1 (2 hilos): todos los Searchers siembran generacion 0 y
        // search_time la sube a 1 -- las entradas nuevas llevan generacion 1.
        buscar_lazy_smp(
            &b,
            None,
            3,
            2,
            &tt,
            &generacion,
            mask,
            true,
            true,
            0,
            &[],
            Arc::clone(&stop),
            None,
        );
        let g1 = generaciones_presentes(&tt);
        assert!(
            g1.contains(&1),
            "la 1a llamada debe escribir entradas de generacion 1, vimos {:?}",
            g1
        );

        // Llamada 2: el contador compartido avanza; los hilos de ESTA llamada
        // deben escribir generacion 2, distinta de la llamada anterior.
        buscar_lazy_smp(
            &b,
            None,
            3,
            2,
            &tt,
            &generacion,
            mask,
            true,
            true,
            0,
            &[],
            Arc::clone(&stop),
            None,
        );
        let g2 = generaciones_presentes(&tt);
        assert!(
            g2.contains(&2),
            "la 2a llamada debe escribir generacion 2 (aging vivo), vimos {:?}",
            g2
        );
        assert!(
            !g1.contains(&2),
            "la generacion 2 no debe existir tras la primera llamada"
        );
    }

    // BUG 2 (fix): la posicion RAIZ nunca se revisaba por tablas reclamables
    // (repeticion o regla de 50) antes de arrancar la profundizacion
    // iterativa. Si la raiz YA era tablas, la busqueda reportaba un
    // score/PV incorrecto (no-cero). Ahora search_fixed_depth y search_time
    // la detectan de inmediato y retornan el score de tablas sin expandir
    // ningun nodo.
    #[test]
    fn raiz_ya_tablas_por_repeticion_retorna_draw_sin_buscar() {
        let b =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 10 1").unwrap();
        let root_eval = crear_eval_state(&b);

        // Referencia: el score de tablas que el propio motor usaria dentro
        // de la busqueda (draw_score). En una posicion balanceada debe ser 0.
        let mut s_ref = Searcher::new(16);
        s_ref.reiniciar_nnue(&b);
        let esperado = draw_score(&b, &root_eval, s_ref.nnue_de(&root_eval));
        assert_eq!(esperado, 0, "startpos balanceada: tablas deben puntuar 0");

        // La posicion actual ya aparecio UNA vez antes en la partida real
        // (2da aparicion => reclamable, mismo criterio que negamax).
        let mut s = Searcher::new(16);
        s.set_game_history(vec![b.zobrist]);

        let (mv, sc, nodos) = s.search_fixed_depth(&b, 6);
        assert!(mv.is_none(), "tabla en la raiz: no debe haber mejor jugada");
        assert_eq!(sc, esperado, "score de tablas inmediato");
        assert_eq!(nodos, 0, "debe reconocer la tabla sin expandir nodos");

        let mut s2 = Searcher::new(16);
        s2.set_game_history(vec![b.zobrist]);
        let (mv2, sc2, prof) = s2.search_time(&b, None, 6, |_, _, _, _| {});
        assert!(mv2.is_none(), "tabla en la raiz: no debe haber mejor jugada");
        assert_eq!(sc2, esperado, "score de tablas inmediato");
        assert_eq!(prof, 0, "debe reconocer la tabla sin profundizar");
    }
}
