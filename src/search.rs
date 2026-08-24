// Negamax + poda alfa-beta + iterative deepening + quiescence + TT.
// Primera version jugable de la Fase 3: SEE, null-move, killers/history y
// LMR quedan para una siguiente pasada si el tiempo alcanza (documentado
// en el reporte final de la sesion).

use crate::board::Board;
use crate::eval::{
    EvalState, crear_eval_state, evaluate_classical_with_state, evaluate_with_state,
};
use crate::movegen::{
    MAX_MOVES, MoveList, generate_captures_legal, generate_captures_legal_into_con_jaque,
    generate_legal, generate_legal_into, generate_legal_into_con_jaque,
};
use crate::types::{Move, MoveFlag, PieceType};
use arrayvec::ArrayVec;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

pub const INFINITO: i32 = 30_000;
pub const MATE: i32 = 29_000;
const MAX_PLY: u32 = 128;
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
    // "Estuvo en PV alguna vez": distinto de es_pv (ventana actual). Persiste
    // en la TT para que, aunque el nodo se revisite luego con ventana nula
    // (por ejemplo por LMR/null-move desde otro camino), la busqueda sepa
    // que esa posicion fue parte de una linea principal y trate RFP/LMR con
    // mas cuidado ahi.
    was_pv: bool,
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

/// Techo DURO del reloj para la busqueda en curso, en ms (0 = sin techo).
///
/// Gestion de tiempo de dos presupuestos: `search_time` recibe el
/// presupuesto "optimo" y este global lleva el "maximo". Es global (y no un
/// parametro) porque lo tienen que ver TODOS los caminos de busqueda,
/// incluidos los hilos que crea `buscar_lazy_smp` por dentro. Lo escribe el
/// bucle UCI al parsear cada `go`, siempre antes de arrancar la busqueda.
pub static TIEMPO_MAXIMO_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Fija el techo duro del reloj para el proximo `go`. `None` lo desactiva.
pub fn fijar_tiempo_maximo(ms: Option<u64>) {
    TIEMPO_MAXIMO_MS.store(ms.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

#[derive(Debug)]
pub struct TimeUp;

const MAX_KILLER_PLY: usize = 176; // margen sobre MAX_PLY para cubrir extensiones de jaque

// Centinela para eval_stack: "no hay eval estatica en este ply" (nodo en
// jaque). Ninguna evaluacion real puede valer i32::MIN.
const EVAL_INVALIDA: i32 = i32::MIN;

/// Margen por ply de la poda RFP (reverse futility), en centipeones de la
/// escala INTERNA del motor (que va ~1.6x inflada respecto de 100cp = peon,
/// ver `peso_bullet()` en neural.rs).
///
/// Era una constante fija de 120, calibrada cuando la eval era el hibrido
/// clasica+red de 512 neuronas. Desde entonces cambiaron las dos cosas: el
/// motor pasa a modo puro (solo red) y la red es de 1024. Una red mas
/// precisa justifica podar MAS agresivo (se le puede confiar mas a la eval
/// estatica), asi que el 120 quedo como deuda de calibracion.
///
/// Se vuelve ajustable con `MITTENS_RFP_MARGEN` SIN cambiar el default: sin
/// la variable el comportamiento es bit-identico al de antes.
fn rfp_margen_por_ply() -> i32 {
    static CACHE: OnceLock<i32> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("MITTENS_RFP_MARGEN")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| *v >= 20 && *v <= 400)
            .unwrap_or(120)
    })
}

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

// `Board::corr_hash` guarda los tres hashes parciales con 16 bits cada uno.
// Eso alcanza para indexar las tablas de correction history mientras el
// enmascarado use como mucho 16 bits. Si algun dia se agranda CORR_SIZE por
// encima de 65536, esto falla EN COMPILACION en vez de silenciosamente
// indexar con bits que no existen.
const _: () = assert!(
    CORR_SIZE <= 1 << 16,
    "CORR_SIZE no entra en los 16 bits por ranura de Board::corr_hash"
);

/// Hash Zobrist solo de los peones (estructura de peones).
///
/// Ya no se recorre el tablero: `Board` lo mantiene incrementalmente, con el
/// mismo XOR de claves `piece_square` que este bucle hacia (ver
/// `Board::corr_hash`). Mismos bits bajos exactos, coste cero.
#[inline(always)]
fn hash_peones(b: &Board) -> u64 {
    b.corr_hash & 0xFFFF
}

/// Hash Zobrist de las piezas que NO son peones de un color dado.
/// Mantenido incrementalmente en la ranura `1 + color` de `Board::corr_hash`.
#[inline(always)]
fn hash_no_peones(b: &Board, color: usize) -> u64 {
    debug_assert!(color < 2);
    (b.corr_hash >> (16 * (1 + color))) & 0xFFFF
}

/// Actualizacion con "gravedad" (Stockfish): el bonus crece con la
/// profundidad y con el error, y se amortigua a medida que la entrada se
/// aleja de cero -- acota la tabla sin resets bruscos.
fn corrhist_update(entry: &mut i32, diff: i32, depth: i32) {
    let bonus = (diff * depth / 8).clamp(-CORR_MAX, CORR_MAX);
    *entry += bonus - *entry * bonus.abs() / 512;
    *entry = (*entry).clamp(-CORR_MAX, CORR_MAX);
}

/// Techo de las tablas de historia (mismo orden de magnitud que Stockfish).
const MAX_HIST: i32 = 16384;

/// Bonus/malus de historia por profundidad.
///
/// FIX DE ESCALA (sesion cand_busqueda_fable): antes era `depth * depth`
/// (25 a profundidad 5, 100 a profundidad 10), una magnitud minuscula
/// frente al techo MAX_HIST=16384 -- pero los CONSUMIDORES de estas tablas
/// estan calibrados a escala Stockfish: la poda por historia exige
/// `stat_score < -4000*depth` y el ajuste proporcional de LMR divide por
/// 8000. Con bonus de 25-100 por corte las entradas casi nunca llegaban a
/// esas magnitudes, asi que ambas tecnicas existian en el codigo pero casi
/// NUNCA disparaban (muertas por escala). Formula estandar tipo Stockfish:
/// lineal en depth con tope, para que unas decenas de cortes ya muevan una
/// entrada al rango que los umbrales esperan.
#[inline]
fn hist_bonus(depth: i32) -> i32 {
    (160 * depth - 100).clamp(20, 1600)
}

/// Actualizacion con GRAVEDAD de una tabla de historia, mismo patron que
/// `corrhist_update`: el bonus se amortigua a medida que la entrada se aleja
/// de cero, asi la tabla se satura suavemente en vez de crecer sin limite.
/// Antes se hacia `entry += depth*depth` a secas, que con profundidades
/// altas dejaba unas pocas entradas dominando todo el ordenamiento.
#[inline]
fn hist_update(entry: &mut i32, bonus: i32) {
    let bonus = bonus.clamp(-MAX_HIST, MAX_HIST);
    *entry += bonus - *entry * bonus.abs() / MAX_HIST;
    *entry = (*entry).clamp(-MAX_HIST, MAX_HIST);
}

// 6 tipos de pieza x 64 casilleros, dos veces (jugada rival + jugada propia).
const CONT_HIST_SIZE: usize = 6 * 64 * 6 * 64;

#[inline]
fn cont_idx(prev_pt: usize, prev_to: usize, pt: usize, to: usize) -> usize {
    ((prev_pt * 64 + prev_to) * 6 + pt) * 64 + to
}

// Capture history: 6 tipos de pieza que mueve x 64 destinos x 6 tipos de
// victima. Aprende que capturas concretas suelen ser buenas mas alla de lo
// que dice el material puro (SEE), p.ej. "caballo x peon en d5" en
// estructuras donde ese peon sostiene todo el centro.
const CAPT_HIST_SIZE: usize = 6 * 64 * 6;

#[inline]
fn capt_idx(pt: usize, to: usize, victima: usize) -> usize {
    (pt * 64 + to) * 6 + victima
}

/// Tipo de pieza capturada por `mv`. En al paso la victima es un peon aunque
/// no haya nada en la casilla destino -- mismo criterio que usa `see()`.
#[inline]
fn victima_de(b: &Board, mv: &Move) -> usize {
    if mv.flag == MoveFlag::EnPassant {
        PieceType::Pawn as usize
    } else {
        b.piece_at(mv.to).map(|(_, pt)| pt as usize).unwrap_or(0)
    }
}

/// Si el rival ya ataca `sq` ANTES de jugar el movimiento (tablero en la
/// posicion actual). Usado para elegir la tabla de historial: una jugada
/// silenciosa hacia una casilla amenazada tiene un significado tactico
/// distinto de una hacia una casilla segura.
#[inline]
fn casilla_amenazada(b: &Board, sq: crate::types::Square) -> bool {
    b.is_square_attacked_by(sq, b.turn.opposite())
}

/// Insertion sort ESTABLE de `moves[inicio..n]` por `claves[inicio..n]`
/// (las dos listas se mueven en paralelo). Estable = claves iguales conservan
/// el orden original, igual que el sort que se usaba antes: asi la secuencia
/// final de jugadas es exactamente la misma.
/// `sees` es el arreglo paralelo de SEE por jugada (ver campo see_negamax):
/// se permuta EXACTAMENTE igual que `claves` para que quede sincronizado con
/// `moves` despues del ordenamiento.
fn ordenar_estable(moves: &mut [Move], claves: &mut [i32], sees: &mut [i32], inicio: usize, n: usize) {
    // IMPLEMENTACION: se ordena un arreglo AUXILIAR de u64 con la clave en la
    // parte alta y el indice original en los 8 bits bajos, y recien al final
    // se aplica la permutacion resultante a las tres listas.
    //
    // Por que: el insertion sort movia TRES arreglos en paralelo dentro del
    // bucle interno (Move + clave + SEE = 3 lecturas y 3 escrituras por
    // desplazamiento). Medido con contadores en `bench 14`: 4.664.099
    // elementos ordenados y 40.026.541 desplazamientos (8,6 por elemento), o
    // sea ~240 millones de accesos a memoria solo para desplazar. Con la
    // clave y el indice EMPAQUETADOS en una sola palabra el desplazamiento
    // cuesta una lectura y una escritura sobre UNA sola corriente contigua, y
    // reacomodar las tres listas pasa a ser lineal (una pasada) en vez de
    // cuadratico.
    //
    // Sigue siendo el MISMO orden final: comparar los u64 empaquetados
    // equivale a comparar (clave, indice_original) lexicograficamente, que es
    // exactamente lo que hace un sort ESTABLE ascendente por clave. El sesgo
    // de 2^31 solo convierte el i32 con signo en un u64 sin signo conservando
    // el orden.
    let m = n.saturating_sub(inicio);
    if m < 2 {
        return;
    }
    debug_assert!(m <= MAX_MOVES, "rango de orden mas grande que MAX_MOVES");
    const SESGO: i64 = 1 << 31;
    let mut emp: ArrayVec<u64, MAX_MOVES> = ArrayVec::new();
    for i in inicio..n {
        emp.push(((((claves[i] as i64) + SESGO) as u64) << 8) | (i - inicio) as u64);
    }
    for i in 1..m {
        let k = emp[i];
        let mut j = i;
        while j > 0 && emp[j - 1] > k {
            emp[j] = emp[j - 1];
            j -= 1;
        }
        emp[j] = k;
    }
    // Permutacion en una pasada. Las tres listas quedan sincronizadas igual
    // que antes (las claves tambien se reacomodan: hoy nadie las lee despues
    // de ordenar, pero dejarlas desordenadas seria una trampa para el
    // proximo que las use).
    let mut mv_tmp: ArrayVec<Move, MAX_MOVES> = ArrayVec::new();
    let mut cl_tmp: ArrayVec<i32, MAX_MOVES> = ArrayVec::new();
    let mut se_tmp: ArrayVec<i32, MAX_MOVES> = ArrayVec::new();
    for &p in emp.iter() {
        let o = inicio + (p & 0xFF) as usize;
        mv_tmp.push(moves[o]);
        cl_tmp.push(claves[o]);
        se_tmp.push(sees[o]);
    }
    moves[inicio..n].copy_from_slice(&mv_tmp);
    claves[inicio..n].copy_from_slice(&cl_tmp);
    sees[inicio..n].copy_from_slice(&se_tmp);
}

/// ¿La jugada `mv` da jaque al rey rival? Equivalente exacto a
/// `b.make_move(mv).in_check(b.turn.opposite())`, pero SIN copiar el tablero
/// completo (~144 bytes): solo se arma la ocupacion post-jugada (u64) y el
/// arreglo de piezas propias (48 bytes) que necesitan los lookups de ataque,
/// y se replica `is_square_attacked_by` sobre esa posicion virtual. En la
/// busqueda negamax los guardas de poda (futility, LMP, SEE-prune) solo
/// quieren saber si la jugada da jaque ANTES de descartarla; antes cada uno
/// hacía su propio `b.make_move(mv)` para consultar eso (hasta 3 copias de
/// 144 bytes por jugada en los nodos poco profundos, que son la mayoria del
/// arbol). Maneja capturas, al paso, enroque y promociones.
#[inline]
fn da_jaque_sin_copiar(b: &Board, mv: &Move) -> bool {
    use crate::bitboard::{
        bishop_attacks, bit, king_attacks, knight_attacks, pawn_attacks, rook_attacks,
    };
    use crate::types::{Color, file_of, make_square, rank_of};

    let us = b.turn;
    let them = us.opposite();
    let ksq = b.king_square(them);
    let (_, pt) = b
        .piece_at(mv.from)
        .expect("da_jaque_sin_copiar: no hay pieza en 'from'");
    let pieza_final = mv.promotion.unwrap_or(pt);

    // Posicion virtual post-jugada: ocupacion + piezas propias ajustadas.
    let mut occ = (b.occupied & !bit(mv.from)) | bit(mv.to);
    let mut piezas = b.pieces[us as usize];
    piezas[pt as usize] &= !bit(mv.from);
    piezas[pieza_final as usize] |= bit(mv.to);
    if mv.flag == MoveFlag::EnPassant {
        let ep = make_square(file_of(mv.to), rank_of(mv.from));
        occ &= !bit(ep);
        piezas[PieceType::Pawn as usize] &= !bit(ep);
    }
    let (rf, rt) = match (mv.flag, us) {
        (MoveFlag::CastleKing, Color::White) => (make_square(7, 0), make_square(5, 0)),
        (MoveFlag::CastleKing, Color::Black) => (make_square(7, 7), make_square(5, 7)),
        (MoveFlag::CastleQueen, Color::White) => (make_square(0, 0), make_square(3, 0)),
        (MoveFlag::CastleQueen, Color::Black) => (make_square(0, 7), make_square(3, 7)),
        _ => (0, 0),
    };
    if mv.flag == MoveFlag::CastleKing || mv.flag == MoveFlag::CastleQueen {
        occ = (occ & !bit(rf)) | bit(rt);
        piezas[PieceType::Rook as usize] &= !bit(rf);
        piezas[PieceType::Rook as usize] |= bit(rt);
    }

    // Misma logica que Board::is_square_attacked_by(ksq, us).
    if pawn_attacks(them, ksq) & piezas[PieceType::Pawn as usize] != 0 {
        return true;
    }
    if knight_attacks(ksq) & piezas[PieceType::Knight as usize] != 0 {
        return true;
    }
    if king_attacks(ksq) & piezas[PieceType::King as usize] != 0 {
        return true;
    }
    let alfil_dama = piezas[PieceType::Bishop as usize] | piezas[PieceType::Queen as usize];
    if alfil_dama != 0 && bishop_attacks(ksq, occ) & alfil_dama != 0 {
        return true;
    }
    let torre_dama = piezas[PieceType::Rook as usize] | piezas[PieceType::Queen as usize];
    if torre_dama != 0 && rook_attacks(ksq, occ) & torre_dama != 0 {
        return true;
    }
    false
}

/// Mascaras de jaque de UN NODO (no de una jugada). `da_jaque_sin_copiar` es
/// exacto pero caro (reconstruye la ocupacion virtual y hace hasta cinco
/// consultas de ataque, dos de ellas magicas) y la busqueda lo llamaba una vez
/// por jugada: medido con contadores en `bench 14`, 4.742.255 llamadas de las
/// cuales solo 49.363 -- el 1,0% -- daban TRUE. O sea: el 99% del trabajo mas
/// caro de los guardas de poda era para contestar "no, no da jaque".
///
/// Estas mascaras contestan ese 99% con dos AND de bitboard, y se calculan una
/// sola vez por nodo (perezosamente: los nodos que nunca preguntan no pagan
/// nada):
///
///  - `por_tipo[pt]` = casillas DESDE LAS QUE una pieza propia de tipo `pt`
///    ataca al rey rival. Una jugada normal da jaque DIRECTO si y solo si su
///    casilla destino esta en la mascara de su tipo.
///  - `descubridores` = piezas propias que son el UNICO bloqueo entre un
///    deslizante propio y el rey rival: moverlas puede destapar un jaque.
///
/// Es una SOBREAPROXIMACION deliberada: si la mascara dice "candidata" se
/// llama igual a `da_jaque_sin_copiar` para la respuesta exacta (p.ej. una
/// pieza descubridora que se mueve DENTRO de la misma linea no da jaque).
/// Nunca puede decir "no" sobre una jugada que si da jaque, asi que el
/// resultado que ve la busqueda es bit a bit el mismo de antes.
#[derive(Clone, Copy)]
struct MascarasJaque {
    por_tipo: [u64; 6],
    descubridores: u64,
}

/// Construye las mascaras del nodo. Coste: dos lookups magicos desde el rey
/// rival + `pinned_pieces` con los roles invertidos (deslizantes PROPIOS
/// contra el rey RIVAL, bloqueadores propios) -- el mismo calculo que ya hace
/// el generador legal, pero mirando al otro rey.
fn mascaras_jaque(b: &Board) -> MascarasJaque {
    use crate::bitboard::{bishop_attacks, knight_attacks, pawn_attacks, pinned_pieces, rook_attacks};
    let us = b.turn;
    let them = us.opposite();
    let ksq = b.king_square(them);
    let occ = b.occupied;
    let ra = rook_attacks(ksq, occ);
    let ba = bishop_attacks(ksq, occ);
    let mut por_tipo = [0u64; 6];
    // Las casillas desde las que un peon NUESTRO ataca `ksq` son exactamente
    // las que ataca un peon DE ELLOS parado en `ksq` (mismo truco que usa
    // `see::atacantes_a`).
    por_tipo[PieceType::Pawn as usize] = pawn_attacks(them, ksq);
    por_tipo[PieceType::Knight as usize] = knight_attacks(ksq);
    por_tipo[PieceType::Bishop as usize] = ba;
    por_tipo[PieceType::Rook as usize] = ra;
    por_tipo[PieceType::Queen as usize] = ra | ba;
    // El rey nunca puede dar jaque directo (los reyes no pueden tocarse: una
    // jugada que dejara al rey propio adyacente al rival seria ilegal).
    por_tipo[PieceType::King as usize] = 0;
    let nuestras_torres = b.pieces[us as usize][PieceType::Rook as usize]
        | b.pieces[us as usize][PieceType::Queen as usize];
    let nuestros_alfiles = b.pieces[us as usize][PieceType::Bishop as usize]
        | b.pieces[us as usize][PieceType::Queen as usize];
    let descubridores = pinned_pieces(
        ksq,
        b.occupied_co[us as usize],
        nuestras_torres,
        nuestros_alfiles,
        occ,
    );
    MascarasJaque {
        por_tipo,
        descubridores,
    }
}

/// Respuesta EXACTA a "¿`mv` da jaque?", igual que `da_jaque_sin_copiar`, pero
/// descartando primero con las mascaras del nodo las jugadas que no pueden
/// darlo.
///
/// Las jugadas con reglas especiales (al paso, enroque, promocion) pasan
/// siempre al camino exacto: cada una mueve o quita una pieza EXTRA (el peon
/// comido al paso, la torre del enroque) o cambia el tipo de la pieza que
/// llega, y ninguna de esas tres cosas la ven las mascaras. Son una minoria
/// de las jugadas, asi que mandarlas al camino caro no cuesta casi nada.
#[inline]
fn da_jaque_con_mascaras(b: &Board, mv: &Move, m: &MascarasJaque) -> bool {
    use crate::bitboard::bit;
    if mv.promotion.is_some()
        || matches!(
            mv.flag,
            MoveFlag::EnPassant | MoveFlag::CastleKing | MoveFlag::CastleQueen
        )
    {
        return da_jaque_sin_copiar(b, mv);
    }
    if bit(mv.from) & m.descubridores == 0 {
        // No puede destapar nada: solo queda el jaque DIRECTO, y para eso la
        // mascara del tipo de la pieza es exacta (si la casilla de origen
        // tapara el rayo entre el destino y el rey rival, la pieza ya estaria
        // dando jaque desde el origen -- posicion imposible con el rival al
        // no estar en jaque).
        let pt = match b.piece_at(mv.from) {
            Some((_, pt)) => pt as usize,
            None => return da_jaque_sin_copiar(b, mv),
        };
        if bit(mv.to) & m.por_tipo[pt] == 0 {
            return false;
        }
    }
    da_jaque_sin_copiar(b, mv)
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
/// Sugerencia de prefetch de LECTURA a la cache L1.
///
/// En ARM64 es la instruccion `prfm pldl1keep`, un HINT puro: no lee ni
/// escribe memoria, no puede fallar (una direccion invalida simplemente se
/// ignora), no toca registros ni banderas. Se emite con `asm!` porque el
/// intrinseco `core::arch::aarch64::_prefetch` todavia es inestable en Rust
/// 1.97. En cualquier otra arquitectura es un no-op.
#[inline(always)]
fn prefetch_lectura(p: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) p,
            options(nostack, readonly, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = p;
}

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

// bits  0..18 (18): jugada
// bits 18..34 (16): score
// bits 34..41 ( 7): profundidad
// bits 41..43 ( 2): flag de TT
// bits 43..47 ( 4): generacion (0..15; antes 5 bits/0..31, se recorto uno
//                    para robarle a este campo un bit gratis -- solo cambia
//                    cada cuantas busquedas se reinicia el aging, no afecta
//                    corretitud)
// bit  47     ( 1): was_pv -- "estuvo en PV alguna vez" (persistente,
//                    distinto de es_pv que es la ventana ACTUAL)
// bit  48     ( 1): "ocupado"
// bits 49..64 (15): verificacion de clave
const TT_WAS_PV: u64 = 1 << 47;

fn tt_empaquetar(entry: &TTEntry, key: u64, generation: u8) -> u64 {
    let mv = tt_empaquetar_move(entry.best) & 0x3FFFF; // 18 bits
    let score = (entry.score as i16 as u16) as u64; // 16 bits, con signo preservado
    let depth = (entry.depth.clamp(0, 127) as u64) & 0x7F; // 7 bits
    let flag = (entry.flag as u64) & 0x3; // 2 bits
    let generacion = (generation as u64) & 0xF; // 4 bits
    let was_pv = if entry.was_pv { TT_WAS_PV } else { 0 };
    let verif = (key >> (64 - 15)) & 0x7FFF; // 15 bits altos de la clave real
    mv | (score << 18)
        | (depth << 34)
        | (flag << 41)
        | (generacion << 43)
        | was_pv
        | TT_OCUPADO
        | (verif << 49)
}

/// Datos del OCUPANTE de un casillero de la TT compartida que necesita la
/// regla de reemplazo, decodificados SIN exigir que la clave coincida.
///
/// `tt_desempaquetar` devuelve None cuando los bits de verificacion no
/// coinciden (es lo correcto para LEER: esa entrada es de otra posicion).
/// Pero para DECIDIR EL REEMPLAZO hace falta saber que hay en el casillero
/// aunque sea de otra posicion -- si no, toda colision de clave parece un
/// casillero vacio y se pisa siempre, que es justo lo contrario de la
/// politica documentada en `tt_store`.
struct OcupanteTT {
    depth: i32,
    flag: TTFlag,
    generation: u8,
    misma_clave: bool,
}

fn tt_ocupante(paquete: u64, key: u64) -> Option<OcupanteTT> {
    if paquete & TT_OCUPADO == 0 {
        return None;
    }
    Some(OcupanteTT {
        depth: ((paquete >> 34) & 0x7F) as i32,
        flag: tt_flag_desde_u64((paquete >> 41) & 0x3),
        generation: ((paquete >> 43) & 0xF) as u8,
        misma_clave: ((paquete >> 49) & 0x7FFF) == ((key >> (64 - 15)) & 0x7FFF),
    })
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
    let generation = ((paquete >> 43) & 0xF) as u8;
    let was_pv = paquete & TT_WAS_PV != 0;
    Some(TTEntry { key, depth, score, flag, best: mv, generation, was_pv })
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
    /// Techo DURO opcional del reloj (gestion de tiempo de dos presupuestos,
    /// estilo timeman.cpp de Stockfish). `search_time` recibe el presupuesto
    /// "optimo" (objetivo normal); este campo, si esta puesto, es el
    /// "maximo": el motor puede estirarse hasta aca cuando la posicion lo
    /// pide (score que cae, mejor jugada inestable) pero JAMAS lo cruza.
    /// None = comportamiento clasico (maximo == optimo).
    pub tiempo_maximo_ms: Option<u64>,
    /// Techo DURO opcional de nodos ("go nodes N" de UCI). Se revisa en el
    /// mismo chequeo periodico que el deadline de tiempo (check_time, cada
    /// 256 nodos DENTRO del arbol, no solo entre iteraciones de
    /// profundizacion) -- si solo se revisara entre iteraciones, el
    /// crecimiento exponencial de nodos por profundidad (factor de
    /// ramificacion) haria que el motor se pase por varias veces el limite
    /// antes de que el chequeo entre depths llegue a ejecutarse.
    pub nodes_limit: Option<u64>,
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
    // Capture history: historial analogo al de las quiets pero para capturas,
    // indexado [pieza_que_mueve][destino][victima]. Solo se usa como DESEMPATE
    // dentro del carril que ya define el SEE (ver clave_orden_movimiento).
    capture_history: Vec<i32>,
    // Counter moves: la mejor respuesta silenciosa conocida contra cada
    // jugada rival (pieza, destino). Se prueba en el ordenamiento despues
    // de los killers.
    counter_moves: Vec<Option<Move>>,
    pub modo_lmr: bool,
    // Desactivable solo para comparacion A/B en pruebas (MITTENS_NO_ASPIRATION=1)
    // -- en juego real siempre queda activado, la tecnica en si es segura por
    // construccion (ensancha hasta ventana completa si hace falta).
    pub modo_aspiration: bool,
    // Singular extensions: activadas por defecto desde el h2h de 250
    // partidas a 20ms (55.0% para singular -- justo en el umbral, no un
    // margen amplio como LMP/history malus, pero cumple el criterio).
    // MITTENS_SINGULAR=0 las desactiva si hiciera falta comparar de nuevo.
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
    // Buffer de claves de orden POR PLY para el bucle principal de negamax.
    // Tiene que ser por ply (no uno solo compartido) porque el nodo padre
    // sigue usando sus claves mientras los hijos buscan. Va en el heap del
    // Searcher y no en la pila del frame: una array local de 1KB se
    // inicializaria a cero en CADA nodo, y eso solo ya se comia la ganancia.
    claves_negamax: Vec<[i32; MAX_MOVES]>,
    // Paso 2 del SEE deduplicado: SEE de cada jugada del bucle principal de
    // negamax, calculado UNA sola vez en el mismo momento que la clave de
    // orden (ver claves_negamax arriba) para no recalcularlo despues en el
    // SEE-prune del bucle. Centinela i32::MIN para jugadas que no son
    // capturas. MISMO ciclo de vida y tamano que claves_negamax, y debe
    // permutarse EXACTAMENTE igual en cada reordenamiento/rotacion de
    // `moves` -- si se desincroniza, el SEE queda pegado a la jugada
    // equivocada (bug silencioso). Ver debug_assert en el punto de consumo
    // (SEE-prune) que detecta esa desincronizacion en tests.
    see_negamax: Vec<[i32; MAX_MOVES]>,
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
    // Lazy SMP: escalonamiento de profundidad (skip arrays al estilo
    // Stockfish clasico). `Some((size, phase))` hace que este hilo AYUDANTE
    // se saltee bloques de iteraciones de la profundizacion: salta la
    // profundidad d cuando ((d + phase) / size) % 2 == 1 (d=1 nunca se
    // salta). El efecto es que en todo momento ~la mitad de los ayudantes
    // esta buscando 1-2 plies POR DELANTE del hilo principal, y sus entradas
    // de TT (mas profundas que lo que el principal ya tiene) le aceleran las
    // iteraciones. Esta era exactamente la pieza que faltaba: medido el
    // 2026-08-20, sin esto 8 hilos quemaban 5-6x los nodos de 1 hilo para
    // ganar ~0-1 ply de profundidad efectiva (ver HALLAZGOS_SMP_2026-08-20.md).
    // Reemplaza a la diversificacion anterior (swap de las 2 primeras
    // jugadas de la raiz en hilos impares), que desperdiciaba la ventana de
    // aspiration buscando la 2a mejor jugada primero sin adelantar a nadie.
    pub salto_smp: Option<(u32, u32)>,
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
    /// Mientras esta en true, `check_time` NO levanta `self.stop`: ni por
    /// deadline ni por la bandera externa. Se usa SOLO durante la primera
    /// iteracion de profundizacion (ver search_time) para garantizar que la
    /// profundidad 1 siempre termine y `mejor_mv` sea una jugada realmente
    /// buscada, nunca el fallback de generacion. Cuesta microsegundos.
    blindar_stop: bool,
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
            tiempo_maximo_ms: None,
            nodes_limit: None,
            stop: false,
            tt_generation: 0,
            killers: vec![[None, None]; MAX_KILLER_PLY],
            history: Box::new([[0i32; 64]; 64]),
            history_amenaza: Box::new([[0i32; 64]; 64]),
            cont_history: vec![0i32; CONT_HIST_SIZE],
            cont_history_2: vec![0i32; CONT_HIST_SIZE],
            capture_history: vec![0i32; CAPT_HIST_SIZE],
            counter_moves: vec![None; 6 * 64],
            // Activado por defecto: el torneo h2h de esta sesion confirmo
            // +80 ELO (61.3% en 40 partidas) con la reescritura PVS -- ver
            // resultados_lmr_h2h.txt en ~/mi-motor. MITTENS_LMR=0 lo desactiva
            // explicitamente para pruebas comparativas.
            modo_lmr: std::env::var("MITTENS_LMR").as_deref() != Ok("0"),
            modo_aspiration: std::env::var("MITTENS_NO_ASPIRATION").as_deref() != Ok("1"),
            modo_singular: std::env::var("MITTENS_SINGULAR").as_deref() != Ok("0"),
            qsearch_nnue: true,
            nnue: None,
            nnue_classical_depth: 0,
            game_history: Vec::new(),
            path: [0u64; MAX_PATH],
            path_len: 0,
            claves_negamax: vec![[0i32; MAX_MOVES]; MAX_PLY as usize + 2],
            see_negamax: vec![[i32::MIN; MAX_MOVES]; MAX_PLY as usize + 2],
            lmr_intentos: 0,
            lmr_reintentos: 0,
            hindsight_parent_eval: vec![0; MAX_KILLER_PLY],
            hindsight_reduction: vec![0; MAX_KILLER_PLY],
            eval_stack: vec![EVAL_INVALIDA; MAX_KILLER_PLY],
            corr_pawn: vec![0; 2 * CORR_SIZE],
            corr_nonpawn: [vec![0; 2 * CORR_SIZE], vec![0; 2 * CORR_SIZE]],
            corr_cont: vec![0; 6 * 64 * 2],
            salto_smp: None,
            root_moves_filtro: None,
            null_move_r_extra: 0,
            external_stop: None,
            blindar_stop: false,
        }
    }

    /// Se lleva las tablas que acumulan entre jugadas (ver TablasPersistentes).
    /// CONSUME el Searcher: se usa justo antes de tirarlo al final de un hilo
    /// de Lazy SMP, de modo que no puede quedar un Searcher a medio vaciar.
    pub fn extraer_memoria(self) -> MemoriaHilo {
        MemoriaHilo(Some(TablasPersistentes {
            history: self.history,
            history_amenaza: self.history_amenaza,
            cont_history: self.cont_history,
            cont_history_2: self.cont_history_2,
            capture_history: self.capture_history,
            counter_moves: self.counter_moves,
            corr_pawn: self.corr_pawn,
            corr_nonpawn: self.corr_nonpawn,
            corr_cont: self.corr_cont,
        }))
    }

    /// Reinstala tablas guardadas de una jugada anterior de ESTE mismo hilo.
    /// Si algun tamaño no coincide con el que espera este binario (no deberia
    /// pasar: son constantes de compilacion) se ignora la memoria entera y el
    /// Searcher se queda con sus tablas en cero, que siempre es correcto.
    pub fn instalar_memoria(&mut self, m: MemoriaHilo) {
        let t = match m.0 {
            Some(t) => t,
            None => return,
        };
        let compatible = t.cont_history.len() == self.cont_history.len()
            && t.cont_history_2.len() == self.cont_history_2.len()
            && t.capture_history.len() == self.capture_history.len()
            && t.counter_moves.len() == self.counter_moves.len()
            && t.corr_pawn.len() == self.corr_pawn.len()
            && t.corr_nonpawn[0].len() == self.corr_nonpawn[0].len()
            && t.corr_nonpawn[1].len() == self.corr_nonpawn[1].len()
            && t.corr_cont.len() == self.corr_cont.len();
        if !compatible {
            return;
        }
        self.history = t.history;
        self.history_amenaza = t.history_amenaza;
        self.cont_history = t.cont_history;
        self.cont_history_2 = t.cont_history_2;
        self.capture_history = t.capture_history;
        self.counter_moves = t.counter_moves;
        self.corr_pawn = t.corr_pawn;
        self.corr_nonpawn = t.corr_nonpawn;
        self.corr_cont = t.corr_cont;
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
            tiempo_maximo_ms: None,
            nodes_limit: None,
            stop: false,
            killers: vec![[None, None]; MAX_KILLER_PLY],
            history: Box::new([[0i32; 64]; 64]),
            history_amenaza: Box::new([[0i32; 64]; 64]),
            cont_history: vec![0i32; CONT_HIST_SIZE],
            cont_history_2: vec![0i32; CONT_HIST_SIZE],
            capture_history: vec![0i32; CAPT_HIST_SIZE],
            counter_moves: vec![None; 6 * 64],
            tt_generation: 0,
            modo_lmr,
            modo_aspiration: std::env::var("MITTENS_NO_ASPIRATION").as_deref() != Ok("1"),
            modo_singular: std::env::var("MITTENS_SINGULAR").as_deref() != Ok("0"),
            qsearch_nnue: true,
            nnue: None,
            nnue_classical_depth: 0,
            game_history: Vec::new(),
            path: [0u64; MAX_PATH],
            path_len: 0,
            claves_negamax: vec![[0i32; MAX_MOVES]; MAX_PLY as usize + 2],
            see_negamax: vec![[i32::MIN; MAX_MOVES]; MAX_PLY as usize + 2],
            lmr_intentos: 0,
            lmr_reintentos: 0,
            hindsight_parent_eval: vec![0; MAX_KILLER_PLY],
            hindsight_reduction: vec![0; MAX_KILLER_PLY],
            eval_stack: vec![EVAL_INVALIDA; MAX_KILLER_PLY],
            corr_pawn: vec![0; 2 * CORR_SIZE],
            corr_nonpawn: [vec![0; 2 * CORR_SIZE], vec![0; 2 * CORR_SIZE]],
            corr_cont: vec![0; 6 * 64 * 2],
            salto_smp: None,
            root_moves_filtro: None,
            null_move_r_extra: 0,
            external_stop: None,
            blindar_stop: false,
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
    ///
    /// UNICO punto del motor que entrega una referencia al acumulador, y por
    /// eso es aqui donde se paga la actualizacion perezosa: `entrar_hijo`
    /// solo anota el delta de cada ply, y la pasada real (ponerse al dia con
    /// la cadena pendiente) se hace recien cuando alguien pide leer. Si un
    /// camino nuevo consiguiera el acumulador sin pasar por aca,
    /// `AcumBullet::evaluar` lo caza con un debug_assert.
    #[inline]
    fn nnue_de(&mut self, eval_state: &EvalState) -> Option<&crate::neural::NnueAccumulator> {
        if eval_state.nnue_activo() {
            let acum = self.nnue.as_mut()?;
            acum.asegurar_actualizado();
            Some(acum)
        } else {
            None
        }
    }

    /// True si este nodo evalua por el camino "red bullet pura", el unico
    /// que puede usar la cache de eval por zobrist. NO toca el acumulador
    /// (solo mira que arquitectura hay cargada), asi que se puede consultar
    /// ANTES de materializar -- que es justo lo que permite que un acierto
    /// de cache no pague ninguna pasada del acumulador.
    #[inline]
    fn usa_cache_eval(&self, eval_state: &EvalState) -> bool {
        eval_state.nnue_activo()
            && match self.nnue.as_ref() {
                Some(acum) => acum.pura(),
                None => false,
            }
    }

    /// True si en ESTA configuracion la parte clasica de la evaluacion puede
    /// entrar en el puntaje de algun nodo. Cuando es false, mantener el
    /// acumulador clasico por delta en cada nodo es trabajo puro perdido
    /// (era ~10% del tiempo de busqueda segun el perfilado): la unica funcion
    /// que aun la consulta es `draw_score`, que la recalcula desde el
    /// tablero en las pocas posiciones de repeticion/regla de 50.
    ///
    /// Se exige que las TRES condiciones se cumplan para poder saltarla:
    ///   - hay acumulador NNUE y su red es "pura" (evalua sola),
    ///   - la quiescence usa NNUE (si no, su stand-pat es clasico puro),
    ///   - no hay nodos que suelten la NNUE por profundidad
    ///     (`NNUEClassicalDepth`), que evaluarian con la clasica.
    #[inline]
    fn clasica_necesaria(&self) -> bool {
        if self.nnue_classical_depth > 0 || !self.qsearch_nnue {
            return true;
        }
        match self.nnue.as_ref() {
            Some(acum) => !acum.pura(),
            None => true,
        }
    }

    #[inline]
    fn evaluar_completo(&mut self, b: &Board, eval_state: &EvalState) -> i32 {
        // Cache de eval por zobrist: SOLO en el camino "red bullet pura"
        // (el default de produccion), donde la eval es funcion pura de la
        // posicion -- ver src/eval_cache.rs. En cualquier otro camino
        // (clasica, hibrido, red de amenazas) se evalua directo, como antes.
        //
        // OJO AL ORDEN: la consulta a la cache va ANTES de pedir el
        // acumulador. Un acierto devuelve el valor sin materializar nada, y
        // esa es la mitad del ahorro de la actualizacion perezosa: el nodo
        // no paga la pasada del acumulador, y si ademas ninguno de sus
        // descendientes lee, tampoco la pagan ellos.
        if self.usa_cache_eval(eval_state) {
            if let Some(v) = crate::eval_cache::probe(b.zobrist) {
                // En builds de debug, cada acierto se verifica contra la
                // eval recomputada: si esto falla, la cache esta
                // devolviendo un valor que NO es el de la posicion.
                debug_assert_eq!(
                    v,
                    {
                        let nnue = self.nnue_de(eval_state);
                        evaluate_with_state(b, eval_state, nnue)
                    },
                    "eval_cache devolvio un valor distinto del recomputado"
                );
                return v;
            }
            let nnue = self.nnue_de(eval_state);
            let v = evaluate_with_state(b, eval_state, nnue);
            crate::eval_cache::store(b.zobrist, v);
            return v;
        }
        let nnue = self.nnue_de(eval_state);
        evaluate_with_state(b, eval_state, nnue)
    }

    #[inline]
    fn evaluar_quiescence(&mut self, b: &Board, eval_state: &EvalState) -> i32 {
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
                acum.entrar(antes, despues);
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
                acum.salir(antes, despues);
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
            if self.clasica_necesaria() {
                eval_state.despues_de_jugada(antes, despues)
            } else {
                eval_state.despues_de_jugada_sin_clasica(antes, despues)
            }
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
        } else if self.clasica_necesaria() {
            eval_state.despues_de_jugada(antes, despues)
        } else {
            eval_state.despues_de_jugada_sin_clasica(antes, despues)
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
        for v in self.capture_history.iter_mut() {
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
    /// Refina la eval estatica con el score guardado en la TT.
    ///
    /// Hallazgo de Stockfish: si hay entrada en la TT, su score viene de una
    /// BUSQUEDA de verdad, asi que es mucho mas preciso que la eval estatica
    /// para alimentar razoring / futilidad / null-move. Solo se adopta
    /// cuando la cota de la entrada es coherente con la direccion del
    /// cambio (un fail-high solo puede SUBIR la estimacion, un fail-low solo
    /// bajarla) y lejos de puntajes de mate, donde no es comparable con una
    /// eval estatica.
    #[inline]
    fn eval_con_tt(&self, eval: i32, entrada: Option<&TTEntry>) -> i32 {
        let Some(e) = entrada else { return eval };
        if e.score.abs() >= MATE - 1000 {
            return eval;
        }
        match e.flag {
            TTFlag::Exact => e.score,
            TTFlag::Beta if e.score > eval => e.score,
            TTFlag::Alpha if e.score < eval => e.score,
            _ => eval,
        }
    }

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
            // Las capturas tienen su propia tabla: el orden grueso lo sigue
            // dando SEE, esto solo afina entre capturas de valor parecido.
            hist_update(
                &mut self.capture_history[capt_idx(pt_mv, mv.to as usize, victima_de(b, &mv))],
                hist_bonus(depth),
            );
            return; // no tocan killers/history/counter/cont (son para quiets)
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
            hist_update(
                &mut self.history_amenaza[mv.from as usize][mv.to as usize],
                hist_bonus(depth),
            );
        } else {
            hist_update(&mut self.history[mv.from as usize][mv.to as usize], hist_bonus(depth));
        }
        if let Some((prev_pt, prev_to)) = prev {
            let idx = cont_idx(prev_pt, prev_to, pt_mv, mv.to as usize);
            hist_update(&mut self.cont_history[idx], hist_bonus(depth));
            self.counter_moves[prev_pt * 64 + prev_to] = Some(mv);
        }
        if let Some((p2_pt, p2_to)) = prev2 {
            let idx = cont_idx(p2_pt, p2_to, pt_mv, mv.to as usize);
            hist_update(&mut self.cont_history_2[idx], hist_bonus(depth));
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

    /// Pide al procesador que traiga a cache la linea de la TT donde caeria
    /// `key`, SIN leerla. Se llama justo despues de `make_move`, o sea unos
    /// cientos de ciclos antes de que el hijo haga su `tt_probe`: en ese
    /// hueco (que se llena con la actualizacion NNUE del acumulador, lo mas
    /// caro del nodo) la linea llega sola y el probe deja de esperar a
    /// memoria. `tt_probe` fue el 4% del tiempo en el perfilado y es casi
    /// todo latencia de fallo de cache: la TT es de decenas/cientos de MB y
    /// el indice es un hash, o sea acceso aleatorio puro.
    ///
    /// `prfm` es una instruccion HINT del ARM: no lee, no escribe, no falla
    /// nunca (ni con una direccion invalida) y no cambia ningun registro ni
    /// bandera. Por construccion no puede alterar el resultado de la
    /// busqueda: el arbol, el conteo de nodos y el puntaje son identicos.
    #[inline(always)]
    fn tt_prefetch(&self, key: u64) {
        let idx = self.tt_index(key);
        let p: *const u8 = match &self.tt {
            TablaTransposicion::Local(tt) => tt.as_ptr().wrapping_add(idx) as *const u8,
            TablaTransposicion::Compartida(tt) => tt.as_ptr().wrapping_add(idx) as *const u8,
        };
        prefetch_lectura(p);
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
        was_pv: bool,
    ) {
        // Copiar la generacion antes del closure: el closure se usa mientras
        // `self.tt` esta prestado mutablemente, asi que no puede capturar
        // `self` por referencia.
        let generacion = self.tt_generation;
        // Antes: una colision de OTRA clave siempre pisaba, sin mirar
        // profundidad. Eso dejaba que las escrituras masivas de quiescence
        // (depth=0, en casi todos los nodos) desalojaran constantemente las
        // entradas profundas de negamax que caian en el mismo casillero,
        // reduciendo la TT efectiva de la busqueda principal. Ahora una
        // colision de otra clave se trata igual que una de la MISMA clave:
        // solo reemplaza si es al menos tan profunda (o si la entrada previa
        // es de una generacion vieja, que siempre se descarta).
        //
        // OJO (bug corregido): esta regla se evaluaba sobre el resultado de
        // `tt_desempaquetar`, que devuelve None cuando los bits de
        // verificacion NO coinciden. Es decir: en la TT COMPARTIDA -- la
        // unica que se usa en produccion -- toda colision de otra clave
        // parecia un casillero VACIO y se pisaba siempre, sin mirar la
        // profundidad. La politica descrita arriba solo estaba viva en la TT
        // Local (un solo hilo, camino de tests). Ahora el ocupante se decodifica
        // con `tt_ocupante`, que no exige que la clave coincida.
        let reemplazar = |slot: Option<OcupanteTT>| match slot {
            None => true,
            Some(existing) if existing.generation != generacion => true,
            Some(existing) => {
                depth > existing.depth
                    || (depth == existing.depth
                        && (!existing.misma_clave
                            || (flag == TTFlag::Exact && existing.flag != TTFlag::Exact)))
            }
        };
        let entry = TTEntry {
            key,
            depth,
            score: score_to_tt(score, ply),
            flag,
            best,
            generation: generacion,
            was_pv,
        };
        let idx = self.tt_index(key);
        match &mut self.tt {
            TablaTransposicion::Local(tt) => {
                let ocupante = tt[idx].map(|e| OcupanteTT {
                    depth: e.depth,
                    flag: e.flag,
                    generation: e.generation,
                    misma_clave: e.key == key,
                });
                if reemplazar(ocupante) {
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
                let actual = tt_ocupante(tt[idx].load(Ordering::Relaxed), key);
                if reemplazar(actual) {
                    tt[idx].store(tt_empaquetar(&entry, key, generacion), Ordering::Relaxed);
                }
            }
        }
    }

    fn check_time(&mut self) -> Result<(), TimeUp> {
        self.nodes += 1;
        if !self.stop && !self.blindar_stop && (self.nodes == 1 || self.nodes & 255 == 0) {
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
            // Limite de nodos ("go nodes N"): mismo chequeo periodico (cada
            // 256 nodos) que ya se usa para el deadline de tiempo, no solo
            // entre iteraciones de profundizacion -- ver comentario en
            // `nodes_limit`.
            if !self.stop
                && let Some(lim) = self.nodes_limit
                && self.nodes >= lim
            {
                self.stop = true;
            }
        }
        if self.stop { Err(TimeUp) } else { Ok(()) }
    }

    fn order_moves(&mut self, b: &Board, moves: &mut [Move], tt_move: Option<Move>) {
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
        // Mapa de casillas atacadas por el rival, construido perezosamente
        // UNA vez por nodo y compartido entre todas las jugadas silenciosas
        // (antes: is_square_attacked_by por cada jugada, ~5 lookups cada una).
        // Equivalencia exacta garantizada por Board::attack_map.
        amenazas: &mut Option<u64>,
    ) -> i32 {
        if Some(*mv) == tt_move {
            return -1_000_000;
        }
        if mv.is_capture() {
            let see = see_precalculado.unwrap_or_else(|| crate::see::see(b, mv));
            let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
            // CLAMP obligatorio: el capture history crece sin cota (bonus
            // depth*depth), y sin acotarlo una captura mala podria colarse
            // delante de la jugada de la TT. Con el SEE multiplicado por 8 el
            // rango material ocupa [0, 7200], asi que un desempate de +-400
            // solo reordena capturas de valor CASI IGUAL: nunca saca a una
            // captura de su carril.
            let ch = self.capture_history[capt_idx(pt_mv, mv.to as usize, victima_de(b, mv))]
                .clamp(-400, 400);
            if see >= 0 {
                return -(10_000 + see * 8 + ch);
            }
            return 1000 - see - ch;
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
        let mapa = *amenazas.get_or_insert_with(|| b.attack_map(b.turn.opposite()));
        let h = if crate::bitboard::bit(mv.to) & mapa != 0 {
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
        // Escalado /16: con el fix de escala de hist_bonus las tablas ahora
        // se mueven de verdad en el rango +-16384, y la suma cruda (hasta
        // +-49k) invadiria los carriles de capturas buenas (-10k..-17k) y
        // killers (-3000). Dividida por 16 queda acotada a ~+-3072, ordenando
        // los quiets entre si con buena granularidad sin sacar a ninguno de
        // su carril salvo casos extremos.
        -((h + ch + ch2) / 16)
    }

    /// Igual que order_moves pero ademas usa killers/history (por ply) para
    /// ordenar las jugadas silenciosas -- capturas/TT siguen mandando.
    /// `prev` es (pieza, casillero_destino) de la jugada rival que llevo a
    /// esta posicion (None si no se conoce, p.ej. en la raiz) -- alimenta la
    /// continuation history para las jugadas silenciosas. `prev2` es la
    /// jugada PROPIA de 2 plies atras -- alimenta el follow-up history.
    fn order_moves_ply(
        &mut self,
        b: &Board,
        moves: &mut [Move],
        tt_move: Option<Move>,
        ply: u32,
        prev: Option<(usize, usize)>,
        prev2: Option<(usize, usize)>,
    ) {
        // SIN asignacion de heap: sort_by_cached_key creaba un Vec<(clave,
        // indice)> temporal por nodo. Aqui se precomputan las claves una sola
        // vez en el buffer FIJO del Searcher (pila, reutilizable) y se ordena
        // con insertion sort manual. El numero de jugadas suele ser <40 y el
        // insertion sort ESTABLE conserva exactamente el mismo orden final
        // que el sort estable de la std (claves iguales mantienen el orden
        // relativo original).
        let n = moves.len();
        debug_assert!(n <= MAX_MOVES);
        // El buffer de claves es local y SIN inicializar: solo se escriben las
        // `n` entradas que se usan. Antes era un campo del Searcher que se
        // COPIABA entero a la pila al entrar (`let mut claves =
        // self.claves_orden_movimiento`) y se copiaba de vuelta al salir --
        // 2 x 1 KB de memmove por llamada, aunque la lista tuviera 3 jugadas.
        // (La copia existia solo para no tener un prestamo mutable de `self`
        // vivo mientras se llama a `clave_orden_movimiento`, que toma `&self`;
        // con el buffer local ese conflicto no existe.)
        let mut claves: ArrayVec<i32, MAX_MOVES> = ArrayVec::new();
        let mut amenazas: Option<u64> = None;
        for mv in moves.iter() {
            claves.push(
                self.clave_orden_movimiento(b, mv, tt_move, ply, prev, prev2, None, &mut amenazas),
            );
        }
        // Insertion sort estable sobre (moves, claves) en paralelo.
        for i in 1..n {
            let mv = moves[i];
            let k = claves[i];
            let mut j = i;
            while j > 0 && claves[j - 1] > k {
                moves[j] = moves[j - 1];
                claves[j] = claves[j - 1];
                j -= 1;
            }
            moves[j] = mv;
            claves[j] = k;
        }
    }

    fn quiescence(
        &mut self,
        b: &Board,
        eval_state: &EvalState,
        mut alpha: i32,
        beta: i32,
        ply: u32,
        // Profundidad DENTRO de quiescence (no confundir con `ply`, el ply
        // absoluto del arbol completo). Siempre 0 en las llamadas desde
        // FUERA de quiescence (negamax, ProbCut); se incrementa en 1 en cada
        // llamada recursiva DENTRO de quiescence. Unico proposito: permitir
        // que SOLO el primer nivel (qdepth == 0) pruebe jaques silenciosos
        // (ver mas abajo) -- a partir de qdepth >= 1 nunca se generan, eso es
        // lo que evita la explosion combinatoria de jaques encadenados.
        qdepth: i32,
    ) -> Result<i32, TimeUp> {
        self.check_time()?;
        // Quiescence también puede cruzar una secuencia de 50 plies sin
        // captura ni peón. No usar el stand-pat allí: por regla es tablas.
        if b.halfmove_clock >= 100 {
            return Ok(draw_score(b, eval_state, self.nnue_de(eval_state)));
        }

        // TT en quiescence: cualquier entrada (todas tienen depth >= 0, y
        // quiescence equivale a depth <= 0) sirve para cortar aqui con las
        // mismas reglas de cota que en negamax. Ahorra la eval NNUE completa
        // y toda la generacion/orden de capturas en posiciones ya resueltas.
        let key = b.zobrist;
        // Igual que en negamax: el prefetch viaja en paralelo con el probe
        // de la TT para que el stand-pat (si no hay corte) encuentre la
        // linea de la cache de eval ya caliente.
        crate::eval_cache::prefetch(key);
        let alpha_orig = alpha;
        let mut tt_mv: Option<Move> = None;
        if let Some(mut entry) = self.tt_probe(key) {
            entry.score = score_from_tt(entry.score, ply);
            tt_mv = entry.best;
            // Misma guarda de la regla de 50 que en negamax (ver alli): con
            // el contador alto, un score guardado con el contador bajo puede
            // prometer un resultado que la regla de 50 ya no permite.
            if b.halfmove_clock < 90 {
                match entry.flag {
                    TTFlag::Exact => return Ok(entry.score),
                    TTFlag::Beta if entry.score >= beta => return Ok(entry.score),
                    TTFlag::Alpha if entry.score <= alpha => return Ok(entry.score),
                    _ => {}
                }
            }
        }
        let en_jaque = b.in_check(b.turn);

        // En jaque no existe "stand pat": quedarse quieto es ilegal. Se deben
        // buscar TODAS las evasiones legales, incluidas jugadas silenciosas.
        if en_jaque {
            let mut evasiones = MoveList::new();
            generate_legal_into_con_jaque(b, &mut evasiones, true);
            if evasiones.is_empty() {
                return Ok(-MATE + ply as i32);
            }
            self.order_moves_ply(b, &mut evasiones, tt_mv, ply, None, None);
            if ply >= MAX_PLY {
                // Tope defensivo contra secuencias patologicas de jaques. Aun
                // detectamos mate arriba; para posiciones no terminales usamos
                // la evaluacion estatica en vez de desbordar la pila.
                return Ok(self.evaluar_quiescence(b, eval_state));
            }
            let mut best = -INFINITO;
            let mut best_mv: Option<Move> = None;
            for &mv in evasiones.iter() {
                let next = b.make_move(&mv);
                self.tt_prefetch(next.zobrist);
                let next_eval = self.siguiente_estado_quiescence(eval_state, b, &next);
                // Ligar el resultado ANTES de deshacer, y recien despues
                // aplicar el `?`: asi ningun retorno temprano se salta el
                // undo del acumulador NNUE.
                let res = self.quiescence(&next, &next_eval, -beta, -alpha, ply + 1, qdepth + 1);
                self.salir_hijo(&next_eval, b, &next);
                let sc = -res?;
                if sc > best {
                    best = sc;
                    best_mv = Some(mv);
                }
                alpha = alpha.max(sc);
                if alpha >= beta {
                    break;
                }
            }
            let flag = if best >= beta {
                TTFlag::Beta
            } else if best <= alpha_orig {
                TTFlag::Alpha
            } else {
                TTFlag::Exact
            };
            self.tt_store(key, 0, best, ply, flag, best_mv, false);
            return Ok(best);
        }

        // CORRECTION HISTORY tambien en el stand-pat de quiescence: es la
        // misma eval estatica que en negamax, asi que arrastra el mismo
        // sesgo sistematico y merece la misma correccion. Sin esto, negamax
        // y quiescence trabajaban con escalas ligeramente distintas.
        // Se pasa `prev = None` porque quiescence no recibe la jugada previa;
        // se aplican igual las correcciones por estructura de peones y por
        // piezas, que son las de mayor peso.
        let stand_pat = {
            let raw = self.evaluar_quiescence(b, eval_state);
            self.eval_corregida(b, raw, None)
        };
        if ply >= MAX_PLY {
            return Ok(stand_pat);
        }
        if stand_pat >= beta {
            self.tt_store(key, 0, stand_pat, ply, TTFlag::Beta, None, false);
            return Ok(beta);
        }
        alpha = alpha.max(stand_pat);

        // generate_captures_legal ya filtra a capturas/promociones sin
        // generar ni legalizar las jugadas silenciosas (que aqui se
        // descartarian de todos modos) -- evita el trabajo mas caro de
        // legalizar jugadas que nunca se van a buscar. Si NO hay capturas
        // legales, con eso solo no se puede distinguir "posicion tranquila
        // normal" de "ahogado real": para ese caso, y SOLO ese, se consulta
        // existe_jugada_legal, que corta en cuanto encuentra UNA jugada legal
        // (antes se generaba y legalizaba la lista COMPLETA solo para mirar
        // si quedaba vacia -- en cada hoja tranquila de quiescence).
        let mut capturas = MoveList::new();
        generate_captures_legal_into_con_jaque(b, &mut capturas, false);
        if capturas.is_empty()
            && !crate::movegen::existe_jugada_legal_con_jaque(b, false)
        {
            return Ok(0); // ahogado
        }
        // Igual que la lista principal: quiescence vive en las hojas y no
        // debe asignar un Vec por nodo solo para conservar el SEE calculado.
        let mut moves: ArrayVec<(Move, Option<i32>), MAX_MOVES> = ArrayVec::new();
        for mv in capturas.iter().copied() {
            let see = mv.is_capture().then(|| crate::see::see(b, &mv));
            moves.push((mv, see));
        }
        // En quiescence la poda SEE se aplica despues del ordenamiento. Llevar
        // el resultado junto a la jugada evita calcular el mismo SEE dos veces.
        //
        // ORDEN: antes esto era `moves.sort_by_key(|..| clave_orden_movimiento(..))`.
        // `sort_by_key` vuelve a llamar a la funcion de clave en CADA
        // comparacion (O(n log n) llamadas), no una vez por elemento; y
        // `clave_orden_movimiento` no es barata (piece_at, capture history,
        // mapa de amenazas). Aca la clave se calcula UNA vez por jugada y el
        // orden se hace con el mismo insertion sort ESTABLE que usa negamax.
        // El orden final es identico: `sort_by_key` tambien es estable y la
        // funcion de clave es determinista (el memo `amenazas_q` devuelve
        // siempre el mismo mapa).
        let mut amenazas_q: Option<u64> = None;
        let mut claves_q: ArrayVec<i32, MAX_MOVES> = ArrayVec::new();
        for (mv, see) in moves.iter() {
            claves_q.push(self.clave_orden_movimiento(
                b, mv, tt_mv, ply, None, None, *see, &mut amenazas_q,
            ));
        }
        for i in 1..moves.len() {
            let par = moves[i];
            let k = claves_q[i];
            let mut j = i;
            while j > 0 && claves_q[j - 1] > k {
                moves[j] = moves[j - 1];
                claves_q[j] = claves_q[j - 1];
                j -= 1;
            }
            moves[j] = par;
            claves_q[j] = k;
        }

        let mut best = stand_pat;
        let mut best_mv: Option<Move> = None;
        // Mascaras de jaque del nodo (perezosas): el bucle de abajo necesita
        // saber si cada captura da jaque ANTES de decidir si la poda.
        let mut mascaras_cap: Option<MascarasJaque> = None;
        for &(mv, see) in moves.iter() {
            // ORDEN: primero se decide el jaque y se aplican las podas, y
            // RECIEN DESPUES se construye el tablero hijo. Antes se hacia
            // `b.make_move` de entrada para poder preguntar
            // `next.in_check(next.turn)`, asi que toda captura podada pagaba
            // igual la copia del tablero: medido con contadores en `bench 14`,
            // de 258.290 `make_move` de este bucle 155.088 (el 60,0%) eran de
            // jugadas que se descartaban dos lineas mas abajo.
            // `da_jaque_con_mascaras` da exactamente la misma respuesta que
            // `next.in_check(next.turn)` (el rival es el lado que mueve en el
            // hijo) sin construir nada.
            let da_jaque = da_jaque_con_mascaras(
                b,
                &mv,
                mascaras_cap.get_or_insert_with(|| mascaras_jaque(b)),
            );

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

            // La jugada sobrevivio a las dos podas: recien aca vale la pena
            // construir el hijo.
            let next = b.make_move(&mv);
            self.tt_prefetch(next.zobrist);
            let next_eval = self.siguiente_estado_quiescence(eval_state, b, &next);
            let res = self.quiescence(&next, &next_eval, -beta, -alpha, ply + 1, qdepth + 1);
            self.salir_hijo(&next_eval, b, &next);
            let sc = -res?;
            if sc > best {
                best = sc;
                best_mv = Some(mv);
            }
            alpha = alpha.max(sc);
            if alpha >= beta {
                break;
            }
        }
        // Jaques silenciosos: SOLO en el primer nivel de quiescence
        // (qdepth == 0), y solo si las capturas no cortaron ya (alpha <
        // beta). Nunca se recursiona con qdepth 0 de nuevo -- las llamadas
        // de mas abajo usan qdepth + 1 -- asi que este bloque JAMAS se
        // ejecuta en qdepth >= 1. Eso es lo que evita la explosion de nodos:
        // sin este freno cada jaque silencioso podria a su vez probar sus
        // propios jaques silenciosos, y los de esos, encadenando sin fin.
        if qdepth == 0 && alpha < beta && stand_pat + 150 > alpha {
            let mut jaques_probados = 0usize;
            let mut jaques_lista = MoveList::new();
            generate_legal_into_con_jaque(b, &mut jaques_lista, false);
            // Este bucle era el mayor consumidor de `da_jaque_sin_copiar`:
            // recorre la lista legal COMPLETA (34,3 jugadas por entrada,
            // medido) preguntando una por una si dan jaque, y casi ninguna
            // lo da. Con las mascaras del nodo el descarte cuesta dos AND.
            let mascaras_q = mascaras_jaque(b);
            for mv in jaques_lista.iter().copied() {
                if jaques_probados >= 5 {
                    break;
                }
                // Las capturas/promociones ya se probaron arriba.
                if mv.is_capture() || mv.promotion.is_some() {
                    continue;
                }
                if !da_jaque_con_mascaras(b, &mv, &mascaras_q) {
                    continue;
                }
                // Filtro barato antes del caro: SEE solo sobre las que ya
                // dan jaque. Descarta jaques que regalan material sin
                // compensacion.
                if crate::see::see(b, &mv) < 0 {
                    continue;
                }
                jaques_probados += 1;

                let next = b.make_move(&mv);
                self.tt_prefetch(next.zobrist);
                let next_eval = self.siguiente_estado_quiescence(eval_state, b, &next);
                let res = self.quiescence(&next, &next_eval, -beta, -alpha, ply + 1, qdepth + 1);
                self.salir_hijo(&next_eval, b, &next);
                let sc = -res?;
                if sc > best {
                    best = sc;
                    best_mv = Some(mv);
                }
                alpha = alpha.max(sc);
                if alpha >= beta {
                    break;
                }
            }
        }

        let flag = if best >= beta {
            TTFlag::Beta
        } else if best <= alpha_orig {
            TTFlag::Alpha
        } else {
            TTFlag::Exact
        };
        self.tt_store(key, 0, best, ply, flag, best_mv, false);
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

        // Hindsight reductions, adaptado al LMR entero de Mittens. Si una
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
            return self.quiescence(b, eval_state, alpha, beta, ply, 0);
        }

        let alpha_orig = alpha;
        let key = b.zobrist;
        // Prefetch del casillero de la cache de eval: para cuando la eval
        // estatica se pida (RFP/razoring/eval_stack, mas abajo), la linea ya
        // viajo en paralelo con el probe de la TT de aca al lado.
        crate::eval_cache::prefetch(key);
        let mut tt_move = None;
        let mut tt_entry_full: Option<TTEntry> = None;
        if let Some(mut entry) = self.tt_probe(key) {
            entry.score = score_from_tt(entry.score, ply);
            tt_move = entry.best;
            tt_entry_full = Some(entry);
            // En nodos PV (ventana no nula) NO se corta por la TT: la PV
            // tiene que salir de una busqueda real, no de una entrada
            // guardada que puede venir de otra ventana o de otro camino. En
            // esos nodos la entrada solo aporta `tt_move` para el orden.
            let nodo_pv = beta > alpha + 1;
            // GUARDA DE LA REGLA DE 50 (faltaba): el zobrist NO incluye el
            // halfmove_clock, asi que la MISMA entrada de TT sirve a la misma
            // posicion con el contador de la regla de 50 en 5 o en 95. Un
            // "gano en N" guardado con el contador bajo es sencillamente
            // falso cuando quedan pocas jugadas reversibles: la partida es
            // tablas antes de que ese plan se complete. Stockfish desactiva
            // el corte por TT con rule50_count() >= 90 por exactamente este
            // motivo; aca no habia ninguna guarda. Solo se pierde el ATAJO:
            // la busqueda real de ese nodo si respeta la regla (el chequeo
            // halfmove_clock >= 100 esta al entrar a negamax y a quiescence).
            if entry.depth >= depth && !nodo_pv && b.halfmove_clock < 90 {
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
        // "Estuvo en PV alguna vez": la ventana ACTUAL puede ser nula (por
        // LMR/null-move desde otro camino) pero si la entrada de TT de este
        // nodo quedo marcada was_pv en una visita anterior con ventana
        // abierta, sigue siendo una posicion "interesante" -- se trata con
        // mas cuidado en RFP/LMR aunque ahora mismo no sea nodo PV.
        let fue_pv = es_pv || tt_entry_full.as_ref().is_some_and(|e| e.was_pv);

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
        let rfp_margen_base: i32 = rfp_margen_por_ply();
        if !en_jaque && !es_pv && depth <= RFP_PROF_MAX && beta.abs() < MATE - 1000 {
            let raw =
                *static_eval_cache.get_or_insert_with(|| self.evaluar_completo(b, eval_state));
            let static_eval =
                self.eval_con_tt(self.eval_corregida(b, raw, prev), tt_entry_full.as_ref());
            // improving: con mejora el margen se achica (poda mas agresivo);
            // sin mejora se agranda (mas conservador, poda menos).
            let mut margen_ply = if improving { rfp_margen_base * 3 / 5 } else { rfp_margen_base };
            // fue_pv (persistente, ver tt_store/was_pv): un nodo que fue PV
            // alguna vez merece mas cuidado aunque ahora la ventana sea
            // nula -- se agranda el margen (poda menos agresiva), igual que
            // ya se hace con "improving".
            if fue_pv {
                margen_ply += margen_ply / 4;
            }
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
            let static_eval =
                self.eval_con_tt(self.eval_corregida(b, raw, prev), tt_entry_full.as_ref());
            // improving: con mejora el margen se achica; sin mejora se agranda
            // (mas conservador, se razorea menos).
            let margen_base = if improving { RAZOR_MARGEN_BASE * 3 / 5 } else { RAZOR_MARGEN_BASE };
            let margen = margen_base + RAZOR_MARGEN_POR_PLY * depth;
            if static_eval + margen <= alpha {
                let sc = self.quiescence(b, eval_state, alpha, alpha + 1, ply, 0)?;
                if sc <= alpha {
                    return Ok(sc);
                }
            }
        }

        // El zobrist de ESTE nodo entra al path ANTES de null-move y ProbCut:
        // ambos lanzan busquedas hijas, y si una linea dentro de esos
        // subarboles vuelve exactamente a esta posicion, tiene que detectarse
        // como repeticion/tablas. Antes se empujaba recien despues de ambos
        // bloques, asi que esos subarboles podian devolver un puntaje de
        // "ganado" en lo que en realidad eran tablas por repeticion y producir
        // un corte beta falso. El pop correspondiente se hace en todas las
        // salidas posteriores de la funcion.
        if self.path_len < MAX_PATH {
            self.path[self.path_len] = b.zobrist;
            self.path_len += 1;
        }

        // Null-move pruning: si "pasar el turno" y aun asi el rival no supera
        // beta, la posicion ya es tan buena que se poda sin generar jugadas
        // reales. Desactivado en jaque, en finales de solo peones (riesgo de
        // zugzwang) y cerca de puntajes de mate (poco fiable ahi).
        // NMP "guiado por evaluacion" (concepto tomado del analisis de
        // Obsidian ~3636 CCRL y Quanticade Cronus ~3624 CCRL; reimplementado
        // desde cero en Rust, no es codigo portado).
        //
        // Diferencia de fondo con la version anterior de Mittens: antes se
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
        //   Mittens previo: R = 2 (+1 en depth>=6, +1 en depth>=12) -> tope 4
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
            let eval_nmp =
                self.eval_con_tt(self.eval_corregida(b, raw, prev), tt_entry_full.as_ref());
            if eval_nmp >= beta {
                let r_eval = ((eval_nmp - beta) / NULL_MOVE_EVAL_DIV).min(NULL_MOVE_EVAL_MAX);
                let r_adaptativo = NULL_MOVE_R_BASE + depth / NULL_MOVE_DEPTH_DIV + r_eval;
                let r = (r_adaptativo + self.null_move_r_extra).clamp(2, (depth - 1).max(2));
                let r = r.min(depth - 1).max(1);
                let next = b.make_null_move();
                self.tt_prefetch(next.zobrist);
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
                    self.path_len = self.path_len.saturating_sub(1);
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
            let sdepth_guarda = (depth - 4).max(1);
            // Guarda de TT: si ya hay una entrada para esta posicion buscada a
            // profundidad >= la que usaria la sonda, y su puntaje queda por
            // debajo del umbral de ProbCut, la sonda fallaria seguro. Saltarla
            // y ahorrar todo ese trabajo.
            let saltar_probcut = tt_entry_full
                .as_ref()
                .is_some_and(|e| e.depth >= sdepth_guarda && e.score < probcut_beta);
            let mut capturas = MoveList::new();
            generate_captures_legal_into_con_jaque(b, &mut capturas, false);
            if saltar_probcut {
                capturas.clear();
            }
            // Igual que en quiescence: calcular el SEE una sola vez por
            // captura y llevarlo junto a la jugada evita recalcularlo para
            // ordenar y de nuevo para el filtro de abajo.
            let mut capturas_see: ArrayVec<(Move, i32), MAX_MOVES> = ArrayVec::new();
            for mv in &capturas {
                capturas_see.push((*mv, crate::see::see(b, mv)));
            }
            capturas_see.sort_by_key(|(_, see)| -see);
            let sdepth = (depth - 4).max(1);
            let mut probcut_corto = false;
            let mut probcut_score = 0;
            let mut probcut_move: Option<Move> = None;
            for (mv, see) in &capturas_see {
                if *see < 0 {
                    continue;
                }
                let next = b.make_move(mv);
                self.tt_prefetch(next.zobrist);
                if next.in_check(next.turn) {
                    continue;
                }
                let next_eval = self.siguiente_estado_busqueda(eval_state, b, &next, sdepth);
                let res_rapido =
                    self.quiescence(&next, &next_eval, -probcut_beta, -probcut_beta + 1, ply + 1, 0);
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
                    probcut_move = Some(*mv);
                    break;
                }
            }
            if probcut_corto {
                // Guardar el corte en la TT: la sonda equivale a una busqueda
                // fail-high a profundidad sdepth+1, asi que si se revisita este
                // nodo no hace falta repetir el trabajo.
                self.tt_store(
                    key,
                    sdepth + 1,
                    probcut_score,
                    ply,
                    TTFlag::Beta,
                    probcut_move,
                    false,
                );
                self.path_len = self.path_len.saturating_sub(1);
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

        // Profundidad minima para activar la verificacion de singularidad.
        const SE_PROF_MIN: i32 = 8;
        let se_activable =
            self.modo_singular && !en_sondeo_se && !en_jaque && ply > 0 && depth >= SE_PROF_MIN;
        // NOTA HISTORICA: se probo una "generacion por etapas" completa
        // (TT sola -> capturas -> silenciosas) al estilo Stockfish. Medida
        // h2h dio 42.0% (+21 =42 -37): regresion clara, asi que se revirtio
        // por completo. Aca se usa siempre el camino clasico: lista completa
        // generada de una, con ordenamiento diferido (seleccion perezosa).
        // Buffer de claves de ESTE ply (ver comentario del campo).
        let pl_claves = ply as usize;
        // Mapa de casillas atacadas por el rival, del NODO: lo construye
        // perezosamente la primera jugada silenciosa que lo necesita para su
        // clave de orden, y a partir de ahi lo comparten tanto el resto de
        // las claves como el registro de quiets buscados (mas abajo). Antes
        // ese registro llamaba a `casilla_amenazada` -- o sea
        // `is_square_attacked_by`, cinco consultas de ataque -- por CADA
        // jugada silenciosa efectivamente buscada, teniendo el mapa completo
        // del nodo ya calculado al lado. La equivalencia es exacta: es la
        // misma que ya usa `clave_orden_movimiento` (ver Board::attack_map).
        let mut amenazas_nodo: Option<u64> = None;
        let mut moves;
        let n_moves;
        let mut lista_ordenada;
        {
            moves = MoveList::new();
            generate_legal_into_con_jaque(b, &mut moves, en_jaque);
            if moves.is_empty() {
                self.path_len = self.path_len.saturating_sub(1);
                return Ok(if en_jaque { -MATE + ply as i32 } else { 0 });
            }
            // ORDEN DE JUGADAS BAJO DEMANDA (seleccion perezosa): claves
            // calculadas todas de una, ordenamiento diferido hasta que la
            // primera jugada no corta.
            n_moves = moves.len();
            {
                let amenazas = &mut amenazas_nodo;
                for j in 0..n_moves {
                    let mv_j = &moves[j];
                    // SEE calculado UNA sola vez aca (paso 2 del SEE
                    // deduplicado) y reutilizado tanto por
                    // clave_orden_movimiento como por el SEE-prune mas abajo
                    // en el bucle principal. Centinela i32::MIN para jugadas
                    // que no son capturas.
                    let see_j = mv_j.is_capture().then(|| crate::see::see(b, mv_j));
                    let k = self.clave_orden_movimiento(
                        b, mv_j, tt_move, ply, prev, prev2, see_j, amenazas,
                    );
                    self.claves_negamax[pl_claves][j] = k;
                    self.see_negamax[pl_claves][j] = see_j.unwrap_or(i32::MIN);
                }
            }
            lista_ordenada = false;
        }

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
        let mut jugada_singular: Option<Move> = None;
        // Cuanto se ajusta la profundidad de la jugada de la TT cuando la
        // verificacion singular dice algo sobre ella. Rango posible:
        //   +2 -> doble extension (singular Y con brecha grande, nodo no-PV)
        //   +1 -> singular normal (comportamiento historico)
        //   -2 -> extension NEGATIVA: no es singular y ademas el valor de la
        //         TT ya supera beta, asi que el nodo casi seguro corta igual
        //         y no vale la pena gastar profundidad completa en esa rama.
        // Solo se usa si jugada_singular == Some(esa jugada).
        let mut ext_singular: i32 = 0;
        // La verificacion de singularidad recorre TODAS las jugadas antes del
        // bucle principal y corta apenas una alcanza la ventana: su costo (y
        // el contenido de la TT que deja) depende del orden en que las
        // recorre. Para que el comportamiento sea identico al de antes, en
        // los (pocos) nodos donde la verificacion puede activarse se ordena
        // la lista completa de una, como se hacia siempre.
        if se_activable {
            ordenar_estable(
                &mut moves,
                &mut self.claves_negamax[pl_claves],
                &mut self.see_negamax[pl_claves],
                0,
                n_moves,
            );
            lista_ordenada = true;
        }
        if se_activable {
            if let (Some(entry), Some(tmv)) = (tt_entry_full, tt_move) {
                if entry.depth >= depth - 3
                    && entry.flag == TTFlag::Beta
                    && entry.score.abs() < MATE - 1000
                    && moves.contains(&tmv)
                {
                    let margen = 2 * depth;
                    let sbeta = entry.score - margen;
                    let sdepth = (depth - 1) / 2;
                    let mut mejor_otra = -INFINITO;
                    let mut se_timed_out = false;
                    for mv in &moves {
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
                    if se_timed_out {
                        // Ni singular ni multicut: con corte por tiempo el
                        // valor de mejor_otra no es confiable.
                    } else if mejor_otra < sbeta {
                        jugada_singular = Some(tmv);
                        // DOBLE EXTENSION: no solo ninguna otra jugada llega a
                        // sbeta, sino que ni siquiera se le acerca por un
                        // margen adicional. La brecha con la segunda mejor es
                        // GRANDE: señal mucho mas fuerte de que la posicion es
                        // "forzada" y conviene invertir mas busqueda ahi. Solo
                        // en nodos que NO son PV: en PV la profundidad ya es
                        // la mas alta y doblar extensiones ahi es la receta
                        // clasica para explosiones de arbol.
                        const SE_MARGEN2: i32 = 30;
                        ext_singular = if !es_pv && mejor_otra < sbeta - SE_MARGEN2 {
                            2
                        } else {
                            1
                        };
                    } else if mejor_otra >= beta && mejor_otra.abs() < MATE - 1000 {
                        // MULTICUT: la jugada de la TT ya fallo alto
                        // (entry.flag == TTFlag::Beta) y ADEMAS otra jugada
                        // distinta supera beta en la busqueda reducida. Dos
                        // fail-highs independientes: es casi seguro que el
                        // nodo corta, asi que se devuelve sin explorar mas.
                        // No se guarda en la TT: el score viene de una ventana
                        // ajena a beta y contaminaria la tabla.
                        self.path_len = self.path_len.saturating_sub(1);
                        return Ok(mejor_otra);
                    } else if entry.score >= beta {
                        // EXTENSION NEGATIVA: la jugada de la TT NO resulto
                        // singular (otra jugada alcanza sbeta) pero el valor
                        // guardado en la TT ya dice que esta posicion es buena
                        // para el que mueve incluso mas alla de la ventana
                        // actual. No hay corte duro (eso lo cubre el multicut
                        // de arriba), pero si hay evidencia de que el nodo se
                        // resuelve solo: se REDUCE la jugada de la TT en vez
                        // de buscarla a profundidad completa. Stockfish usa -3
                        // aca (y -2 extra en cutNode); Mittens no tiene la
                        // señal de cutNode, asi que se queda con el valor
                        // conservador -2.
                        jugada_singular = Some(tmv);
                        ext_singular = -2;
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
        // Lo mismo para las capturas buscadas sin cortar: (pieza, destino,
        // victima), para aplicarles el malus de capture history.
        let mut capts_buscadas: [(u8, u8, u8); 32] = [(0, 0, 0); 32];
        let mut n_capts_buscadas = 0usize;
        // Mascaras de jaque del NODO (ver MascarasJaque): perezosas, se
        // calculan la primera vez que un guarda de poda pregunta si una
        // jugada da jaque y sirven para todas las jugadas del nodo.
        let mut mascaras_nodo: Option<MascarasJaque> = None;
        let mut idx_siguiente = 0usize;
        'jugadas: loop {
            // Se agotaron las jugadas: fin del bucle.
            if idx_siguiente >= n_moves {
                break 'jugadas;
            }
            let idx = idx_siguiente;
            idx_siguiente += 1;
            if !lista_ordenada {
                if idx == 0 {
                    // Solo se busca la MEJOR jugada (un barrido lineal) y se
                    // la rota al frente. Si corta beta aca -- el caso mas
                    // comun en un arbol bien ordenado -- el resto de la lista
                    // nunca se ordena.
                    let mut mejor_j = 0usize;
                    for j in 1..n_moves {
                        if self.claves_negamax[pl_claves][j]
                            < self.claves_negamax[pl_claves][mejor_j]
                        {
                            mejor_j = j;
                        }
                    }
                    if mejor_j != 0 {
                        let m = moves[mejor_j];
                        let k = self.claves_negamax[pl_claves][mejor_j];
                        let s = self.see_negamax[pl_claves][mejor_j];
                        for j in (1..=mejor_j).rev() {
                            moves[j] = moves[j - 1];
                            self.claves_negamax[pl_claves][j] =
                                self.claves_negamax[pl_claves][j - 1];
                            self.see_negamax[pl_claves][j] = self.see_negamax[pl_claves][j - 1];
                        }
                        moves[0] = m;
                        self.claves_negamax[pl_claves][0] = k;
                        self.see_negamax[pl_claves][0] = s;
                    }
                } else {
                    // No hubo corte en la primera jugada: recien ahora se
                    // ordena el resto, con las MISMAS claves ya calculadas.
                    ordenar_estable(
                        &mut moves,
                        &mut self.claves_negamax[pl_claves],
                        &mut self.see_negamax[pl_claves],
                        1,
                        n_moves,
                    );
                    lista_ordenada = true;
                }
            }
            let mv_actual = moves[idx];
            let mv = &mv_actual;
            // LMR: candidatas a reducir son jugadas silenciosas, tarde en el
            // orden (ya viene de mejor a peor), sin jaque propio ni jaque
            // que dan -- justo donde el orden ya filtra la mayoria de
            // jugadas malas sin gastar profundidad completa.
            //
            // Capturas TARDIAS con SEE negativo (capturas "malas", que ya
            // perdieron su carril de orden frente a las buenas -- ver
            // clave_orden_movimiento) tambien son candidatas, pero recien a
            // partir de un idx bastante mas tardio que las quiets: la
            // captura ya gasto su turno de ser probada a fondo si aparece
            // pronto en el orden. Se usa el SEE ya cacheado en see_negamax
            // (paso 2 del SEE deduplicado) en vez de recalcularlo.
            const LMR_CAPTURAS_MOVES_SIN_REDUCIR: usize = 5;
            let es_reducible = self.modo_lmr
                && !en_jaque
                && depth >= LMR_PROF_MIN
                && mv.promotion.is_none()
                && if mv.is_capture() {
                    idx >= LMR_CAPTURAS_MOVES_SIN_REDUCIR
                        && self.see_negamax[pl_claves][idx] < 0
                } else {
                    idx >= LMR_MOVES_SIN_REDUCIR
                };

            // Cache de "¿la jugada da jaque?" para los 3 guardas de poda de
            // abajo (futility, LMP, SEE-prune): se calcula UNA sola vez por
            // jugada (la primera guarda que lo necesita) con una funcion que
            // NO copia el tablero completo, y se reutiliza en las otras dos.
            // Antes cada guarda hacia su propio `b.make_move(mv)` solo para
            // consultar el jaque: hasta 3 copias de 144 bytes por jugada en
            // los nodos poco profundos, que son la mayoria del arbol.
            let mut da_jaque: Option<bool> = None;

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
                    self.eval_con_tt(self.eval_corregida(b, raw, prev), tt_entry_full.as_ref())
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
                    if !*da_jaque.get_or_insert_with(|| {
                        da_jaque_con_mascaras(
                            b,
                            mv,
                            mascaras_nodo.get_or_insert_with(|| mascaras_jaque(b)),
                        )
                    }) {
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
                if !*da_jaque.get_or_insert_with(|| {
                        da_jaque_con_mascaras(
                            b,
                            mv,
                            mascaras_nodo.get_or_insert_with(|| mascaras_jaque(b)),
                        )
                    }) {
                    continue;
                }
            }

            // PODA DE JUGADAS SILENCIOSAS por SEE y por historia negativa.
            //
            // Las dos usan la profundidad EFECTIVA post-LMR (`lmr_depth`), no
            // la profundidad cruda del nodo: una quiet tardia que de todos
            // modos se iba a buscar reducida a 2 plies no merece el margen de
            // un nodo de profundidad 8. Esa es la diferencia clave respecto
            // de podar por `depth` a secas.
            if !es_pv
                && !en_jaque
                && best_move.is_some()
                && !mv.is_capture()
                && mv.promotion.is_none()
                && beta.abs() < MATE - 1000
            {
                let r_est = tabla_lmr()[(depth as usize).min(63)][(idx + 1).min(63)].max(1);
                let lmr_depth = (depth - r_est).max(0);
                if lmr_depth <= 6 {
                    let pt_q = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                    // (b) Historia claramente negativa: esta jugada ya fallo
                    //     repetidas veces en contextos parecidos.
                    let h = self.history[mv.from as usize][mv.to as usize];
                    let ch = match prev {
                        Some((p_pt, p_to)) => {
                            self.cont_history[cont_idx(p_pt, p_to, pt_q, mv.to as usize)]
                        }
                        None => 0,
                    };
                    let ch2 = match prev2 {
                        Some((p2_pt, p2_to)) => {
                            self.cont_history_2[cont_idx(p2_pt, p2_to, pt_q, mv.to as usize)]
                        }
                        None => 0,
                    };
                    let stat_score = 2 * h + ch + ch2;
                    let hist_mala = stat_score < -4000 * depth;
                    // (a) SEE negativo con margen cuadratico en lmr_depth
                    //     (~-23*lmr_depth^2, calibracion de Stockfish).
                    //     Se evalua PEREZOSAMENTE: si la historia ya condeno
                    //     la jugada no hace falta pagar el SEE, que es lo
                    //     caro de este guarda.
                    let see_malo =
                        || crate::see::see(b, mv) < -23 * lmr_depth * lmr_depth;
                    if (hist_mala || see_malo())
                        && !*da_jaque.get_or_insert_with(|| {
                        da_jaque_con_mascaras(
                            b,
                            mv,
                            mascaras_nodo.get_or_insert_with(|| mascaras_jaque(b)),
                        )
                    })
                    {
                        continue;
                    }
                }
            }

            // SEE pruning en el loop principal (no solo en quiescence): en
            // nodos NO-PV y poca profundidad, una captura con SEE muy
            // negativo rara vez compensa aunque se reduzca por LMR -- se
            // descarta directo sin bajar al hijo.
            const SEE_PRUNE_PROF_MAX: i32 = 7;
            // Margen LINEAL. OJO CON LA CALIBRACION: la version anterior
            // usaba -85*depth/3 (== -28*depth) diciendo que era "estilo
            // Stockfish", pero la constante real de Stockfish para capturas
            // es see_ge(move, -203*depth) y la de Ethereal es -20*depth^2.
            // Con -28*depth el motor descartaba, a partir de profundidad 2,
            // CUALQUIER captura que perdiera mas de ~0.6-2.0 peones (a depth
            // 7 el umbral era -198 contra -980 de Ethereal y -1421 de
            // Stockfish): sacrificios de peon o de calidad desaparecian del
            // arbol en todo nodo no-PV. Se pasa a -100*depth, entre ambas
            // referencias y ~3.5x menos agresivo que antes.
            const SEE_PRUNE_MARGEN_POR_PLY: i32 = -100;
            if !es_pv
                && !en_jaque
                && depth <= SEE_PRUNE_PROF_MAX
                && best_move.is_some()
                && mv.is_capture()
                && mv.promotion.is_none()
                && beta.abs() < MATE - 1000
                && {
                    // SEE ya calculado en el llenado de claves_negamax (paso 2
                    // del SEE deduplicado): se reutiliza el cacheado en vez de
                    // volver a llamar a crate::see::see. El debug_assert
                    // detecta cualquier desincronizacion entre see_negamax y
                    // moves (p.ej. si un futuro cambio rota/ordena la lista
                    // sin permutar see_negamax igual).
                    let see_cacheado = self.see_negamax[pl_claves][idx];
                    debug_assert_eq!(
                        see_cacheado,
                        crate::see::see(b, mv),
                        "SEE cacheado desalineado con la jugada en idx={idx} (ver see_negamax)"
                    );
                    see_cacheado < SEE_PRUNE_MARGEN_POR_PLY * depth
                }
            {
                if !*da_jaque.get_or_insert_with(|| {
                        da_jaque_con_mascaras(
                            b,
                            mv,
                            mascaras_nodo.get_or_insert_with(|| mascaras_jaque(b)),
                        )
                    }) {
                    continue;
                }
            }

            // Registrar esta quiet como "buscada" ANTES de buscarla, para
            // poder penalizarla si otra jugada posterior causa el corte.
            if !mv.is_capture() && mv.promotion.is_none() && n_quiets_buscados < 64 {
                let mapa = *amenazas_nodo.get_or_insert_with(|| b.attack_map(b.turn.opposite()));
                let amenazada = crate::bitboard::bit(mv.to) & mapa != 0;
                debug_assert_eq!(
                    amenazada,
                    casilla_amenazada(b, mv.to),
                    "el mapa de amenazas del nodo no coincide con is_square_attacked_by"
                );
                quiets_buscados[n_quiets_buscados] = (mv.from, mv.to, amenazada);
                n_quiets_buscados += 1;
            }
            if mv.is_capture() && n_capts_buscadas < 32 {
                let pt_c = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                capts_buscadas[n_capts_buscadas] =
                    (pt_c as u8, mv.to, victima_de(b, mv) as u8);
                n_capts_buscadas += 1;
            }

            let pt_mv = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
            let next = b.make_move(mv);
            self.tt_prefetch(next.zobrist);
            let child_prev = Some((pt_mv, mv.to as usize));
            let child_prev2 = prev;
            // ext puede valer -2, 0, +1 o +2 (ver ext_singular arriba). El
            // bloque singular solo corre con depth >= SE_PROF_MIN (8), asi que
            // el peor caso CON ext negativo es depth - 1 - 2 = depth - 3 >= 5.
            //
            // BUG CORREGIDO: la cota de esta assertion era `>= 1`, que es
            // sencillamente falsa en el caso NORMAL (ext == 0). Un nodo con
            // depth == 1 -- el mas comun de todo el arbol -- recurre con
            // depth - 1 + 0 == 0, que es exactamente como se entra a la
            // quiescence: correcto y esperado. La assertion hacia panic ahi.
            // Como es un debug_assert no afecta al binario de release (queda
            // compilada fuera), pero rompia 6 tests de `cargo test` en debug
            // (bench_nps_depth12, smp_tt_generacion_compartida..., los tres de
            // reproduccion_h5c5 y el microbench), dejando la suite en rojo.
            // La cota real que se quiere vigilar es que ext negativo nunca
            // lleve la profundidad hija por debajo de 0.
            let ext = if jugada_singular == Some(*mv) {
                ext_singular
            } else {
                0
            };
            debug_assert!(depth - 1 + ext >= 0, "profundidad hija invalida tras ext");
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
                } else if fue_pv {
                    // No es PV en la ventana actual, pero fue PV alguna vez
                    // (persistido en la TT, ver was_pv): reducir un poco
                    // menos que a un nodo comun, sin llegar al trato de un
                    // nodo PV real.
                    r -= 1;
                }
                if !improving {
                    r += 1;
                }
                if !mv.is_capture() {
                    // AJUSTE PROPORCIONAL (antes era binario: -1 si la suma
                    // daba positivo). Ahora la reduccion se mueve en
                    // proporcion a la senal combinada, asi una quiet con
                    // historia MUY buena se reduce bastante menos que una
                    // apenas positiva, y una con historia mala se reduce mas.
                    // Pesos tomados de movepick.cpp de Stockfish, que combina
                    // (2252*main + 1126*cont[0] + 1093*cont[1]) / 1024:
                    // el history principal pesa el doble que cada
                    // continuation history. Se suma tambien cont_history_2
                    // (follow-up a 2 plies), que antes no se usaba aca.
                    let h = self.history[mv.from as usize][mv.to as usize];
                    let ch = match prev {
                        Some((p_pt, p_to)) => {
                            self.cont_history[cont_idx(p_pt, p_to, pt_mv, mv.to as usize)]
                        }
                        None => 0,
                    };
                    let ch2 = match prev2 {
                        Some((p2_pt, p2_to)) => {
                            self.cont_history_2[cont_idx(p2_pt, p2_to, pt_mv, mv.to as usize)]
                        }
                        None => 0,
                    };
                    let stat_score = 2 * h + ch + ch2;
                    // K elegido para el rango real de las tablas de Mittens
                    // (ya acotadas a +-MAX_HIST por la gravedad): el ajuste
                    // queda acotado a +-2 plies, en el mismo orden de
                    // magnitud que el -1 binario que reemplaza.
                    r -= (stat_score / 8000).clamp(-2, 2);
                } else {
                    // Rama simetrica para capturas tardias con SEE negativo
                    // (unicas capturas que llegan aca, ver es_reducible mas
                    // arriba): reduccion MENOR que la de una quiet en la
                    // misma posicion del indice -- una captura mala sigue
                    // siendo mas prometedora que una quiet tardia, asi que se
                    // resta 1 al calculo base. Si ademas el capture history
                    // de esa captura es positivo (esta linea de capturas dio
                    // buen resultado antes), se resta 1 mas -- igual que el
                    // bonus de history para quiets arriba.
                    r -= 1;
                    let ch_capt = self.capture_history
                        [capt_idx(pt_mv, mv.to as usize, victima_de(b, mv))];
                    if ch_capt > 0 {
                        r -= 1;
                    }
                }
                // "Corrplexity": cuanto mas grande la magnitud de la
                // correction history (mas "sorprendida" esta la NNUE por
                // esta posicion respecto de su eval estatica cruda), menos
                // se reduce -- idea usada por Stormphrax/Alexandria/Sirius/
                // Obsidian. Reusa la MISMA magnitud que ya calcula
                // eval_corregida(), sin estado nuevo. Divisor conservador
                // para que el ajuste quede en el orden de 0-1 plies en el
                // rango tipico de corrhist.
                {
                    let raw_ce = *static_eval_cache
                        .get_or_insert_with(|| self.evaluar_completo(b, eval_state));
                    let corregida = self.eval_corregida(b, raw_ce, prev);
                    const CORRPLEXITY_DIV: i32 = 100;
                    r -= (corregida - raw_ce).abs() / CORRPLEXITY_DIV;
                }
                // El tope se descuenta con la parte NEGATIVA de ext (si la
                // jugada recibio extension negativa, la profundidad hija ya
                // bajo y la reduccion no puede comerse lo que queda). Con
                // ext >= 0 el tope es identico al historico (depth - 2).
                let r = r.clamp(1, (depth - 2 + ext.min(0)).max(1));
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
                // Malus de capture history: se aplica SIEMPRE (haya cortado
                // una captura o una quiet), porque las capturas buscadas antes
                // sin exito fallaron igual en ambos casos.
                {
                    let malus = hist_bonus(depth);
                    let pt_cortante = b.piece_at(mv.from).map(|(_, pt)| pt as usize).unwrap_or(0);
                    let cortante = if mv.is_capture() {
                        Some((pt_cortante as u8, mv.to, victima_de(b, mv) as u8))
                    } else {
                        None
                    };
                    for &(cp, ct, cv) in &capts_buscadas[..n_capts_buscadas] {
                        if Some((cp, ct, cv)) != cortante {
                            hist_update(
                                &mut self.capture_history
                                    [capt_idx(cp as usize, ct as usize, cv as usize)],
                                -malus,
                            );
                        }
                    }
                }
                if !mv.is_capture() && mv.promotion.is_none() {
                    let malus = hist_bonus(depth);
                    for &(qf, qt, amenazada) in &quiets_buscados[..n_quiets_buscados] {
                        if (qf, qt) != (mv.from, mv.to) {
                            if amenazada {
                                hist_update(
                                    &mut self.history_amenaza[qf as usize][qt as usize],
                                    -malus,
                                );
                            } else {
                                hist_update(&mut self.history[qf as usize][qt as usize], -malus);
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
        self.tt_store(key, depth, best_score, ply, flag, best_move, fue_pv);

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
        // Red de seguridad (ver from_fen): si por cualquier via llega a la
        // busqueda una posicion ilegal (el bando que acaba de mover dejo su
        // rey en jaque), NO buscar -- devolver sin jugada en vez de dejar que
        // generate_legal ofrezca la captura del rey rival y make_move aborte
        // el proceso (panic=abort en release). En produccion la posicion se
        // rechaza antes, en from_fen; esto es defensa en profundidad.
        if b.in_check(b.turn.opposite()) {
            eprintln!(
                "MITTENS: posicion ilegal en la raiz de busqueda -- se devuelve sin jugada. FEN: {}",
                b.to_fen()
            );
            return (None, 0, 0);
        }
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
        // OJO (fix bestmove 0000): reportar el score de tablas NO debe
        // significar quedarse sin jugada. Antes se hacia `return (None, ...)`
        // y el UCI imprimia "bestmove 0000" en cualquier posicion repetida,
        // perdiendo la partida. Ahora se busca normalmente y solo se
        // SOBRESCRIBE el score al final.
        let raiz_tablas = self.posicion_raiz_tablas(b);
        let sc_tablas = draw_score(b, &root_eval, self.nnue_de(&root_eval));
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
                    let sc = if raiz_tablas { sc_tablas } else { mejor_sc };
                    return (mejor_mv.or(Some(actual_mv)), sc, self.nodes);
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
        let sc_final = if raiz_tablas { sc_tablas } else { mejor_sc };
        (mejor_mv, sc_final, self.nodes)
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
        mut on_info: impl FnMut(i32, i32, u64, u64, &[Move]),
    ) -> (Option<Move>, i32, i32) {
        // Red de seguridad (ver from_fen y search_fixed_depth): nunca buscar
        // una posicion ilegal; devolver sin jugada (bestmove 0000) en vez de
        // abortar el proceso por la captura del rey rival.
        if b.in_check(b.turn.opposite()) {
            eprintln!(
                "MITTENS: posicion ilegal en la raiz de busqueda -- se devuelve sin jugada. FEN: {}",
                b.to_fen()
            );
            return (None, 0, 0);
        }
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
        // OJO (fix bestmove 0000): igual que en search_fixed_depth, tablas en
        // la raiz NO puede devolver "sin jugada" -- el UCI lo imprimia como
        // "bestmove 0000" (equivalente a abandonar) en cualquier posicion
        // repetida o con la regla de 50 cumplida. Se sigue buscando una
        // jugada normal y solo se sobrescribe el score reportado.
        let raiz_tablas = self.posicion_raiz_tablas(b);
        let sc_tablas = draw_score(b, &root_eval, self.nnue_de(&root_eval));

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
                on_info(1, 0, 0, 0, &[mv]);
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
                on_info(1, sc, 0, 0, &[mv]);
                return (Some(mv), sc, 1);
            }
        }

        let inicio = Instant::now();
        // GESTION DE TIEMPO DE DOS PRESUPUESTOS (estilo timeman.cpp):
        //   - `optimo` = movetime_ms: el objetivo normal, y la base sobre la
        //     que se aplican los factores reactivos del corte blando.
        //   - `maximo` = self.tiempo_maximo_ms (si main.rs lo puso): techo
        //     DURO. El deadline se ancla aca, nunca al optimo, asi la
        //     busqueda puede estirarse cuando la posicion se pone fea. Si
        //     nadie puso un maximo, maximo == optimo (comportamiento previo).
        let optimo_ms = movetime_ms;
        let techo_global = match TIEMPO_MAXIMO_MS.load(std::sync::atomic::Ordering::Relaxed) {
            0 => None,
            v => Some(v),
        };
        // ¿Hay un RELOJ REAL detras de este presupuesto? main.rs solo publica
        // un techo duro (fijar_tiempo_maximo) cuando el `go` trae
        // wtime/btime. Con `go movetime X` no hay techo: el presupuesto ES el
        // limite y el tiempo que no se usa NO se ahorra para despues, se
        // pierde. Por eso el corte blando (que existe para GUARDAR reloj para
        // las jugadas siguientes) solo tiene sentido con reloj real; con
        // movetime fijo cortar al 55-85% del presupuesto era regalar entre un
        // 15% y un 45% del tiempo de pensar en cada jugada.
        let hay_reloj_real = self.tiempo_maximo_ms.or(techo_global).is_some();
        let maximo_ms = match (movetime_ms, self.tiempo_maximo_ms.or(techo_global)) {
            (Some(opt), Some(max)) => Some(max.max(opt)),
            (Some(opt), None) => Some(opt),
            (None, _) => None,
        };
        self.deadline = maximo_ms.map(|ms| {
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
        // Senales reactivas del time management (ver el corte blando abajo).
        // `inestabilidad` acumula cambios de mejor jugada con decaimiento;
        // `caida_score` es cuanto bajo el score respecto de la iteracion
        // anterior; `esfuerzo_mejor` es el % de nodos que se fue a la mejor
        // jugada en la ultima iteracion.
        let mut inestabilidad: i32 = 0;
        let mut caida_score: i32 = 0;
        let mut esfuerzo_mejor: i32 = 0;

        for d in 1..=max_depth {
            // Escalonamiento Lazy SMP (ver `salto_smp`): los hilos ayudantes
            // saltan bloques de iteraciones para ir POR DELANTE del hilo
            // principal y alimentar la TT compartida con entradas mas
            // profundas. d=1 nunca se salta: garantiza `mejor_mv` valido
            // (mismo contrato que `blindar_stop`).
            if let Some((size, phase)) = self.salto_smp
                && d > 1
                && ((d as u32 + phase) / size) % 2 == 1
            {
                continue;
            }
            // Blindaje de la profundidad 1 (ver `blindar_stop`): sin esto, un
            // "stop" que llega en el primer milisegundo abortaba antes de
            // puntuar una sola jugada de raiz y se devolvia el fallback de
            // generacion (a2a3 en la posicion inicial). A partir de d=2 el
            // corte es normal, porque ya hay una jugada buscada que devolver.
            self.blindar_stop = d == 1;
            self.reiniciar_nnue(b);
            let mut moves = generate_legal(b);
            if let Some(filtro) = &self.root_moves_filtro {
                moves.retain(|m| filtro.contains(m));
            }
            if moves.is_empty() {
                break;
            }
            self.order_moves_ply(b, &mut moves, mejor_mv, 0, None, None);

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

            // Mejor jugada con la que ARRANCA esta iteracion. Se guarda
            // porque `mejor_mv` puede actualizarse a mitad de iteracion (ver
            // el fail-high de la raiz mas abajo) y la senal de estabilidad
            // del PV tiene que comparar contra el valor de la iteracion
            // ANTERIOR, no contra ese adelanto.
            let mv_al_entrar = mejor_mv;
            let mut actual_mv;
            let mut actual_sc;
            let mut nodos_mejor: u64;
            let mut nodos_iter: u64;
            let mut timed_out = false;
            let mut ancho = VENTANA_INICIAL;

            loop {
                let mut alpha = vent_alpha;
                actual_mv = moves[0];
                actual_sc = -INFINITO;
                // ESFUERZO DE NODOS (factor (c) del time management): nodos
                // gastados en la mejor jugada vs. el total de la iteracion.
                // Si casi todos los nodos se fueron a la jugada que termino
                // ganando, la eleccion es "obvia" y se puede cortar antes.
                nodos_mejor = 0;
                nodos_iter = 0;
                if self.path_len < MAX_PATH { self.path[self.path_len] = b.zobrist; self.path_len += 1; }
                for (idx, mv) in moves.iter().enumerate() {
                    let nodos_antes = self.nodes;
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
                    let nodos_mv = self.nodes.saturating_sub(nodos_antes);
                    nodos_iter = nodos_iter.saturating_add(nodos_mv);
                    if sc > actual_sc {
                        actual_sc = sc;
                        actual_mv = *mv;
                        nodos_mejor = nodos_mv;
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
                    // La ventana se recentra sobre el score de la iteracion
                    // ANTERIOR (no sobre el fail-high), igual que siempre.
                    vent_beta = mejor_sc.saturating_add(ancho).min(INFINITO);
                    // BUG ARREGLADO: una jugada que falla ALTO en la raiz es
                    // casi seguro la mejor, pero antes `mejor_mv` recien se
                    // actualizaba al terminar la re-busqueda con la ventana
                    // ancha. Si el reloj se acababa en el medio, la busqueda
                    // devolvia la jugada VIEJA y tiraba el hallazgo. Ahora se
                    // adopta de inmediato y ademas se rota al frente de la
                    // lista para que la re-busqueda la pruebe primero.
                    mejor_mv = Some(actual_mv);
                    mejor_sc = actual_sc;
                    mejor_prof = d;
                    if let Some(pos) = moves.iter().position(|m| *m == actual_mv) {
                        if pos > 0 {
                            let m = moves[pos];
                            moves.remove(pos);
                            moves.insert(0, m);
                        }
                    }
                    continue;
                }
                break; // adentro de la ventana (o ya en ventana completa): valor confiable
            }
            if timed_out {
                break;
            }
            if mv_al_entrar == Some(actual_mv) {
                pv_estable += 1;
            } else {
                pv_estable = 0;
                // Inestabilidad acumulada con decaimiento: cada cambio de
                // mejor jugada suma, y el decaimiento de abajo la va
                // olvidando si la posicion se asienta.
                inestabilidad += 100;
            }
            // Decaimiento (~25% por iteracion) para que un cambio viejo no
            // siga estirando el reloj para siempre.
            inestabilidad = inestabilidad * 3 / 4;
            // Caida de score respecto de la iteracion anterior: senal fuerte
            // de que la jugada elegida se esta cayendo y conviene pensar mas.
            caida_score = if d >= 2 && mejor_sc.abs() < MATE - 1000 && actual_sc.abs() < MATE - 1000
            {
                (mejor_sc - actual_sc).max(0)
            } else {
                0
            };
            esfuerzo_mejor = if nodos_iter > 0 {
                (nodos_mejor.saturating_mul(100) / nodos_iter) as i32
            } else {
                0
            };
            mejor_mv = Some(actual_mv);
            mejor_sc = actual_sc;
            mejor_prof = d;
            // La raiz nunca pasa por negamax (el loop la maneja aparte), asi
            // que sin esto la TT no tiene entrada para ella y extraer_pv()
            // no puede ni arrancar a caminarla -- se guardaria recien al
            // final de toda la busqueda (ver mas abajo) y cada "info"
            // intermedio imprimiria "pv" vacio. Se repite aca, una vez por
            // iteracion, solo para que el reporte informativo tenga de donde
            // arrancar; no afecta la busqueda en si (el valor final se
            // vuelve a escribir igual al terminar el loop).
            self.tt_store(b.zobrist, mejor_prof, mejor_sc, 0, TTFlag::Exact, mejor_mv, true);
            // El PV mostrado en "info" es puramente informativo (no
            // participa de la busqueda): se reconstruye caminando la TT
            // desde la raiz, acotado a la profundidad ya completada para no
            // arrastrar basura de iteraciones futuras sin terminar.
            let pv_info = self.extraer_pv(b, mejor_prof.max(1) as usize);
            on_info(d, mejor_sc, self.nodes, inicio.elapsed().as_millis() as u64, &pv_info);

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
            // FACTORES REACTIVOS sobre el presupuesto optimo (en centesimas).
            // Arrancan en 100 (= sin cambio) y se multiplican entre si.
            let mut factor: u64 = 100;
            // (a) El score cayo respecto de la iteracion anterior: estirar
            //     hasta ~1.7x, proporcional a la caida a partir de 30cp.
            if caida_score >= 30 {
                let extra = (caida_score as u64).saturating_sub(30) * 70 / 120;
                factor = factor * (100 + extra.min(70)) / 100;
            }
            // (b) La mejor jugada viene cambiando: estirar hasta ~1.5x.
            if inestabilidad > 0 {
                factor = factor * (100 + (inestabilidad as u64 / 2).min(50)) / 100;
            }
            // (c) Jugada "obvia": mas del 90% de los nodos se fueron a la
            //     mejor jugada y el PV ya viene estable -> cortar antes.
            if esfuerzo_mejor > 90 && pv_estable >= 3 && caida_score == 0 {
                factor = factor * 75 / 100;
            }
            // Clamp: nunca menos del 40% ni mas del 200% del objetivo.
            let factor = factor.clamp(40, 200);
            if let Some(ms) = optimo_ms
                && ms > 25
                && hay_reloj_real
            {
                let umbral = ms.saturating_mul(fraccion_corte) / 100;
                let umbral = umbral.saturating_mul(factor) / 100;
                // El corte blando NUNCA puede autorizar mas alla del techo
                // duro: si el factor lo estira por encima del maximo, manda
                // el maximo (el deadline lo corta igual, pero asi no se
                // arranca una iteracion condenada a abortarse).
                let umbral = match maximo_ms {
                    Some(max) => umbral.min(max.saturating_mul(fraccion_corte) / 100),
                    None => umbral,
                };
                if inicio.elapsed().as_millis() as u64 > umbral {
                    break;
                }
            }
        }
        // El blindaje es estrictamente local a la iteracion 1; no puede
        // quedar activo y volverse un "stop ignorado" en la busqueda de la
        // jugada siguiente (el Searcher de un solo hilo se reutiliza).
        self.blindar_stop = false;
        // La raiz nunca pasa por negamax (el loop de arriba la maneja
        // aparte), asi que sin esto la TT no tiene entrada para ella y
        // extraer_pv() no puede ni arrancar a caminarla. Guardarla aca no
        // afecta la busqueda en si (pasa DESPUES del loop).
        if let Some(mv) = mejor_mv {
            self.tt_store(b.zobrist, mejor_prof, mejor_sc, 0, TTFlag::Exact, Some(mv), true);
        }
        let sc_final = if raiz_tablas { sc_tablas } else { mejor_sc };
        (mejor_mv, sc_final, mejor_prof)
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

/// Tablas de un Searcher que, por diseño, ACUMULAN informacion entre jugadas
/// de la misma partida (a diferencia de killers, que solo valen dentro del
/// arbol de una busqueda). Ver el comentario de `Searcher::history`: "history
/// SI persiste entre jugadas de la partida, igual que la TT".
///
/// Con un solo hilo eso ya ocurria: el loop UCI guarda el Searcher en
/// `searcher_slot` y lo reutiliza en cada "go". Con Lazy SMP no: cada llamada
/// a `buscar_lazy_smp` creaba Searchers nuevos y los tiraba al terminar, asi
/// que con Threads>1 el history/continuation/corrhist arrancaba EN CERO en
/// cada jugada de la partida. Este tipo es el vehiculo para sacar esas tablas
/// del Searcher del hilo antes de tirarlo y volver a meterlas en el Searcher
/// del mismo indice de hilo en la jugada siguiente.
///
/// No contiene nada dependiente de la posicion, del reloj ni de la TT, asi
/// que reinstalarlo nunca puede producir una busqueda invalida: en el peor
/// caso el ordenamiento de jugadas arranca con estadisticas de la jugada
/// anterior, que es exactamente el objetivo.
pub struct TablasPersistentes {
    history: Box<[[i32; 64]; 64]>,
    history_amenaza: Box<[[i32; 64]; 64]>,
    cont_history: Vec<i32>,
    cont_history_2: Vec<i32>,
    capture_history: Vec<i32>,
    counter_moves: Vec<Option<Move>>,
    corr_pawn: Vec<i32>,
    corr_nonpawn: [Vec<i32>; 2],
    corr_cont: Vec<i32>,
}

/// Memoria persistente de UN hilo de Lazy SMP entre llamadas a "go".
/// `None` = todavia no hay nada guardado (primera jugada de la partida, o
/// recien limpiado por "ucinewgame"): el hilo arranca con las tablas en cero
/// que ya trae un Searcher recien creado.
#[derive(Default)]
pub struct MemoriaHilo(Option<TablasPersistentes>);

/// Conjunto de memorias, una por indice de hilo de Lazy SMP.
///
/// PROPIEDAD EXCLUSIVA, SIN LOCKS: el pool vive en el loop UCI; en cada "go"
/// cada `MemoriaHilo` se SACA del pool (mem::take) y se MUEVE al hilo que le
/// corresponde, que la devuelve al terminar junto con su resultado. Mientras
/// la busqueda corre, cada memoria tiene un unico dueño (su hilo) y el pool
/// queda con huecos vacios, asi que no hay dos hilos que puedan tocar la
/// misma tabla ni hace falta Mutex alguno.
#[derive(Default)]
pub struct PoolMemoriaSMP {
    hilos: Vec<MemoriaHilo>,
}

impl PoolMemoriaSMP {
    pub fn nuevo() -> PoolMemoriaSMP {
        PoolMemoriaSMP { hilos: Vec::new() }
    }

    /// Olvida todo lo aprendido. Se llama en "ucinewgame": el history de una
    /// partida no debe contaminar la siguiente.
    pub fn limpiar(&mut self) {
        self.hilos.clear();
    }

    /// Asegura que haya una ranura por hilo. Subir Threads con setoption crea
    /// ranuras vacias (los hilos nuevos arrancan de cero, como siempre);
    /// bajarlo deja las de mas sin usar, y si se vuelve a subir se reencuentran
    /// con su historial -- en ningun caso se comparte una ranura entre hilos.
    fn reservar(&mut self, n: usize) {
        while self.hilos.len() < n {
            self.hilos.push(MemoriaHilo::default());
        }
    }

    fn tomar(&mut self, i: usize) -> MemoriaHilo {
        std::mem::take(&mut self.hilos[i])
    }

    fn devolver(&mut self, i: usize, m: MemoriaHilo) {
        if i < self.hilos.len() {
            self.hilos[i] = m;
        }
    }

    /// Solo para tests: cuantas ranuras tienen algo guardado.
    pub fn ranuras_con_datos(&self) -> usize {
        self.hilos.iter().filter(|m| m.0.is_some()).count()
    }

    /// Solo para tests: magnitud total del history guardado en todas las
    /// ranuras. Cero significa "no se aprendio nada" (o se limpio).
    pub fn magnitud_history(&self) -> i64 {
        self.hilos
            .iter()
            .filter_map(|m| m.0.as_ref())
            .map(|t| {
                t.history
                    .iter()
                    .flat_map(|f| f.iter())
                    .map(|v| (*v as i64).abs())
                    .sum::<i64>()
            })
            .sum()
    }
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
    pool: &mut PoolMemoriaSMP,
    nodes_limit: Option<u64>,
) -> (Option<Move>, i32, u64, Vec<ResultadoHilo>) {
    // Aging compartido de la TT (ver set_tt_generacion): el contador vive en
    // el llamador (junto a smp_tt, que persiste entre jugadas) y avanza UNA
    // vez por llamada. Todos los hilos de esta llamada se siembran con el
    // MISMO valor, asi que las entradas que escriben llevan una generacion
    // DISTINTA a la de jugadas anteriores y la regla de reemplazo de tt_store
    // puede expulsarlas de inmediato (aging vivo en Lazy SMP).
    let generacion = generacion_compartida.fetch_add(1, Ordering::Relaxed) & 0x1F;
    // Una ranura de memoria persistente por hilo (ver PoolMemoriaSMP).
    pool.reservar(n_hilos.max(1));
    if n_hilos <= 1 {
        let mut s = Searcher::new_con_tt_compartida(Arc::clone(tt), tt_mask, modo_lmr);
        s.instalar_memoria(pool.tomar(0));
        s.set_tt_generacion(generacion);
        s.set_qsearch_nnue(qsearch_nnue);
        s.set_nnue_classical_depth(nnue_classical_depth);
        s.set_external_stop(Some(external_stop));
        s.set_game_history(game_history.to_vec());
        s.root_moves_filtro = root_moves_filtro;
        s.nodes_limit = nodes_limit;
        let (mv, sc, prof) = s.search_time(b, movetime_ms, max_depth, |_, _, _, _, _| {});
        let nodos = s.nodes;
        pool.devolver(0, s.extraer_memoria());
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
            // La memoria del hilo i se MUEVE dentro de su hilo y vuelve con su
            // resultado: durante la busqueda tiene un unico dueño, sin locks.
            let memoria = pool.tomar(i);
            std::thread::spawn(move || {
                let mut s = Searcher::new_con_tt_compartida(tt, tt_mask, modo_lmr);
                s.instalar_memoria(memoria);
                s.set_tt_generacion(generacion);
                s.set_qsearch_nnue(qsearch_nnue);
                s.set_nnue_classical_depth(nnue_classical_depth);
                s.root_moves_filtro = root_moves_filtro;
                // Escalonamiento de profundidad para los AYUDANTES (i >= 1):
                // tabla de (size, phase) al estilo de los SkipSize/SkipPhase
                // del Stockfish clasico. El hilo 0 (principal) nunca salta;
                // los ayudantes se saltean bloques de iteraciones y quedan
                // 1-2 plies por delante, llenando la TT compartida con
                // entradas mas profundas que aceleran al principal. Antes de
                // esto la unica diversificacion era swap(0,1) en la raiz de
                // los hilos impares -- medido el 2026-08-20: 8 hilos ganaban
                // ~0-1 ply sobre 1 hilo (ver HALLAZGOS_SMP_2026-08-20.md).
                const SALTOS: [(u32, u32); 20] = [
                    (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3),
                    (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2),
                    (4, 3), (4, 4), (4, 5), (5, 0), (5, 1), (5, 2),
                ];
                s.salto_smp = if i == 0 {
                    None
                } else {
                    Some(SALTOS[(i - 1) % SALTOS.len()])
                };
                s.null_move_r_extra = match i % 3 {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                s.set_external_stop(Some(external_stop));
                s.set_game_history(game_history);
                // NOTA: en Lazy SMP real (n_hilos > 1) cada hilo recibe el
                // MISMO limite N -- no hay un contador de nodos compartido
                // entre hilos, asi que el total agregado puede llegar hasta
                // n_hilos * N en el peor caso, no exactamente N. Sigue
                // siendo un techo acotado (antes no habia limite alguno);
                // acotar el TOTAL exacto entre hilos requeriria un contador
                // atomico compartido, fuera de alcance de este fix.
                s.nodes_limit = nodes_limit;
                let (mv, sc, prof) =
                    s.search_time(&board_copy, movetime_ms, max_depth, |_, _, _, _, _| {});
                let nodos = s.nodes;
                (
                    ResultadoHilo {
                        mv,
                        score: sc,
                        profundidad: prof,
                        nodos,
                    },
                    s.extraer_memoria(),
                )
            })
        })
        .collect();

    let resultados: Vec<ResultadoHilo> = handles
        .into_iter()
        .enumerate()
        .map(|(i, h)| {
            let (r, memoria) = h.join().expect("hilo de busqueda con panic");
            pool.devolver(i, memoria);
            r
        })
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

    /// BUG: un "stop" (o cualquier comando que aborte la busqueda) que llega
    /// ANTES de que termine la profundidad 1 dejaba `mejor_mv` en el
    /// fallback, que es la PRIMERA jugada en orden de generacion -- a2a3 en
    /// la posicion inicial. O sea: "go" seguido de "stop" inmediato hacia
    /// jugar a2a3. Ocurre de verdad cuando un GUI corta al instante (usuario
    /// aprieta parar, perdida por tiempo, cambio de posicion). El contrato
    /// UCI es devolver la mejor jugada ENCONTRADA, y para eso la
    /// profundidad 1 tiene que completarse siempre: cuesta microsegundos.
    #[test]
    fn stop_inmediato_devuelve_jugada_buscada_no_la_primera_generada() {
        let b = Board::from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
            .expect("fen valido");
        // Bandera de stop ya levantada ANTES de arrancar: el caso extremo.
        let flag = Arc::new(AtomicBool::new(true));
        let mut s = Searcher::new(16);
        s.set_external_stop(Some(Arc::clone(&flag)));
        let (mv, _sc, prof) = s.search_time(&b, Some(5_000), 64, |_, _, _, _, _| {});
        let mv = mv.expect("siempre debe haber jugada");
        assert!(
            prof >= 1,
            "la profundidad 1 debe completarse pese al stop, se obtuvo {}",
            prof
        );
        // a2a3 es la primera jugada legal generada en esta posicion: si sale
        // esa, es el fallback sin buscar, no una eleccion de la busqueda.
        assert_ne!(
            mv.to_uci(),
            "a2a3",
            "devolvio el fallback de generacion en vez de una jugada buscada"
        );
    }

    /// `da_jaque_sin_copiar` decide si una jugada da jaque SIN construir el
    /// tablero resultante (reconstruye a mano la ocupacion virtual, el
    /// enroque, el peon comido al paso y la promocion). Es una optimizacion
    /// caliente: de ella dependen CINCO decisiones de poda distintas en
    /// quiescence y negamax. Un falso negativo poda una jugada de jaque que
    /// no se debia podar; un falso positivo desactiva podas que si
    /// correspondian. En ambos casos se pierde fuerza en silencio, sin
    /// crashear ni fallar ningun test -- justo el tipo de bug que solo
    /// aparece como "el motor juega peor y no sabemos por que".
    ///
    /// No tenia ninguna prueba. Esta la contrasta contra la verdad de fondo
    /// (hacer la jugada de verdad y preguntar si el rival quedo en jaque)
    /// para TODAS las jugadas legales de un arbol de posiciones elegidas por
    /// cubrir los casos que la funcion reconstruye a mano.
    fn jaque_coincide_con_verdad(b: &Board) {
        let mascaras = mascaras_jaque(b);
        for mv in generate_legal(b) {
            let rapido = da_jaque_sin_copiar(b, &mv);
            // Verdad de fondo: tras la jugada, el bando que ahora mueve
            // (el rival) esta en jaque?
            let despues = b.make_move(&mv);
            let real = despues.in_check(despues.turn);
            assert_eq!(
                rapido,
                real,
                "da_jaque_sin_copiar dijo {} para {} en {} (la verdad es {})",
                rapido,
                mv.to_uci(),
                b.to_fen(),
                real
            );
            // El filtro por mascaras del nodo tiene que dar EXACTAMENTE lo
            // mismo: es una sobreaproximacion que descarta rapido y, cuando
            // duda, delega en el camino exacto. Un solo falso negativo aca
            // (una jugada que da jaque y la mascara descarta) cambiaria
            // decisiones de poda en silencio.
            let con_mascaras = da_jaque_con_mascaras(b, &mv, &mascaras);
            assert_eq!(
                con_mascaras,
                real,
                "da_jaque_con_mascaras dijo {} para {} en {} (la verdad es {})",
                con_mascaras,
                mv.to_uci(),
                b.to_fen(),
                real
            );
        }
    }

    fn explorar_jaques(b: &Board, profundidad: u32) {
        jaque_coincide_con_verdad(b);
        if profundidad == 0 {
            return;
        }
        for mv in generate_legal(b) {
            explorar_jaques(&b.make_move(&mv), profundidad - 1);
        }
    }

    #[test]
    fn da_jaque_sin_copiar_coincide_con_hacer_la_jugada() {
        let posiciones = [
            // Posicion inicial: el caso trivial y masivo.
            Board::startpos(),
            // "Kiwipete": enroques por ambos lados, clavadas y descubiertos.
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -")
                .expect("fen valido"),
            Board::from_fen("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2")
                .expect("fen valido"),
            // Final con torres y peones: descubiertos por lineas largas.
            Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").expect("fen valido"),
            // Promociones (incluida promocion con captura) y enroque negro.
            Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
                .expect("fen valido"),
            // --- Casos que la funcion reconstruye A MANO y que las
            // posiciones "normales" de arriba NO ejercitan (comprobado
            // rompiendo cada rama a proposito: sin estos FEN el test seguia
            // pasando con el bug puesto). ---
            //
            // JAQUE DESCUBIERTO POR CAPTURA AL PASO: exd6 al paso despeja
            // DOS casillas de la fila 5 a la vez (sale el peon blanco de e5
            // y desaparece el negro de d5), y la torre a5 pasa a dar jaque
            // al rey h5. Si la funcion no quita el peon comido, no ve el
            // jaque.
            Board::from_fen("8/8/8/R2pP2k/8/8/8/K7 w - d6 0 1").expect("fen valido"),
            // JAQUE AL ENROCAR: tras O-O la TORRE llega a f1 y da jaque al
            // rey de f8. El jaque lo da la torre, no el rey que se movio,
            // asi que solo se ve si se reubica la torre del enroque.
            Board::from_fen("5k2/8/8/8/8/8/8/4K2R w K - 0 1").expect("fen valido"),
            // Lo mismo del lado de dama: tras O-O-O la torre llega a d1 y
            // da jaque al rey de d8.
            Board::from_fen("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1").expect("fen valido"),
            // PROMOCION QUE DA JAQUE: el peon corona y la pieza NUEVA es la
            // que da jaque; con el tipo de pieza sin actualizar no se ve.
            Board::from_fen("8/1P6/8/8/8/7k/8/K7 w - - 0 1").expect("fen valido"),
        ];
        for b in &posiciones {
            explorar_jaques(b, 2);
        }
    }

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

    // Round-trip COMPLETO por la TT compartida (empaquetado real en un u64):
    // el test viejo de mate solo probaba score_to_tt/score_from_tt aislados,
    // sin pasar por el empaquetado a i16 ni por el casillero atomico. Aca se
    // guarda de verdad en la tabla y se recupera, para cada ply y cada tipo
    // de score (mate a favor, mate en contra, score normal, +-INFINITO).
    #[test]
    fn tt_compartida_roundtrip_scores_de_mate_por_ply() {
        let (tt, mask) = construir_tt(1);
        let mut s = Searcher::new_con_tt_compartida(Arc::clone(&tt), mask, true);
        let mut clave = 0x1u64;
        for ply in [0u32, 1, 7, 31, 63, 127, 175] {
            for score in [
                MATE - 1,
                MATE - 12,
                MATE - 999,
                -MATE + 1,
                -MATE + 9,
                -MATE + 999,
                0,
                123,
                -4567,
                INFINITO,
                -INFINITO,
            ] {
                clave = clave.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                s.clear_hash();
                s.tt_store(clave, 5, score, ply, TTFlag::Exact, None, false);
                let e = s
                    .tt_probe(clave)
                    .unwrap_or_else(|| panic!("no se recupero la entrada (ply={ply}, score={score})"));
                assert_eq!(
                    score_from_tt(e.score, ply),
                    score,
                    "round-trip roto por la TT compartida: ply={ply} score={score}"
                );
            }
        }
    }

    // Round-trip de TODAS las combinaciones de jugada por el empaquetado de
    // 18 bits de la TT: 64x64 casilleros x 6 flags x (sin promocion + las 6
    // piezas). Si algun campo se pisa con otro, este test lo caza.
    #[test]
    fn tt_roundtrip_de_todas_las_jugadas_posibles() {
        let flags = [
            MoveFlag::Quiet,
            MoveFlag::Capture,
            MoveFlag::DoublePush,
            MoveFlag::EnPassant,
            MoveFlag::CastleKing,
            MoveFlag::CastleQueen,
        ];
        let promos = [
            None,
            Some(PieceType::Pawn),
            Some(PieceType::Knight),
            Some(PieceType::Bishop),
            Some(PieceType::Rook),
            Some(PieceType::Queen),
            Some(PieceType::King),
        ];
        for from in 0u8..64 {
            for to in 0u8..64 {
                if from == to {
                    continue; // sentinel de "sin jugada", no es una jugada real
                }
                for flag in flags {
                    for promotion in promos {
                        let mv = Move { from, to, promotion, flag };
                        let empaquetada = tt_empaquetar_move(Some(mv));
                        assert!(empaquetada <= 0x3FFFF, "la jugada no entra en 18 bits: {mv:?}");
                        assert_eq!(
                            tt_desempaquetar_move(empaquetada),
                            Some(mv),
                            "round-trip de jugada roto: {mv:?}"
                        );
                    }
                }
            }
        }
        assert_eq!(tt_desempaquetar_move(tt_empaquetar_move(None)), None);
    }

    // La politica de reemplazo documentada en tt_store ("una colision de otra
    // clave se trata igual que una de la MISMA clave: solo reemplaza si es al
    // menos tan profunda") debe valer en LAS DOS tablas. Este test la exige
    // en la COMPARTIDA, que es la unica que se usa en produccion (main.rs y
    // lib.rs siempre construyen construir_tt()).
    #[test]
    fn tt_compartida_colision_menos_profunda_no_pisa() {
        let (tt, mask) = construir_tt(1);
        let mut s = Searcher::new_con_tt_compartida(Arc::clone(&tt), mask, true);
        // Misma posicion en la tabla, bits de verificacion (49..64) distintos.
        let k1 = (0x1234u64 & mask as u64) | (0x1111u64 << 49);
        let k2 = (0x1234u64 & mask as u64) | (0xABCDu64 << 49);
        s.tt_store(k1, 12, 50, 0, TTFlag::Exact, None, false);
        // Menos profunda (tipica escritura de quiescence, depth=0): NO debe
        // desalojar la entrada profunda de negamax.
        s.tt_store(k2, 1, 20, 0, TTFlag::Alpha, None, false);
        assert_eq!(
            s.tt_probe(k1).map(|e| e.depth),
            Some(12),
            "una colision menos profunda desalojo la entrada profunda"
        );
        // Al menos tan profunda: SI debe pisar.
        s.tt_store(k2, 12, 20, 0, TTFlag::Alpha, None, false);
        assert_eq!(s.tt_probe(k2).map(|e| e.depth), Some(12));
        assert!(s.tt_probe(k1).is_none());
    }

    #[test]
    fn tt_colision_de_otra_clave_se_reemplaza() {
        // Una colision de OTRA clave solo pisa si es al menos tan profunda:
        // si no, una entrada de quiescence (depth=0) desalojaria
        // constantemente las entradas profundas de negamax que compartan
        // casillero, encogiendo la TT efectiva de la busqueda principal.
        let mut s = Searcher::new(1);
        let k1 = 0x10u64;
        let k2 = k1.wrapping_add((s.tt_mask as u64) + 1);
        s.tt_store(k1, 12, 50, 0, TTFlag::Exact, None, false);
        // Menos profunda: NO debe pisar.
        s.tt_store(k2, 1, 20, 0, TTFlag::Alpha, None, false);
        assert_eq!(s.tt_probe(k1).map(|e| e.depth), Some(12));
        assert!(s.tt_probe(k2).is_none());
        // Al menos tan profunda: SI debe pisar.
        s.tt_store(k2, 12, 20, 0, TTFlag::Alpha, None, false);
        assert!(s.tt_probe(k1).is_none());
        assert_eq!(s.tt_probe(k2).map(|e| e.depth), Some(12));
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
        s.tt_store(0xABCD, 10, 55, 0, TTFlag::Exact, Some(mv1), false);
        let e1 = s.tt_probe(0xABCD).expect("deberia encontrarse (misma clave)");
        assert_eq!(e1.depth, 10);
        assert_eq!(e1.score, 55);
        assert_eq!(e1.flag, TTFlag::Exact);
        assert_eq!(e1.best, Some(mv1));

        // Roundtrip con score negativo y jugada CON promocion (probando los
        // 3 bits de promocion del empaquetado, no solo el caso None).
        let mv2 = Move { from: 48, to: 56, promotion: Some(PieceType::Queen), flag: MoveFlag::Capture };
        s.tt_store(0x9999, 3, -1200, 0, TTFlag::Beta, Some(mv2), false);
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
        s.tt_store(k1, 12, 50, 0, TTFlag::Exact, None, false);
        // Igual que en la TT Local: una colision MENOS profunda no desaloja
        // (antes si lo hacia, ver tt_ocupante); una al menos tan profunda si.
        s.tt_store(k2, 1, 20, 0, TTFlag::Alpha, None, false);
        assert_eq!(s.tt_probe(k1).map(|e| e.depth), Some(12));
        assert!(s.tt_probe(k2).is_none());
        s.tt_store(k2, 12, 20, 0, TTFlag::Alpha, None, false);
        assert!(s.tt_probe(k1).is_none());
        assert_eq!(s.tt_probe(k2).map(|e| e.depth), Some(12));
    }

    #[test]
    fn quiescence_detecta_mate_en_jaque() {
        let b = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut s = Searcher::new(1);
        let eval_state = crear_eval_state(&b);
        let score = s
            .quiescence(&b, &eval_state, -INFINITO, INFINITO, 3, 0)
            .unwrap();
        assert_eq!(score, -MATE + 3);
    }

    #[test]
    fn quiescence_respeta_regla_de_cincuenta() {
        // Posicion LEGAL (la anterior tenia el rey negro en jaque con blancas
        // al turno -- misma clase de posicion ilegal que el bug de captura de
        // rey, que ahora from_fen rechaza). Dama en d4: no ataca e8.
        let b = Board::from_fen("4k3/8/8/8/3Q4/8/8/4K3 w - - 100 1").unwrap();
        let mut s = Searcher::new(1);
        let eval_state = crear_eval_state(&b);
        assert_eq!(
            s.quiescence(&b, &eval_state, -INFINITO, INFINITO, 0, 0)
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

        // Un solo pool para las dos llamadas, como en el loop UCI real.
        let mut pool = PoolMemoriaSMP::nuevo();

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
            &mut pool,
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
            &mut pool,
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

    /// BUG: con Threads>1, `buscar_lazy_smp` creaba Searchers nuevos en cada
    /// "go" y los tiraba al terminar, asi que el history/continuation/corrhist
    /// arrancaba EN CERO en cada jugada de la partida -- justo lo contrario de
    /// lo documentado en el campo `history` ("SI persiste entre jugadas de la
    /// partida") y de lo que ya hacia el camino de un solo hilo, que reutiliza
    /// el Searcher guardado en `searcher_slot`.
    ///
    /// Antes del arreglo el pool no existia; con el arreglo, tras un "go" cada
    /// hilo deja su memoria en su ranura y la siguiente llamada la reinstala.
    #[test]
    fn lazy_smp_el_history_sobrevive_entre_llamadas_a_go() {
        let b = Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1")
            .unwrap();
        let (tt, mask) = construir_tt(16);
        let generacion = AtomicU8::new(0);
        let stop = Arc::new(AtomicBool::new(false));
        let mut pool = PoolMemoriaSMP::nuevo();

        assert_eq!(
            pool.ranuras_con_datos(),
            0,
            "un pool nuevo no tiene nada guardado"
        );

        let mut buscar = |pool: &mut PoolMemoriaSMP| {
            buscar_lazy_smp(
                &b,
                Some(60),
                6,
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
                pool,
                None,
            );
        };

        buscar(&mut pool);
        assert_eq!(
            pool.ranuras_con_datos(),
            2,
            "tras un go con 2 hilos, cada hilo debe haber devuelto su memoria"
        );
        let tras_primera = pool.magnitud_history();
        assert!(
            tras_primera > 0,
            "la busqueda debe haber aprendido algo de history, vimos {}",
            tras_primera
        );

        // Segunda llamada con el MISMO pool: el history no se reinicia, se
        // sigue acumulando sobre lo aprendido en la jugada anterior.
        buscar(&mut pool);
        assert_eq!(pool.ranuras_con_datos(), 2);
        assert!(
            pool.magnitud_history() > 0,
            "el history debe seguir vivo tras la segunda llamada"
        );

        // "ucinewgame" tiene que borrarlo: el history de una partida no debe
        // contaminar la siguiente.
        pool.limpiar();
        assert_eq!(pool.ranuras_con_datos(), 0, "limpiar() olvida todo");
        assert_eq!(pool.magnitud_history(), 0);
    }

    /// Round-trip de las tablas persistentes: lo que sale de un Searcher entra
    /// intacto en otro. Es el mecanismo que usa el pool entre jugadas.
    #[test]
    fn memoria_de_hilo_va_y_vuelve_intacta() {
        let mut origen = Searcher::new(1);
        origen.history[12][34] = 4321;
        origen.cont_history[7] = -99;
        let memoria = origen.extraer_memoria();

        let mut destino = Searcher::new(1);
        assert_eq!(destino.history[12][34], 0, "arranca en cero");
        destino.instalar_memoria(memoria);
        assert_eq!(destino.history[12][34], 4321);
        assert_eq!(destino.cont_history[7], -99);

        // Una memoria vacia (primera jugada / recien limpiada) no toca nada.
        let mut virgen = Searcher::new(1);
        virgen.history[12][34] = 7;
        virgen.instalar_memoria(MemoriaHilo::default());
        assert_eq!(
            virgen.history[12][34], 7,
            "instalar una memoria vacia no debe borrar lo que ya habia"
        );
    }
}

#[cfg(test)]
mod regresion_posicion_ilegal {
    use super::*;

    // FEN del bug reportado: rey negro en e7 bajo ataque directo de la torre
    // blanca en h7, siendo el turno de BLANCAS. En una posicion legal el
    // bando que NO mueve nunca puede estar en jaque (la jugada anterior fue
    // ilegal). Antes del fix, generate_legal ofrecia "h7e7" (captura del
    // rey negro, legal para blancas) y make_move panicaba -- fatal con
    // panic=abort en release.
    const FEN_ILEGAL: &str = "8/4k2R/3b4/5p2/5pr1/3N1K2/5P2/8 w - - 18 70";

    #[test]
    fn from_fen_rechaza_posicion_ilegal() {
        assert!(
            Board::from_fen(FEN_ILEGAL).is_err(),
            "un FEN con el rey del bando que no mueve en jaque debe ser rechazado"
        );
    }

    #[test]
    fn from_fen_acepta_posicion_legal_equivalente() {
        // Misma estructura pero rey negro en e8 (NO en jaque con blancas al turno).
        let legal = Board::from_fen("4k3/7R/3b4/5p2/5pr1/3N1K2/5P2/8 w - - 18 70")
            .expect("la variante legal debe aceptarse");
        assert!(!legal.in_check(crate::types::Color::Black));
    }

    #[test]
    fn busqueda_sobre_posicion_ilegal_no_aborta() {
        // Defensa en profundidad: si una posicion ilegal llegara igual a la
        // busqueda (p.ej. por una API que construye el Board a mano o que lo
        // muta sin pasar por from_fen), debe devolver sin jugada en vez de
        // abortar por la captura del rey. Se fabrica la ilegalidad volteando
        // el turno sobre una posicion de jaque legal: con blancas al turno,
        // el rey negro (bando que acaba de mover) queda en jaque => ilegal.
        let mut ilegal = Board::from_fen("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
            .expect("jaque a negras con turno de negras es una posicion legal");
        assert!(ilegal.in_check(crate::types::Color::Black));
        ilegal.turn = crate::types::Color::White;
        assert!(ilegal.in_check(ilegal.turn.opposite()));

        let mut s = Searcher::new(64);
        let (mv, sc, nodes) = s.search_fixed_depth(&ilegal, 8);
        assert!(mv.is_none(), "posicion ilegal: no debe haber jugada");
        assert_eq!(nodes, 0);
        assert_eq!(sc, 0);
        let (mv2, sc2, _) = s.search_time(&ilegal, None, 8, |_, _, _, _, _| {});
        assert!(mv2.is_none());
        assert_eq!(sc2, 0);
    }
}

#[cfg(test)]
mod verificacion_reproduccion {
    use super::*;

    const FEN_ILEGAL: &str = "8/4k2R/3b4/5p2/5pr1/3N1K2/5P2/8 w - - 18 70";
    const FEN_LEGAL: &str = "4k3/7R/3b4/5p2/5pr1/3N1K2/5P2/8 w - - 18 70";

    // Lento en perfil debug (depth 15 en codigo no optimizado): se corre
    // explicitamente con `cargo test --release --lib verificacion_reproduccion`.
    #[test]
    #[ignore]
    fn reproduccion_depth_8_10_12_15_no_crashea() {
        // El FEN ilegal ahora es rechazado en from_fen: "position fen ..."
        // del protocolo UCI imprime "info string error de FEN" y continua.
        assert!(Board::from_fen(FEN_ILEGAL).is_err());

        // La variante LEGAL equivalente debe producir bestmove razonable a
        // las mismas profundidades sin abortar (verifica que el arbol de
        // busqueda en si sigue limpio).
        let b = Board::from_fen(FEN_LEGAL).unwrap();
        let mut s = Searcher::new(64);
        for d in [8, 10, 12, 15] {
            let (mv, sc, nodes) = s.search_fixed_depth(&b, d);
            assert!(mv.is_some(), "depth {}: debe haber bestmove", d);
            let uci = mv.unwrap().to_uci();
            println!("depth {}: bestmove {} score {} nodes {}", d, uci, sc, nodes);
            assert!(
                generate_legal(&b).iter().any(|m| m.to_uci() == uci),
                "depth {}: bestmove {} no esta en los movimientos legales de la raiz",
                d,
                uci
            );
        }
    }
}

#[cfg(test)]
mod reproduccion_h5c5 {
    use super::*;

    // FEN reportado en el incidente real: negras al turno, dama negra en c5,
    // h5 VACIO. El motor devolvio "bestmove h5c5" (origen inexistente). Ese
    // FEN en si es ILEGAL (Board::from_fen lo rechaza, correctamente): el bug
    // real es que el tablero interno del motor habia DIVERGIDO del real (dama
    // en h5 en vez de c5) porque una jugada corrupta del stream UCI se habia
    // descartado en silencio y la busqueda siguio sobre el tablero inventado.
    // Por eso este modulo NO intenta construir ese FEN: reproduce el camino
    // real (un stream de jugadas con una invalida en el medio) y verifica que
    // con el parche la posicion se rechaza completa, en vez de producir una
    // jugada ilegal despues.
    const FEN_H5C5: &str = "8/3P1kp1/5p1p/2q5/8/8/1P1Q2P1/6K1 b - - 6 40";

    // Secuencia TODA legal desde la posicion inicial (Ruy Lopez: 1.e4 e5
    // 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 dxc6 5.Nxe5 Qd4). Termina con negras al
    // turno: dama negra en d4, caballo negro b8 YA capturado.
    const MOVES_LEGALES: &[&str] = &[
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6", "f3e5", "d8d4",
    ];

    // La misma secuencia pero con una jugada corrupta en el MEDIO: en vez de
    // "b5c6" (la captura blanca del caballo), el stream trae "b8d7" -- un
    // movimiento NEGRO en el turno de BLANCAS, ademas desde una casilla b8
    // ya vacia. Es justo el escenario del incidente: un driver rival o una
    // condicion de carrera manda una jugada inaplicable en ese punto exacto.
    // Con la logica vieja ese error se descartaba en silencio y TODO lo que
    // venia despues (que dependia de la captura b5xc6) se descuadraba.
    const MOVES_CORRUPTOS: &[&str] = &[
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b8d7", "d7c6", "f3e5", "d8d4",
    ];

    fn legal_raiz(b: &Board) -> Vec<String> {
        generate_legal(b).iter().map(|m| m.to_uci()).collect()
    }

    /// Documenta el comportamiento VIEJO (el bug): una jugada invalida se
    /// descartaba EN SILENCIO y las siguientes se aplicaban sobre un tablero
    /// ya divergente. Devuelve (tablero final, jugadas descartadas).
    fn aplicar_viejo_descartando_invalidas(mut board: Board, moves_uci: &[&str]) -> (Board, usize) {
        let mut descartadas = 0usize;
        for uci in moves_uci {
            match crate::parse_uci_move(&board, uci) {
                Some(mv) => board = board.make_move(&mv),
                None => descartadas += 1,
            }
        }
        (board, descartadas)
    }

    #[test]
    fn secuencia_legal_completa_aplica_y_bestmove_es_legal_1_hilo_y_smp() {
        // Sanity del propio test: MOVES_LEGALES es una secuencia valida de
        // principio a fin, y la busqueda (1 hilo y lazy_smp 4 hilos) sobre la
        // posicion resultante devuelve siempre un bestmove legal.
        let mut b = Board::from_fen(crate::STARTPOS).unwrap();
        let mut hist: Vec<u64> = Vec::new();
        assert!(
            crate::aplicar_moves_position(&mut b, &mut hist, MOVES_LEGALES),
            "la secuencia de referencia debe ser 100% legal"
        );
        assert_eq!(hist.len(), MOVES_LEGALES.len());
        let legales = legal_raiz(&b);

        let mut s = Searcher::new(64);
        for movetime in [20u64, 100, 500] {
            let (mv, _sc, _prof) = s.search_time(&b, Some(movetime), 64, |_, _, _, _, _| {});
            let uci = mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string());
            assert!(
                legales.contains(&uci),
                "1 hilo, movetime {}: bestmove {} NO esta en los movimientos legales",
                movetime,
                uci
            );
        }

        let (tt, mask) = construir_tt(64);
        let generacion = std::sync::atomic::AtomicU8::new(0);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut pool = PoolMemoriaSMP::nuevo();
        for movetime in [20u64, 100, 500] {
            let (mv, _sc, _nodos, _res) = buscar_lazy_smp(
                &b,
                Some(movetime),
                64,
                4,
                &tt,
                &generacion,
                mask,
                true,
                true,
                0,
                &hist,
                stop.clone(),
                None,
                &mut pool,
                None,
            );
            let uci = mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string());
            assert!(
                legales.contains(&uci),
                "lazy_smp, movetime {}: bestmove {} NO esta en los movimientos legales",
                movetime,
                uci
            );
        }
    }

    #[test]
    fn stream_con_jugada_corrupta_en_medio_se_rechaza_y_conserva_ultima_valida() {
        // CAMINO REAL del bug: un stream 'position startpos moves ...' con una
        // jugada invalida en el medio. Con el parche, aplicar_moves_position
        // debe rechazar la posicion COMPLETA (devuelve false) sin tocar board
        // ni game_history -- nunca aplicar jugadas parciales en silencio.
        let mut b = Board::from_fen(crate::STARTPOS).unwrap();
        let mut hist: Vec<u64> = Vec::new();
        let ok = crate::aplicar_moves_position(&mut b, &mut hist, MOVES_CORRUPTOS);
        assert!(
            !ok,
            "una jugada invalida en el medio debe rechazar TODA la posicion (no aplicar parciales)"
        );
        let startpos = Board::from_fen(crate::STARTPOS).unwrap();
        assert_eq!(
            b.zobrist, startpos.zobrist,
            "el tablero debe quedar intacto (se conserva la ultima posicion valida)"
        );
        assert!(hist.is_empty(), "el historial debe quedar intacto tras el rechazo");

        // Control positivo: la misma secuencia SIN la jugada corrupta si aplica
        // y produce un tablero distinto al inicial (el rechazo no es porque la
        // secuencia entera sea basura, sino por la jugada corrupta puntual).
        let mut b2 = Board::from_fen(crate::STARTPOS).unwrap();
        let mut hist2: Vec<u64> = Vec::new();
        assert!(crate::aplicar_moves_position(&mut b2, &mut hist2, MOVES_LEGALES));
        assert_ne!(b2.zobrist, startpos.zobrist);
    }

    #[test]
    fn comportamiento_viejo_descartaba_en_silencio_y_producia_bestmove_ilegal() {
        // Demuestra que el test SIN el parche habria detectado el bug real:
        // con la logica VIEJA (descartar la jugada invalida y seguir), el
        // tablero interno divergia del real y ofrecia jugadas ILEGALES en la
        // partida real -- exactamente la clase de 'bestmove h5c5' del
        // incidente (mover desde una casilla vacia en el tablero real).
        let (b_viejo, descartadas) =
            aplicar_viejo_descartando_invalidas(Board::from_fen(crate::STARTPOS).unwrap(), MOVES_CORRUPTOS);
        assert!(
            descartadas >= 1,
            "el flujo viejo descarta en silencio la jugada corrupta (y las que dependen de ella)"
        );

        let b_real = {
            let mut br = Board::from_fen(crate::STARTPOS).unwrap();
            let mut hr = Vec::new();
            assert!(crate::aplicar_moves_position(&mut br, &mut hr, MOVES_LEGALES));
            br
        };
        assert_ne!(
            b_viejo.zobrist, b_real.zobrist,
            "el tablero interno divergia del real tras descartar en silencio"
        );

        let legales_viejo = legal_raiz(&b_viejo);
        let legales_real = legal_raiz(&b_real);
        // En el tablero divergente la captura b5xc6 NUNCA ocurrio: el caballo
        // negro sigue en c6. Por eso "c6d4" es legal ahi pero ILEGAL en el
        // tablero real (ahi c6 lo ocupa un peon negro, no un caballo) -- igual
        // que h5c5 era "legal" en el tablero interno corrupto del incidente y
        // no existia en el tablero real.
        assert!(
            legales_viejo.contains(&"c6d4".to_string()),
            "el tablero divergente debe ofrecer jugadas del caballo fantasma en c6 (c6d4)"
        );
        assert!(
            !legales_real.contains(&"c6d4".to_string()),
            "c6d4 debe ser ILEGAL en el tablero real (en c6 hay un peon, no un caballo)"
        );
    }

    #[test]
    fn repeticion_en_la_raiz_devuelve_jugada_legal_no_0000() {
        // Regresion del bug "bestmove 0000": con la posicion de la raiz ya
        // repetida (o con la regla de 50 cumplida) el motor devolvia None y
        // el UCI imprimia "bestmove 0000", que en la practica es abandonar.
        // Debe devolver una jugada legal y reportar score de tablas.
        let mut b = Board::from_fen(crate::STARTPOS).unwrap();
        let mut hist: Vec<u64> = Vec::new();
        let repeticion = ["g1f3", "g8f6", "f3g1", "f6g8"];
        assert!(crate::aplicar_moves_position(&mut b, &mut hist, &repeticion));
        let legales = legal_raiz(&b);

        let mut s = Searcher::new(16);
        s.set_game_history(hist.clone());
        let (mv, sc, _) = s.search_time(&b, Some(200), 64, |_, _, _, _, _| {});
        let uci = mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string());
        assert!(
            legales.contains(&uci),
            "repeticion en la raiz: bestmove {} no es legal",
            uci
        );
        assert_eq!(sc, 0, "score de tablas esperado en posicion repetida");

        // Regla de 50 jugadas cumplida: mismo contrato.
        let b50 = Board::from_fen("8/8/4k3/8/8/4K3/6R1/8 w - - 100 120").unwrap();
        let legales50 = legal_raiz(&b50);
        let mut s50 = Searcher::new(16);
        let (mv50, _sc50, _) = s50.search_time(&b50, Some(200), 64, |_, _, _, _, _| {});
        let uci50 = mv50.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string());
        assert!(
            legales50.contains(&uci50),
            "regla de 50 en la raiz: bestmove {} no es legal",
            uci50
        );
    }

    #[test]
    fn tras_rechazo_la_busqueda_lazy_smp_usa_el_tablero_sano() {
        // El rechazo ocurre en el parser de "position", comun a los caminos de
        // 1 hilo y lazy_smp: un stream corrupto nunca llega a la busqueda.
        // Aqui se verifica el contrato completo: tras un rechazo, el tablero
        // vuelve a la ultima posicion valida y la busqueda multihilo posterior
        // corre sobre ella y devuelve bestmove legal.
        let mut b = Board::from_fen(crate::STARTPOS).unwrap();
        let mut hist: Vec<u64> = Vec::new();
        assert!(crate::aplicar_moves_position(&mut b, &mut hist, MOVES_LEGALES));
        let b_previa = b;
        let hist_previa = hist.clone();
        let legales = legal_raiz(&b_previa);

        // Un stream corrupto RELATIVO a esta posicion (empieza desde startpos,
        // inaplicable aqui): debe rechazarse y restaurar el par completo.
        assert!(!crate::aplicar_moves_position(&mut b, &mut hist, MOVES_CORRUPTOS));
        assert_eq!(b.zobrist, b_previa.zobrist, "tablero restaurado tras rechazo");
        assert_eq!(hist.len(), hist_previa.len(), "historial restaurado tras rechazo");

        let (tt, mask) = construir_tt(64);
        let generacion = std::sync::atomic::AtomicU8::new(0);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut pool = PoolMemoriaSMP::nuevo();
        let (mv, _sc, _nodos, _res) = buscar_lazy_smp(
            &b,
            Some(100),
            64,
            4,
            &tt,
            &generacion,
            mask,
            true,
            true,
            0,
            &hist,
            stop.clone(),
            None,
            &mut pool,
            None,
        );
        let uci = mv.map(|m| m.to_uci()).unwrap_or_else(|| "0000".to_string());
        assert!(
            legales.contains(&uci),
            "lazy_smp sobre la posicion restaurada: bestmove {} ilegal",
            uci
        );
    }
}

#[cfg(test)]
mod validacion_movegen_from_to {
    use super::*;
    use crate::types::{Color, PieceType, file_of, rank_of};

    const FEN_H5C5: &str = "8/3P1kp1/5p1p/2q5/8/8/1P1Q2P1/6K1 b - - 6 40";

    /// Para cada movimiento legal generado verifica el invariante basico:
    /// hay una pieza del bando correcto en `from`, y la geometria del
    /// movimiento coincide con esa pieza (fila/columna/diagonal de las
    /// deslizantes, L del caballo, etc.). Un swap de from/to rompe este
    /// invariante en el 100% de los casos (pieza ausente en 'from').
    fn validar_geometria(b: &Board, mv: &Move) -> Option<String> {
        let (color, pt) = b.piece_at(mv.from)?;
        if color != b.turn {
            return Some(format!("pieza de color equivocado en from={}", mv.to_uci()));
        }
        if mv.flag == MoveFlag::CastleKing || mv.flag == MoveFlag::CastleQueen {
            // El enroque se valida con reyes/torres; solo chequeamos rey.
            if pt != PieceType::King {
                return Some(format!("enroque sin rey: {}", mv.to_uci()));
            }
            return None;
        }
        let (f1, r1) = (file_of(mv.from) as i32, rank_of(mv.from) as i32);
        let (f2, r2) = (file_of(mv.to) as i32, rank_of(mv.to) as i32);
        let df = (f2 - f1).abs();
        let dr = (r2 - r1).abs();
        let ok = match pt {
            PieceType::Pawn => {
                let dir = if b.turn == Color::White { 1 } else { -1 };
                let avance = (r2 - r1) == dir && df == 0
                    || (r2 - r1) == 2 * dir && df == 0
                    || (r2 - r1) == dir && df == 1; // captura o al paso
                let captura = df == 1 && (r2 - r1) == dir;
                (avance && (df == 0 || captura)) || (captura && mv.flag == MoveFlag::Capture)
                    || (mv.flag == MoveFlag::EnPassant && captura)
            }
            PieceType::Knight => (df == 1 && dr == 2) || (df == 2 && dr == 1),
            PieceType::Bishop => df == dr && df != 0,
            PieceType::Rook => (df == 0) != (dr == 0) && (df != 0 || dr != 0),
            PieceType::Queen => (df == dr && df != 0) || ((df == 0) != (dr == 0) && (df != 0 || dr != 0)),
            PieceType::King => df <= 1 && dr <= 1,
        };
        if !ok {
            Some(format!(
                "{}: geometria invalida para pieza {:?} (df={}, dr={})",
                mv.to_uci(),
                pt,
                df,
                dr
            ))
        } else {
            None
        }
    }

    #[test]
    fn todo_movimiento_generado_tiene_pieza_en_from_y_geometria_valida() {
        let fens = [
            FEN_H5C5,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
            "8/8/3k4/8/8/4K3/8/8 w - - 0 1",
            "k7/8/8/8/8/8/8/K6Q w - - 0 1",
            "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
        ];
        for fen in fens {
            let b = match Board::from_fen(fen) {
                Ok(b) => b,
                Err(_) => continue, // el FEN ilegal del incidente no se puede construir
            };
            for mv in generate_legal(&b) {
                if let Some(err) = validar_geometria(&b, &mv) {
                    panic!("FEN {} -> {}: {}", fen, mv.to_uci(), err);
                }
            }
        }
    }

    #[test]
    fn no_hay_pieza_en_from_nunca() {
        // Sobre el FEN del incidente (que from_fen rechaza por ilegal), no
        // podemos construir el tablero; pero la propiedad importante es que
        // ningun movimiento generado apunte a una casilla vacia como origen.
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "k7/8/8/8/8/8/8/K6Q b - - 0 1", // dama h1 clava al rey a8: turno negro (legal)
        ] {
            let b = Board::from_fen(fen).unwrap();
            for mv in generate_legal(&b) {
                assert!(
                    b.piece_at(mv.from).is_some(),
                    "{}: 'from' vacio (swap from/to?)",
                    mv.to_uci()
                );
            }
        }
    }
}
