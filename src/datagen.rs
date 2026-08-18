//! Generacion de datos de entrenamiento por self-play, NATIVA (sin Python).
//!
//! Por que existe: el generador anterior era un script de Python que manejaba
//! el tablero con python-chess y hablaba con el motor por una tuberia, una
//! jugada a la vez. Medido: 250 posiciones/seg, con el motor ocioso (el motor
//! solo tarda 0.28 ms por jugada a profundidad 7). O sea que el cuello de
//! botella era Python, no la busqueda. Ademas no se podia correr en un
//! celular sin Termux. Esto lo hace todo dentro del propio binario del motor,
//! asi que corre igual en la Mac y en cualquier Android via `adb shell`.
//!
//! ---------------------------------------------------------------------
//! FORMATO DE SALIDA (bulletformat, 32 bytes por posicion)
//! ---------------------------------------------------------------------
//! OJO, aca es donde ya nos quemamos dos veces (dos entrenamientos enteros
//! perdidos, 16/16 y 46/46 derrotas en SPRT, por confundir la perspectiva).
//! La convencion, verificada leyendo el parser de bulletformat 1.8.0:
//!
//!   * Cuando mueven NEGRAS, bullet GUARDA EL TABLERO VOLTEADO
//!     (`square ^= 56`, `piece ^= 8`): la posicion siempre se almacena como
//!     si el que mueve fuera blanco.
//!   * En cambio NO voltea `score` NI `result`: los toma tal cual del
//!     archivo.
//!
//! Conclusion: `score` y `result` van SIEMPRE desde la perspectiva DEL QUE
//! MUEVE, no desde blancas.
//!
//!   - `score`: lo que devuelve la busqueda ya es relativo al que mueve
//!     (negamax), asi que se guarda directo.
//!   - `result`: el resultado de la partida se conoce desde BLANCAS
//!     (1-0 / 0-1 / tablas), asi que en cada posicion con negras al turno
//!     hay que INVERTIRLO (`1.0 - r`). Si no, la mitad de las etiquetas
//!     quedan al reves -- exactamente el bug que ya costo dos entrenamientos.
//!
//! El empaquetado replica al pie de la letra el de `convertir_cand.py`, que
//! es el que produjo el dataset del fine-tune que SI gano su SPRT.

use crate::board::Board;
use crate::movegen::generate_legal;
use crate::search::{MATE, Searcher};
use crate::types::{Color, PieceType};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Jugadas aleatorias al inicio de cada partida. Sin esto todas las partidas
/// serian identicas (el motor es determinista) y el dataset tendria una sola
/// posicion repetida millones de veces.
const PLIES_ALEATORIOS: usize = 8;

/// Tope de plies por partida; evita que un final tablas se eternice.
const MAX_PLIES: usize = 320;

/// Posiciones con |score| por encima de esto no se guardan: son mates o
/// posiciones ya decididas, donde la eval no aporta senal util y solo
/// desbalancea el dataset.
const LIMITE_SCORE: i32 = 3000;

/// Generador aleatorio propio (xorshift64*). Se evita meter la dependencia
/// `rand` solo para esto: hace falta que el binario cross-compile a Android
/// sin arrastrar nada nuevo.
struct Rng(u64);

impl Rng {
    fn nuevo(semilla: u64) -> Rng {
        // Una semilla 0 dejaria el generador clavado en 0 para siempre.
        Rng(semilla | 1)
    }
    fn siguiente(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn hasta(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.siguiente() % n as u64) as usize }
    }
}

/// Una posicion pendiente de etiquetar: se conoce su score al momento de
/// jugarla, pero el resultado de la partida recien al terminar.
struct Muestra {
    board: Board,
    score: i32,
}

/// Empaqueta una posicion al formato de bullet (32 bytes).
///
/// `resultado_stm` es 0.0 / 0.5 / 1.0 YA convertido a la perspectiva del que
/// mueve (ver la nota de arriba sobre la convencion).
fn empaquetar(b: &Board, score: i32, resultado_stm: f32, salida: &mut Vec<u8>) {
    let negras = b.turn == Color::Black;
    // (casilla, color, tipo) con la casilla y el color ya volteados si mueven
    // negras, para que la posicion quede siempre "como si moviera blanco".
    let mut piezas: Vec<(u8, u8, u8)> = Vec::with_capacity(32);
    for color in [Color::White, Color::Black] {
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            let mut bb = b.pieces[color as usize][pt as usize];
            while bb != 0 {
                let sq = bb.trailing_zeros() as u8;
                bb &= bb - 1;
                let (mut s, mut c) = (sq, color as u8);
                if negras {
                    s ^= 56;
                    c ^= 1;
                }
                piezas.push((s, c, pt as u8));
            }
        }
    }
    // El orden de los nibbles de `pcs` tiene que coincidir con el orden
    // ascendente de bits de `occ`: bullet los lee en paralelo.
    piezas.sort_unstable();

    let mut occ: u64 = 0;
    let mut pcs = [0u8; 16];
    let (mut ksq, mut oksq) = (0u8, 0u8);
    for (i, (sq, col, pt)) in piezas.iter().enumerate() {
        occ |= 1u64 << sq;
        pcs[i / 2] |= ((col << 3) | pt) << (4 * (i % 2));
        if *pt == PieceType::King as u8 {
            if *col == 0 {
                ksq = *sq;
            } else {
                oksq = sq ^ 56;
            }
        }
    }

    let score = score.clamp(-32000, 32000) as i16;
    let result = (2.0 * resultado_stm) as u8;

    salida.extend_from_slice(&occ.to_le_bytes());
    salida.extend_from_slice(&pcs);
    salida.extend_from_slice(&score.to_le_bytes());
    salida.push(result);
    salida.push(ksq);
    salida.push(oksq);
    salida.extend_from_slice(&[0u8; 3]); // relleno hasta 32 bytes
}

/// Juega una partida completa contra si mismo y devuelve sus posiciones ya
/// etiquetadas con score Y resultado real.
fn jugar_partida(s: &mut Searcher, rng: &mut Rng, profundidad: i32, buf: &mut Vec<u8>) -> usize {
    let mut b = Board::startpos();

    // Apertura aleatoria (no se guarda: son posiciones sin evaluar).
    for _ in 0..PLIES_ALEATORIOS {
        let moves = generate_legal(&b);
        if moves.is_empty() {
            return 0;
        }
        let i = rng.hasta(moves.len());
        b = b.make_move(&moves[i]);
    }
    // Si la apertura aleatoria ya termino la partida, se descarta.
    if generate_legal(&b).is_empty() {
        return 0;
    }

    let mut muestras: Vec<Muestra> = Vec::with_capacity(MAX_PLIES);
    let mut historial: Vec<u64> = Vec::with_capacity(MAX_PLIES);
    // Resultado desde BLANCAS (se convierte por posicion al empaquetar).
    let resultado_blancas: f32;

    loop {
        let moves = generate_legal(&b);
        if moves.is_empty() {
            // Mate o ahogado: el que no tiene jugadas y esta en jaque, pierde.
            resultado_blancas = if b.in_check(b.turn) {
                if b.turn == Color::White { 0.0 } else { 1.0 }
            } else {
                0.5
            };
            break;
        }
        if b.halfmove_clock >= 100 || muestras.len() >= MAX_PLIES {
            resultado_blancas = 0.5;
            break;
        }
        // Repeticion (3 veces la misma posicion = tablas).
        if historial.iter().filter(|&&h| h == b.zobrist).count() >= 2 {
            resultado_blancas = 0.5;
            break;
        }
        // OJO -- CONVENIO DEL HISTORIAL (bug que corrompio el primer lote):
        // `game_history` debe traer las posiciones ANTERIORES, NUNCA la
        // actual. Asi lo arma el bucle UCI en `aplicar_moves_position`:
        // empuja el zobrist y DESPUES aplica la jugada, o sea que la
        // posicion en la que se va a pensar queda FUERA del historial.
        //
        // Al empujar la posicion actual antes de buscar,
        // `posicion_raiz_tablas` la encontraba en el path y declaraba
        // repeticion, y `search_fixed_depth` sobrescribia el score con el de
        // tablas. Resultado medido: 50% de las posiciones con score 0 y
        // picos en +-200 (que es CONTEMPT_PENALIZACION, no una evaluacion
        // real). O sea, la mitad del dataset eran scores inventados.
        s.set_game_history(historial.clone());
        let (mv, score, _) = s.search_fixed_depth(&b, profundidad);
        let Some(mv) = mv else {
            resultado_blancas = 0.5;
            break;
        };

        // Filtros estandar de datagen: en jaque la eval estatica es ruido, y
        // los scores de mate no aportan senal a una red de evaluacion.
        if !b.in_check(b.turn) && score.abs() < LIMITE_SCORE && score.abs() < MATE - 1000 {
            muestras.push(Muestra { board: b, score });
        }
        // Recien AHORA la posicion pasa a ser "anterior".
        historial.push(b.zobrist);
        b = b.make_move(&mv);
    }

    for m in &muestras {
        // AQUI esta la conversion critica de perspectiva.
        let r = if m.board.turn == Color::White {
            resultado_blancas
        } else {
            1.0 - resultado_blancas
        };
        empaquetar(&m.board, m.score, r, buf);
    }
    muestras.len()
}

/// `mittens datagen --salida F [--partidas N] [--prof D] [--hilos T] [--semilla S]`
pub fn run_datagen(args: &[String]) {
    let leer = |nombre: &str| -> Option<String> {
        args.iter()
            .position(|a| a == nombre)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let salida = leer("--salida").unwrap_or_else(|| "datagen.data".to_string());
    let partidas: u64 = leer("--partidas").and_then(|v| v.parse().ok()).unwrap_or(1000);
    let profundidad: i32 = leer("--prof").and_then(|v| v.parse().ok()).unwrap_or(7);
    let hilos: usize = leer("--hilos").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let semilla: u64 = leer("--semilla")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15));

    println!(
        "datagen: {} partidas, profundidad {}, {} hilos -> {}",
        partidas, profundidad, hilos, salida
    );

    let hechas = Arc::new(AtomicU64::new(0));
    let posiciones = Arc::new(AtomicU64::new(0));
    let parar = Arc::new(AtomicBool::new(false));
    let t0 = std::time::Instant::now();

    std::thread::scope(|scope| {
        for h in 0..hilos {
            // Cada hilo escribe SU PROPIO archivo: sin candados y sin
            // mezclar escrituras. Se concatenan al final (`cat`), el formato
            // es de registros de 32 bytes sin cabecera.
            let ruta = if hilos == 1 {
                salida.clone()
            } else {
                format!("{}.parte{}", salida, h)
            };
            let hechas = Arc::clone(&hechas);
            let posiciones = Arc::clone(&posiciones);
            let parar = Arc::clone(&parar);
            scope.spawn(move || {
                let Ok(f) = std::fs::File::create(&ruta) else {
                    eprintln!("no se pudo crear {}", ruta);
                    return;
                };
                // Buffer chico y vaciado DESPUES DE CADA PARTIDA. Con el
                // buffer de 1 MB que tenia antes, en una corrida larga no se
                // veia avanzar el archivo durante ~9 minutos (todo vivia en
                // memoria) y un corte de luz o un celular que se desconecta
                // se llevaba hasta 32 mil posiciones por hilo. Una escritura
                // de ~3,5 KB por partida no cuesta nada al lado de los
                // ~0,4 s que toma jugarla.
                let mut out = std::io::BufWriter::with_capacity(1 << 14, f);
                // TT chica por hilo: en datagen a profundidad fija no hace
                // falta mucha, y en un celular la memoria es escasa.
                let mut s = Searcher::new(16);
                let mut rng = Rng::nuevo(semilla ^ ((h as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)));
                let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
                while !parar.load(Ordering::Relaxed) {
                    if hechas.fetch_add(1, Ordering::Relaxed) >= partidas {
                        break;
                    }
                    buf.clear();
                    let n = jugar_partida(&mut s, &mut rng, profundidad, &mut buf);
                    if out.write_all(&buf).is_err() || out.flush().is_err() {
                        eprintln!("error escribiendo {}", ruta);
                        parar.store(true, Ordering::Relaxed);
                        break;
                    }
                    let total = posiciones.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
                    if h == 0 && total % 5_000 < n as u64 {
                        let seg = t0.elapsed().as_secs_f64().max(0.001);
                        println!(
                            "  {} posiciones, {:.0} pos/seg",
                            total,
                            total as f64 / seg
                        );
                        let _ = std::io::stdout().flush();
                    }
                }
                let _ = out.flush();
            });
        }
    });

    let seg = t0.elapsed().as_secs_f64().max(0.001);
    let total = posiciones.load(Ordering::Relaxed);
    println!(
        "LISTO: {} posiciones en {:.1}s ({:.0} pos/seg){}",
        total,
        seg,
        total as f64 / seg,
        if hilos > 1 {
            format!(" -- concatenar {}.parte0..{}", salida, hilos - 1)
        } else {
            String::new()
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// El registro DEBE medir exactamente 32 bytes: bullet lee el archivo
    /// como un array plano de structs de 32 bytes, sin cabecera. Un byte de
    /// mas o de menos desalinea TODO el dataset a partir de la primera
    /// posicion, sin ningun error visible.
    #[test]
    fn el_registro_mide_32_bytes() {
        let mut v = Vec::new();
        empaquetar(&Board::startpos(), 25, 1.0, &mut v);
        assert_eq!(v.len(), 32);
    }

    /// La conversion de perspectiva del RESULTADO es el punto exacto donde
    /// ya se perdieron dos entrenamientos enteros. bullet voltea el tablero
    /// cuando mueven negras pero NO voltea el resultado, asi que el
    /// resultado tiene que llegar ya convertido a "el que mueve".
    ///
    /// Misma posicion, mismo bando ganador (blancas), pero una con blancas
    /// al turno y otra con negras: el byte de resultado tiene que salir
    /// OPUESTO (2 = gana el que mueve, 0 = pierde el que mueve).
    #[test]
    fn el_resultado_se_guarda_desde_el_que_mueve() {
        let blancas_mueven = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").expect("fen");
        let negras_mueven = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1").expect("fen");

        let ganan_blancas = 1.0f32;
        let mut a = Vec::new();
        empaquetar(&blancas_mueven, 100, ganan_blancas, &mut a);
        let mut b = Vec::new();
        // Con negras al turno, "ganan blancas" es una DERROTA para el que
        // mueve: quien llama tiene que pasar 1.0 - r.
        empaquetar(&negras_mueven, -100, 1.0 - ganan_blancas, &mut b);

        assert_eq!(a[26], 2, "con blancas al turno y ganando debe ser 2");
        assert_eq!(b[26], 0, "con negras al turno y perdiendo debe ser 0");
    }

    /// Con negras al turno el tablero se guarda VOLTEADO (square ^ 56), asi
    /// que una posicion y su espejo exacto deben producir los mismos bits de
    /// ocupacion. Si el volteo faltara, la red veria dos posiciones
    /// distintas para lo que es la misma situacion.
    #[test]
    fn el_tablero_se_voltea_con_negras_al_turno() {
        // Peon blanco en e2 con blancas al turno, y su espejo: peon negro en
        // e7 con negras al turno. Para el que mueve son la MISMA posicion.
        let a = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").expect("fen");
        let b = Board::from_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1").expect("fen");
        let (mut va, mut vb) = (Vec::new(), Vec::new());
        empaquetar(&a, 50, 1.0, &mut va);
        empaquetar(&b, 50, 1.0, &mut vb);
        assert_eq!(va[0..8], vb[0..8], "la ocupacion deberia quedar identica");
        // byte 27 = ksq (0..7 occ, 8..23 pcs, 24..25 score, 26 result, 27 ksq)
        assert_eq!(va[27], vb[27], "y el ksq del que mueve tambien");
    }

    /// El score guardado tiene que ser una EVALUACION REAL de la busqueda,
    /// no el score de tablas.
    ///
    /// Bug real que corrompio el primer lote de ~4,3 millones de posiciones:
    /// se metia la posicion actual en `game_history` ANTES de buscarla, asi
    /// que el motor la veia como repeticion y devolvia el score de tablas.
    /// Medido en los datos generados: 50% de los scores eran exactamente 0
    /// y habia picos en +-200 (CONTEMPT_PENALIZACION), en vez de una
    /// distribucion continua de evaluaciones.
    ///
    /// Un dataset asi entrena la red a creer que la mitad de las posiciones
    /// del ajedrez estan empatadas. No falla ningun test "de compilar", no
    /// crashea, y solo se nota entrenando y perdiendo el SPRT.
    #[test]
    fn los_scores_son_evaluaciones_reales_no_tablas() {
        let mut s = Searcher::new(8);
        let mut rng = Rng::nuevo(4242);
        let mut buf = Vec::new();
        let mut n = 0;
        // Varias partidas para tener muestra suficiente.
        for _ in 0..6 {
            n += jugar_partida(&mut s, &mut rng, 4, &mut buf);
        }
        assert!(n > 50, "muestra demasiado chica para concluir: {}", n);
        let scores: Vec<i32> = (0..n)
            .map(|i| i16::from_le_bytes([buf[i * 32 + 24], buf[i * 32 + 25]]) as i32)
            .collect();

        let ceros = scores.iter().filter(|&&x| x == 0).count();
        let pct_ceros = 100.0 * ceros as f64 / n as f64;
        assert!(
            pct_ceros < 25.0,
            "demasiados scores exactamente 0 ({:.1}%): senal de que se esta \
             guardando el score de TABLAS en vez de la evaluacion real",
            pct_ceros
        );

        // El contempt (+-200) no debe aparecer como si fuera una evaluacion.
        let contempt = scores.iter().filter(|&&x| x.abs() == 200).count();
        let pct_contempt = 100.0 * contempt as f64 / n as f64;
        assert!(
            pct_contempt < 5.0,
            "{:.1}% de los scores valen exactamente +-200 (CONTEMPT_PENALIZACION): \
             se esta guardando score de tablas, no evaluacion",
            pct_contempt
        );
    }

    /// Una partida real tiene que producir posiciones, y todas del tamano
    /// correcto. Ademas comprueba que no se guardan scores de mate.
    #[test]
    fn una_partida_produce_registros_validos() {
        let mut s = Searcher::new(8);
        let mut rng = Rng::nuevo(12345);
        let mut buf = Vec::new();
        let n = jugar_partida(&mut s, &mut rng, 4, &mut buf);
        assert!(n > 0, "la partida no produjo ninguna posicion");
        assert_eq!(buf.len(), n * 32, "algun registro no mide 32 bytes");
        for i in 0..n {
            let sc = i16::from_le_bytes([buf[i * 32 + 24], buf[i * 32 + 25]]);
            assert!(
                (sc as i32).abs() < LIMITE_SCORE,
                "se guardo un score fuera de rango: {}",
                sc
            );
            assert!(buf[i * 32 + 26] <= 2, "resultado invalido");
        }
    }
}
