// Cache global de evaluacion estatica NNUE, indexada por zobrist.
//
// MOTIVO (perfilado con `sample` sobre `bench 18`, binario release con
// simbolos, 2026-08-22): `evaluate_with_state` -- que en el modo puro por
// defecto es esencialmente el producto punto SCReLU de la red bullet
// (2*1024 terminos) -- era el 27% del tiempo de busqueda, el item mas caro
// del perfil, por encima de `aplicar_features_desde` (17%) y del propio
// `negamax` (9%). La busqueda evalua la MISMA posicion muchas veces:
// transposiciones (la TT no guarda la eval estatica: cada casillero es un
// unico u64 sin bits libres), re-busquedas de LMR/aspiration sobre el mismo
// subarbol, y la quiescence que rebota entre pocas capturas. Cada una de
// esas repeticiones paga el producto punto completo otra vez.
//
// La cache guarda el resultado EXACTO de esa evaluacion por posicion. Un
// acierto devuelve bit a bit el mismo i32 que habria devuelto
// `evaluate_with_state`, asi que la busqueda toma exactamente las mismas
// decisiones: mismo arbol, mismos nodos, misma jugada -- solo que mas
// rapido. (Verificado: `bench 14` da conteos de nodos IDENTICOS con la
// cache puesta y quitada; en builds de debug ademas hay un debug_assert que
// recomputa la eval en cada acierto y compara.)
//
// CUANDO ES CORRECTO CACHEAR: solo cuando la eval es funcion pura de la
// posicion. Eso pasa exactamente en el camino "red bullet pura" (el default
// de produccion desde 2026-08-13): eval = round(peso * red(posicion)) +
// TEMPO, donde `peso`, `pura` y `scale` son OnceLock fijados al arranque
// del proceso, y la red cargada solo puede cambiar via NNUEPath -- por eso
// `neural::cargar_de_datos` invalida esta cache al cargar una red nueva.
// Los demas caminos (clasica, hibrido clasica+red, red de amenazas) NO
// pasan por la cache: `Searcher::evaluar_completo` solo la consulta cuando
// hay acumulador activo y `pura()` es true.
//
// FORMATO: un unico AtomicU64 por casillero, igual de lockless que la TT
// compartida (src/search.rs): los 48 bits altos del zobrist como
// verificacion + la eval como i16 en los 16 bits bajos. Cargas y guardados
// Relaxed: un u64 alineado se lee/escribe atomico entero, no hay entradas
// "rotas" ni hace falta candado. Politica de reemplazo: siempre reemplazar
// (lo ultimo buscado es lo mas util, igual que una cache de eval clasica).
//
// La palabra 0 significa "vacio". Una entrada real solo seria 0 si los 48
// bits altos del zobrist fueran 0 Y la eval fuera 0 (probabilidad ~2^-48):
// se trataria como fallo de cache y se recalcularia -- pierde un acierto,
// nunca da un valor incorrecto.
//
// Evals fuera de rango i16 (imposibles en la practica: la red da +/- unos
// pocos miles de centipeones) simplemente NO se guardan, para que el limite
// del empaquetado jamas pueda alterar un valor.
//
// COMPARTIDA entre hilos (Lazy SMP): todos los hilos evaluan con la misma
// red, asi que dos hilos que escriben el mismo casillero escriben el mismo
// valor para la misma clave; una carrera solo puede dejar la entrada de uno
// u otro, ambas correctas. Compartirla tambien hace que la cache persista
// entre jugadas de la partida (como la TT), donde la superposicion de
// posiciones entre una busqueda y la siguiente es enorme.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Mascara de los 48 bits de verificacion (los altos del zobrist).
const TAG_MASK: u64 = !0xFFFF;

/// Tamano por defecto en MB. 16 MB = 2^21 casilleros de 8 bytes: bastante
/// mas chico que la TT por defecto (64 MB) porque cada acierto ahorra "solo"
/// un producto punto (~cientos de ns), no un subarbol; no hace falta
/// retener posiciones viejas por mucho tiempo.
const DEFAULT_MB: usize = 16;
const MAX_MB: usize = 4096;

/// `None` = cache deshabilitada (MITTENS_EVAL_CACHE_MB=0).
fn slots() -> Option<&'static [AtomicU64]> {
    static CACHE: OnceLock<Option<Vec<AtomicU64>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mb = std::env::var("MITTENS_EVAL_CACHE_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map(|v| v.min(MAX_MB))
                .unwrap_or(DEFAULT_MB);
            if mb == 0 {
                return None;
            }
            // Redondea HACIA ABAJO a potencia de dos para que el indice sea
            // un AND (mb>=1 garantiza al menos 2^17 casilleros; con mb=16,
            // el default, quedan exactamente 2^21).
            let x = mb * 1024 * 1024 / 8;
            let n = 1usize << (usize::BITS - 1 - x.leading_zeros());
            Some((0..n).map(|_| AtomicU64::new(0)).collect())
        })
        .as_deref()
}

#[inline(always)]
fn indice(slots: &[AtomicU64], key: u64) -> usize {
    (key as usize) & (slots.len() - 1)
}

/// Busca la eval cacheada para esta posicion. `None` = no esta (o la cache
/// esta deshabilitada) y el llamador debe evaluar de verdad.
#[inline]
pub fn probe(key: u64) -> Option<i32> {
    let slots = slots()?;
    let w = slots[indice(slots, key)].load(Ordering::Relaxed);
    if w != 0 && (w ^ key) & TAG_MASK == 0 {
        Some((w & 0xFFFF) as u16 as i16 as i32)
    } else {
        None
    }
}

/// Guarda la eval de esta posicion (siempre-reemplazar). Si el valor no
/// cabe en i16 no se guarda nada: preferible perder un acierto que guardar
/// un valor alterado.
#[inline]
pub fn store(key: u64, eval: i32) {
    let Some(slots) = slots() else { return };
    let Ok(e16) = i16::try_from(eval) else { return };
    let w = (key & TAG_MASK) | (e16 as u16 as u64);
    slots[indice(slots, key)].store(w, Ordering::Relaxed);
}

/// Pista de prefetch para el casillero de esta clave. Se emite al ENTRAR al
/// nodo (junto al probe de la TT, que es el otro acceso aleatorio a memoria
/// del nodo) para que, cuando `probe` se llame de verdad unas decenas de ns
/// despues (tras el corte por TT, los chequeos de tablas, etc.), la linea ya
/// este en cache. Es solo una pista al hardware: no lee ni escribe nada
/// observable, y en arquitecturas sin soporte es un no-op.
#[inline(always)]
pub fn prefetch(key: u64) {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if let Some(slots) = slots() {
        let p = slots[indice(slots, key)].as_ptr();
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) p,
                options(nomem, nostack, preserves_flags)
            );
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = key;
}

/// Vacia la cache. Se llama al cargar una red nueva (NNUEPath): los valores
/// cacheados fueron calculados con la red anterior y ya no describen nada.
pub fn flush() {
    if let Some(slots) = slots() {
        for s in slots {
            s.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// La cache es GLOBAL y `cargo test` corre en paralelo: sin esto,
    /// `flush_vacia` podria borrar la entrada de otro test entre su store y
    /// su probe. Cada test toma el candado primero.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn probe_store_roundtrip() {
        let _g = SERIAL.lock().unwrap();
        if slots().is_none() {
            return; // deshabilitada por MITTENS_EVAL_CACHE_MB=0
        }
        // Claves con bits altos no nulos (una entrada con tag 0 y eval 0 se
        // confunde con "vacio" a proposito; ver comentario del modulo).
        let k = 0xDEAD_BEEF_1234_5678u64;
        store(k, -321);
        assert_eq!(probe(k), Some(-321));
        store(k, 250);
        assert_eq!(probe(k), Some(250));
    }

    #[test]
    fn clave_distinta_no_pisa() {
        let _g = SERIAL.lock().unwrap();
        if slots().is_none() {
            return;
        }
        // Mismo casillero (mismos bits bajos), distinto tag: el segundo
        // store reemplaza y el probe del primero debe FALLAR, no devolver
        // el valor del otro.
        let a = 0x1111_0000_0000_ABCDu64;
        let b = 0x2222_0000_0000_ABCDu64;
        store(a, 77);
        store(b, -5);
        assert_eq!(probe(b), Some(-5));
        assert_eq!(probe(a), None);
    }

    #[test]
    fn eval_fuera_de_i16_no_se_guarda() {
        let _g = SERIAL.lock().unwrap();
        if slots().is_none() {
            return;
        }
        let k = 0xCAFE_F00D_0000_0001u64;
        store(k, 40000); // no cabe en i16: no debe guardarse
        assert_eq!(probe(k), None);
        store(k, i16::MAX as i32);
        assert_eq!(probe(k), Some(i16::MAX as i32));
    }

    #[test]
    fn flush_vacia() {
        let _g = SERIAL.lock().unwrap();
        if slots().is_none() {
            return;
        }
        let k = 0xABCD_0123_4567_89EFu64;
        store(k, 42);
        assert_eq!(probe(k), Some(42));
        flush();
        assert_eq!(probe(k), None);
    }
}
