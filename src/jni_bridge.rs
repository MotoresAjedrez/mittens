//! Puente JNI para Android.
//!
//! Envuelve las funciones C simples de `ffi.rs` (`mimotor_new`,
//! `mimotor_enviar`, ...) con la convencion de nombres y los tipos que
//! exige JNI, para que Kotlin/Java las pueda declarar como `external fun`.
//!
//! Clase Kotlin esperada:
//!
//! ```kotlin
//! package com.tavito.mimotor
//!
//! object MimotorNative {
//!     init { System.loadLibrary("mimotor_core") }
//!     external fun nativeNew(): Long
//!     external fun nativeEnviar(handle: Long, comando: String)
//!     external fun nativeLeerLinea(handle: Long): String?
//!     external fun nativeLeerLineaEsperando(handle: Long, timeoutMs: Long): String?
//!     external fun nativeLiberar(handle: Long)
//! }
//! ```
//!
//! Nombres de simbolo resultantes (JNI resuelve por nombre exacto):
//!   Java_com_tavito_mimotor_MimotorNative_nativeNew
//!   Java_com_tavito_mimotor_MimotorNative_nativeEnviar
//!   Java_com_tavito_mimotor_MimotorNative_nativeLeerLinea
//!   Java_com_tavito_mimotor_MimotorNative_nativeLeerLineaEsperando
//!   Java_com_tavito_mimotor_MimotorNative_nativeLiberar
//!
//! IMPORTANTE: el paquete y el nombre de la clase no llevan ningun
//! caracter `_`, asi que no hace falta el escape `_1` de JNI. Si el otro
//! agente renombra el paquete o la clase, hay que renombrar estos
//! simbolos igual o Java lanzara UnsatisfiedLinkError.
//!
//! Nota sobre panics: el perfil `release` de este crate usa
//! `panic = "abort"`, asi que `catch_unwind` no puede atrapar nada; el
//! codigo de aqui esta escrito para no entrar en panico nunca (nada de
//! `unwrap`/`expect`/indexado), que es la unica garantia real.

use crate::ffi::{
    Motor, mimotor_enviar, mimotor_leer_linea, mimotor_leer_linea_esperando,
    mimotor_liberar_string, mimotor_new,
};
use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString};
use jni::sys::{jlong, jstring};
use std::ffi::{CString, c_char};

/// Convierte un `*mut c_char` recien devuelto por `ffi.rs` en un `jstring`,
/// liberando siempre la memoria del lado Rust. Devuelve `null` de Java si
/// no habia linea o si la conversion fallo.
fn c_char_a_jstring(env: &mut JNIEnv, puntero: *mut c_char) -> jstring {
    if puntero.is_null() {
        return JObject::null().into_raw();
    }
    // Recuperar el contenido como String antes de devolver la memoria.
    let texto = unsafe { CString::from_raw(puntero) }
        .to_string_lossy()
        .into_owned();
    // `from_raw` ya libero la memoria al soltar el CString; no llamar a
    // mimotor_liberar_string sobre el mismo puntero (seria doble free).
    match env.new_string(texto) {
        Ok(s) => s.into_raw(),
        Err(_) => JObject::null().into_raw(),
    }
}

/// Arranca el motor. Devuelve un handle opaco (`Motor*` como `jlong`),
/// o 0 si algo salio mal.
///
/// Solo tiene sentido llamarlo UNA vez por proceso: internamente redirige
/// los file descriptors 0 y 1 del proceso hacia pipes propios.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tavito_mimotor_MimotorNative_nativeNew(
    _env: JNIEnv,
    _clase: JClass,
) -> jlong {
    let motor: *mut Motor = mimotor_new();
    motor as jlong
}

/// Envia un comando UCI (sin salto de linea) al motor.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tavito_mimotor_MimotorNative_nativeEnviar(
    mut env: JNIEnv,
    _clase: JClass,
    handle: jlong,
    comando: JString,
) {
    if handle == 0 || comando.is_null() {
        return;
    }
    // get_string devuelve UTF-8 modificado de Java; to_string_lossy lo
    // normaliza sin posibilidad de panico.
    let texto: String = match env.get_string(&comando) {
        Ok(java_str) => java_str.to_string_lossy().into_owned(),
        Err(_) => return,
    };
    // Los comandos UCI no llevan NUL; si por lo que sea viniera uno,
    // CString::new falla y simplemente se ignora el comando.
    let Ok(c_comando) = CString::new(texto) else {
        return;
    };
    mimotor_enviar(handle as *mut Motor, c_comando.as_ptr());
}

/// Lee la siguiente linea ya emitida por el motor, sin bloquear.
/// Devuelve `null` si todavia no hay nada.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tavito_mimotor_MimotorNative_nativeLeerLinea(
    mut env: JNIEnv,
    _clase: JClass,
    handle: jlong,
) -> jstring {
    if handle == 0 {
        return JObject::null().into_raw();
    }
    let puntero = mimotor_leer_linea(handle as *mut Motor);
    c_char_a_jstring(&mut env, puntero)
}

/// Igual que nativeLeerLinea, pero espera hasta `timeout_ms` milisegundos.
/// Devuelve `null` si se agoto el tiempo sin recibir nada.
///
/// OJO desde Kotlin: bloquea el hilo que la llama; usarla en un
/// Dispatchers.IO / hilo aparte, nunca en el hilo principal de la UI.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tavito_mimotor_MimotorNative_nativeLeerLineaEsperando(
    mut env: JNIEnv,
    _clase: JClass,
    handle: jlong,
    timeout_ms: jlong,
) -> jstring {
    if handle == 0 {
        return JObject::null().into_raw();
    }
    let espera = if timeout_ms < 0 { 0u64 } else { timeout_ms as u64 };
    let puntero = mimotor_leer_linea_esperando(handle as *mut Motor, espera);
    c_char_a_jstring(&mut env, puntero)
}

/// Libera el motor. Al soltar el canal de entrada, el `uci_loop()` ve un
/// EOF en su "stdin" y termina solo. Despues de esto el handle no se
/// puede volver a usar.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tavito_mimotor_MimotorNative_nativeLiberar(
    _env: JNIEnv,
    _clase: JClass,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut Motor));
    }
}

// Referencia a mimotor_liberar_string para que el enlazador no la
// descarte: la app iOS/C la usa, y aqui la memoria se libera via
// CString::from_raw en c_char_a_jstring.
#[allow(dead_code)]
fn _referencia_liberar_string() {
    let _ = mimotor_liberar_string as *const ();
}
