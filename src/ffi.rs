//! Interfaz C para embeber el motor en apps (iOS, etc.) sin tocar el
//! `uci_loop()` original: se crean un par de pipes, se redirigen los file
//! descriptors 0 (stdin) y 1 (stdout) del proceso hacia ellos, y se corre
//! `uci_loop()` sin cambios en un hilo aparte -- exactamente como si le
//! llegaran comandos UCI por la entrada estandar de siempre.
use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::c_char;
use std::os::unix::io::RawFd;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

unsafe extern "C" {
    fn pipe(fds: *mut RawFd) -> i32;
    fn dup(fd: RawFd) -> RawFd;
    fn dup2(oldfd: RawFd, newfd: RawFd) -> RawFd;
    fn close(fd: RawFd) -> i32;
    fn fdopen(fd: RawFd, mode: *const c_char) -> *mut std::os::raw::c_void;
}

pub struct Motor {
    entrada_tx: Sender<String>,
    salida_rx: Receiver<String>,
}

fn crear_pipe() -> (RawFd, RawFd) {
    let mut fds: [RawFd; 2] = [0, 0];
    unsafe {
        pipe(fds.as_mut_ptr());
    }
    (fds[0], fds[1]) // (lectura, escritura)
}

/// Crea el motor: arranca uci_loop() en un hilo con stdin/stdout
/// redirigidos a pipes internos, y dos hilos puente que traducen esos
/// pipes a canales de Rust normales (mpsc), mucho mas comodos de usar
/// desde FFI que file descriptors crudos.
#[unsafe(no_mangle)]
pub extern "C" fn mimotor_new() -> *mut Motor {
    let (entrada_lectura, entrada_escritura) = crear_pipe();
    let (salida_lectura, salida_escritura) = crear_pipe();

    // Guarda una copia del stdin/stdout reales del proceso host antes de
    // pisarlos, por si algun dia hace falta restaurarlos.
    unsafe {
        let _stdin_original = dup(0);
        let _stdout_original = dup(1);
        dup2(entrada_lectura, 0);
        dup2(salida_escritura, 1);
        close(entrada_lectura);
        close(salida_escritura);
    }

    // Hilo puente: canal Rust -> escribe cada linea al extremo de escritura
    // del pipe de entrada (que ahora es efectivamente "stdin" del proceso).
    let (entrada_tx, entrada_rx) = channel::<String>();
    thread::spawn(move || {
        let mut archivo = unsafe {
            use std::os::unix::io::FromRawFd;
            std::fs::File::from_raw_fd(entrada_escritura)
        };
        for linea in entrada_rx {
            let _ = archivo.write_all(linea.as_bytes());
            let _ = archivo.write_all(b"\n");
            let _ = archivo.flush();
        }
    });

    // Hilo puente: lee linea por linea del extremo de lectura del pipe de
    // salida (que ahora es efectivamente "stdout" del proceso) y las manda
    // por un canal Rust normal.
    let (salida_tx, salida_rx) = channel::<String>();
    thread::spawn(move || {
        use std::os::unix::io::FromRawFd;
        let archivo = unsafe { std::fs::File::from_raw_fd(salida_lectura) };
        let mut lector = std::io::BufReader::new(archivo);
        let mut buffer = Vec::new();
        loop {
            buffer.clear();
            use std::io::BufRead;
            match lector.read_until(b'\n', &mut buffer) {
                Ok(0) => break,
                Ok(_) => {
                    let linea = String::from_utf8_lossy(&buffer).trim_end().to_string();
                    if salida_tx.send(linea).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    // El motor en si, corriendo su uci_loop() de siempre sin ninguna
    // modificacion -- cree que esta hablando por la stdin/stdout reales.
    thread::spawn(|| {
        crate::uci_loop();
    });

    let motor = Box::new(Motor {
        entrada_tx,
        salida_rx,
    });
    Box::into_raw(motor)
}

/// Envia un comando UCI (sin salto de linea) al motor.
#[unsafe(no_mangle)]
pub extern "C" fn mimotor_enviar(motor: *mut Motor, comando: *const c_char) {
    if motor.is_null() || comando.is_null() {
        return;
    }
    let motor = unsafe { &*motor };
    let c_str = unsafe { CStr::from_ptr(comando) };
    if let Ok(texto) = c_str.to_str() {
        let _ = motor.entrada_tx.send(texto.to_string());
    }
}

/// Devuelve la siguiente linea que el motor ya escribio (o NULL si todavia
/// no hay ninguna nueva). El llamador es dueno del string devuelto y debe
/// liberarlo con mimotor_liberar_string().
#[unsafe(no_mangle)]
pub extern "C" fn mimotor_leer_linea(motor: *mut Motor) -> *mut c_char {
    if motor.is_null() {
        return std::ptr::null_mut();
    }
    let motor = unsafe { &*motor };
    match motor.salida_rx.try_recv() {
        Ok(linea) => match CString::new(linea) {
            Ok(cstring) => cstring.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Igual que mimotor_leer_linea pero espera hasta timeout_ms si todavia no
/// hay nada, en vez de devolver NULL de inmediato.
#[unsafe(no_mangle)]
pub extern "C" fn mimotor_leer_linea_esperando(
    motor: *mut Motor,
    timeout_ms: u64,
) -> *mut c_char {
    if motor.is_null() {
        return std::ptr::null_mut();
    }
    let motor = unsafe { &*motor };
    match motor
        .salida_rx
        .recv_timeout(std::time::Duration::from_millis(timeout_ms))
    {
        Ok(linea) => match CString::new(linea) {
            Ok(cstring) => cstring.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mimotor_liberar_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

// Evita warning de "extern fdopen never used" -- reservado por si se
// necesita mas adelante depurar los pipes crudos con stdio de C.
#[allow(dead_code)]
fn _referencia_fdopen() {
    let _ = fdopen as *const ();
}
