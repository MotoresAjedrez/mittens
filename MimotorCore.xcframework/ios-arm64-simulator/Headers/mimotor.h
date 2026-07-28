/*
 * mimotor.h -- Interfaz C del motor de ajedrez MiMotorTal (Rust).
 *
 * Copia canonica. Debe coincidir exactamente con las funciones
 * `extern "C"` declaradas en src/ffi.rs.
 *
 * Se enlaza contra libmimotor_core.a (ver MimotorCore.xcframework).
 */
#ifndef MIMOTOR_H
#define MIMOTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Handle opaco al motor. */
typedef struct Motor Motor;

/*
 * Arranca el motor y devuelve su handle.
 *
 * OJO: solo debe llamarse UNA vez por proceso. Internamente redirige los
 * descriptores 0 (stdin) y 1 (stdout) del proceso hacia pipes propios para
 * poder reutilizar el uci_loop() original sin modificarlo. Llamarlo dos
 * veces dejaria el proceso con los descriptores cruzados.
 *
 * El handle vive durante todo el proceso; no hay funcion para destruirlo.
 */
Motor *mimotor_new(void);

/*
 * Envia una linea de comando UCI al motor (por ejemplo "uci", "isready",
 * "position startpos", "go movetime 1000").
 *
 * El comando NO debe llevar '\n' final: la funcion lo agrega.
 * Es no bloqueante. Ignora silenciosamente punteros nulos.
 */
void mimotor_enviar(Motor *motor, const char *comando);

/*
 * Devuelve la siguiente linea de salida del motor, o NULL si en este
 * instante no hay ninguna disponible (no bloquea nunca).
 *
 * El llamador pasa a ser dueno del string y debe liberarlo con
 * mimotor_liberar_string(). La linea viene sin el '\n' final.
 */
char *mimotor_leer_linea(Motor *motor);

/*
 * Igual que mimotor_leer_linea(), pero si todavia no hay linea espera hasta
 * timeout_ms milisegundos antes de rendirse. Devuelve NULL si se agoto el
 * tiempo. El string devuelto tambien se libera con mimotor_liberar_string().
 */
char *mimotor_leer_linea_esperando(Motor *motor, uint64_t timeout_ms);

/*
 * Libera un string devuelto por mimotor_leer_linea() o
 * mimotor_leer_linea_esperando(). Acepta NULL sin hacer nada.
 */
void mimotor_liberar_string(char *s);

#ifdef __cplusplus
}
#endif

#endif /* MIMOTOR_H */
