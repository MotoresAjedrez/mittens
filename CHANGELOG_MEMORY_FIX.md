# Correccion de memoria de arranque

## Diagnostico

Los logs de produccion muestran varios `EngineTerminatedError` con `exit code:
-9` durante `uci`, antes de recibir `uciok`. Eso apunta a SIGKILL del sistema
por presion de memoria, no a una fuga durante la busqueda.

La ruta UCI construia un `Searcher::new(tt_mb)` y, ademas, una segunda TT para
Lazy SMP. Con `Hash=512` eran dos tablas independientes. Al cambiar Hash o la
evaluacion, la reserva de la tabla nueva podia solaparse con la vieja.

## Cambio

- El Searcher principal y Lazy SMP comparten una sola `Arc<SharedTT>`.
- Hash/Clear Hash/ucinewgame liberan la tabla anterior antes de reservar la
  nueva, reduciendo el pico de memoria.
- No se cambio la busqueda, la evaluacion ni la generacion de jugadas.

## Validacion

- `cargo fmt -- --check`: OK
- `cargo check --locked`: OK (warnings preexistentes)
- `cargo test --locked`: 28/28 OK
- `cargo build --release --locked`: OK
- Smoke UCI con `Hash=512`, `Threads=4`, NNUE activa: 5/5 OK
- H2H contra SHA de produccion `205a580c...`: en curso, con checkpoint en
  `results/memory_fix_tt_shared/state.json`.

La candidata no se despliega automaticamente.
