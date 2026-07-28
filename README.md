# Mittens (MiMotor Tal)

![logo](logo.png)

**ELO: 1**

Motor de ajedrez UCI escrito en Rust. Esta carpeta contiene el codigo fuente completo, los pesos neuronales proporcionados y una serie de reparaciones de correctitud, reloj y robustez.

## Compilar en macOS

Requiere Rust con soporte para edition 2024.

```bash
cd mi-motor-rust-produccion
cargo build --release
```

El ejecutable quedara en:

```text
target/release/mi-motor-rust
```

Pruebas recomendadas antes de conectarlo al bot:

```bash
cargo fmt --check
cargo check
cargo test
cargo run --release -- perft
cargo run --release -- matetest
cargo run --release -- seetest
cargo run --release -- repetitiontest
```

La suite `perft` completa llega a profundidades altas y puede tardar. Para una prueba corta:

```bash
cargo test perft_inicio_hasta_tres
```

## Uso UCI

Opciones principales:

- `Hash`: memoria aproximada de la tabla de transposicion en MiB.
- `Clear Hash`: borra la tabla de transposicion.
- `Move Overhead`: margen en milisegundos para GUI, red y sistema operativo.
- `Threads`: hilos de Lazy SMP.
- `Personalidad`: `tal` o `universal`.
- `BookPath` y `OwnBook`: libro Polyglot.
- `SyzygyPath`: tablas de finales.
- `NNUEPath`: ruta al archivo de pesos NNUE (arquitectura actual: `pesos_amenazas_prueba.bin`, 5378 entradas).
- `UseNNUE`: activa la evaluacion hibrida despues de cargar pesos validos.

Ejemplo de NNUE (ruta relativa a esta carpeta):

```text
setoption name NNUEPath value pesos_amenazas_prueba.bin
setoption name UseNNUE value true
```

Una carga fallida ya no borra una red valida que estuviera cargada. El motor muestra un checksum al aceptar los pesos.

## Ayudante para una posicion

```bash
./mover_fen_rust.sh "FEN" 5000
```

Variables opcionales:

```bash
MIMOTOR_BIN=/ruta/al/motor MIMOTOR_HILOS=4 ./mover_fen_rust.sh "FEN" 5000
```

LMR queda activado salvo que se defina expresamente `MIMOTOR_LMR=0`.

## Archivos binarios

- `pesos_amenazas_prueba.bin`: pesos de la arquitectura NNUE ACTUAL en produccion (5378 entradas: 770 base + 4608 features de amenaza, `5378 -> 256 -> 32 -> 1`). Es el que hay que usar en `NNUEPath`.
- `pesos_v1.bin`: fixture de la arquitectura VIEJA (770 entradas, `770 -> 256 -> 32 -> 1`), conservado solo porque dos tests unitarios en `src/neural.rs` (`rechaza_nan_sin_panico`, `checksum_es_estable`) lo usan via `include_bytes!` para probar el validador de bytes -- no representa la red que juega hoy.
- `performance.bin`: conservado tal como fue proporcionado. El codigo actual no lo carga directamente.

## Estado de validacion de este paquete

`cargo test` corre 31/31 pruebas, incluido un fuzz de ~1920 posiciones aleatorias comparando el acumulador NNUE incremental contra el recalculo completo (ver `src/neural.rs`). El manifiesto `MANIFEST_SHA256.txt` se regenera con `shasum -a 256 * src/* > MANIFEST_SHA256.txt` cada vez que cambia el codigo o los pesos -- si no coincide con los archivos actuales, es que el proyecto se modifico despues de la ultima regeneracion, no necesariamente que este corrupto.
