# 🐱 Mittens

![logo](logo.png)

**ELO: 1**

Un motor de ajedrez escrito en Rust, con red neuronal NNUE propia — nacido sin saber ni mover un peón en código, y ahora juega solo, piensa solo, y a veces sacrifica piezas sin avisar.

No es Stockfish. No pretende serlo. Pero pelea.

## ¿Qué trae por dentro?

- 🧠 Evaluación híbrida: clásica + red neuronal NNUE
- ⚡ Búsqueda negamax con poda alfa-beta, tabla de transposición, quiescence search
- 🎮 Protocolo UCI — funciona con cualquier interfaz gráfica de ajedrez
- 🤖 Juega en vivo en Lichess

## Cómo usarlo

1. Clona el repo.
2. Compílalo:
   ```bash
   cargo build --release
   ```
3. Ábrelo con cualquier interfaz UCI (Arena, CuteChess, BanksiaGUI...) o directo por terminal:
   ```bash
   ./target/release/mi-motor-rust
   uci
   ```
4. Escribe `go` y reza.

Mittens no explica sus jugadas. No pide perdón por los sacrificios que no calculó bien. Simplemente juega — y a veces, sin querer, hace algo brillante.

¿Le ganas? Cuéntanoslo. ¿Te gana? También cuéntanoslo, mejor con captura de pantalla.

---

Si te gusta el proyecto, una ⭐ ayuda mucho a que más gente lo descubra.

## Soporte para Android

Mittens corre en Android via un puente JNI (`src/jni_bridge.rs`), probado en
un celular real (arm64-v8a, Android 16): handshake UCI, carga de la NNUE
completa, búsqueda multi-hilo y `stop` funcionando dentro del runtime ART.

Las librerías nativas ya compiladas para las 3 ABIs (`arm64-v8a`,
`armeabi-v7a`, `x86_64`) estan en [`android-jnilibs/`](android-jnilibs/),
listas para copiar a `app/src/main/jniLibs/` de cualquier proyecto Android
(`minSdk 24+`).

Guia completa de integracion (clase Kotlin exacta, como cargar los pesos
NNUE, advertencias de threading, y que se verifico realmente):
[`ANDROID_JNI_README.md`](ANDROID_JNI_README.md).

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
