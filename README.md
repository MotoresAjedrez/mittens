# 🐱 Mittens 

![logo](logo.png)
Note: Its ai coded
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
   ./target/release/mittens
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
target/release/mittens
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
- `BookPath` y `OwnBook`: libro Polyglot. El libro `performance.bin` va **embebido en el
  ejecutable pero APAGADO por defecto** (`OwnBook` es `default false`): el motor busca
  normalmente desde la jugada 1 salvo que se active con
  `setoption name OwnBook value true` (o `MITTENS_CON_LIBRO=1`). Un `BookPath` explicito
  sustituye al libro embebido.
- `SyzygyPath`: tablas de finales.
- `UseNNUE`: **activado por defecto**. La red va EMBEBIDA en el ejecutable
  (`pesos_bullet_1024_buckets8.bin`, formato bullet `768 -> 1024`x2 -> 1 SCReLU
  con 8 output buckets), asi que el motor evalua con NNUE sin necesidad de
  configurar nada ni de que el .bin viaje junto al binario. Antes el valor por
  defecto era `false` con `NNUEPath` vacio: cualquier tester que no
  configurara la ruta hacia jugar al motor con evaluacion clasica pura, mucho
  mas debil.
- `NNUEPath`: opcional, solo para reemplazar la red embebida por otra.

Ejemplo de red alternativa (ruta relativa a esta carpeta):

```text
setoption name NNUEPath value pesos_bullet_1024_buckets8.bin
setoption name UseNNUE value true
```

Una carga fallida ya no borra una red valida que estuviera cargada. El motor muestra un checksum al aceptar los pesos.

## Ayudante para una posicion

```bash
./mover_fen_rust.sh "FEN" 5000
```

Variables opcionales:

```bash
MITTENS_BIN=/ruta/al/motor MITTENS_HILOS=4 ./mover_fen_rust.sh "FEN" 5000
```

LMR queda activado salvo que se defina expresamente `MITTENS_LMR=0`.

## Archivos binarios

- `pesos_bullet_1024_buckets8.bin`: pesos de la red NNUE ACTUAL en produccion
  (formato bullet, `768 -> 1024`x2 -> 1 SCReLU, 8 output buckets). Va embebida
  en el ejecutable via `include_bytes!` desde `src/neural.rs`.
- `pesos_amenazas_prueba.bin`: fixture histórico de la red Bullet de 512
  neuronas usado por scripts antiguos; no es la red de 1024 embebida actual.
- `pesos_v1.bin`: fixture de la arquitectura VIEJA (770 entradas, `770 -> 256 -> 32 -> 1`), conservado solo porque dos tests unitarios en `src/neural.rs` (`rechaza_nan_sin_panico`, `checksum_es_estable`) lo usan via `include_bytes!` para probar el validador de bytes -- no representa la red que juega hoy.
- `performance.bin`: libro de aperturas Polyglot (1.487.264 bytes = 92.954 entradas,
  77.872 posiciones unicas). Va EMBEBIDO en el ejecutable via `include_bytes!`
  (`src/polyglot.rs`), pero **apagado por defecto**: se instala en la primera busqueda
  solo si `OwnBook` esta en `true`. Antes era codigo muerto: solo se cargaba si alguien
  pasaba `MITTENS_BOOK_PATH` o `setoption name BookPath`, asi que en la practica el motor
  jugaba siempre sin libro. Auditado en la rama `ronda4_libro`: el repertorio es sano
  (mediana -4 cp a depth 16, ninguna linea peor que -1.00) y con el libro encendido marco
  56.2% en 160 partidas a 150 ms contra el mismo motor sin libro.

## Estado de validacion de este paquete

`cargo test --release --locked` ejecuta las pruebas unitarias y del protocolo UCI,
incluidas las regresiones de mate en el límite de 50 movimientos, ahogado y
combinaciones de profundidad, reloj y `searchmoves`. Los duelos H2H y las sondas
que requieren otros motores están marcados `ignore`; se ejecutan expresamente con
`cargo test --release --test h2h_mittens_vs_termino -- --ignored --nocapture`
(o el nombre de la integración correspondiente), con los binarios externos disponibles.
Las pruebas de los acumuladores incluyen recorridos aleatorios comparando la
actualización incremental con el recálculo completo. El manifiesto `MANIFEST_SHA256.txt` se regenera con `shasum -a 256 * src/* > MANIFEST_SHA256.txt` cada vez que cambia el codigo o los pesos -- si no coincide con los archivos actuales, es que el proyecto se modifico despues de la ultima regeneracion, no necesariamente que este corrupto.
Vibecoded

## Ryzen 9 6900HX: Windows x64 y AVX2

La NNUE Bullet usa AVX2 de 256 bits para actualizar acumuladores, calcular
SCReLU y fusionar actualización y evaluación cuando AVX2 está habilitado
al compilar. Las compilaciones portables x86-64 conservan SSE2; ARM conserva NEON.

En un Windows con Rust MSVC y las herramientas de C++ instaladas:

```powershell
$env:RUSTFLAGS = "-C target-cpu=znver3 -C target-feature=+avx2"
cargo build --release --locked --features tournament --target x86_64-pc-windows-msvc --bin mittens
```

`znver3` orienta el código a Zen 3, base usada por el compilador para el
6900HX (Zen 3+). La feature `tournament` fija `Contempt` a 0 por defecto.
Sin esa feature se conserva el valor histórico 200; ahora puede ajustarse por UCI.
Las compilaciones de torneo se identifican en UCI como `Mittens CEDR Windows`,
`Mittens CEDR Linux` o `Mittens CEDR macOS`, según el sistema.
La NNUE embebida permanece activada por defecto (`UseNNUE=true`).

Ajustes del torneo: `Threads=4`, `Hash=2048`, `Contempt=0`, `OwnBook=false`.
Configurar 3 minutos + 3 segundos y Perfect 2021 en la GUI. Para tablas
Syzygy de hasta 5 piezas, indicar la carpeta en `SyzygyPath`.

Las pruebas de AVX2 comparan los resultados con el cálculo escalar exacto,
incluidos pesos extremos y actualizaciones incrementales. La emulación
verifica corrección; la velocidad debe medirse en el Ryzen real.
