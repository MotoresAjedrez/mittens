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

## Compilación y Guía de Rendimiento por CPU (Mac, Intel, AMD)

Mittens cuenta con optimizaciones vectoriales y despacho dinámico de instrucciones según el procesador anfitrión:
- **Apple Silicon (ARM64):** soporte nativo para **`USDOT` (`FEAT_I8MM`)** en Apple M2/M3/M4/M5 con bloques de 32 neuronas y elevación a núcleos de alto rendimiento (P-cores vía QoS de macOS), con fallback a **NEON** de 128 bits en Apple M1.
- **Procesadores Intel (x86-64):** aceleración **AVX2** de 256 bits (`vpmaddwd`) para la red neuronal NNUE y ataques de piezas deslizantes por hardware con **BMI2 (`_pext_u64`)** a ~3 ciclos de reloj.
- **Procesadores AMD Ryzen (x86-64):** detección automática de microarquitectura en el arranque. En **Zen 3 / Zen 4 / Zen 5** activa `PEXT` acelerado por hardware; en **Zen 1 / Zen+ / Zen 2** desactiva `PEXT` automáticamente y usa **Magic Bitboards** (evitando la penalización de ~18 ciclos del microcódigo lento de Zen 1/2).
- **CPUs x86-64 heredadas:** fallback vectorizado **SSE2** fusionado (`pmaddwd`) para correr en cualquier PC sin fallar por `Illegal instruction`.

Requiere Rust con soporte para edición 2024 (`rustup update`).

---

### 1. 🍏 Apple Silicon & Mac (M1 a M5, Intel Mac)

#### Para Apple Silicon M2, M3, M4, M5 (y iPad Pro M2/M4)
Aprovecha el kernel `USDOT` (`i8mm`) optimizado con `UZP1`/`UZP2`, reuso de pila NNUE y pool persistente SMP:
```bash
# Compilación estándar (aplica automáticamente .cargo/config.toml con +i8mm):
cargo build --release --bin mittens

# O especificando la arquitectura manualmente:
RUSTFLAGS="-C target-cpu=apple-m2 -C target-feature=+i8mm" cargo build --release --bin mittens
```
*El ejecutable quedará en `target/release/mittens`.*

#### Para Apple Silicon M1 (o iPad Air/Pro M1)
El silicio M1 no dispone de la instrucción `i8mm`, pero ejecuta el kernel `NEON` vectorizado de 128 bits a máxima velocidad:
```bash
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release --bin mittens
```

#### Para Mac con procesador Intel (x86-64)
```bash
# Si tu Mac Intel soporta AVX2:
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin mittens

# O binario universal compatible con cualquier Mac Intel:
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release --bin mittens
```

---

### 2. 🔷 Procesadores Intel (Core i3 / i5 / i7 / i9 / Core Ultra / Xeon)

Los procesadores Intel desde la 4.ª generación (Haswell, 2013) en adelante cuentan con **AVX2** y **BMI2 (`PEXT`)** nativo y de baja latencia (3 ciclos).

#### Compilación local optimizada (máximo rendimiento en la máquina actual):
- **En Linux / macOS / WSL:**
  ```bash
  RUSTFLAGS="-C target-cpu=native" cargo build --release --bin mittens
  ```
- **En Windows (PowerShell con MSVC):**
  ```powershell
  $env:RUSTFLAGS = "-C target-cpu=native"
  cargo build --release --bin mittens
  ```

#### Binario portátil optimizado para Intel moderno (Haswell o superior):
```bash
# Linux:
RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+avx2,+bmi2" cargo build --release --bin mittens

# Windows (MSVC):
$env:RUSTFLAGS = "-C target-cpu=x86-64-v3 -C target-feature=+avx2,+bmi2"
cargo build --release --target x86_64-pc-windows-msvc --bin mittens
```

---

### 3. 🔴 Procesadores AMD Ryzen (Zen 1, Zen+, Zen 2, Zen 3, Zen 4, Zen 5)

> [!IMPORTANT]
> **Comportamiento arquitectónico de `PEXT` (BMI2) en AMD:**
> - En **AMD Ryzen series 1000, 2000 y 3000 (Zen 1, Zen+, Zen 2)**, AMD implementó la instrucción `_pext_u64` mediante microcódigo emulado que toma **~18 ciclos de reloj**, frente a los **~3 ciclos** de los *Magic Bitboards*. Forzar `+bmi2` en esos procesadores frena el motor entre un 15 % y un 25 %.
> - En **AMD Ryzen series 5000, 7000, 8000 y 9000 (Zen 3, Zen 4, Zen 5)**, `PEXT` está implementado directamente en hardware a máxima velocidad.

Mittens incorpora **despacho inteligente en runtime**: detecta la familia de procesador al arrancar y usa `PEXT` solo si es Intel o AMD Zen 3+ (`Family >= 0x19`), cayendo automáticamente a Magic Bitboards en chips Zen 1/Zen 2.

#### Para AMD Ryzen Zen 3, Zen 4 o Zen 5 (Serie 5000 / 7000 / 8000 / 9000):
```bash
# Linux:
RUSTFLAGS="-C target-cpu=znver3 -C target-feature=+avx2" cargo build --release --bin mittens

# Windows (PowerShell):
$env:RUSTFLAGS = "-C target-cpu=znver3 -C target-feature=+avx2"
cargo build --release --target x86_64-pc-windows-msvc --bin mittens
```

#### Para AMD Ryzen Zen 1, Zen+ o Zen 2 (Serie 1000 / 2000 / 3000):
```bash
# Evita forzar BMI2 por compilador para que utilice Magic Bitboards:
RUSTFLAGS="-C target-cpu=znver1 -C target-feature=+avx2,-bmi2" cargo build --release --bin mittens
```

---

### 4. 🌐 Binario Universal x86-64 (Recomendado para CCRL, CEDR o Distribución General)

Si deseas un solo ejecutable que corra en cualquier PC (Intel o AMD) y aproveche siempre lo mejor de cada chip sin crashear en equipos antiguos:

```bash
# Linux / macOS x86-64:
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release --bin mittens

# Windows (PowerShell con MSVC):
$env:RUSTFLAGS = "-C target-cpu=x86-64-v2"
cargo build --release --target x86_64-pc-windows-msvc --bin mittens
```

**Matriz de selección automática en tiempo de ejecución:**
| Procesador anfitrión | Red Neuronal (NNUE) | Ataques de Torres y Alfiles |
| :--- | :--- | :--- |
| **Intel con AVX2 + BMI2** (Haswell+) | AVX2 256 bits (`vpmaddwd`) | PEXT en hardware (3 ciclos) |
| **AMD Zen 3 / Zen 4 / Zen 5** | AVX2 256 bits (`vpmaddwd`) | PEXT en hardware (3 ciclos) |
| **AMD Zen 1 / Zen+ / Zen 2** | AVX2 256 bits (`vpmaddwd`) | **Magic Bitboards automáticos** (evita microcódigo lento) |
| **CPUs x86-64 antiguas (sin AVX2)** | SSE2 128 bits (`pmaddwd`) | Magic Bitboards |

---

### 5. 🧪 Verificación y Pruebas del Motor

Tras compilar, puedes validar la integridad y velocidad del binario:

```bash
# Suite de pruebas unitarias (144 tests exactos):
cargo test --release --lib

# Pruebas de velocidad de nodos (bench):
cargo run --release -- bench 12    # Referencia estándar: 604,743 nodos
cargo run --release -- bench 16    # Referencia extendida: 2,761,927 nodos

# Pruebas de validación de reglas y ajedrez:
cargo run --release -- perft
cargo run --release -- matetest
cargo run --release -- seetest
cargo run --release -- repetitiontest
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


## Mittens CCRL: Windows and Linux

The `ccrl` feature provides the tournament preset: UCI name `Mittens CCRL
Windows` or `Mittens CCRL Linux`, `Hash=256` MiB, `Threads=1`, `Contempt=0`,
`OwnBook=false` and `UseNNUE=true` at startup. The environment variable
`MITTENS_CON_LIBRO=1` does not enable the book in this preset. The GUI can
still configure options through UCI. The 1024-neuron, 8-output-bucket NNUE
is embedded in the executable and enabled by default.

Three build approaches are available:

| Variant | CPU requirements | NNUE and sliding attacks |
|---|---|---|
| **Universal (Recommended)** | x86-64 baseline (`x86-64-v2`) | Runtime dispatch: AVX2 on modern chips, SSE2 on older; PEXT on Intel/Zen3+, Magic Bitboards on AMD Zen 1/2 |
| Portable | x86-64 baseline with SSE2 | SSE2 NNUE; magic bitboards |
| Fixed AVX2/BMI2 | x86-64-v3 and operating-system AVX support | AVX2 NNUE; PEXT/BMI2 attack lookup |

The **Universal** variant is the easiest and safest choice for rating list operators (CCRL, CEDR): a single binary automatically runs at peak speed on both Intel and AMD Zen 3+ without crashing on older systems and without the 18-cycle PEXT slowdown on AMD Zen 1/Zen 2.

```sh
# Universal CCRL Linux build:
RUSTFLAGS="-C target-cpu=x86-64-v2" \
cargo build --release --locked --features ccrl --target x86_64-unknown-linux-musl --bin mittens
```

Example Linux builds (musl target and linker must be installed):

```sh
RUSTFLAGS="-C target-cpu=x86-64" \
cargo build --release --locked --features ccrl --target x86_64-unknown-linux-musl --bin mittens

RUSTFLAGS="-C target-cpu=x86-64-v3" \
cargo build --release --locked --features ccrl,intel-bmi2 --target x86_64-unknown-linux-musl --bin mittens
```

For Windows PowerShell with a configured MinGW toolchain:

```powershell
$env:RUSTFLAGS = "-C target-cpu=x86-64"
cargo build --release --locked --features ccrl --target x86_64-pc-windows-gnu --bin mittens

$env:RUSTFLAGS = "-C target-cpu=x86-64-v3"
cargo build --release --locked --features ccrl,intel-bmi2 --target x86_64-pc-windows-gnu --bin mittens
```

The internal feature name `intel-bmi2` enables PEXT whenever the x86-64
build also enables BMI2. It is not a CPU-brand check. The attack table is
built in PEXT order and exhaustively compared against independent ray
casting. Other builds retain magic bitboards. The NNUE uses integer
arithmetic; no speedup is attributed to unused floating-point instructions.
AVX-512, VNNI and AMX are not enabled.

### Portable NNUE: fused SSE2 evaluation

The portable x86-64 build fuses common accumulator updates with SCReLU and
output evaluation, using SSE2 PADDW/PSUBW, PMULLW and PMADDWD instructions.
It consumes updated neurons directly from registers while storing the child
accumulator, avoiding a separate copy and output pass. Integer results remain
exact: the two perspectives are reduced separately into i64. Null moves and
less common update shapes retain the general fallback. The AVX2 and NEON
paths remain available on their respective targets. No different network
or additional CPU extensions are required for the SSE2 path.

### CCRL 40/15 setup

Configure a REPEATING time control of 40 moves in 15 minutes, calibrated for
the tester's machine using Stockfish 10 as specified by CCRL. Pondering OFF,
`OwnBook=false`, book and position learning OFF. Use the same generic book
for all engines, with at most 12 moves per side. Set 256 or 512 MiB of search
hash per single-CPU engine; for a 4-CPU entry, up to four times that amount.
The operator selects `Threads=1` or `Threads=4`.

Provide external Syzygy tables through `SyzygyPath`; no tablebases or opening
book are included. There is no separate EGTB hash option: search Hash does
not replace the 32/64 MiB EGTB setting in the CCRL conditions. Coordinate that
requirement with the tester/controller. A package does not imply CCRL acceptance.

References:

- [PEXT/BMI2 in Rust](https://doc.rust-lang.org/stable/core/arch/x86_64/fn._pext_u64.html)
- [SSE2 PMADDWD](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_madd_epi16.html)
- [CCRL 40/15 conditions](https://computerchess.org.uk/4040/about.html)
