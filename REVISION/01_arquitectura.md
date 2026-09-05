# 01 — Arquitectura del motor Mittens

Revisión de arquitectura basada en lectura directa de `Cargo.toml`, `README.md`,
`src/main.rs`, `src/lib.rs` y el listado de `src/`. Citas en formato `archivo:línea`.
(Estado: PARCIAL — falta la segunda mitad de `src/lib.rs`, líneas 1001–2224, y las
cabeceras de los módulos.)

## Visión general

Mittens es un motor de ajedrez UCI escrito en Rust (paquete `mittens` v0.8.0,
edition 2024; `Cargo.toml:1-5`). Según el README combina:

- Evaluación híbrida clásica + red neuronal NNUE (`README.md:13`).
- Búsqueda negamax con poda alfa-beta, tabla de transposición y quiescence
  (`README.md:14`).
- Protocolo UCI estándar para conectar con GUIs (`README.md:15`).

## Empaquetado: un crate, dos artefactos

`Cargo.toml:7-10` define una biblioteca `mimotor_core` con
`crate-type = ["rlib", "staticlib", "cdylib"]` y raíz en `src/lib.rs`:

- `rlib` para enlazar el binario UCI de consola.
- `staticlib`/`cdylib` para consumo nativo externo (iOS/macOS FFI y Android JNI).

El binario `mittens` es un cascarón deliberado: `src/main.rs:10-12` solo llama a
`mimotor_core::run_cli()`. El comentario de `src/main.rs:1-9` explica que antes
era una copia literal de `lib.rs` y las dos copias divergieron, así que toda la
lógica (bucle UCI, subcomandos, parsers) vive en la biblioteca.

Perfiles de compilación (`Cargo.toml:23-35`): release con `opt-level = 3`,
`lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, `strip = true`; y un
perfil `profiling` que hereda release pero conserva símbolos (`strip = false`,
`debug = 1`).

## Módulos de `src/` (borrador, pendiente de verificar cabeceras)

Declarados en `src/lib.rs:1-26`:

- Núcleo de representación: `types`, `bitboard`, `board`, `zobrist`, `movegen`, `see`.
- Evaluación: `eval` (clásica), `eval_cache`, `neural`, `bullet_net`,
  `bullet_net_amenazas` (red NNUE formato bullet y acumuladores).
- Búsqueda: `search` (negamax/alfa-beta, TT, quiescence, Lazy SMP).
- Aperturas y finales: `polyglot`, `polyglot_random`, `syzygy`.
- Utilidades y verificación: `perft`, `pruebas_consistencia`, `datagen`.
- Integración: `ffi` (`#[cfg(unix)]`, `src/lib.rs:19-23`) y `jni_bridge`
  (`#[cfg(target_os = "android")]`, `src/lib.rs:24-26`).

Además `src/lib.rs` contiene el CLI: `run_simple` (`src/lib.rs:95`), suite perft
(`src/lib.rs:126`), `run_divide` (`src/lib.rs:189`), bench SMP
(`src/lib.rs:199`), bench estándar con NNUE embebida (`src/lib.rs:270-337`),
diagnósticos de LMR y de singular extensions (`src/lib.rs:339`, `src/lib.rs:412`),
tests de mates (`src/lib.rs:501`), de apertura (`src/lib.rs:547`), de finales
(`src/lib.rs:706`), de repetición (`src/lib.rs:771`) y de SEE con oráculo de
fuerza bruta (`src/lib.rs:864`).

## Flujo UCI → búsqueda → evaluación (borrador)

Lo ya verificado:

- Parseo estricto de jugadas UCI: `parse_uci_move` (`src/lib.rs:601-620`)
  convierte texto a `Move` contra `movegen::generate_legal`;
  `aplicar_moves_position` (`src/lib.rs:637-662`) aplica el stream de `position
  ... moves ...` de forma atómica y estricta: ante una jugada ilegal rechaza
  todo y conserva la última posición válida (fix del incidente `bestmove
  h5c5`, documentado en `src/lib.rs:622-636`).
- El manejo de `go`/`stop` usa un hilo de búsqueda con una bandera atómica
  compartida (`AtomicBool`) y reciclaje del `Searcher` o del pool Lazy SMP
  (comentario `src/lib.rs:995-1000`).
- Búsqueda mono-hilo vía `search::Searcher` (`src/lib.rs:29`); multi-hilo vía
  `search::buscar_lazy_smp` con TT compartida construida por
  `search::construir_tt` y un `PoolMemoriaSMP` (`src/lib.rs:207-229`).
- Presentación UCI: `formatear_score_uci` (`src/lib.rs:72-85`) convierte la
  escala interna a `cp`/`mate N`; el factor 1.6 de `escala_uci`
  (`src/lib.rs:56-65`) compensa que la evaluación interna suma la eval clásica
  completa y la red bullet completa (material contado dos veces,
  `src/lib.rs:45-50`) y es solo cosmético.
- Evaluación híbrida: `evaluate_with_state` suma clásica + red NNUE
  (`src/lib.rs:45-47`). La red en producción es bullet `768 -> 512x2 -> 1`
  SCReLU con 8 output buckets, embebida con `include_bytes!` desde
  `src/neural.rs` (según `README.md:103-109` y `README.md:137-139`), cargada
  por `neural::cargar_embebida` + `neural::set_activa` (`src/lib.rs:283-288`).
  `NNUEPath` permite sustituirla (`README.md:110`).
- Libro Polyglot `performance.bin` embebido vía `include_bytes!` en
  `src/polyglot.rs`, apagado por defecto (`OwnBook=false`;
  `README.md:97-101`, `README.md:145-152`).
- Tablas Syzygy vía opción `SyzygyPath` (`README.md:102`) y módulo `syzygy`.

Pendiente de leer: el cuerpo de `uci_loop` y `run_cli` (líneas 1001–2224 de
`src/lib.rs`) para documentar el despacho exacto de comandos, la gestión de
tiempo y el ponderado de `Threads`.

## Dependencias externas

De `Cargo.toml:12-21`:

- `arrayvec = "0.7.8"` — colecciones de capacidad fija (listas de jugadas).
- `rand = "0.10.2"` — aleatoriedad (datagen / polyglot_random).
- `shakmaty = "0.30.1"` — modelo de reglas de ajedrez (usado por syzygy).
- `shakmaty-syzygy = "0.28.1"` — lectura de tablebases Syzygy.
- `jni = "0.21"` — SOLO cuando el target es Android
  (`[target.'cfg(target_os = "android")'.dependencies]`, `Cargo.toml:20-21`),
  para no arrastrarla en macOS/iOS.

## Targets

- **Binario UCI de consola** (`mittens`): `src/main.rs` → `run_cli()`.
- **Android (JNI)**: `src/jni_bridge.rs`, condicional a
  `cfg(target_os = "android")` (`src/lib.rs:24-26`). Probado en arm64-v8a con
  Android 16; `.so` precompiladas para arm64-v8a/armeabi-v7a/x86_64 en
  `android-jnilibs/`, `minSdk 24+` (`README.md:40-53`).
- **iOS/macOS (FFI C)**: `src/ffi.rs`, condicional a `cfg(unix)`
  (`src/lib.rs:19-23`); usa pipes de Unix (`dup2`/`RawFd`). Hay un
  `MimotorCore.xcframework/` en el repo con slices `ios-arm64` y
  `ios-arm64-simulator`, y headers en `include/` (`mimotor.h`,
  `module.modulemap`).

## Diagrama de flujo (borrador)

```
GUI/terminal
   |  stdin: "uci" / "setoption" / "position" / "go" / "stop"
   v
+-----------------------------------------------+
| src/main.rs -> mimotor_core::run_cli()        |
| src/lib.rs: uci_loop (lectura de comandos)    |
|   - position -> parse_uci_move /              |
|     aplicar_moves_position (estricto)         |
|   - go -> spawnea hilo de búsqueda            |
|     (AtomicBool stop compartido)              |
+-----------------------------------------------+
   |                                    |
   v (1 hilo)                           v (N hilos, Threads>1)
search::Searcher::search_time    search::buscar_lazy_smp
   |                                 (TT compartida + pool)
   v                                    |
negamax alfa-beta + quiescence <---------+
   |  usa: movegen, see, zobrist, polyglot (libro), syzygy (finales)
   v
evaluación híbrida = eval clásica (eval.rs, con eval_cache)
                   + NNUE bullet (neural.rs + bullet_net.rs, acumulador
                     incremental, pesos embebidos include_bytes!)
   |
   v
info depth/score (formatear_score_uci, escala 1.6) -> bestmove
```

Fin del borrador parcial.
