# Revisión del motor Mittens (v0.8.0) — informe final

Autor de la revisión: Claude Fable 5.1, con el informe de arquitectura de Kimi K3
(`REVISION/01_arquitectura.md`) como base. Entorno de verificación: Android/Termux
(proot Ubuntu, aarch64, 8 núcleos), Rust 1.96, clon `cd028be`.
Citas en formato `archivo:línea` sobre el árbol tal como se clonó.

## 1. Resumen ejecutivo

- El motor está **bien construido para su tamaño**: NNUE bullet 768→1024x2→1 con 8
  output buckets, acumulador con pila por ply y NEON, TT lockless empaquetada en un
  u64, y una búsqueda moderna (PVS, aspiración, RFP, razoring, NMP guiado por eval,
  ProbCut, IIR, singular extensions con multicut, LMR tabulado, LMP, futilidad por
  profundidad post-LMR, correction history, history/continuation/capture history,
  hindsight reductions, Lazy SMP con saltos de profundidad). La cobertura de tests
  es alta para un proyecto de este tipo (119 tests, perft, fuzz del acumulador, SEE
  contra oráculo, protocolo UCI).
- **Compila y juega en este teléfono** una vez sorteadas las rutas de linker de una Mac
  concreta que trae el `.cargo/config.toml` (B3, sección 6). 117 de 118 tests pasan; el único fallo era un test h2h con rutas
  absolutas de macOS, ya corregido en este árbol.
- **Bug real encontrado**: el envejecimiento (aging) de la TT compartida usa 5 bits
  en el contador y 4 bits en la entrada empaquetada, así que con `Threads>1` la
  política de reemplazo se degrada a "siempre reemplazar" durante la mitad de la
  partida. Corregido en este árbol (sección 3, B1).
- Hay deuda de **documentación divergente** (README describe una red 512 que ya no es
  la embebida; comentarios que afirman que la TT compartida es la única en producción
  cuando `Threads=1` usa la TT local) y de **calibración**: 14 perillas por variable de
  entorno con valores "históricos" sin barrer.
- La candidata construida a partir de esta revisión **gana el duelo contra la base**:
  +27,9 ± 19,6 Elo en 700 partidas a 25 000 nodos (sección 7). La versión final
  (candidata 4, sección 9) añade una TT por cubetas y votación entre hilos: **+36 ± 21**
  sobre la candidata 3 a un hilo y **57,5 %** a 8 hilos por reloj.

## 2. Arquitectura (complemento al informe de Kimi)

Lo que el informe parcial de Kimi dejó pendiente (segunda mitad de `src/lib.rs`):

- **Despacho UCI** (`src/lib.rs:1213-2064`): `stop` e `isready` se atienden sin tocar el
  estado (`lib.rs:1263-1287`); cualquier otro comando primero detiene y recupera la
  búsqueda activa (`detener_y_recuperar`, `lib.rs:1013-1034`). `position` es atómico
  y estricto: si una jugada del stream es ilegal se restaura tablero e historial
  previos (`lib.rs:1521-1571`).
- **Dos backends de TT según `Threads`** (`lib.rs:1218-1233`, `1447-1467`): con un hilo
  se usa `Searcher::new` (TT local `Vec<Option<TTEntry>>`, ~24-32 bytes por
  casillero); con más de uno, la TT compartida empaquetada (8 bytes por casillero,
  `search.rs:657-800`). El comentario del test `tt_compartida_colision_menos_profunda_no_pisa`
  (`search.rs:4720-4724`) afirma lo contrario; está desactualizado.
- **Gestión de tiempo de dos presupuestos** (`lib.rs:1058-1146`, `search.rs:3730-4100`):
  objetivo = reloj/movestogo + 0.8·incremento, máximo = min(80 % del reloj, 4×objetivo,
  `Max Move Time`). El corte blando es adaptativo (55-85 % según estabilidad del PV,
  multiplicado por factores de caída de score, inestabilidad y esfuerzo) y **solo
  actúa con reloj real**; con `go movetime` se gasta el presupuesto entero. Los hilos
  ayudantes de Lazy SMP no administran el reloj (`search.rs:4460-4500`).
- **`go depth` y `go nodes`** enrutan por Lazy SMP cuando `Threads>1` (`lib.rs:1596-1690`);
  el límite de nodos se revisa cada 256 nodos dentro del árbol (`search.rs:1720-1760`).
- **MultiPV** es una simulación secuencial con `root_moves_filtro` (`lib.rs:1815-1895`):
  no soporta `stop` a mitad de camino (documentado en el propio código).
- **Evaluación**: en producción es "red pura": `round(peso·red) + TEMPO(12)`
  (`eval.rs:1509-1550`), con `material_insuficiente` y `final_aplastante` como cortes
  reglamentarios antes de la red (`eval.rs:~1180-1300`). La eval clásica solo se usa en
  `draw_score` (contempt) y en modos híbridos. Hay una cache global de eval por zobrist
  (`eval_cache.rs`, 16 MB, lockless) válida solo en modo red pura.
- **FFI/JNI** (`ffi.rs`, `jni_bridge.rs`): `mimotor_new` redirige los fd 0 y 1 del
  proceso a pipes y corre `uci_loop()` en un hilo; el puente JNI está escrito sin
  `unwrap`/indexado porque el perfil release usa `panic=abort`.

## 3. Bugs y defectos encontrados

Severidad: **A** afecta la fuerza o corrección en juego; **B** portabilidad/robustez;
**C** documentación o cosmético.

### B1 (A) Aging de la TT compartida roto la mitad del tiempo
- `search.rs:743` empaqueta la generación en **4 bits** (`& 0xF`), pero el contador se
  enmascara con **5 bits** en `search.rs:1248` (`set_tt_generacion`), `3595`, `3763`
  (`search_fixed_depth`/`search_time`) y `4382` (`buscar_lazy_smp`).
- Efecto: con generación 16..31 ninguna entrada coincide con la generación actual, y
  `tt_store` (`search.rs:1676-1689`) trata todo casillero como "viejo": las escrituras
  de quiescence (depth 0) desalojan entradas profundas. Es exactamente el bug que el
  comentario de `tt_store` dice haber corregido, reabierto por la mitad de las jugadas.
- Solo afecta a `Threads>1` (con un hilo la TT local guarda la generación como `u8`).
- El test `smp_tt_generacion_compartida_avanza_entre_llamadas` (`search.rs:4928`) leía
  `(raw >> 43) & 0x1F`, que mezcla el bit `was_pv` (bit 47): por eso no lo detectó.
- **Corregido en este árbol**: máscara `0xF` en los cuatro sitios y en el test.

### B2 (B) Tests h2h con rutas absolutas de otra máquina
- `tests/h2h_mittens_vs_debil.rs:10-11` y `tests/h2h_mittens_vs_termino.rs:14-15` apuntan
  a `/Users/Tavito/...`; `cargo test` falla en cualquier otra máquina (así se observó
  aquí: 117 ok, 1 fallo).
- **Corregido en este árbol**: `MITTENS_H2H_DIR` / `MITTENS_H2H_PESOS` sobreescriben las
  rutas y el test se omite con aviso si el directorio no existe.

### B3 (B) `.cargo/config.toml` con rutas de una sola máquina
- El repo sí declara `+dotprod` por target en `.cargo/config.toml` (correcto y bien
  comentado: `neural.rs:81-120` usa `sdot` en `asm!` sin camino escalar). El problema es
  que la sección `[target.aarch64-linux-android]` (y las de armv7/x86_64 Android y
  Windows) fija `linker` y `ar` con rutas absolutas del NDK instalado en una Mac
  concreta (`/opt/homebrew/Caskroom/android-ndk/29/...`). En cualquier otra máquina cuyo
  host sea Android (Termux, como aquí) el build muere antes de enlazar.
- Solución aplicada aquí (sin tocar el repo): `CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER`
  y `..._AR` por variable de entorno, que tienen prioridad sobre el archivo.
- Recomendación: dejar en el archivo versionado solo `rustflags`; los `linker`/`ar` van a
  `~/.cargo/config.toml` de cada desarrollador o a variables de entorno. Y para ARM64
  anteriores a ARMv8.2, un camino escalar de respaldo para `sdot` (hoy no compila).

### B4 (A, menor) Hindsight compara escalas distintas
- El padre guarda la eval **corregida por corrhist y refinada por TT**
  (`search.rs:3345`, vía `fut_eval`), pero el hijo compara contra su eval **cruda**
  (`search.rs:2277`, `evaluar_completo`). La señal `eval_delta` mezcla por tanto el
  cambio real de posición con la magnitud de la corrección del padre.

### B5 (A, menor) Calibración de la quiescence en escala clásica
- El delta pruning (`search.rs:2113`) usa `valor_pieza` (100..900) y margen 250 sobre una
  eval que va ~1.6× inflada (`peso_bullet`), es decir el margen efectivo es ~156 cp
  reales y las piezas se subestiman un 37 %. Igual con `see < -50` (`search.rs:2101`).
- El stand-pat devuelve `beta` pero guarda `stand_pat` en la TT (`search.rs:2032-2035`):
  mezcla fail-hard y fail-soft sin motivo.

### B6 (A, discutible) Extensión de jaque incondicional
- `search.rs:2258`: todo jaque hasta ply 40 extiende un ply completo. Los motores de
  referencia la eliminaron o la condicionan (SEE ≥ 0). Sospechoso de inflar el árbol en
  posiciones con jaques baratos; requiere medición (no se tocó en la candidata).

### B7 (B) FFI sin comprobación de errores ni reentrada
- `ffi.rs:39-50`: `pipe()`/`dup2()` sin comprobar el retorno; los `dup(0)`/`dup(1)` se
  filtran; una segunda llamada a `mimotor_new` rompe la primera. Documentar como
  "una vez por proceso" está bien, pero conviene un guard estático.

### B8 (C) Documentación divergente
- `README.md:104,115,137-141`: dice que la red embebida es `pesos_bullet_512_buckets8.bin`;
  `neural.rs:1303` embebe `pesos_bullet_1024_buckets8.bin` (y `bullet_net.rs:90-95` aún
  la llama "PENDIENTE_ENTRENAR_1024").
- `README.md:156`: "31/31 pruebas"; hoy son 119.
- `search.rs:4720-4724` y `4901-4906`: "la compartida es la única que se usa en
  producción" — falso con `Threads=1` (`lib.rs:1233`).
- El comentario del campo `tt_generation` (`search.rs:861-864`) dice `0..31`; el layout
  real es `0..15`.

### B9 (C) Varios menores
- `valor_pieza` en `search.rs:1036` duplica `VALOR` de `see.rs:14`.
- `syzygy::init` y `polyglot::init` no admiten recarga (`syzygy.rs:33`, `polyglot.rs:68`):
  cambiar `SyzygyPath`/`BookPath` en caliente devuelve error.
- `set_tt_generacion` y los `Searcher::new*` repiten ~40 líneas de inicialización
  (`search.rs:1041-1095` vs `1160-1215`); una función común evitaría divergencias.
- Con `Threads=1` en modo SMP interno (`buscar_lazy_smp` con `n_hilos<=1`) se crea un
  `Searcher` nuevo por `go` con ~260 KB de buffers (`search.rs:4386-4406`).

## 4. Rendimiento

Medido aquí (aarch64, un hilo, binario release con `+dotprod`):

| posición (bench 10) | nodos | nps |
|---|---|---|
| inicial | 50 143 | 761 k |
| medio juego abierto | 38 212 | 732 k |
| medio juego cerrado | 65 240 | 763 k |
| táctico | 139 106 | 1 081 k |
| final de torres | 25 261 | 1 538 k |

- Los puntos calientes ya están atendidos: capa de salida en NEON con cuatro
  acumuladores i64 (`bullet_net.rs:690-780`), actualización fusionada del acumulador
  (`bullet_net.rs:233-350`), prefetch de TT y de la cache de eval, `da_jaque_sin_copiar`
  para las guardas de poda, SEE calculado una sola vez por jugada.
- **Oportunidades**:
  1. Unificar la TT: usar el formato empaquetado de 8 bytes también con un hilo (hoy la
     TT local gasta 3-4× más memoria por entrada y duplica el código de `tt_store`).
  2. Guardar la eval estática en la TT (hoy no cabe: la cache de eval externa lo
     compensa parcialmente).
  3. Los jaques silenciosos de quiescence (`search.rs:2140-2180`) generan la lista legal
     completa en cada hoja de primer nivel. La medición de la sección 7 indica que
     apagarlos **pierde** fuerza a nodos fijos, así que el costo se paga por algo; la
     mejora sería generar solo jaques (generador dedicado) en vez de la lista completa.
  4. Ordenación de la raíz por nodos gastados en la iteración anterior (hoy solo
     `tt_move` + history), práctica estándar que estabiliza las ventanas de aspiración.
  5. `decaer_history` (`search.rs:1409-1440`) divide TODO entre 2 en cada `go`, incluido
     el corrhist; los motores de referencia no decaen el corrhist. Medir.

## 5. Deuda técnica

- **14 perillas por variable de entorno** (`MITTENS_RFP_MARGEN`, `_SE_MARGEN`, `_SE_PROF_MIN`,
  `_FUT_LMRDEPTH`, `_ROOT_LMR`, `_NMP_BASE/_DIV/_EVAL_MAX`, `_RFP_PROF_MAX`, `_IIR_TT`,
  `_ASPIRACION`, `_QCHECKS`, `_SE_PODA`, `_SE_DOBLE`, más `_LMR`, `_SINGULAR`,
  `_NO_ASPIRATION`, `_EVAL_CACHE_MB`, `_BULLET_SCALE`, `_HILOS`, `_CON_LIBRO`...). Son útiles
  para barrer, pero varias siguen con valores "históricos" sin medir. Tras calibrar,
  deberían pasar a constantes (o a opciones UCI si se quieren exponer).
- **Código dormido**: `bullet_net_amenazas.rs` (1 646 líneas) y la `RedNeural` de 5378
  entradas en `neural.rs` (~1 200 líneas) no están en producción; sus tests se omiten
  si faltan fixtures externos. Mantenerlos cuesta y confunde (`README` y comentarios
  los mencionan como si estuvieran vivos).
- **Documentos de hallazgos en la raíz** (8 archivos `.md` + `MANIFEST_SHA256.txt`,
  `medir/`, `results_sprt/`): valiosos, pero deberían vivir en `docs/` con un índice.
- **`Box::leak` de redes** al recargar `NNUEPath` (`neural.rs:1323-1340`): aceptable, pero
  documentarlo como límite (no recargar en bucle).
- **Dos copias de la lógica de sondeo raíz** (`search_fixed_depth` y `search_time`,
  `search.rs:3576-3720` y `3730-4100`): el bench y el juego pueden divergir en silencio
  (por ejemplo `search_fixed_depth` no decae el history).
- **`Cargo.toml`** sin `rust-version`; sin CI.

## 6. Estado real de compilación y tests (en este entorno)

- `cargo build --release` con el `.cargo/config.toml` del repo intacto y el linker del
  NDK sobreescrito por entorno (B3): **OK** (1 min 30 s desde cero, 30 s incremental).
  Sin `+dotprod` el build falla con 32 errores "instruction requires: dotprod".
- `cargo test --release -- --test-threads=1` (árbol original): **117 ok, 1 fallo, 1 ignorado**.
  El fallo: `duelo_mittens_vs_debil` (ruta `/Users/Tavito/...` inexistente). Con el
  cambio B2 el test se omite con aviso.
- El binario responde correctamente al protocolo UCI (uci/isready/position/go nodes,
  `info depth ... score cp ... pv`, `bestmove`) y carga la NNUE embebida
  (`checksum 52d9075b0d495dce`, 1024 neuronas, 8 buckets).
- El arnés `medir/sprt_diverso.py --smoke-test` pasa (python-chess 1.11.2).

## 7. Candidata y duelo h2h

Método: `medir/sprt_diverso.py` del propio repo (25 000 nodos por jugada, `Threads=1`,
`Hash=128`, banco de 1 200 aperturas, un motor y una TT nuevos por partida). Baseline =
binario compilado del árbol original (`cd028be`). Las variantes de la fase de cribado se
probaron con la MISMA base más una variable de entorno (las perillas ya existían), sobre
las aperturas 0-99; las confirmaciones usan aperturas distintas.

### 7.1 Cribado (200 partidas por variante, IC95% ≈ ±37 Elo)

| variante | perilla | score | Elo |
|---|---|---|---|
| singular desde profundidad 6 | `MITTENS_SE_PROF_MIN=6` | 59,3 % | +65 ± 38 |
| LMR en la raíz desde la jugada 4 | `MITTENS_ROOT_LMR=4` | 54,8 % | +33 ± 36 |
| IIR también con TT superficial | `MITTENS_IIR_TT=4` | 52,0 % | +14 ± 36 |
| ventana de aspiración 25 | `MITTENS_ASPIRACION=25` | 52,0 % | +14 ± 38 |
| singular desde profundidad 5 | `MITTENS_SE_PROF_MIN=5` | 51,8 % | +12 ± 37 |
| singular desde profundidad 4 (75 partidas) | `MITTENS_SE_PROF_MIN=4` | 50,0 % | 0 ± 61 |
| jaques quietos en quiescence apagados | `MITTENS_QCHECKS=0` | 45,5 % | −31 ± 39 |
| RFP hasta profundidad 10 (70 partidas) | `MITTENS_RFP_PROF_MAX=10` | 45,0 % | −35 ± 63 |
| null-move R = 4 + d/3 (78 partidas) | `MITTENS_NMP_BASE=4 _DIV=3` | 42,3 % | −54 ± 61 |

Lección del cribado: 200 partidas NO bastan para separar +10 de +60 Elo; la singular
desde 6 salió +65 en el cribado y **+1,7 ± 22 en 600 partidas de confirmación**
(candidata 2, abajo). Solo sirven para descartar lo claramente negativo.

### 7.2 Candidatas confirmadas

| candidata | contenido | partidas | resultado |
|---|---|---|---|
| 1 | fix TT + singular 6 + LMR raíz 4 + ventana 25 | 600 (aperturas 100-399) | +187 =243 −170, 51,4 %, **+9,8 ± 21,5** |
| 2 | fix TT + singular 6 | 600 (aperturas 400-699) | +186 =231 −183, 50,3 %, +1,7 ± 21,8 |
| 1 (total) | ídem candidata 1 | 1 500 (aperturas 100-399 y 700-1149) | +480 =627 −393, 52,9 %, **+20,2 ± 13,4** |
| **3 (final)** | candidata 1 + reducción LMR extra en nodos de corte (`cut_node`) | 700 (aperturas 400-699 y 1150-1199) | +230 =296 −174, 54,0 %, **+27,9 ± 19,6** |

(El fix de la TT no influye a `Threads=1`; está en todas las candidatas por corrección,
no por fuerza.)

### 7.3 Veredicto

**La candidata final (3) gana el h2h contra la base**: 54,0 % en 700 partidas,
+27,9 ± 19,6 Elo (el intervalo de confianza del 95 % no incluye el cero; el LLR del
SPRT con H1 = +5 Elo no llegó a cruzar su límite, así que el arnés lo etiqueta como
"ambiguo" en el sentido estricto de Wald). La candidata 1, que es la 3 sin el cambio
de `cut_node`, ya ganaba con +20,2 ± 13,4 en 1 500 partidas, lo que respalda que la
ganancia es real y no un artefacto de un tramo de aperturas.

Contenido exacto de la candidata final (todo en `src/search.rs`, más los tests; ver
`REVISION/candidata_fable.patch`):

1. Fix B1: máscara de generación de la TT a 4 bits (`set_tt_generacion`,
   `search_fixed_depth`, `search_time`, `buscar_lazy_smp` y el test que la vigila).
2. `se_prof_min` 8 → 6, `root_lmr_desde` 0 → 4, `ventana_aspiracion` 50 → 25
   (las perillas de entorno siguen funcionando para volver atrás sin recompilar).
3. Nuevo parámetro `cut_node` en `negamax` (semántica de Stockfish: los hijos de un
   cut-node son all-nodes y viceversa, la primera jugada de un nodo PV no es
   cut-node, la sonda reducida de LMR siempre lo es) y `r += 1` en LMR cuando el nodo
   es de corte esperado. Trece sitios de llamada actualizados.
4. Tests h2h portables (B2).

Qué NO cambió: red, evaluación, generador, gestión de tiempo, formato de la TT.

Cómo reproducir el duelo en cualquier máquina:

```
cargo build --release            # en Termux: linker/ar del NDK por variable de entorno, ver B3
python3 medir/sprt_diverso.py ./target/release/mittens pesos_bullet_1024_buckets8.bin \
    /ruta/al/binario/base pesos_bullet_1024_buckets8.bin cand_final 0 5 0.05 0.05 25000 600 400
```

Advertencia honesta: la medición es a **nodos fijos** (25 000). El cambio de `cut_node`
y el LMR en la raíz reducen el árbol, así que a reloj real deberían valer lo mismo o
más; la ventana de aspiración estrecha cuesta re-búsquedas y conviene confirmarla a
tiempo real (por ejemplo 10 s + 0,1 s) antes de publicar una versión.

## 8. Plan de mejoras (10 puntos, en orden)

1. Aplicar los fixes B1 y B2 (hechos aquí) y añadir un test que escriba 20 generaciones
   y verifique que la política de reemplazo sigue prefiriendo profundidad.
2. Sacar los `linker`/`ar` de la Mac del `.cargo/config.toml` versionado (B3) y dar a
   `sdot` un camino escalar de respaldo o detección en tiempo de ejecución.
3. Calibrar las perillas con `medir/sprt_diverso.py` a nodos fijos (empezando por las
   que la sección 7 señala como positivas) y convertirlas en constantes.
4. Unificar la TT en el formato empaquetado para uno y varios hilos; borrar la TT local.
5. Corregir la escala de la quiescence (B5) y la de hindsight (B4).
6. Medir la extensión de jaque (B6): condicionarla a SEE ≥ 0 o a profundidad baja.
7. Ordenación de la raíz por conteo de nodos; verificación de null-move a profundidad
   alta (hoy solo protege finales de rey y peones).
8. Sincronizar README (red 1024, número de tests, lista real de opciones UCI) y
   los comentarios señalados en B8; mover los `.md` de hallazgos a `docs/`.
9. Retirar o aislar en un feature de Cargo las redes dormidas (`bullet_net_amenazas`,
   `RedNeural`) para reducir 2 800 líneas del árbol de producción.
10. CI mínima (GitHub Actions): `cargo fmt --check`, `cargo clippy`, `cargo test`
    en x86_64 y aarch64, y `bench` con conteo de nodos fijo como test de regresión.

## 9. Multinúcleo (Lazy SMP): medición y mejoras

Pedido posterior a la revisión: "aprovechar todos los núcleos". Máquina de medición: el
mismo teléfono (8 núcleos Qualcomm, 6 a 3,6 GHz + 2 a 4,6 GHz), que **se estrangula
térmicamente** tras varios minutos de carga: los nps absolutos varían 2-3× entre
mediciones y solo valen las comparaciones hechas una detrás de otra en las mismas
condiciones.

### 9.1 Punto de partida (candidata 3, `medir/escalado_smp.py`, 3 s por posición)

| posición | 1 hilo | 2 hilos | 4 hilos | 8 hilos |
|---|---|---|---|---|
| medio abierto | d20, 0,58 Mnps | d22, 3,2 Mnps | d22, 5,1 Mnps | d22, 7,5 Mnps |
| medio cerrado | d19, 1,4 Mnps | d20, 3,1 Mnps | d18, 5,0 Mnps | d22, 7,7 Mnps |
| táctico | d20, 0,66 Mnps | d22, 3,1 Mnps | d21, 4,5 Mnps | d22, 6,8 Mnps |

Lectura: el escalado de nodos es sano (8 hilos ≈ 4,7× los nodos de un hilo rápido;
el resto lo pierde el propio teléfono por frecuencia y ancho de banda), y la
profundidad sube ~+2 plies con 8 hilos, que es lo esperable de un Lazy SMP correcto.
El esquema (hilo principal + ayudantes con saltos de profundidad, reloj administrado
solo por el principal, aborto de ayudantes) ya estaba bien resuelto por el autor. Lo
que quedaba por mejorar no era el reparto de trabajo sino **la tabla compartida** en la
que los 8 hilos escriben.

### 9.2 Cambios (candidata 4 = candidata 3 + esto)

1. **TT por cubetas de 4 vías, la misma para 1 y N hilos** (`search.rs`, `CubetaTT`,
   `construir_tt`, `tt_probe`, `tt_store`). Antes: un casillero directo por índice, y
   con 8 hilos y la quiescence escribiendo en todos los nodos, cada colisión de
   índice expulsaba una entrada entera; además el camino de un hilo usaba OTRA tabla
   (`Vec<Option<TTEntry>>`, ~24 bytes por entrada) con su propia política. Ahora: cubeta
   de 32 bytes alineada (nunca cruza una línea de cache), reemplazo por
   `profundidad − 8·edad` dentro de la cubeta, misma clave se sobreescribe salvo que la
   nueva sea ≥4 plies más superficial (regla de Stockfish). Se eliminó la TT local y
   con ella el bug B1 deja de tener dónde reaparecer.
2. **Votación entre hilos** para elegir la jugada final (`buscar_lazy_smp`): cada hilo
   vota por su jugada con peso `(score − score_mín + 14) · profundidad`, en vez de
   quedarse con el hilo más profundo a secas.
3. (Ya en la candidata 3) el fix B1 del envejecimiento, que solo afectaba a
   `Threads > 1`.

### 9.3 Verificación

- **Un hilo, nodos fijos** (arnés de la sección 7, 600 partidas, aperturas 0-299):
  candidata 4 contra candidata 3: **+193 =276 −131, 55,2 %, +36,0 ± 20,5 Elo**. La TT
  por cubetas gana también mono-hilo.
- **NPS mono-hilo en igualdad de condiciones** (`bench 12`, posición táctica,
  alternando binarios): candidata 3 = 0,50 Mnps, candidata 4 = 0,54 Mnps; nodos para
  profundidad 12: 299 k contra 174 k.
- **8 hilos, reloj real** (`medir/h2h_reloj.py`, 5 s + 0,05 s, `Hash 256`, 100
  partidas, motores alternando, así que cada uno dispone de los 8 núcleos mientras
  piensa): candidata 4 contra candidata 3 **+25 =65 −10, 57,5 %** (≈ +52 Elo, IC95%
  ≈ ±40), sin banderas, mismo reloj medio por jugada (0,116 s y 0,115 s).
- Suite de tests: 117 ok en la librería (incluye el nuevo
  `tt_cubeta_expulsa_la_menos_valiosa` y las adaptaciones de los tests de colisión a
  la semántica de cubetas).

### 9.4 Recomendaciones de operación multinúcleo

- Con 8 hilos el motor produce 6-8 Mnps: `Hash 64` (el default) se llena en ~1 s.
  Para el bot (`Threads: 8`) usar `Hash 512` o más; a 3 s por jugada, 1 GB no sobra.
- En teléfonos, `Threads` = núcleos grandes + medianos (aquí 8) es correcto, pero el
  estrangulamiento térmico hace que a partir del minuto 3-5 de partida el nps caiga a
  la mitad: no comparar resultados de dos partidas consecutivas como si fueran
  iguales.
- `medir/escalado_smp.py` mide profundidad a tiempo fijo y es muy ruidoso con una
  sola repetición; usar `reps ≥ 3` y, para decidir, `h2h_reloj.py`.
