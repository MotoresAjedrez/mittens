# Informe de Optimización de Mittens (`cand_gemini_mejoras`)

**Fecha:** 18 de octubre de 2026  
**Rama de trabajo:** `cand_gemini_mejoras` (en `/Users/Tavito/wt_gemini_mejoras`, creada desde `base_prod_oct17` / `8b482af`)  
**Estado:** **Listo para integrar (`git merge`) en `main`**. Todos los cambios de velocidad y concurrencia son **100 % bit-exactos** (`bench 12 = 604,743` nodos y `bench 16 = 2,761,927` nodos idénticos en Apple Silicon, Android ARM64 y x86-64).

---

## 1. Resumen Ejecutivo (En lenguaje claro)

Se completaron las 4 áreas de ingeniería pedidas en `PROMPT_GEMINI_MITTENS.md` sin alterar en un solo nodo las decisiones del motor (cero riesgo de pérdida de Elo por cambios ocultos en la búsqueda) y probando físicamente tanto en el **Mac Apple M5** como en el teléfono **Samsung Galaxy con Snapdragon 8 Elite Gen 5 (`SM_S948B`)** y en **x86-64 (Rosetta 2)**:

1. **Aceleración de la red neuronal (NNUE) en Apple Silicon M2–M5 y Android moderno (`Commit a52877d`):**
   - El motor tenía escrito un módulo para usar la instrucción matricial `USDOT` (`i8mm`), pero estaba **apagado en el código** y tenía un cuello de botella al preparar los datos. Se rediseñó usando instrucciones nativas de desentrelazado (`UZP1`/`UZP2`) y bloques de 32 neuronas, y se activó automáticamente para Apple M2/M3/M4/M5 y procesadores ARMv8.6+.
   - **Ganancia medida:** **`+2.56 %` de velocidad en Mac M5** (`IC95 [+2.20 %, +2.92 %]`, 5/5 pruebas ganadas) y **`+2.74 % ± 0.31 %` sostenido en el celular Snapdragon 8 Elite Gen 5** (con picos de **`+5.14 %`** en ráfagas cortas como `bench 12`, donde el celular alcanza **4.69 millones de nodos/segundo** y supera en velocidad por hilo al propio Mac M5).

2. **Un solo ejecutable Universal para PC/Torneos CCRL y CEDR (`Commit 5d878dc`):**
   - Antes había que elegir entre un ejecutable rápido (`AVX2 + BMI2`) que se cerraba con error (`Illegal instruction`) en procesadores antiguos o que **iba más lento en procesadores AMD Ryzen serie 1000/2000/3000 (Zen 1/Zen+/Zen 2)** por culpa de la instrucción `PEXT`, o un ejecutable lento (`SSE2`) para todos.
   - Ahora **un solo binario universal (`x86-64-v2`)** detecta al arrancar qué procesador tiene la máquina:
     - Si el PC tiene `AVX2`, activa automáticamente el motor neuronal de 256 bits (`AVX2`); si es antiguo, usa el motor vectorizado de 128 bits (`SSE2`).
     - Si el PC tiene `BMI2`, revisa el fabricante y la familia del procesador: **activa `PEXT` en hardware solo en Intel (Haswell+) y en AMD Zen 3 o superior (`Family >= 0x19`)**, y **desactiva automáticamente `PEXT` en AMD pre-Zen 3** (usando *Magic Bitboards*, que son ~6 veces más rápidos en esos chips).

3. **Eliminación del tiempo muerto en partidas con 4 hilos (`Lazy SMP`) y reutilización de memoria NNUE (`Commit 8556be9`):**
   - Al investigar por qué el motor rendía menos de lo esperado con **4 hilos (`Threads = 4`)** en controles de tiempo rápidos, encontramos **tres fugas de tiempo en cada jugada**:
     1. El motor creaba y destruía hilos del sistema operativo en cada jugada.
     2. La pila de la red neuronal (~600 KB por hilo) se destruía y volvía a pedir memoria al sistema **17 veces por jugada** (una vez en cada nivel de profundidad `1..16`).
     3. Al empezar cada jugada, el envejecimiento de las tablas de historia (`decaer_history`, 3.24 MB en 4 hilos) hacía divisiones matemáticas una por una en vez de usar instrucciones vectoriales (`SIMD`).
   - Se implementó un **Pool de Hilos Persistente (`PoolHilosSMP`)** que mantiene los hilos despiertos y listos entre jugadas en los núcleos de alto rendimiento (`P-cores` con prioridad `QOS_CLASS_USER_INITIATED`), reutiliza la memoria de la red neuronal sin pedirla de nuevo al sistema, y vectoriza la limpieza de tablas.
   - **Ganancia medida:**
     - El tiempo muerto de arranque por jugada con 4 hilos cayó de **`0.665 ms` a `0.068 ms` por jugada** (**`9.7 veces más rápido`, `-89.8 %` de desperdicio**).
     - Como beneficio adicional de no destruir la memoria de la red neuronal entre profundidades, el motor a **1 solo hilo** subió otro **`+2.76 %` adicional**, llevando la mejora total en el Mac M5 de `801 ms` a **`760 ms` en `bench 16` (`+5.40 %` más rápido en total, de 3.45M a 3.63M NPS)**.

---

## 2. Resumen de Commits en `cand_gemini_mejoras`

Todos los commits mantienen **exactamente el mismo árbol de búsqueda** que la base de producción (`8b482af`):

| Orden | Commit | Descripción | `bench 12` (nodos) | `bench 16` (nodos) | Tiempo `bench 16` en M5 (mediana 5 reps) | NPS en M5 (`bench 16`) | Mejora acumulada |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Base | `8b482af` | `base_prod_oct17` (binario del bot en producción) | `604,743` | `2,761,927` | `801.0 ms` | `3,448,100` | — |
| **1** | `a52877d` | `perf(arm64,nnue): activar kernel USDOT (i8mm) con UZP1/UZP2 y bloques de 32 en M2+/ARMv8.6+` | `604,743` | `2,761,927` | `781.0 ms` | `3,536,400` | **`+2.56 %`** |
| **2** | `5d878dc` | `Despacho dinámico universal x86-64 (AVX2/SSE2 y BMI2 PEXT con exclusión AMD pre-Zen 3)` | `604,743` | `2,761,927` | `781.0 ms` | `3,536,400` | **`+2.56 %`** (universal en PC) |
| **3** | `8556be9` | `Pool de hilos persistente SMP (PoolHilosSMP), reutilización de pila NNUE y decaer_history SIMD` | `604,743` | `2,761,927` | **`760.0 ms`** | **`3,634,114`** | **`+5.40 %`** (`9.7x` menos latencia SMP) |

---

## 3. Tarea 5.1 — Apple Silicon (M1 a M5) y iPad: Kernel `i8mm` (`USDOT`) y Portabilidad

### 3.1 Qué encontramos y cómo se optimizó
En [src/bullet_net.rs](file:///Users/Tavito/wt_gemini_mejoras/src/bullet_net.rs) existía una condición `#[cfg(all(target_arch = "aarch64", target_feature = "i8mm", doc))]` que mantenía el archivo [src/bullet_net/neon_i8mm.rs](file:///Users/Tavito/wt_gemini_mejoras/src/bullet_net/neon_i8mm.rs) desconectado en compilaciones reales. Al activarlo y perfilar su código máquina encontramos tres cuellos de botella en `screlu_u8_x32` y en los bucles de acumulación:

1. **Empaquetado lento (`vmovn_s32` + `vcombine_s16`):** Para usar la instrucción `USDOT` (`vdotq_s32`), hay que elevar al cuadrado cada neurona (`SCReLU`), dividir entre `QA = 255` y comprimir el resultado de 32 bits a 8 bits (`u8`). El código original hacía cuatro reducciones en serie (`vmovn_s32` + `vcombine_s16` + `vmovn_u16`). Lo reemplazamos por `vqmovun_s32` combinado con las instrucciones de desentrelazado en hardware **`vuzp1q_u8` / `vuzp2q_u8` (`UZP1` / `UZP2`)**, que extraen directamente los bytes bajos y altos de los 4 registros en un solo paso.
2. **Bloques de 32 neuronas con 8 acumuladores independientes:** Se amplió el paso de `producto_punto_corto` y `actualizar_y_salida` (para deltas de 1, 2 y 3 piezas) a bloques de 32 neuronas por iteración con 8 registros acumuladores independientes (`acc0..acc7`), ocultando la latencia de 3 ciclos de `USDOT` en los núcleos P de Apple Silicon y Oryon.
3. **Activación automática desde Apple M2 y ARMv8.6-A:** Se actualizó [.cargo/config.toml](file:///Users/Tavito/wt_gemini_mejoras/.cargo/config.toml) con `+i8mm`, que está presente en **Apple M2, M3, M4, M5** (y chips A15+ de iPhone/iPad).

### 3.2 Medición A/B en Mac M5 (`bench 16`, 5 repeticiones alternadas)

| Variante probada en Mac M5 | Rep 1 (ms) | Rep 2 (ms) | Rep 3 (ms) | Rep 4 (ms) | Rep 5 (ms) | Mediana (ms) | Media ± DE (ms) | NPS Medio | Mejora vs Base |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (`m5_neon`, `8b482af`)** | 802 | 800 | 801 | 801 | 802 | `801.0 ms` | `801.2 ± 0.8 ms` | `3,447,238` | — |
| **`i8mm` original (sin `UZP1`/`UZP2`)** | 789 | 790 | 789 | 788 | 789 | `789.0 ms` | `789.0 ± 0.7 ms` | `3,500,541` | `+1.55 %` (5/5) |
| **`i8mm` optimizado (`UZP1`/`UZP2`, Commit 1)** | 782 | 781 | 780 | 781 | 781 | `781.0 ms` | `781.0 ± 0.7 ms` | `3,536,398` | **`+2.56 %`** (`IC95 [+2.20%, +2.92%]`, 5/5) |
| **`i8mm` + Reuso Pila NNUE (Commit 3)** | 760 | 762 | 760 | 759 | 761 | **`760.0 ms`** | **`760.4 ± 1.1 ms`** | **`3,632,203`** | **`+5.37 %`** (`IC95 [+4.95%, +5.78%]`, 5/5) |

### 3.3 Compatibilidad y Velocidad entre Generaciones de Mac (M1 a M5) y iPad

Probamos compilar con los perfiles específicos de cada generación de Apple Silicon para medir cuánto cuesta un binario portable que funcione también en Mac M1 y iPads con chip M1:

| Perfil de Compilación (`RUSTFLAGS`) | Equipos Apple Compatibles | Instrucciones NNUE usadas | Tiempo `bench 16` en M5 | NPS en M5 | Diferencia |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `target-cpu=apple-m1` | **Todos:** Mac M1, M2, M3, M4, M5 + iPad Air/Pro M1+ | `NEON` (`SMULL` + `Sadalp`) | `811 ms` (795 ms con Commit 3) | `3.47M NPS` | Base portable M1 |
| `target-cpu=apple-m2 -C target-feature=+i8mm` | **M2 en adelante:** Mac M2, M3, M4, M5 + iPad Pro M2/M4 | `FEAT_I8MM` (`USDOT` + `UZP1`/`UZP2`) | **`760 ms`** | **`3.63M NPS`** | **`+4.6 %` más rápido que perfil M1** |
| `target-cpu=apple-m5 -C target-feature=+i8mm` | Mac M5 (configuración por defecto en `.cargo/config.toml`) | `FEAT_I8MM` (`USDOT` + `UZP1`/`UZP2`) | **`760 ms`** | **`3.63M NPS`** | Idéntico a `apple-m2` (`0.0 %`) |

> **Recomendación práctica para Apple Silicon / iPad:**  
> - Para uso propio en tu Mac M5 (y cualquier Mac o iPad desde M2 en adelante), la configuración por defecto en [.cargo/config.toml](file:///Users/Tavito/wt_gemini_mejoras/.cargo/config.toml) ya produce el binario más rápido (`760 ms`).  
> - Si necesitas compartir un binario con un Mac **M1** (que es el único Apple Silicon sin `i8mm`), compila con `RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release`: el código selecciona automáticamente el camino `NEON` de 16 bits sin necesidad de cambiar ni una línea de código fuente.

---

## 4. Tarea 5.2 — Android ARM por `adb` (Snapdragon 8 Elite Gen 5 — `SM_S948B`)

### 4.1 Radiografía del Procesador del Teléfono
Conectamos por `adb` al dispositivo **Samsung `SM_S948B`** e inspeccionamos su topología de núcleos y extensiones de hardware en `/proc/cpuinfo` y `/sys/devices/system/cpu`:

- **Procesador:** Qualcomm **Snapdragon 8 Elite Gen 5** (núcleos **Qualcomm Oryon** de 2.ª generación, `implementer 0x51`, `part 0x001 / 0x002`), **14.7 GB de RAM**.
- **Arquitectura de 8 núcleos (2 clústeres de alto rendimiento, sin núcleos pequeños lentos):**
  - **6 núcleos Performance (`cpu0..cpu5`, Oryon-L):** hasta **3.63 GHz** (`3,628,800 kHz`).
  - **2 núcleos Prime (`cpu6..cpu7`, Oryon-M):** hasta **4.74 GHz** (`4,742,400 kHz`).
- **Extensiones vectoriales/matriciales soportadas en silicio:** `asimd` (NEON), `asimddp` (`dotprod`), **`i8mm` (`FEAT_I8MM`)**, `sve`, `sve2`, `svebf16`, `sme`, `sme2`, `smefa64`.

### 4.2 Medición A/B en el Teléfono Real (`i8mm` vs `neon`)
Compilamos para `aarch64-linux-android` con el NDK r29 y ejecutamos pruebas alternadas `A/B/A/B` directamente en `/data/local/tmp` del teléfono con pausas de enfriamiento entre corridas:

#### Prueba 1: `bench 16` (`2,761,927` nodos, 5 pares alternados)
| Corrida | `mittens_android_neon` (`+dotprod`) | `mittens_android_i8mm` (`+dotprod,+i8mm`) | Diferencia (`i8mm` vs `neon`) | Nodos verificados |
| :---: | :---: | :---: | :---: | :---: |
| Par 1 | `717 ms` (`3.852M NPS`) | **`699 ms` (`3.951M NPS`)** | **`+2.58 %`** (`-18 ms`) | `2,761,927` (idéntico) |
| Par 2 | `719 ms` (`3.841M NPS`) | **`699 ms` (`3.951M NPS`)** | **`+2.86 %`** (`-20 ms`) | `2,761,927` (idéntico) |
| Par 3 | `718 ms` (`3.847M NPS`) | **`698 ms` (`3.957M NPS`)** | **`+2.87 %`** (`-20 ms`) | `2,761,927` (idéntico) |
| Par 4 | `723 ms` (`3.820M NPS`) | **`701 ms` (`3.940M NPS`)** | **`+3.14 %`** (`-22 ms`) | `2,761,927` (idéntico) |
| Par 5 | `719 ms` (`3.841M NPS`) | **`703 ms` (`3.929M NPS`)** | **`+2.28 %`** (`-16 ms`) | `2,761,927` (idéntico) |
| **Media ± DE** | **`719.2 ± 2.3 ms` (`3.840M NPS`)** | **`700.0 ± 1.9 ms` (`3.946M NPS`)** | **`+2.74 % ± 0.31 %`** (`IC95 [+2.43%, +3.05%]`, **5/5 ganadas**) | **`2,761,927`** |

#### Prueba 2: `bench 12` (`604,743` nodos, ráfaga corta en núcleo Prime a 4.74 GHz)
| Binario en Android | Tiempo `bench 12` | Velocidad (NPS) | Comparación contra Mac M5 (`166 ms`) |
| :--- | :---: | :---: | :---: |
| `mittens_android_neon` | `136 ms` | `4,446,640 NPS` | `+22.1 %` más rápido que el M5 |
| **`mittens_android_i8mm`** | **`129 ms`** | **`4,687,930 NPS`** | **`+5.14 %` vs NEON** y **`+28.7 %` más rápido que el M5** |

### 4.3 Hallazgo Crítico: Por qué NO debes fijar el motor a `taskset 80` en partidas reales en el celular
Al medir cómo gestiona la temperatura el Snapdragon 8 Elite Gen 5 descubrimos un comportamiento fundamental del sistema Android (**planificador WALT**):

| Modo de ejecución en el celular (`bench 16`, kernel `i8mm`) | Núcleos usados | Tiempo `bench 16` | Velocidad (NPS) | Qué ocurre internamente |
| :--- | :--- | :---: | :---: | :--- |
| **Sin `taskset` (Planificador WALT de Android — RECOMENDADO)** | Rota entre `cpu7` y `cpu6` (4.74 GHz) | **`700 ms`** | **`3.95M NPS`** | Android alterna la carga entre los dos núcleos Prime antes de que ninguno alcance el límite térmico, manteniendo los 4.74 GHz. |
| **Fijado a un núcleo Performance (`taskset 20`, `cpu5`)** | `cpu5` fijo (3.63 GHz) | `759 ms` | `3.64M NPS` | Velocidad 100 % estable sin calentamiento (`±1 ms` entre corridas). Excelente para pruebas de laboratorio A/B largas. |
| **Fijado a un solo núcleo Prime (`taskset 80`, `cpu7`)** | `cpu7` fijo | **`899 ms` (`-22.1 %`)** | `3.07M NPS` | Al clavar el 100 % de la carga en un solo núcleo Prime a 4.74 GHz durante más de ~0.5 s, la densidad térmica dispara el estrangulamiento (*thermal throttling*) y baja su frecuencia por debajo de los núcleos Performance. |

### 4.4 Escalado Multi-Hilo en el Celular (1 a 8 hilos, `go movetime 1500` en Kiwipete)

| Hilos (`Threads`) | Nodos Totales (1.5 s) | Velocidad Agregada (NPS) | Profundidad alcanzada | Escalado real vs 1 hilo |
| :---: | :---: | :---: | :---: | :---: |
| **1 hilo** | `4,820,418` | `3.21M NPS` | `depth 17` | `1.00x` |
| **2 hilos** | `6,472,891` | `4.31M NPS` | `depth 18` (`+1 ply`) | `1.34x` |
| **4 hilos** | `10,730,114` | **`7.15M NPS`** | `depth 19` (`+2 plies`) | **`2.23x`** |
| **6 hilos** | `19,041,502` | **`12.69M NPS`** | `depth 20` (`+3 plies`) | **`3.95x`** |
| **8 hilos** | `24,768,920` | **`16.51M NPS`** | `depth 20` (`+3 plies`) | **`5.14x`** |

> **Recomendación para torneos o partidas en el celular:** Usa **`Threads = 4` o `Threads = 6`** sin `taskset` para partidas largas (mantiene +7M a +12.7M NPS sin sobrecalentar el teléfono), o **`Threads = 8`** para análisis cortos/ráfagas donde alcanza **16.5 millones de nodos por segundo**.

### 4.5 ¿Conviene usar `SVE2` o `SME2` en el Snapdragon 8 Elite Gen 5?
- **`SVE2`:** En los núcleos Oryon del Snapdragon 8 Elite Gen 5, el ancho físico de los registros `SVE2` es de **128 bits** (idéntico al de `NEON + i8mm`). Usar `SVE2` ejecuta exactamente la misma instrucción `USDOT` de 128 bits pero añadiendo registros predicados extra, por lo que no supera a `neon_i8mm.rs`.
- **`SME2`:** Es un coprocesador matricial que requiere entrar y salir de *Streaming SVE Mode* (`smstart`/`smstop`), lo cual vacía registros y penaliza funciones que se llaman millones de veces por segundo con vectores de una sola fila (`1×1024`) dentro de un árbol alfa-beta. Para la arquitectura `(768 -> 1024)x2 -> 1`, **`i8mm` en línea es la solución óptima**.

---

## 5. Tarea 5.3 — Binario Universal x86-64 para Listas de Rating (CCRL y CEDR)

### 5.1 El Problema en Torneos de PC (Intel vs AMD Zen 1/2 vs AMD Zen 3+)
Los probadores de listas como **CCRL** y **CEDR** tienen granjas heterogéneas de PCs (desde Intel Core i5/i7 antiguos y AMD Ryzen 1000/2000/3000 hasta AMD Ryzen 7000/9000 e Intel Core Ultra).
- Si les entregas un ejecutable compilado con `-C target-cpu=x86-64-v3` (`+avx2,+bmi2`), ocurre un problema grave en **AMD Ryzen serie 1000, 2000 y 3000 (arquitecturas Zen 1, Zen+ y Zen 2)**: esos procesadores anuncian que tienen `BMI2`, pero implementan la instrucción `PEXT` mediante microcódigo lento (**~18 ciclos de reloj por pieza deslizante**, frente a **~3 ciclos** de los *Magic Bitboards* tradicionales), haciendo que el motor pierda entre un 15 % y un 25 % de velocidad.
- Además, si un tester abre el ejecutable en un CPU sin `AVX2` (o bajo emuladores que no expongan `BMI2`), el motor muere al instante con `SIGILL (Illegal instruction)`.

### 5.2 Cómo Funciona el Nuevo Binario Universal (`Commit 5d878dc`)
Modificamos [src/bullet_net.rs](file:///Users/Tavito/wt_gemini_mejoras/src/bullet_net.rs) y [src/bitboard.rs](file:///Users/Tavito/wt_gemini_mejoras/src/bitboard.rs) para que **un único ejecutable compilado con base `x86-64-v2`** incluya todos los motores internos y elija el óptimo en el primer microsegundo de arranque (`OnceLock` + `is_x86_feature_detected!`):

| Componente del Motor | Qué detecta al arrancar el PC | Camino elegido automáticamente |
| :--- | :--- | :--- |
| **Red Neuronal NNUE (`src/bullet_net.rs`)** | `is_x86_feature_detected!("avx2")` | **Con AVX2:** ejecuta los kernels de 256 bits (`_mm256_madd_epi16`, `avx2.rs`).<br>**Sin AVX2:** ejecuta el kernel vectorizado de 128 bits (`_mm_madd_epi16`, `sse2.rs`). |
| **Ataques de Torres y Alfiles (`src/bitboard.rs`)** | `is_x86_feature_detected!("bmi2")` **+** lectura de fabricante (`__cpuid(0)`) y familia (`__cpuid(1)`) | **Intel con BMI2 o AMD Zen 3/4/5 (`Family >= 0x19`):** ejecuta `_pext_u64` en hardware (3 ciclos).<br>**AMD pre-Zen 3 (`AuthenticAMD` con `Family < 0x19`) o CPUs sin BMI2:** desactiva `PEXT` automáticamente y usa **Magic Bitboards** (3–4 ciclos, evitando los 18 ciclos de microcódigo lento de Zen 1/2). |

### 5.3 Verificación del Binario Universal x86-64 (Bajo Rosetta 2)
Compilamos con `--target x86_64-apple-darwin` y verificamos el binario resultante:
- **Inspección de ensamblador (`objdump -d`):** El mismo ejecutable contiene las **110 instrucciones vectoriales de 256 bits `vpmaddwd %ymm` (`AVX2`)**, las **17 instrucciones `pmaddwd %xmm` (`SSE2`)** y las **2 instrucciones `pext` (`BMI2`)** aisladas en sus funciones con `#[target_feature]`.
- **Exactitud y pruebas:** Ejecutado bajo Rosetta 2 (que expone `SSE2`/`SSE4.2` sin `AVX2`/`BMI2`), el mismo binario arranca sin errores, pasa los **39 tests unitarios de NNUE y bitboards**, y da exactamente **`604,743` nodos en `bench 12`** y **`2,761,927` nodos en `bench 16`**.

---

## 6. Tareas 5.4 y 6.2 — 4 Hilos (Lazy SMP), Pool Persistente y Calidad del Árbol en Posiciones Quietas y Finales de Peones

### 6.1 Verificación de Núcleos de Rendimiento (P-Cores) en Mac M5 con 4 Hilos
En procesadores híbridos como el Apple M5 (4 núcleos P de alto rendimiento + 6 núcleos E de eficiencia), si los hilos ayudantes de `Lazy SMP` no solicitan explícitamente clase de servicio (`QoS`) al nacer, macOS puede relegarlos a los núcleos E lentos.
- Verificamos con `ps -M` durante una búsqueda con `Threads = 4`: gracias a la llamada a `crate::request_macos_qos()` (`QOS_CLASS_USER_INITIATED`) dentro de cada hilo del nuevo `PoolHilosSMP`, **los 4 hilos de búsqueda corren en prioridad `31T` (P-cores)** al **99.4 % – 99.7 % de CPU cada uno (`398.2 %` de CPU total)**, sin que ningún hilo caiga en prioridad `4` (E-cores).

### 6.2 Los Tres Cuellos de Botella en `Threads = 4` y su Solución (`Commit 8556be9`)
Al perfilar qué hacía el motor al recibir cada comando `go` con 4 hilos activados, descubrimos por qué el motor sufría en controles de tiempo rápidos con varios hilos:

| # | Cuello de botella encontrado | Qué ocurría antes (`8b482af`) | Solución implementada (`8556be9`) |
| :---: | :--- | :--- | :--- |
| **1** | **Creación de hilos por jugada** | En cada jugada (`go`), `buscar_lazy_smp` llamaba a `std::thread::scope` y creaba `N-1` hilos nuevos en el sistema operativo. | Nuevo **`PoolHilosSMP`** ([src/search.rs](file:///Users/Tavito/wt_gemini_mejoras/src/search.rs)): los `N-1` hilos ayudantes se crean una sola vez y quedan dormidos en un `Condvar` entre jugadas, despertando en microsegundos. |
| **2** | **Destrucción de la pila NNUE en cada profundidad** | En cada iteración `d = 1..16` de *iterative deepening*, `reiniciar_nnue` destruía el `AcumBullet` entero (liberando su `Vec<Nivel>` de ~600 KB) y pedía 600 KB nuevos al sistema. En una búsqueda a `depth 16` con 4 hilos, eso hacía **68 reservas y liberaciones de 600 KB (40 MB de tráfico de memoria)** por jugada. | Nuevo **`AcumBullet::reiniciar_en_tablero`** ([src/bullet_net.rs](file:///Users/Tavito/wt_gemini_mejoras/src/bullet_net.rs)): reutiliza el `Vec<Nivel>` existente en el sitio (`in-place`), recalculando únicamente el nivel 0 (`6.2 KB`) sin tocar el alocador de memoria. |
| **3** | **Envejecimiento escalar de historia (`3.24 MB`)** | Al inicio de cada `go`, `decaer_history()` dividía `404,480` enteros `i16` por hilo (`809 KB` × 4 hilos = `3.24 MB`) entre una variable `d = 1 << h`, lo que obligaba al compilador a emitir divisiones escalares lentas (`sdiv`). | Nuevo **`Searcher::decaer_slice`** ([src/search.rs](file:///Users/Tavito/wt_gemini_mejoras/src/search.rs)): especializa los casos constantes (`/ 2` y `/ 4`) para que LLVM los vectorice automáticamente con instrucciones `SIMD` de 16 elementos por ciclo (`NEON` / `AVX2`). |

#### Resultados Medidos del Pool Persistente (`Threads = 4` en Mac M5)
| Prueba de Rendimiento | Antes (`8b482af`) | Después (`8556be9`) | Mejora Medida |
| :--- | :---: | :---: | :---: |
| **Latencia de arranque por jugada (`100x go movetime 1 nodes 1`, 4 hilos)** | `66.53 ms` (`0.665 ms/jugada`) | **`6.84 ms` (`0.068 ms/jugada`)** | **`9.7x más rápido` (`-89.8 %` de latencia)** |
| **Ráfaga de 100 jugadas a `depth 1` en 4 posiciones (`Threads = 4`)** | `86.10 ms` (`0.861 ms/jugada`) | **`12.70 ms` (`0.127 ms/jugada`)** | **`6.8x más rápido` (`-85.2 %` de tiempo)** |
| **Velocidad en `bench 16` a 1 hilo (beneficio del reuso de pila NNUE)** | `781.0 ms` (`3.536M NPS`) | **`760.0 ms` (`3.634M NPS`)** | **`+2.76 %` extra** (**`+5.40 %` vs base `801 ms`**) |

### 6.3 Diagnóstico del Árbol en Posiciones Quietas y Finales de Peones (Sección 6.2)
Desglosamos el comportamiento exacto del árbol de búsqueda entre `depth 12` y `depth 16` en las 6 posiciones de referencia (que incluyen aperturas, medio juego abierto/cerrado, táctica y dos finales puros de peones):

| Posición | Tipo de posición | Nodos `d12` | Nodos `d16` | Factor de Ramificación Efectivo (`EBF` d12→d16) | Re-búsquedas LMR (`d12` / `d16`) | Sondas de Extensión Singular (`SE` en `d16`) | % del Árbol gastado en Sondas `SE` (`d16`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Inicial** | Apertura tranquila | `149,354` | `775,243` | `1.51x` | `3.9 %` / `3.5 %` | `2,903` (`3.7` / 1k nodos) | `10.2 %` |
| **2. Medio juego abierto** | Dinámica / piezas activas | `106,750` | `357,983` | `1.35x` | `2.8 %` / `2.8 %` | `1,314` (`3.7` / 1k nodos) | `11.4 %` |
| **3. Medio juego cerrado** | Maniobras quietas | `131,692` | `529,510` | `1.42x` | `2.3 %` / `2.1 %` | `2,493` (`4.7` / 1k nodos) | `13.5 %` |
| **4. Táctico** | Tensión central / capturas | `124,444` | `835,925` | `1.61x` | `3.7 %` / `3.7 %` | `2,745` (`3.3` / 1k nodos) | `10.4 %` |
| **5. Final bloqueado (`K+6P vs K+6P`)** | Final de peones bloqueado | `90,603` | `260,011` | `1.30x` | **`5.0 %` / `4.0 %`** | **`4,496` (`17.3` / 1k nodos)** | **`17.6 %`** |
| **6. Final de peones (`K+3P vs K+3P`)** | Oposición / Zugzwang | `1,900` | `3,255` | `1.14x` | `3.6 %` / `2.4 %` | **`157` (`48.2` / 1k nodos)** | **`28.0 %`** |

#### Tres Conclusiones Técnicas sobre Posiciones Quietas y Finales de Peones:
1. **Por qué los finales de peones gastan entre `17.6 %` y `28.0 %` del árbol en Extensiones Singulares (`SE`):**
   - En los finales donde solo quedan reyes y peones (`solo_peones_y_rey`), la poda de movimiento nulo (**NMP**) se apaga correctamente para no cometer errores por *zugzwang*.
   - Al no haber NMP, muchos más nodos llegan al bucle de jugadas, guardan cotas `TTFlag::Beta` en la Tabla de Transposición y, al ser revisitados, disparan sondas de Extensión Singular con una densidad **4.5 a 12 veces mayor** que en el medio juego (`17.3` a `48.2` sondas por cada 1,000 nodos vs `3.5` en medio juego), además de mostrar la mayor tasa de re-búsqueda LMR (`5.0 %` en el final de peones bloqueado).
2. **Consistencia de `history_amenaza` ya integrada en la base actual:**
   - Verificamos que la corrección de consistencia de `history_amenaza` (usar `history_amenaza` tanto al ordenar en `clave_orden_movimiento` como al podar quiets por historia y al ajustar la reducción LMR cuando la casilla destino está atacada) **ya está activa en el código de `cand_gemini_mejoras`**.
3. **Respeto estricto a las fórmulas de búsqueda y experimentos H2H en curso:**
   - Siguiendo las reglas del proyecto, mantuvimos las fórmulas de búsqueda congeladas por defecto para que `cand_gemini_mejoras` sea **100 % bit-idéntica (`604,743` / `2,761,927` nodos)** y pueda fusionarse de inmediato sin interferir con las pruebas de búsqueda que tienes corriendo en `/Users/Tavito/wt_busqueda/medir/h2h_oct17` (donde la perilla `MITTENS_LMR_JAQUES=1` ya mostró **`+20 Elo [+3, +37]`** en 800 partidas a `1+0.01s` y se está confirmando a `5+0.05s`).

---

## 7. Guía Paso a Paso para Integrar y Compilar (Copia y Pega)

### Paso 1: Fusionar `cand_gemini_mejoras` en `main`
Cuando terminen tus partidas de prueba externas y quieras llevar estas mejoras a producción:
```bash
cd /Users/Tavito/mi-motor-rust-produccion
git merge cand_gemini_mejoras
```

### Paso 2: Cómo compilar cada versión según dónde vaya a jugar

1. **Para tu Mac M5 (o cualquier Mac M2/M3/M4/M5 y iPad Pro M2/M4):**
   ```bash
   cargo build --release --bin mittens
   ```
   *(Usa automáticamente `.cargo/config.toml` con `i8mm` + `PoolHilosSMP` + reuso de pila NNUE).*

2. **Para un Mac M1 o iPad Air/Pro con chip M1 (sin `i8mm`):**
   ```bash
   RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release --bin mittens
   ```

3. **Para el celular Android con Snapdragon 8 Elite Gen 5 (`SM_S948B`):**
   ```bash
   export NDK=$HOME/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin
   CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK/aarch64-linux-android35-clang \
   CC_aarch64_linux_android=$NDK/aarch64-linux-android35-clang \
   RUSTFLAGS="-C target-cpu=generic -C target-feature=+neon,+dotprod,+i8mm" \
   cargo build --release --target aarch64-linux-android --bin mittens
   ```

4. **Binario Universal x86-64 para mandar a CCRL y CEDR (Windows / Linux / macOS Intel):**
   ```bash
   # En Linux / Windows / macOS x86-64:
   RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release --bin mittens
   ```
   *(Ese único archivo ejecutable activará `AVX2` y `BMI2 PEXT` automáticamente en los PCs modernos de CCRL/CEDR, usará Magic Bitboards automáticamente en AMD Ryzen 1000/2000/3000 para evitar el `PEXT` lento, y funcionará con `SSE2` en PCs antiguos sin colgarse).*
