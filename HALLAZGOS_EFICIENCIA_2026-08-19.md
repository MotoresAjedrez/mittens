# Hallazgos de eficiencia y calibración — 2026-08-19

Sesión de optimización de Mittens. Todo lo de acá está **medido**, no estimado.

---

## 1. EL BENCH NO USABA LA RED NEURONAL (arreglado)

**El hallazgo más importante de la sesión.**

`neural::cargar_embebida()` se llamaba **únicamente dentro de `uci_loop()`**
(src/lib.rs). El subcomando `mittens bench` construía su `Searcher` sin pasar
nunca por ahí, así que:

- `ACTIVA` quedaba en `false` y el almacenamiento de la red vacío,
- `crear_acumulador()` devolvía `None`,
- `self.nnue` del Searcher quedaba en `None` toda la búsqueda,
- **el bench buscaba con evaluación clásica pura.**

Verificado de dos formas: el bench no imprimía la línea de checksum que sí
imprime la ruta UCI, y con contadores temporales el bench daba **0
actualizaciones de acumulador NNUE** contra 18.067 evaluaciones.

### Por qué importa

`mittens bench` es la métrica rápida que este proyecto viene usando para
validar cambios ("bench idéntico" como test de regresión, conteos de nodos
para calibrar cutnode/LMR/amenazas/conthist). **Todas esas mediciones se
hicieron sobre un motor distinto del que juega**: sin red, el árbol, el orden
de jugadas y sobre todo los márgenes de poda —que están calibrados para la
escala de la red— se comportan de otra manera.

Esto explica algo que la memoria del proyecto ya había registrado sin
entender la causa: *"el conteo de nodos a profundidad fija resultó ser una
métrica NO confiable en este motor, da resultados no monótonos"*. No era solo
inestabilidad de búsqueda: el bench literalmente ejercitaba otra función de
evaluación.

Ejemplo concreto medido en esta sesión, barriendo el margen de RFP con el
bench viejo: 90 → 31.538 nodos, 120 → 29.637, 150 → 45.374. No monótono y sin
sentido físico, porque se estaba podando con márgenes de escala NNUE sobre una
eval clásica.

### El arreglo

`run_bench` ahora carga la red embebida y la activa antes de buscar.

**Efecto medido** (mismas posiciones, misma profundidad):

| | Bench viejo (clásica) | Bench nuevo (NNUE real) |
|---|---|---|
| Nodos | 29.637 | **38.814** |
| NPS | ~2,05 M | ~826 k |

El árbol cambia un 31%. **Los números de bench anteriores a este commit no son
comparables con los nuevos.**

---

## 2. Dónde se va el tiempo (perfilado real, con símbolos)

Perfilado con `sample` sobre una búsqueda real de 25 s, binario compilado con
`-C debuginfo=2` en un target aparte. Reparto por hoja de pila:

| Función | % del hilo de búsqueda |
|---|---|
| `RedBullet::aplicar_features` (actualizar acumulador) | **34,5 %** |
| `eval::evaluate_with_state` (capa de salida) | 15,1 % |
| `search::negamax` | 7,4 % |
| `board::make_move` | 6,3 % |
| `clave_orden_movimiento` | 5,6 % |
| `NnueAccumulator::aplicar_jugada` | 4,5 % |
| resto (orden, TT, movegen, SEE) | < 4 % c/u |

**La NNUE es ~54 % del tiempo de búsqueda**, y un tercio entero se va en una
sola función.

### Cosas que ya estaban bien (verificadas, no tocar)

- **La eval clásica ya es perezosa** en modo puro: hay un comentario en
  `evaluate_with_state` documentando que ese trabajo muerto era ~15 % y ya se
  eliminó.
- **El acumulador NO se clona por nodo**: vive en el `Searcher` y se
  actualiza in-place al bajar y se deshace al volver. El comentario en
  `search.rs:683` documenta que copiarlo costaba 2-4 KB de memcpy por nodo.
- **El bucle NEON de `aplicar_features` ya está bien escrito**: procesa
  bloques de 32 i16 (4 registros), carga el acumulador una vez y aplica todas
  las features antes de guardarlo. No hay micro-optimización obvia.

---

## 3. LA OPORTUNIDAD GRANDE: pila de acumuladores por ply

Medido con contadores temporales sobre el bench con NNUE:

- actualizaciones de acumulador: **25.314**
- evaluaciones completas: **21.680**
- **evaluaciones / actualización = 0,856**

Dos conclusiones opuestas y las dos importan:

1. **Diferir la actualización (lazy update) rinde poco.** El 86 % de las
   actualizaciones termina evaluando, así que solo se ahorraría ~14 %. No
   justifica el riesgo del refactor.

2. **Pero cada jugada paga DOS actualizaciones**, no una: `entrar_hijo`
   aplica el delta al bajar y `salir_hijo` lo aplica invertido al volver
   (`aplicar_jugada(despues, antes)`). Ambas cuestan lo mismo.

   Con una **pila de acumuladores indexada por ply** —escribir el hijo en
   `acc[ply+1]` en vez de mutar in-place— **el deshacer desaparece por
   completo**: volver es decrementar un índice, coste cero.

   Eso elimina ~la mitad de `aplicar_features`, o sea **~17 % del tiempo total
   de búsqueda**, y es **bit-idéntico en resultado** (los mismos valores de
   acumulador, solo guardados en otro lado). Es la clase de mejora sin riesgo
   de debilitar: solo más NPS.

   Coste en memoria: `MAX_PLY × 2 × H_MAX × 2 bytes` ≈ 512 KB por hilo con
   H_MAX=1024. Aceptable.

### IMPLEMENTADO Y MEDIDO (rama `pila-acumulador`)

Se implementó. `AcumBullet` pasa de un buffer único mutado in-place a una
pila de niveles (`Vec<Nivel>` + índice `nivel`), y se agregó
`aplicar_features_desde` (variante NEON que lee del padre y escribe en el
hijo en una sola pasada, sin memcpy previo).

- `entrar(antes, despues)`: escribe el nivel siguiente leyendo del padre.
- `salir()`: decrementa el índice. **Coste cero.**

**Resultado medido** (mismas 4 posiciones, profundidad 15, máquina
descargada, mejor de 3 corridas):

| | Tiempo | NPS |
|---|---|---|
| Sin pila (in-place + undo) | 0,969 s | 1.880.327 |
| **Con pila** | **0,836 s** | **2.179.594** |

**Mismos 1.821.950 nodos en ambos** — o sea que es bit-idéntico y la
comparación es válida. **+15,9 % de NPS** (−13,7 % de tiempo), muy cerca del
~17 % predicho por el perfilado.

Validación: bench bit-idéntico (38.814 nodos exactos) y **102 tests pasan**,
incluidos dos nuevos que cubren específicamente el camino de pila
(`pila_entrar_salir_coincide_con_recalculo` y
`pila_soporta_profundidad_y_desanda_bien`), porque los tests viejos solo
ejercitaban `aplicar_jugada` in-place, que ya no es lo que usa la búsqueda.

---

## 4. Calibración: el peso de la red quedó viejo

`MITTENS_PESO_BULLET` tiene default **1.6**, calibrado en 2026-08-13 para la
red de **512** neuronas y el modo puro. El 2026-08-19 se desplegó la red de
**1024** y nadie recalibró.

Medido a nodos fijos (20.000 nodos/jugada, libro real de 200 aperturas, cada
apertura jugada con ambos colores):

| Peso | Resultado vs 1.6 | Partidas |
|---|---|---|
| 1.2 | ~27 % (mucho peor) | 12 únicas* |
| **1.8** | **55,0 %** (+52 =28 -40) | 120 únicas |
| 2.0 | ~46 % | 12 únicas* |

\* Las corridas de 1.2 y 2.0 salieron de una versión del script con un error
(ver abajo) y valen poco; la de 1.8 sí es limpia.

**1.8 va ganando pero no es concluyente**: con 120 partidas el error estándar
es ~4,5 %, así que 55 % está a ~1,1 desviaciones de 50 %. Hay un segundo tramo
de 120 aperturas nuevas corriendo para confirmarlo o descartarlo.

`MITTENS_RFP_MARGEN` (perilla nueva, ver abajo): 90 vs 120 dio **51,7 %** en
120 partidas — dentro del ruido, no promete.

### Error metodológico encontrado en el propio arnés de pruebas

La primera versión del script de h2h usaba 12 aperturas con
`apertura = i % 12` y `color = i % 2`. Como 12 es par, el ciclo real es de 12
partidas: **pedir 60 jugaba 12 partidas únicas repetidas 5 veces**. Verificado
comparando los resultados: idénticos carácter por carácter.

Arreglado usando el libro real de 200 FEN con emparejamiento explícito
(`apertura = i//2`, `color = i%2`), o sea 400 partidas únicas disponibles.
**Cualquier medición anterior a ese arreglo hay que descartarla.**

---

## 5. Cambios de código hechos en esta sesión

1. **`src/lib.rs` — `run_bench` carga la NNUE embebida.** Cambia el número de
   nodos del bench; documentado en el propio comentario.
2. **`src/search.rs` — `MITTENS_RFP_MARGEN`**, perilla nueva para el margen de
   RFP (antes constante fija de 120, sin forma de A/B testearla). Default
   idéntico: sin la variable el bench daba 29.637 nodos exactos, bit a bit
   igual que antes del cambio.
3. Instrumentación temporal de contadores: **revertida** tras medir.

Tests: **100 pasan, 0 fallan.**

---

## 6. Pendiente / no medido todavía

- **Gestión de tiempo**: en muerte súbita `movestogo` cae a **30** por defecto
  (`src/lib.rs:1633`), o sea que reparte el reloj en 30 partes. Los motores
  fuertes suelen usar 20-25. Puede valer Elo, pero **solo se puede medir por
  reloj**, y durante esta sesión había un entrenamiento comiendo CPU que
  habría corrompido cualquier medición temporal. Queda para una máquina
  descargada.
- **Entrenamiento base 21→24**: el entrenamiento original se cortó en el
  superbatch 20 de 24, dejando sin hacer la cola de learning-rate bajo del
  coseno (la que más afina). Se retomó en esta sesión.
  - Trampa 1 encontrada: `save_rate: 10` habría hecho que el próximo guardado
    cayera en el superbatch 30 → **una hora de GPU sin guardar nada**. Bajado
    a 1.
  - Trampa 2: `load_from_checkpoint` **no** restaura el contador de
    superbatch; con `start_superbatch: 1` reentrenaba los 24 enteros con el LR
    alto otra vez (~4 h, pisando los pesos buenos). Corregido a 21.
