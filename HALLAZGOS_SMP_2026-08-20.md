# Hallazgos SMP — 2026-08-20

> **ESTADO AL 2026-08-26 — LEER ANTES QUE EL RESTO DEL DOCUMENTO.**
> Los dos problemas de abajo **ya están arreglados en `main`** (commits
> `6d6fda2` / `d425bc6`). Re-medido el 2026-08-26 sobre `main`:
>
> - Con **tiempo de pared igual** (`go movetime 3000`, que no tiene corte
>   blando) 4 hilos llegan **+1.7 plies** más hondo que 1 hilo, y el tiempo
>   para alcanzar una profundidad fija baja **2.5x** con 4 hilos. El "8 hilos
>   ≈ 1 hilo" del punto 1 **ya no reproduce**.
> - `go depth N` **sí** respeta `Threads` (punto 2 arreglado).
>
> Lo que seguía roto era otra cosa: **la gestión de tiempo bajo SMP**. El
> corte blando del reloj lo corría CADA hilo por su cuenta y se hacía join()
> de los N, así que el reloj gastado por jugada era el MÁXIMO de N
> decisiones y se clavaba en el techo duro (medido: 2x el presupuesto con 4
> y 8 hilos). Arreglado en la rama `cand_smp_escalonado` (2026-08-26): el
> reloj lo administra sólo el hilo principal.
>
> Moraleja de medición: el "8 hilos ≈ 1 hilo en profundidad" de este
> documento se midió con `go movetime`, que compara profundidades a tiempo
> igual — está bien. Pero la conclusión "el SMP no rinde" se quedó pegada en
> la memoria del proyecto mucho después de que el arreglo estuviera fusionado.

Investigación de "peces grandes" con la máquina descargada. Todo medido, no
estimado. Binario de `main` (pila de acumuladores + red 1024).

---

## 1. EL PEZ GORDO: Lazy SMP casi no rinde (8 hilos ≈ 1 hilo en profundidad)

Medido con `go movetime 3000` (el camino que SÍ usa `buscar_lazy_smp`),
2 corridas por configuración:

| Posición | 1 hilo (depth / nodos) | 8 hilos (depth / nodos) |
|---|---|---|
| medio abierto | 19 / 1,47M | **19-18** / 10,3M-5,9M |
| medio cerrado | 20-21 / 1,3-1,8M | **22-20** / 10,6M-8,7M |
| táctico | 18 / 1,81M | **18** / 9,5M-8,4M |

Los hilos trabajan (5-6x nodos agregados) pero la profundidad efectiva sube
**~0-1 ply**. Un Lazy SMP sano da +1.5-2.5 plies efectivos con 8 hilos.

### Causa raíz (leída en el código, no especulada)

`buscar_lazy_smp` (search.rs:3754) lanza N hilos idénticos donde:

1. **No hay escalonamiento de profundidad**: todos iteran `for d in 1..=max`
   al mismo ritmo (search.rs:3337). Ningún ayudante se adelanta al principal,
   así que la TT compartida nunca contiene información MÁS PROFUNDA que la
   que el hilo principal ya tiene — solo mejora orden de jugadas, que ya está
   en 90.4%. De ahí el ~0 de ganancia.
2. La única diversificación es `variante_orden_raiz` (swap(0,1) en la raíz,
   hilos impares — que además desperdicia la ventana de aspiration buscando
   la 2ª mejor primero) y `null_move_r_extra` ±1 (i%3).
3. Con 8 hilos, los patrones `i%2`/`i%3` se repiten: **los hilos 6 y 7 son
   duplicados exactos de 0 y 1** (salvo carreras de TT).

### Arreglo propuesto (conocido, barato, alto techo)

Escalonamiento de profundidad estilo Stockfish clásico (skip arrays): los
ayudantes saltan iteraciones según su id para que la mitad esté buscando
d+1/d+2 mientras el principal está en d. Sus entradas de TT (más profundas)
aceleran las iteraciones del principal. Es un cambio localizado en
`search_time`/`buscar_lazy_smp` + parámetro por hilo.

**Impacto**: el bot de Lichess juega con `Threads: 8` (config.yml:86) — hoy
rinde fuerza de ~1 hilo quemando 8 núcleos. +1-2 plies efectivos son decenas
de Elo en juego real multihilo. Validación: h2h por RELOJ (no nodos) 8h-vs-8h
viejo contra nuevo, máquina descargada.

---

## 2. BUG: `go depth N` ignora la opción Threads

`lib.rs:1576`: si el `go` trae "depth", se toma el searcher single-thread
directamente — `n_hilos` ni se consulta. Verificado midiendo: con Threads=4 y
Threads=8 los conteos de nodos son idénticos al último dígito y deterministas
entre corridas (imposible en SMP real).

- Cualquier análisis a profundidad fija desde una GUI usa 1 hilo aunque el
  usuario configure 8.
- Es una trampa de medición: un "escalado SMP" medido con `go depth` da
  resultados sin sentido (nos pasó hoy; misma clase de trampa que el bench
  sin NNUE del 19-ago).

Arreglo: enrutar `go depth` con n_hilos>1 por `buscar_lazy_smp`
(max_depth=N, movetime=None).

---

## 3. Menor: en SMP no se imprime `info` incremental

El camino movetime multihilo imprime UNA línea `info` al final (lib.rs:1922).
Durante la búsqueda el motor se ve "mudo" en GUIs. Cosmético pero fácil de
notar en análisis largos.

---

## 4. Sospecha anotada (NO conclusión): árbol peor con TT compartida

Con `go depth 17` (single-thread por el bug #2), la misma posición dio
667k nodos con TT Local y 1.64M con TT Compartida (misma profundidad).
Medido UNA vez, en UNA posición — puede ser ruido de forma de TT. Si se
retoma: medir en las 200 aperturas del libro antes de creer nada.

---

## 5. Oportunidades de la revisión de código (2026-08-20, mientras corre el h2h)

Revisados los caminos calientes del perfil post-pila (capa de salida NNUE,
movegen, TT, orden). Lo que YA está bien y no hay que tocar: capa de salida
con NEON de 4 acumuladores (exacta), MoveList en stack (ArrayVec, sin heap
por nodo), mapa de amenazas perezoso y SEE deduplicado en el orden.

### 5a. TT con buckets (candidato fuerte)

La TT es direct-mapped de UN slot (`key & mask`, un u64 por casillero,
search.rs:1296). Los motores fuertes usan buckets de 2-4 vías con reemplazo
por (profundidad, edad) DENTRO del bucket. Hoy además la política de aging es
"generación distinta → pisar al instante", que mata entradas profundas de la
jugada anterior apenas empieza la nueva búsqueda. Con 8 hilos martillando la
misma tabla esto pega más — y conecta con la sospecha del árbol 2.5x peor
con TT compartida (punto 4). Cambio localizado en tt_probe/tt_store +
construir_tt; validar por reloj, no por nodos.

### 5b. Generador dedicado de jaques quietos en quiescence

En qdepth==0 con `stand_pat + 150 > alpha`, quiescence llama
`generate_legal_into` COMPLETO (lo más caro del movegen: legalidad de todas
las jugadas) para quedarse con ≤5 jaques silenciosos (search.rs:1748). Un
generador dedicado (jaques directos + descubiertos hacia el rey rival)
evitaría generar y filtrar todo. Antes de implementarlo: medir con un
contador qué % de las entradas a quiescence pasa el gate — dimensiona la
ganancia real.

### 5c. Prefetch de TT (micro, 1-3% NPS típico)

No hay prefetch del casillero de TT del hijo antes de recursar (clásico de
Stockfish). En Rust estable en aarch64 requiere inline asm (`prfm`); posible
pero feo. Anotado como micro-oportunidad, no prioridad.

### Lo que NO hacer (ya falló con SPRT, ver memoria del proyecto)

Generación por etapas / puntuación diferida de quiets / términos de orden de
Reckless / conthist-4 / cutnode-LMR: todos medidos negativos o neutros.
La revisión no encontró razones para reabrirlos.
