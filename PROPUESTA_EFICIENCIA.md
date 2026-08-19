# Propuesta: Mittens Eco — variante de eficiencia energética

Fecha: 2026-08-18. Fase de diseño — **sin cambios de código todavía**.

Objetivo: una variante que dé la mayor fuerza posible **por watt / por núcleo / por MB**, para teléfonos y computadoras viejas. No es "recortar todo": es maximizar Elo por unidad de energía.

---

## 1. Diagnóstico: dónde se va la energía hoy

Medido/verificado en esta sesión:

- **Evaluación NNUE por nodo**: la red es `(768->512)x2->1` SCReLU, 8 buckets, ~786 KB de pesos i16. Cada nodo paga la actualización incremental del acumulador (512 i16 por perspectiva) más la capa de salida (1024 multiplicaciones). Es el costo dominante por nodo junto con la generación de jugadas. Referencia: **~502K NPS a 1 hilo en la Mac** (bench estándar `bench_nps_depth12`: 86.700 nodos, 0,17 s).
- **Hilos = calor, no siempre Elo**: evidencia real de hoy en el Red Magic (Snapdragon 8 Elite Gen 5): 8 hilos → 99,6 °C en el chip y 49 °C en batería (zona de degradación); 4 hilos → estable a 41 °C. En móvil el chip clava su frecuencia por térmico de todas formas: los hilos extra generan calor sin NPS sostenido proporcional.
- **Acumulador clásico**: ya está optimizado — `clasica_necesaria()` en `src/search.rs` salta el mantenimiento (~10 % del tiempo) cuando la red es pura. No hay margen fácil ahí.
- **Cuantización**: ya es i16 (QA=255, QB=64) con ruta NEON `+dotprod` en ARM64 (requisito duro de compilación, no tocar el gate). Bajar a i8 el acumulador rompería la precisión del formato actual; tema esencialmente resuelto.
- **Búsqueda/poda**: la memoria del proyecto documenta que la búsqueda está **saturada** — 3 intentos de mejora = 0 Elo, y los 6 términos estilo Reckless fallaron todos con SPRT. **No proponemos tocar orden/poda**: es el camino ya quemado.

Palancas reales que quedan: (a) tamaño de la red, (b) número de hilos, (c) apagar la NNUE en el extremo bajo, (d) gestión de tiempo.

---

## 2. Propuestas

### P1 — Modo Eco de hilos (adaptativo por hardware) — RECOMENDADA PRIMERO

**Qué cambia**: en vez de un `Threads` fijo, el motor detecta el hardware al arrancar (ARM64/Android vs escritorio, núcleos disponibles) y limita el default: en móvil, `min(4, núcleos_grandes)`; opción UCI nueva `EcoMode` (check) que además baja el Hash a 32–64 MB. `MITTENS_HILOS` y `Threads` siguen mandando si el usuario los fija.

- **Costo**: bajo — ~1 día; toca solo el arranque en `src/lib.rs`, nada de búsqueda ni NNUE.
- **Elo perdido**: en móvil, **≈ 0 o incluso positivo**: con throttling térmico, 8 hilos no rinden más que 4 sostenidos (evidencia del Red Magic). En escritorio no cambia nada salvo que se pida EcoMode.
- **Ahorro**: enorme en móvil — de 99,6 °C a ~41 °C, batería fuera de zona de degradación, consumo aproximadamente a la mitad.

### P2 — Red NNUE de 256 neuronas ("mittens-lite")

**Qué cambia**: entrenar con la infraestructura bullet ya existente una red `(768->256)x2->1`, mismos 8 buckets. Mitad de FLOPS por actualización de acumulador y por capa de salida; pesos ~400 KB. `N_OCULTA1` en `src/neural.rs` es una constante — se compila un binario variante, no un runtime switch.

- **Costo**: medio — el datagen nativo (`mittens datagen`) y el pipeline de entrenamiento ya existen; lo caro es el tiempo de entrenamiento + un h2h de validación (ojo: con 160 partidas el error es ±3,5 puntos; usar SPRT o ≥1000 partidas).
- **Elo perdido**: estimación honesta **30–70 Elo** a tiempo fijo en hardware rápido, pero en hardware lento parte se recupera vía +30–50 % NPS (más profundidad por segundo). Es la palanca clásica de los "small nets" (Stockfish usa exactamente esto: red chica para hojas, grande para nodos importantes).
- **Ahorro**: ~40–50 % del costo de evaluación por nodo → menos ciclos, menos calor por nodo.
- **Riesgo**: recordar la lección del fine-tune 8M@30ms (-94 Elo): validar con datos y método ya probados (los 56,5 M del despliegue actual), no con dataset nuevo.

### P3 — Perfil "hardware muy débil": knobs que YA existen

**Qué cambia**: casi nada de código — empaquetar presets sobre opciones existentes:
- `NNUEClassicalDepth 2–4`: suelta la NNUE en hojas superficiales y evalúa con la clásica (ya implementado y gateado en `siguiente_estado_busqueda`).
- `UseNNUE false`: eval 100 % clásica para el extremo bajo (mucho más débil, ~varios cientos de Elo, pero corre en cualquier cosa).

- **Costo**: trivial — documentar y medir; opcionalmente una opción `Perfil` (combo: normal/eco/ultra-eco) que fije los knobs juntos.
- **Elo perdido**: NNUEClassicalDepth 2: estimo 20–60 Elo (hay que medirlo, nunca se validó con h2h serio); UseNNUE off: grande, solo para hardware donde la NNUE no corre decentemente.
- **Ahorro**: NNUEClassicalDepth evita la actualización de acumulador en la mayoría de los nodos hoja (la fracción más numerosa del árbol).

### P4 — Gestión de tiempo de bajo consumo (menor prioridad)

**Qué cambia**: en EcoMode, terminar la iteración antes cuando la mejor jugada está estable N iteraciones (soft-stop más agresivo) y/o cap de profundidad configurable. Gasta menos tiempo de CPU por jugada aceptando la jugada "ya decidida".

- **Costo**: bajo-medio, pero toca time management → necesita h2h de validación.
- **Elo perdido**: 10–30 Elo estimados por soft-stop agresivo; a cambio ~20–30 % menos tiempo de CPU por partida.
- **Nota**: es la propuesta con peor relación evidencia/riesgo; dejarla para después.

---

## 3. Recomendación

1. **P1 (Modo Eco de hilos)** primero: costo mínimo, evidencia térmica real de hoy, cero riesgo de Elo en móvil. Es la definición misma de "lo mismo por menos".
2. **P3 (presets con knobs existentes)** en paralelo: solo medir con h2h qué cuesta realmente `NNUEClassicalDepth 2` — si sale barato, es ahorro gratis para gama baja.
3. **P2 (red 256)** como el proyecto de fondo: es la única palanca grande que queda (la búsqueda está saturada), y el pipeline ya existe. Validar con SPRT, no con 160 partidas.
4. P4 solo si después de 1–3 aún falta ahorro.

Lo que **NO** hacer: tocar orden de jugadas/poda (todo falló SPRT según memoria del proyecto), bajar cuantización a i8, ni quitar el gate `+dotprod` de ARM64.
