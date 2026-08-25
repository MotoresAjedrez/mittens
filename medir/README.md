# `medir/` — herramientas de medicion honesta

Tres herramientas, cada una contesta una pregunta distinta. **No mezclarlas**:
la primera dice si el motor JUEGA mejor, la segunda dice si el ARBOL es mas
chico, la tercera arma el material para la primera.

---

## 1. `sprt_diverso.py` — ¿juega mejor? (el unico juez de fuerza)

```
python3 medir/sprt_diverso.py CANDIDATO_BIN CANDIDATO_PESOS BASELINE_BIN BASELINE_PESOS \
    NOMBRE [elo0=0] [elo1=5] [alpha=0.05] [beta=0.05] [nodos=25000] \
    [max_partidas=2000] [apertura_inicial=0]

python3 medir/sprt_diverso.py --smoke-test    # autochequeo, sin motores
```

### Por que NO se usa mas `sprt_real.py`

El arnes historico traia **20 aperturas fijas** y las recorria asi:

```python
opening = OPENINGS[(index // 2) % len(OPENINGS)]
candidate_white = index % 2 == 0
```

O sea, el ciclo completo son **40 partidas**. Los dos motores son
DETERMINISTAS a nodos fijos y el arnes manda `ucinewgame` (que en Mittens
recrea el `Searcher` entero: TT, history, killers, corrhist) antes de cada
partida. Conclusion: **a partir de la partida 41 se repiten exactamente las
mismas 40 partidas**, jugada por jugada.

El LLR sigue creciendo, pero solo porque cuenta los mismos 40 resultados una y
otra vez. Con suficientes "partidas" SIEMPRE cruza un limite. **Cualquier
veredicto de ese arnes con N > 40 es un artefacto, no evidencia.**
El `--smoke-test` de `sprt_diverso.py` reproduce el bug explicitamente para
que quede documentado en ejecucion, no solo en un comentario.

### Que hace distinto este arnes

1. **Banco grande y balanceado.** 1200 aperturas del libro Polyglot del repo,
   con semilla fija, filtradas para que ninguna arranque ya decidida
   (`|score| <= 90 cp` a 60.000 nodos). 1200 x 2 colores = **2400 partidas
   realmente independientes**. Si se piden mas, avisa y recorta en vez de
   repetir en silencio.

2. **Deteccion de duplicados EN VIVO.** Se hashea `apertura + secuencia
   completa de jugadas` de cada partida terminada. Cada linea imprime
   `distintas=N/M`. Si la fraccion cae por debajo del **60%** (con al menos 40
   partidas jugadas), **aborta**: es la senal de que se volvio a caer en el bug
   de arriba.

3. **Checkpoint / reanudacion** con firma sha256 de binarios, pesos y
   parametros. Reanudar con otro binario aborta en vez de mezclar evidencia.
   Las huellas de partida se guardan tambien, asi el conteo de duplicados
   sobrevive a la reanudacion.

4. **`apertura_inicial`** desplaza el tramo del banco. Sirve para (a) partir un
   test en varios trabajadores paralelos sobre tramos DISJUNTOS -- sumar los
   W/D/L despues es legitimo porque no comparten ni una apertura -- y (b)
   confirmar un resultado sobre aperturas que NO se usaron en el tanteo, para
   que el veredicto no dependa del subconjunto que ya salio favorable.

Ademas del LLR imprime el **Elo estimado con IC95%**, que es lo que hay que
mirar cuando el SPRT termina AMBIGUO (no cruzo ningun limite).

### Reglas de uso

- **Threads=1 en los dos motores** (lo fija el arnes) y **nodos fijos**: asi la
  medicion es inmune a la contencion de CPU y se pueden correr varios
  trabajadores en paralelo sin contaminar el resultado.
- Los resultados van a `results_sprt/NOMBRE/` (PGN, `state.json`,
  `veredicto.txt`).

### `juntar_sprt.py` — sumar trabajadores paralelos

```
python3 medir/juntar_sprt.py results_sprt/NOMBRE_A results_sprt/NOMBRE_B ...
```

Suma los W/D/L y recalcula LLR/Elo. **Aborta** si los trabajadores no midieron
exactamente lo mismo (firma distinta) o si comparten aunque sea una partida
(huella repetida): sumar evidencia repetida es el mismo bug del arnes viejo,
solo que repartido entre procesos.

---

## 2. `nodos_arbol.py` — ¿el arbol es mas chico?

```
python3 medir/nodos_arbol.py MOTOR PESOS medir/aperturas.txt PROFUNDIDAD [N_POSICIONES]
```

Nodos exactos para llegar a una profundidad fija, sobre MUCHAS posiciones.

**Por que no alcanza con `mittens bench N`:** el bench tiene 6 posiciones. Un
cambio en extensiones o podas mueve el arbol de forma caotica posicion por
posicion (una sube 25%, la de al lado baja 25%), asi que con 6 muestras el
total salta sin decir nada. Ejemplo real medido en esta sesion, mismo par de
binarios:

| medicion | veredicto |
|---|---|
| `bench 14` (6 posiciones) | candidato **-24%** nodos |
| `bench 12` (6 posiciones) | candidato **+24%** nodos |
| `nodos_arbol.py` d12, 150 posiciones | candidato **-0,6% ± 6%** (neutro) |

Las dos primeras filas son ruido. Solo la tercera es una medicion.

Manda `ucinewgame` antes de cada posicion, que en Mittens recrea el `Searcher`
completo: equivale a arrancar el motor de cero por posicion, que es la unica
forma de que el conteo sea comparable (ver la nota del proyecto "la TT
contamina mediciones"). Imprime la lista cruda `CRUDO ...` para poder comparar
dos binarios **posicion a posicion** (razon geometrica pareada con IC95%), que
tiene muchisima menos varianza que comparar los totales.

---

## 3. `generar_aperturas.py` — arma el banco

```
python3 medir/generar_aperturas.py medir/aperturas.txt 1200 MOTOR PESOS
```

Camina el libro Polyglot del repo (`performance.bin`) desde la posicion
inicial con semilla fija, eligiendo jugadas de libro **uniformemente** (no
ponderadas por popularidad: ponderar colapsaba el banco en un punado de lineas
de moda y lo que se busca es diversidad), 4 a 16 plies, deduplica por FEN y
descarta las posiciones que ya estan decididas. `medir/aperturas.txt` ya esta
versionado; solo hace falta regenerarlo si se quiere un banco mas grande.
