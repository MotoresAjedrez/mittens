# medir/ — con qué medir cada cosa en Mittens

Regla de oro de este motor: **el `bench` mide un régimen que no se juega.**
`mittens bench` crea un `Searcher::new` por posición y llama a
`search_fixed_depth`, así que arranca con las tablas de historia en CERO y
nunca pasa por `decaer_history()`. Una partida real usa el MISMO Searcher
jugada tras jugada y entra por `search_time`, que envejece las tablas en cada
"go" y las deja calientes de la jugada anterior. Cualquier cambio de tablas de
historia, de orden de jugadas o de la raíz se comporta distinto en cada
régimen. Ya hubo un paquete que el bench aprobó (−17 % de nodos) y el juego
real rechazó (−62 Elo).

| Herramienta | Qué mide | Cuándo usarla |
|---|---|---|
| `depth_en_partida.py` | profundidad alcanzada a **nodos fijos**, en régimen de partida | filtro rápido y grueso; es un entero, hace falta mucha muestra |
| `orden_en_partida.py` | % de cortes beta en la **primera** jugada, en régimen de partida | calidad del ordenamiento; determinista, sin error estadístico |
| `nodos_en_partida.py` | **nodos** a profundidad fija, en régimen de partida | tamaño del árbol, que es la brecha conocida contra Reckless |
| `sprt_diverso.py` | **Elo**, con SPRT sobre partidas reales | la única que decide; todo lo demás es proxy |

Los tres primeros son PROXIES: mejoran o empeoran de forma medible sin que eso
garantice Elo. Sirven para decidir qué merece gastar un SPRT, no para aprobar
un cambio.

## Por qué `sprt_diverso.py` y no `sprt_real.py`

`sprt_real.py` (que no está en el repo, vive suelto en la raíz del proyecto)
juega con un banco fijo de 20 aperturas y manda `ucinewgame` antes de cada
partida. Mittens es determinista a nodos fijos y `ucinewgame` le tira la TT y
el Searcher, así que **la partida 40 es exactamente la partida 0**. Medido
sobre los PGN que dejó:

- `corrplexity_lmr`: 2.843 partidas, **40 distintas** (71× cada una)
- `ttpv_persistente`: 1.562 partidas, **40 distintas** (39×)
- `corrhist_menores_2ply`: 362 partidas, **40 distintas** (9×)

El LLR trata cada repetición como información nueva, así que está inflado por
el factor de repetición. `sprt_diverso.py` usa un banco de 400 aperturas
equilibradas (generadas con semilla fija y filtradas por el motor baseline) y
**cuenta las partidas distintas en cada línea de progreso**, abortando si la
fracción cae por debajo del 60 %.

Ojo con la prueba nula (mismo binario en los dos lados): ahí las dos partidas
de una misma apertura son idénticas con los colores cambiados, así que el
detector marca "20 de 40" y el score es 50,00 % exacto. Es correcto: una
prueba nula no tiene más información que eso.
