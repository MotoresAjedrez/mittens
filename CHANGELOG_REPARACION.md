# Cambios de la reparacion v8

## Correctitud

- Quiescence busca todas las evasiones cuando el rey esta en jaque.
- Quiescence detecta mate y ahogado en el horizonte.
- Las promociones y los jaques no se podan por SEE o delta pruning.
- Scores de mate normalizados al guardar y leer la TT.
- SEE impide recapturas ilegales del rey y suma la ganancia de promociones.
- Zobrist solo incluye en-passant cuando el bando al turno puede capturarlo.
- `perft_divide(..., 0)` ya no desborda.
- FEN valida un rey por color, reyes no adyacentes, peones fuera de filas extremas y coherencia basica de en-passant.
- La segunda carga de libro/Syzygy ya no informa exito falso cuando el almacenamiento de una sola carga ya estaba ocupado.

## Tabla de transposicion

- El calculo de `Hash` cuenta el tamano real del `Mutex` por casillero.
- Se corrigio el redondeo a potencia de dos.
- Una colision de otra clave puede reemplazar el casillero.
- A igual profundidad se prefiere una entrada exacta.
- La TT se reinicia al cambiar personalidad, NNUE o tamano de Hash.
- Nueva opcion UCI `Clear Hash`.

## Reloj

- Nueva opcion UCI `Move Overhead`.
- El motor no inventa un minimo de 50 ms cuando queda menos tiempo.
- El presupuesto nunca supera el reloj util disponible.
- Comprobacion de tiempo desde el primer nodo y luego cada 256 nodos.
- Siempre conserva una jugada legal de emergencia para evitar `bestmove 0000` por falta de tiempo.

## NNUE

- Una carga fallida conserva la red anterior.
- Validacion de tamano, NaN, infinito y magnitudes absurdas.
- Checksum FNV-1a mostrado al cargar.
- Pruebas incrementales para movimiento normal, captura, en-passant, enroque y promocion.

## Herramientas y rendimiento

- `mover_fen_rust.sh` ya no fuerza `MIMOTOR_LMR=0`.
- Perfil release con optimizacion 3, ThinLTO, una unidad de codegen y stripping.
- Nuevas pruebas de regresion para mate-TT, colisiones TT, quiescence, reloj, Zobrist, SEE, NNUE y perft.
