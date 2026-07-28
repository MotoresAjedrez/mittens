#!/bin/bash
# Ayudante: recibe un FEN y un movetime (ms), devuelve la jugada del motor.
# Respeta MIMOTOR_LMR y MIMOTOR_HILOS del entorno. LMR queda activado por
# defecto; antes este script lo apagaba siempre y anulaba una mejora medida.
set -euo pipefail

FEN="${1:?uso: mover_fen_rust.sh 'FEN' [movetime_ms]}"
MOVETIME="${2:-5000}"
BIN="${MIMOTOR_BIN:-$HOME/mi-motor-rust/target/release/mi-motor-rust}"
HILOS="${MIMOTOR_HILOS:-4}"

if [[ ! -x "$BIN" ]]; then
  echo "Motor no encontrado o no ejecutable: $BIN" >&2
  exit 1
fi

printf "uci\nisready\nposition fen %s\ngo movetime %s\nquit\n" "$FEN" "$MOVETIME" \
  | MIMOTOR_HILOS="$HILOS" "$BIN" 2>&1 \
  | awk '/^bestmove / { print $2; exit }'
