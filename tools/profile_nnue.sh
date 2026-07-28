#!/bin/zsh
# Perfil reproducible de una búsqueda UCI con NNUE activa.
# Uso: tools/profile_nnue.sh RUTA_BINARIO RUTA_PESOS PREFIJO_SALIDA [MOVETIME_MS=24000]
set -euo pipefail

BIN=$1
PESOS=$2
PREFIJO=$3
MOVETIME_MS=${4:-24000}
FEN='r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6'
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkfifo "$TMP/in"

"$BIN" <"$TMP/in" >"${PREFIJO}.uci.txt" 2>&1 &
PID=$!
(
  print -r -- 'uci'
  print -r -- 'setoption name Threads value 1'
  print -r -- 'setoption name Hash value 128'
  print -r -- 'setoption name Personalidad value tal'
  print -r -- "setoption name NNUEPath value $PESOS"
  print -r -- 'setoption name UseNNUE value true'
  print -r -- 'isready'
  print -r -- 'ucinewgame'
  print -r -- "position fen $FEN"
  print -r -- "go movetime $MOVETIME_MS"
  # La busqueda puede cortar cerca del 70% del presupuesto. Dejamos al
  # menos siete segundos vivos para que sample capture una muestra util.
  sleep $(( MOVETIME_MS / 1000 - 2 ))
  print -r -- 'quit'
) >"$TMP/in" &

sleep 0.3
/usr/bin/sample "$PID" 7 1 -mayDie -fullPaths -file "${PREFIJO}.sample.txt" || true
wait "$PID" || true

awk '/^info depth / { nodes=$8; ms=$10 } END { if (ms > 0) printf "nodes=%s time_ms=%s nps=%.0f\\n", nodes, ms, nodes*1000/ms; else print "No hubo linea info util" }' "${PREFIJO}.uci.txt"
