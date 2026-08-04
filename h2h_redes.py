#!/usr/bin/env python3
"""H2H entre DOS ARCHIVOS DE PESOS NNUE usando el MISMO binario.

Uso:
  python3 h2h_redes.py PESOS_A PESOS_B [PARTIDAS=100] [MS=500]

A y B reciben identicas opciones UCI (Threads, Hash, UseNNUE) y solo se
diferencian en NNUEPath, de modo que la comparacion aisla la ARQUITECTURA /
los pesos, no la version del codigo. Cada apertura se juega dos veces con
colores invertidos. No despliega ni modifica produccion.

Salida: puntaje de A, error estandar aproximado sqrt(0.25/n) e intervalo 95%.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import chess
import chess.engine

ROOT = pathlib.Path(__file__).resolve().parent
BIN = pathlib.Path(os.environ.get("MIMOTOR_BIN", str(ROOT / "target/release/mi-motor-rust")))
MAX_PLIES = 300

OPENINGS = [
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
    "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6",
    "1. e4 c5 2. Nf3 d6 3. d4 cxd4",
    "1. e4 c5 2. Nf3 Nc6 3. d4 cxd4",
    "1. e4 c5 2. c3 d5 3. exd5 Qxd5",
    "1. e4 e6 2. d4 d5 3. Nc3 Nf6",
    "1. e4 c6 2. d4 d5 3. Nc3 dxe4",
    "1. e4 d5 2. exd5 Qxd5 3. Nc3 Qd8",
    "1. d4 d5 2. c4 e6 3. Nc3 Nf6",
    "1. d4 d5 2. c4 c6 3. Nf3 Nf6",
    "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7",
    "1. d4 Nf6 2. c4 e6 3. Nf3 b6",
    "1. d4 Nf6 2. c4 c5 3. d5 e6",
    "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6",
    "1. c4 c5 2. Nf3 Nf6 3. d4 cxd4",
    "1. Nf3 d5 2. g3 Nf6 3. Bg2 g6",
    "1. Nf3 Nf6 2. c4 g6 3. g3 Bg7",
    "1. e4 e5 2. Nf3 Nf6 3. Nxe5 d6",
    "1. d4 e6 2. c4 f5 3. g3 Nf6",
    "1. e4 g6 2. d4 Bg7 3. Nc3 d6",
]


def configure(engine, weights: pathlib.Path) -> None:
    requested = {
        "Threads": 1,
        "Hash": 128,
        "UseNNUE": True,
        "NNUEPath": str(weights),
    }
    engine.configure({k: v for k, v in requested.items() if k in engine.options})


def board_from_opening(line: str) -> chess.Board:
    b = chess.Board()
    for tok in line.split():
        if tok.endswith("."):
            continue
        b.push_san(tok)
    return b


def play(a, b, opening: str, a_is_white: bool, ms: int) -> float:
    """Devuelve el puntaje de A: 1.0 gana, 0.5 tablas, 0.0 pierde."""
    board = board_from_opening(opening)
    limit = chess.engine.Limit(time=ms / 1000.0)
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        white_to_move = board.turn == chess.WHITE
        engine = a if (white_to_move == a_is_white) else b
        result = engine.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if (outcome.winner == chess.WHITE) == a_is_white else 0.0


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    pesos_a = pathlib.Path(sys.argv[1]).resolve()
    pesos_b = pathlib.Path(sys.argv[2]).resolve()
    partidas = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    for p in (BIN, pesos_a, pesos_b):
        if not p.exists():
            print(f"ERROR: no existe {p}")
            sys.exit(2)

    print(f"Binario : {BIN}")
    print(f"A (nuevo): {pesos_a}  ({pesos_a.stat().st_size} bytes)")
    print(f"B (viejo): {pesos_b}  ({pesos_b.stat().st_size} bytes)")
    print(f"{partidas} partidas a {ms} ms/jugada, 1 hilo, hash 128MB\n", flush=True)

    a = chess.engine.SimpleEngine.popen_uci(str(BIN))
    b = chess.engine.SimpleEngine.popen_uci(str(BIN))
    try:
        configure(a, pesos_a)
        configure(b, pesos_b)
        puntos = 0.0
        w = d = l = 0
        for i in range(partidas):
            opening = OPENINGS[(i // 2) % len(OPENINGS)]
            s = play(a, b, opening, a_is_white=(i % 2 == 0), ms=ms)
            puntos += s
            if s == 1.0:
                w += 1
            elif s == 0.5:
                d += 1
            else:
                l += 1
            n = i + 1
            pct = 100.0 * puntos / n
            print(
                f"[{n}/{partidas}] A: +{w} ={d} -{l}  ({puntos}/{n} = {pct:.1f}%)",
                flush=True,
            )
    finally:
        a.quit()
        b.quit()

    n = partidas
    score = puntos / n
    se = math.sqrt(0.25 / n)
    lo, hi = score - 1.96 * se, score + 1.96 * se
    print("\n==================== RESULTADO ====================")
    print(f"A (nuevo) vs B (viejo): +{w} ={d} -{l} en {n} partidas")
    print(f"Puntaje A: {100*score:.1f}%  (error estandar ~{100*se:.1f}%)")
    print(f"Intervalo 95% aprox: {100*lo:.1f}% .. {100*hi:.1f}%")
    if 0.0 < score < 1.0:
        elo = -400 * math.log10(1 / score - 1)
        print(f"Elo estimado de A sobre B: {elo:+.0f}")
    if lo > 0.5:
        print("VEREDICTO: A es significativamente MEJOR.")
    elif hi < 0.5:
        print("VEREDICTO: A es significativamente PEOR.")
    else:
        print("VEREDICTO: sin diferencia estadisticamente significativa.")


if __name__ == "__main__":
    main()
