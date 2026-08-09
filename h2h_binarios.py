#!/usr/bin/env python3
"""H2H entre DOS BINARIOS distintos (mismas opciones UCI salvo el ejecutable).

Uso:
  python3 h2h_binarios.py BIN_A BIN_B [PARTIDAS=100] [MS=500] [NOMBRE_A] [NOMBRE_B]

Cada apertura se juega dos veces con colores invertidos. No despliega ni
modifica produccion.
"""

from __future__ import annotations

import math
import pathlib
import sys

import chess
import chess.engine

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


def configure(engine) -> None:
    requested = {"Threads": 1, "Hash": 128, "UseNNUE": True}
    if "NNUEPath" in engine.options:
        requested["NNUEPath"] = "/Users/Tavito/mi-motor-rust-produccion/pesos_amenazas_prueba.bin"
    engine.configure({k: v for k, v in requested.items() if k in engine.options})


def board_from_opening(line: str) -> chess.Board:
    b = chess.Board()
    for tok in line.split():
        if tok.endswith("."):
            continue
        b.push_san(tok)
    return b


def play(a, b, opening: str, a_is_white: bool, ms: int) -> float:
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
    bin_a = pathlib.Path(sys.argv[1]).resolve()
    bin_b = pathlib.Path(sys.argv[2]).resolve()
    partidas = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    nombre_a = sys.argv[5] if len(sys.argv) > 5 else "A"
    nombre_b = sys.argv[6] if len(sys.argv) > 6 else "B"

    for p in (bin_a, bin_b):
        if not p.exists():
            print(f"ERROR: no existe {p}")
            sys.exit(2)

    print(f"{nombre_a}: {bin_a}")
    print(f"{nombre_b}: {bin_b}")
    print(f"{partidas} partidas a {ms} ms/jugada, 1 hilo, hash 128MB\n", flush=True)

    a = chess.engine.SimpleEngine.popen_uci(str(bin_a))
    b = chess.engine.SimpleEngine.popen_uci(str(bin_b))
    try:
        configure(a)
        configure(b)
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
                f"[{n}/{partidas}] {nombre_a}: +{w} ={d} -{l}  ({puntos}/{n} = {pct:.1f}%)",
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
    print(f"{nombre_a} vs {nombre_b}: +{w} ={d} -{l} en {n} partidas")
    print(f"Puntaje {nombre_a}: {100*score:.1f}%  (error estandar ~{100*se:.1f}%)")
    print(f"Intervalo 95% aprox: {100*lo:.1f}% .. {100*hi:.1f}%")
    if 0.0 < score < 1.0:
        elo = -400 * math.log10(1 / score - 1)
        print(f"Elo estimado de {nombre_a} sobre {nombre_b}: {elo:+.0f}")
    if lo > 0.5:
        print(f"VEREDICTO: {nombre_a} es significativamente MEJOR.")
    elif hi < 0.5:
        print(f"VEREDICTO: {nombre_a} es significativamente PEOR.")
    else:
        print("VEREDICTO: sin diferencia estadisticamente significativa.")


if __name__ == "__main__":
    main()
