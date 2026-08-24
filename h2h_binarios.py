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

ROOT = pathlib.Path(__file__).resolve().parent

# BANCO DE APERTURAS: el mismo `libro_sprt.epd` que usan sprt_real.py y
# h2h.py (2313 aperturas unicas, ver tools/generar_libro_sprt.py).
#
# ANTES habia 20 aperturas cableadas y el indice se tomaba modulo 20: pasadas
# las 40 partidas se volvia al principio del libro. Como este script juega por
# RELOJ, las partidas repetidas no salen identicas (a nodos fijos SI salian
# identicas, que es lo que rompia sprt_real.py), pero 20 aperturas para 100 o
# 250 partidas dan una muestra bastante mas correlacionada de lo que supone el
# error estandar que despues se usa para decidir.

LIBRO_APERTURAS = ROOT / "libro_sprt.epd"


def cargar_libro() -> list[str]:
    if not LIBRO_APERTURAS.exists():
        raise SystemExit(
            f"Falta el libro de aperturas {LIBRO_APERTURAS}.\n"
            "Generalo con: python3 tools/generar_libro_sprt.py"
        )
    fens = [
        linea.strip()
        for linea in LIBRO_APERTURAS.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.startswith("#")
    ]
    if len(set(fens)) != len(fens):
        raise SystemExit("El libro tiene lineas repetidas: regeneralo.")
    return fens



def configure(engine) -> None:
    requested = {"Threads": 1, "Hash": 128, "UseNNUE": True}
    if "NNUEPath" in engine.options:
        requested["NNUEPath"] = "/Users/Tavito/mi-motor-rust-produccion/pesos_amenazas_prueba.bin"
    engine.configure({k: v for k, v in requested.items() if k in engine.options})


def board_from_opening(fen):
    """Tablero de arranque de una apertura del libro (una FEN por linea)."""
    return chess.Board(fen)


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

    aperturas = cargar_libro()
    if partidas > 2 * len(aperturas):
        raise SystemExit(
            f"ABORTADO: pediste {partidas} partidas pero el libro solo permite "
            f"{2 * len(aperturas)} sin repetir apertura+color.\n"
            "Solucion: python3 tools/generar_libro_sprt.py <mas_aperturas>"
        )

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
            opening = aperturas[i // 2]
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
