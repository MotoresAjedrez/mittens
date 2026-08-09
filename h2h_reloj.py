#!/usr/bin/env python3
"""H2H con RELOJ REAL (wtime/btime/winc/binc) entre dos binarios.

A diferencia de h2h_binarios.py (que manda `go movetime`), este manda un
reloj de verdad, que es lo unico que ejercita la gestion de tiempo.

Uso:
  python3 h2h_reloj.py BIN_A BIN_B [PARTIDAS] [BASE_MS] [INC_MS] [NOM_A] [NOM_B] [CONCURRENCIA]
"""
from __future__ import annotations
import math, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

import chess, chess.engine

MAX_PLIES = 300
OPENINGS = [
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6", "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6",
    "1. e4 c5 2. Nf3 d6 3. d4 cxd4", "1. e4 c5 2. Nf3 Nc6 3. d4 cxd4",
    "1. e4 c5 2. c3 d5 3. exd5 Qxd5", "1. e4 e6 2. d4 d5 3. Nc3 Nf6",
    "1. e4 c6 2. d4 d5 3. Nc3 dxe4", "1. e4 d5 2. exd5 Qxd5 3. Nc3 Qd8",
    "1. d4 d5 2. c4 e6 3. Nc3 Nf6", "1. d4 d5 2. c4 c6 3. Nf3 Nf6",
    "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7", "1. d4 Nf6 2. c4 e6 3. Nf3 b6",
    "1. d4 Nf6 2. c4 c5 3. d5 e6", "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6",
    "1. c4 c5 2. Nf3 Nf6 3. d4 cxd4", "1. Nf3 d5 2. g3 Nf6 3. Bg2 g6",
    "1. Nf3 Nf6 2. c4 g6 3. g3 Bg7", "1. e4 e5 2. Nf3 Nf6 3. Nxe5 d6",
    "1. d4 e6 2. c4 f5 3. g3 Nf6", "1. e4 g6 2. d4 Bg7 3. Nc3 d6",
]

def configure(engine):
    req = {"Threads": 1, "Hash": 128, "UseNNUE": True}
    if "NNUEPath" in engine.options:
        req["NNUEPath"] = "/Users/Tavito/mi-motor-rust-produccion/pesos_amenazas_prueba.bin"
    engine.configure({k: v for k, v in req.items() if k in engine.options})

def board_from_opening(line):
    b = chess.Board()
    for tok in line.split():
        if not tok.endswith("."):
            b.push_san(tok)
    return b

def jugar_una(args):
    """Una partida completa con reloj. Devuelve (puntos_A, perdio_por_tiempo_A)."""
    bin_a, bin_b, opening, a_is_white, base_ms, inc_ms = args
    a = chess.engine.SimpleEngine.popen_uci(bin_a)
    b = chess.engine.SimpleEngine.popen_uci(bin_b)
    try:
        configure(a); configure(b)
        board = board_from_opening(opening)
        # Relojes por COLOR, en segundos.
        reloj = {chess.WHITE: base_ms / 1000.0, chess.BLACK: base_ms / 1000.0}
        inc = inc_ms / 1000.0
        flag = None  # color que perdio por tiempo
        while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
            turno = board.turn
            eng = a if ((turno == chess.WHITE) == a_is_white) else b
            if base_ms == 0:
                # Modo movetime fijo (base_ms=0): NO ejercita la gestion de
                # tiempo, sirve para aislar cambios del arbol de busqueda.
                lim = chess.engine.Limit(time=inc_ms / 1000.0)
            else:
                lim = chess.engine.Limit(
                    white_clock=reloj[chess.WHITE], black_clock=reloj[chess.BLACK],
                    white_inc=inc, black_inc=inc,
                )
            import time as _t
            t0 = _t.time()
            res = eng.play(board, lim)
            usado = _t.time() - t0
            reloj[turno] -= usado
            if base_ms != 0 and reloj[turno] < 0:
                flag = turno
                break
            reloj[turno] += inc
            if res.move is None:
                break
            board.push(res.move)
        if flag is not None:
            gano_a = (flag == chess.WHITE) != a_is_white
            return (1.0 if gano_a else 0.0, not gano_a)
        out = board.outcome(claim_draw=True)
        if out is None or out.winner is None:
            return (0.5, False)
        return (1.0 if (out.winner == chess.WHITE) == a_is_white else 0.0, False)
    finally:
        try: a.quit()
        except Exception: pass
        try: b.quit()
        except Exception: pass

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    bin_a = str(pathlib.Path(sys.argv[1]).resolve())
    bin_b = str(pathlib.Path(sys.argv[2]).resolve())
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    base = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    inc = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    na = sys.argv[6] if len(sys.argv) > 6 else "A"
    nb = sys.argv[7] if len(sys.argv) > 7 else "B"
    conc = int(sys.argv[8]) if len(sys.argv) > 8 else 4

    print(f"{na}: {bin_a}\n{nb}: {bin_b}")
    print(f"{n} partidas, reloj {base}ms + {inc}ms, 1 hilo, concurrencia {conc}\n", flush=True)

    tareas = [(bin_a, bin_b, OPENINGS[(i // 2) % len(OPENINGS)], i % 2 == 0, base, inc)
              for i in range(n)]
    puntos = 0.0; w = d = l = 0; flags = 0; hechas = 0
    with ProcessPoolExecutor(max_workers=conc) as ex:
        for s, perdio_tiempo in ex.map(jugar_una, tareas):
            puntos += s; hechas += 1
            if perdio_tiempo: flags += 1
            if s == 1.0: w += 1
            elif s == 0.5: d += 1
            else: l += 1
            print(f"[{hechas}/{n}] {na}: +{w} ={d} -{l} ({puntos}/{hechas} = {100*puntos/hechas:.1f}%) flags_A={flags}", flush=True)

    score = puntos / n
    se = math.sqrt(0.25 / n)
    print("\n==================== RESULTADO ====================")
    print(f"{na} vs {nb}: +{w} ={d} -{l} en {n} partidas")
    print(f"Puntaje {na}: {100*score:.1f}%  (error estandar ~{100*se:.1f}%)")
    print(f"Partidas que {na} PERDIO POR TIEMPO: {flags}")
    lo, hi = score - 1.96*se, score + 1.96*se
    print(f"Intervalo 95%: {100*lo:.1f}% .. {100*hi:.1f}%")
    if 0.0 < score < 1.0:
        print(f"Elo estimado: {-400*math.log10(1/score - 1):+.0f}")
    if lo > 0.5: print(f"VEREDICTO: {na} es significativamente MEJOR.")
    elif hi < 0.5: print(f"VEREDICTO: {na} es significativamente PEOR.")
    else: print("VEREDICTO: sin diferencia significativa.")

if __name__ == "__main__":
    main()
