#!/usr/bin/env python3
"""h2h con RELOJ REAL (no nodos fijos) entre dos binarios UCI, con Threads>1.

Existe porque `sprt_diverso.py` juega a NODOS FIJOS y con Threads=1: es ciego
por construccion a cualquier cambio de gestion de tiempo o de multihilo. Este
arnes reparte un reloj de verdad (wtime/btime/inc), cuenta banderas y reporta
el reloj promedio consumido por jugada de cada motor.

Uso:
  python3 medir/h2h_reloj.py CAND_BIN BASE_BIN NOMBRE [partidas] [hilos]
          [base_s] [inc_s]
"""
import sys, time, random, pathlib, statistics, json
import chess, chess.engine

CAND = sys.argv[1]
BASE = sys.argv[2]
NOMBRE = sys.argv[3] if len(sys.argv) > 3 else "h2h"
PARTIDAS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
HILOS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
BASE_S = float(sys.argv[6]) if len(sys.argv) > 6 else 30.0
INC_S = float(sys.argv[7]) if len(sys.argv) > 7 else 0.3
# Desplazamiento del banco de aperturas: permite partir el mismo test en
# varios trabajadores sobre tramos DISJUNTOS y sumar los W/D/L despues.
OFFSET = int(sys.argv[8]) if len(sys.argv) > 8 else 0

APER = [l.strip() for l in open(pathlib.Path(__file__).with_name("aperturas.txt")) if l.strip()]
random.Random(20260826).shuffle(APER)

def abrir(bin_):
    e = chess.engine.SimpleEngine.popen_uci(bin_)
    e.configure({"Threads": HILOS, "Hash": 256})
    return e

def jugar(cand, base, fen, cand_blancas):
    tab = chess.Board(fen)
    reloj = {chess.WHITE: BASE_S, chess.BLACK: BASE_S}
    tiempos = {"cand": [], "base": []}
    while not tab.is_game_over(claim_draw=True):
        if len(tab.move_stack) > 200:
            return "1/2-1/2", tiempos, None
        es_cand = (tab.turn == chess.WHITE) == cand_blancas
        motor = cand if es_cand else base
        lim = chess.engine.Limit(white_clock=reloj[chess.WHITE], black_clock=reloj[chess.BLACK],
                                 white_inc=INC_S, black_inc=INC_S)
        t0 = time.time()
        try:
            res = motor.play(tab, lim, info=chess.engine.INFO_NONE)
        except Exception as ex:
            return ("0-1" if cand_blancas == (tab.turn == chess.WHITE) else "1-0"), tiempos, f"error {ex}"
        el = time.time() - t0
        tiempos["cand" if es_cand else "base"].append(el)
        reloj[tab.turn] -= el
        if reloj[tab.turn] < 0:
            # bandera: pierde el que se paso
            perdedor_cand = es_cand
            return ("0-1" if tab.turn == chess.WHITE else "1-0"), tiempos, ("bandera_cand" if perdedor_cand else "bandera_base")
        reloj[tab.turn] += INC_S
        tab.push(res.move)
    return tab.result(claim_draw=True), tiempos, None

def main():
    cand = abrir(CAND); base = abrir(BASE)
    w = d = l = 0
    tc = []; tb = []
    banderas = {"cand": 0, "base": 0}
    salida = pathlib.Path(__file__).parent / f"{NOMBRE}_resultado.txt"
    try:
        for i in range(PARTIDAS):
            fen = APER[(OFFSET + i) % len(APER)]
            cand_blancas = (i % 2 == 0)
            r, t, nota = jugar(cand, base, fen, cand_blancas)
            if nota and nota.startswith("bandera"):
                banderas["cand" if nota.endswith("cand") else "base"] += 1
            tc += t["cand"]; tb += t["base"]
            if r == "1/2-1/2": d += 1
            elif (r == "1-0") == cand_blancas: w += 1
            else: l += 1
            if True:
                n = w + d + l
                sc = (w + 0.5 * d) / n
                print(f"[{n:3d}] cand {w}-{d}-{l}  score={sc*100:.1f}%  "
                      f"t/jugada cand={statistics.mean(tc):.2f}s base={statistics.mean(tb):.2f}s  "
                      f"banderas={banderas}", flush=True)
    finally:
        cand.quit(); base.quit()
    n = max(w + d + l, 1)
    sc = (w + 0.5 * d) / n
    txt = (f"{NOMBRE}: {w}W {d}D {l}L de {n}  score={sc*100:.2f}%\n"
           f"TC={BASE_S}s+{INC_S}s  Threads={HILOS}\n"
           f"t/jugada: cand={statistics.mean(tc):.3f}s  base={statistics.mean(tb):.3f}s\n"
           f"banderas: {json.dumps(banderas)}\n")
    salida.write_text(txt)
    print(txt)

main()
