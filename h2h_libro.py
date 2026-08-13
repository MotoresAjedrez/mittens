#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h2h rapido: con libro (OwnBook+BookPath) vs sin libro, mismo binario/pesos."""
import chess
import chess.engine

BIN = "/Users/Tavito/mi-motor-rust-produccion/target/release/mittens"
PESOS = "/Users/Tavito/mi-motor-rust-produccion/pesos_amenazas_prueba.bin"
BOOK = "/Users/Tavito/mi-motor-rust-produccion/performance.bin"
MOVETIME_MS = 600
MAX_PLIES = 200


def abrir(con_libro):
    e = chess.engine.SimpleEngine.popen_uci([BIN])
    opts = {"UseNNUE": True, "NNUEPath": PESOS, "Threads": 1, "Hash": 64}
    if con_libro:
        opts["BookPath"] = BOOK
        opts["OwnBook"] = True
    else:
        opts["OwnBook"] = False
    e.configure(opts)
    return e


def jugar(con_libro_blancas, libro_eng, sin_libro_eng):
    board = chess.Board()
    limite = chess.engine.Limit(time=MOVETIME_MS / 1000.0)
    jugadas = []
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        motor = libro_eng if ((board.turn == chess.WHITE) == con_libro_blancas) else sin_libro_eng
        r = motor.play(board, limite)
        if r.move is None:
            break
        jugadas.append(board.san(r.move))
        board.push(r.move)
    res = board.outcome(claim_draw=True)
    return jugadas, (res.result() if res else "?")


def main():
    libro_eng = abrir(True)
    sin_libro_eng = abrir(False)
    try:
        for i, (con_libro_blancas) in enumerate([True, False]):
            jugadas, resultado = jugar(con_libro_blancas, libro_eng, sin_libro_eng)
            lado_libro = "blancas" if con_libro_blancas else "negras"
            print(f"\n=== Partida {i+1}: libro juega {lado_libro} ===")
            print("Jugadas:", " ".join(jugadas[:20]), "..." if len(jugadas) > 20 else "")
            print("Resultado:", resultado)
    finally:
        libro_eng.quit()
        sin_libro_eng.quit()


if __name__ == "__main__":
    main()
