#!/usr/bin/env python3
"""Genera el banco de aperturas DIVERSAS y BALANCEADAS para `sprt_diverso.py`.

Por que existe: el arnes historico (`sprt_real.py`) traia 20 aperturas fijas.
Con los dos motores deterministas a nodos fijos y `ucinewgame` entre partidas,
20 aperturas x 2 colores = 40 partidas unicas y TODO lo que venga despues es
una repeticion exacta. El LLR sigue creciendo contando los mismos 40
resultados una y otra vez, asi que cualquier veredicto con N > 40 era un
artefacto.

Que hace este generador:
  1. Camina el libro Polyglot del propio repo (`performance.bin`) desde la
     posicion inicial, eligiendo jugadas de libro al azar con SEMILLA FIJA,
     hasta una profundidad de 4 a 16 plies (tambien al azar).
  2. Deduplica por FEN.
  3. (Opcional, recomendado) filtra por BALANCE: se le pide al motor una
     busqueda corta en cada posicion y se descartan las que ya estan
     decididas (|score| por encima de un umbral). Asi ninguna apertura del
     banco regala la partida a un color.

Uso:
  python3 medir/generar_aperturas.py SALIDA.txt [N=1200] [MOTOR] [PESOS]

Sin MOTOR se salta el filtro de balance (banco solo diverso, no verificado).
"""

from __future__ import annotations

import pathlib
import random
import sys

import chess
import chess.engine
import chess.polyglot

RAIZ = pathlib.Path(__file__).resolve().parent.parent
LIBRO = RAIZ / "performance.bin"
SEMILLA = 20260825
PLIES_MIN = 4
PLIES_MAX = 16
UMBRAL_BALANCE = 90  # centipeones; por encima de esto la apertura ya esta torcida
NODOS_BALANCE = 60_000


def generar_candidatas(objetivo: int) -> list[str]:
    rng = random.Random(SEMILLA)
    vistas: set[str] = set()
    salida: list[str] = []
    with chess.polyglot.open_reader(LIBRO) as libro:
        intentos = 0
        # Se piden bastantes mas de las necesarias: el filtro de balance
        # descarta una parte.
        while len(salida) < objetivo * 3 and intentos < objetivo * 400:
            intentos += 1
            tablero = chess.Board()
            plies = rng.randint(PLIES_MIN, PLIES_MAX)
            ok = True
            for _ in range(plies):
                entradas = list(libro.find_all(tablero))
                if not entradas:
                    ok = False
                    break
                # Muestreo UNIFORME entre las jugadas de libro (no ponderado
                # por popularidad): ponderar por peso colapsaba el banco en un
                # punado de lineas de moda, y lo que se busca aca es DIVERSIDAD
                # -- son 2000+ posiciones distintas lo que hace que cada partida
                # del SPRT sea una observacion independiente.
                elegida = rng.choice(entradas)
                tablero.push(elegida.move)
            if not ok or tablero.is_game_over():
                continue
            fen = tablero.fen()
            if fen in vistas:
                continue
            vistas.add(fen)
            salida.append(fen)
    return salida


def filtrar_balanceadas(
    fens: list[str], motor_path: pathlib.Path, pesos: pathlib.Path | None, objetivo: int
) -> list[str]:
    motor = chess.engine.SimpleEngine.popen_uci([str(motor_path)])
    try:
        opciones = {"Threads": 1, "Hash": 64}
        if pesos is not None:
            opciones["NNUEPath"] = str(pesos)
            opciones["UseNNUE"] = True
        motor.configure({k: v for k, v in opciones.items() if k in motor.options})
        buenas: list[str] = []
        for i, fen in enumerate(fens):
            if len(buenas) >= objetivo:
                break
            tablero = chess.Board(fen)
            info = motor.analyse(tablero, chess.engine.Limit(nodes=NODOS_BALANCE))
            puntaje = info["score"].white()
            if puntaje.is_mate():
                continue
            if abs(puntaje.score()) <= UMBRAL_BALANCE:
                buenas.append(fen)
            if (i + 1) % 50 == 0:
                print(f"  filtradas {len(buenas)} de {i + 1} revisadas", flush=True)
        return buenas
    finally:
        motor.quit()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    salida = pathlib.Path(sys.argv[1])
    objetivo = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    motor = pathlib.Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
    pesos = pathlib.Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None

    print(f"Generando candidatas desde {LIBRO} (semilla {SEMILLA})...")
    candidatas = generar_candidatas(objetivo)
    print(f"  {len(candidatas)} FEN distintos de libro")

    if motor is not None:
        print(f"Filtrando por balance con {motor} (|score| <= {UMBRAL_BALANCE} cp)...")
        candidatas = filtrar_balanceadas(candidatas, motor, pesos, objetivo)
        print(f"  {len(candidatas)} balanceadas")

    candidatas = candidatas[:objetivo]
    if len(candidatas) < objetivo:
        raise SystemExit(
            f"Solo salieron {len(candidatas)} aperturas de {objetivo} pedidas; "
            "bajar el objetivo o aflojar el umbral de balance."
        )
    salida.write_text("\n".join(candidatas) + "\n", encoding="utf-8")
    print(f"Escritas {len(candidatas)} aperturas en {salida}")


if __name__ == "__main__":
    main()
