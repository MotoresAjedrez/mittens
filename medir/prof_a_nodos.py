#!/usr/bin/env python3
"""Profundidad alcanzada con un PRESUPUESTO FIJO DE NODOS, sobre muchas posiciones.

Es la contracara de medir/nodos_arbol.py: alli se fija la profundidad y se
cuentan nodos; aca se fijan los nodos y se mira hasta donde llega. Un arbol mas
chico solo sirve si se convierte en MAS PROFUNDIDAD con el mismo presupuesto --
esta es la medicion que lo comprueba directamente.

Uso: prof_a_nodos.py MOTOR PESOS FENS NODOS [N_POSICIONES]
"""
from __future__ import annotations

import pathlib
import sys

import chess
import chess.engine


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    motor_path = pathlib.Path(sys.argv[1]).resolve()
    pesos = pathlib.Path(sys.argv[2]).resolve()
    fens_path = pathlib.Path(sys.argv[3]).resolve()
    nodos = int(sys.argv[4])
    limite = int(sys.argv[5]) if len(sys.argv) > 5 else 200

    fens = [l.strip() for l in fens_path.read_text().splitlines() if l.strip()][:limite]
    motor = chess.engine.SimpleEngine.popen_uci([str(motor_path)])
    try:
        pedido = {"Threads": 1, "Hash": 128, "NNUEPath": str(pesos), "UseNNUE": True}
        motor.configure({k: v for k, v in pedido.items() if k in motor.options})
        profs: list[int] = []
        for i, fen in enumerate(fens):
            info = motor.analyse(
                chess.Board(fen), chess.engine.Limit(nodes=nodos), game=("prof", i)
            )
            d = int(info.get("depth", 0))
            if d > 0:
                profs.append(d)
        print(f"motor      : {motor_path}")
        print(f"nodos      : {nodos}")
        print(f"posiciones : {len(profs)}")
        print(f"prof. media: {sum(profs)/len(profs):.3f}")
        print("CRUDO " + " ".join(str(d) for d in profs))
    finally:
        motor.quit()


if __name__ == "__main__":
    main()
