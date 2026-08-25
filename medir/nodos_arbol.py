#!/usr/bin/env python3
"""Mide el TAMAÑO DEL ARBOL: nodos exactos para llegar a una profundidad fija,
promediado sobre MUCHAS posiciones.

Por que no alcanza con `mittens bench N`: el bench tiene 6 posiciones. Un
cambio en extensiones/podas mueve el arbol de forma caotica posicion por
posicion (una sola posicion puede subir 25% y la de al lado bajar 25%), asi que
con 6 muestras el numero total salta sin decir nada. Con 150-300 posiciones el
promedio si es estable y comparable entre binarios.

Como se mide sin contaminacion: se manda `ucinewgame` antes de cada posicion.
En Mittens eso RECREA el Searcher entero (TT, history, killers, corrhist), o
sea que equivale a arrancar el motor de cero para cada posicion -- que es la
unica forma de que el conteo de nodos sea comparable (ver la nota del proyecto
"la TT contamina mediciones").

Uso:
  python3 medir/nodos_arbol.py MOTOR PESOS ARCHIVO_FENS PROFUNDIDAD [N_POSICIONES]

Imprime el total de nodos, la media geometrica por posicion y la lista cruda,
para poder comparar dos binarios posicion a posicion.
"""

from __future__ import annotations

import math
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
    profundidad = int(sys.argv[4])
    limite = int(sys.argv[5]) if len(sys.argv) > 5 else 200

    fens = [l.strip() for l in fens_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    fens = fens[:limite]

    motor = chess.engine.SimpleEngine.popen_uci([str(motor_path)])
    try:
        pedido = {"Threads": 1, "Hash": 128, "NNUEPath": str(pesos), "UseNNUE": True}
        motor.configure({k: v for k, v in pedido.items() if k in motor.options})
        total = 0
        logs = 0.0
        nodos_por_pos: list[int] = []
        for i, fen in enumerate(fens):
            tablero = chess.Board(fen)
            info = motor.analyse(
                tablero, chess.engine.Limit(depth=profundidad), game=("nodos", i)
            )
            n = int(info.get("nodes", 0))
            if n <= 0:
                continue
            nodos_por_pos.append(n)
            total += n
            logs += math.log(n)
        media_geo = math.exp(logs / len(nodos_por_pos)) if nodos_por_pos else 0.0
        print(f"motor      : {motor_path}")
        print(f"profundidad: {profundidad}")
        print(f"posiciones : {len(nodos_por_pos)}")
        print(f"NODOS TOTAL: {total}")
        print(f"media geom.: {media_geo:.0f}")
        print("CRUDO " + " ".join(str(n) for n in nodos_por_pos))
    finally:
        motor.quit()


if __name__ == "__main__":
    main()
