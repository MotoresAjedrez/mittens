#!/usr/bin/env python3
"""Audita los PGN de results_sprt/: cuenta partidas REPETIDAS y recalcula el
veredicto usando SOLO las partidas unicas.

POR QUE
-------
Un SPRT solo vale si cada partida es evidencia NUEVA. Con dos motores
deterministas (nodos fijos, 1 hilo, `ucinewgame` entre partidas) y un banco de
aperturas chico, el arnes vuelve al principio del libro y REPITE partidas
jugada por jugada; el LLR las suma como si fueran independientes y "converge"
sin que la muestra haya crecido.

Eso es exactamente lo que le pasaba a `sprt_real.py` antes del arreglo del
libro (20 aperturas cableadas -> 40 partidas posibles). Este script existe
para que ese fallo -- en este arnes o en cualquier otro -- se vea de un
vistazo y no vuelva a pasar inadvertido.

Como leerlo: si "score" no cambia entre el veredicto guardado y el recalculo
sobre unicas, es la firma del bug -- son literalmente las mismas partidas
repetidas, el porcentaje se queda quieto y lo unico que crece es N (y con N,
el LLR).

Uso:  python3 tools/auditar_sprt.py [carpeta_de_resultados]
      (por defecto: results_sprt/)
"""

from __future__ import annotations

import math
import pathlib
import re
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def elo_a_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_a_elo(score: float) -> float:
    if score <= 0.0 or score >= 1.0:
        return float("nan")
    return -400.0 * math.log10(1.0 / score - 1.0)


def llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Mismo GSPRT que sprt_real.py, para poder comparar peras con peras."""
    n = wins + draws + losses
    if n == 0:
        return 0.0
    score = (wins + 0.5 * draws) / n
    var = (
        wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + losses * (0.0 - score) ** 2
    ) / n
    if var < 1e-9:
        if n < 8:
            return 0.0
        var = 1.0 / n
    s0, s1 = elo_a_score(elo0), elo_a_score(elo1)
    return n * (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * var)


def auditar(pgn: pathlib.Path, elo0: float, elo1: float) -> str:
    bloques = re.split(r"\n\n(?=\[Event)", pgn.read_text(errors="replace").strip())
    vistos: set[str] = set()
    wins = draws = losses = 0
    repetidas = 0
    for bloque in bloques:
        cabeceras = dict(re.findall(r'\[(\w+) "([^"]*)"\]', bloque))
        cuerpo = "\n".join(
            linea for linea in bloque.splitlines() if not linea.startswith("[")
        ).strip()
        if cuerpo in vistos:
            repetidas += 1
            continue
        vistos.add(cuerpo)
        resultado = cabeceras.get("Result", "*")
        candidato_blancas = cabeceras.get("White", "") != "baseline"
        if resultado == "1-0":
            if candidato_blancas:
                wins += 1
            else:
                losses += 1
        elif resultado == "0-1":
            if candidato_blancas:
                losses += 1
            else:
                wins += 1
        else:
            # "1/2-1/2" y "*" (partida cortada por el tope de plies, que
            # sprt_real.py puntua 0.5) cuentan como tablas.
            draws += 1

    n = wins + draws + losses
    if n == 0:
        return "  sin partidas puntuables"
    score = (wins + 0.5 * draws) / n
    err = math.sqrt(score * (1.0 - score) / n) if 0.0 < score < 1.0 else 0.0
    bajo = score_a_elo(max(score - 1.96 * err, 1e-9))
    alto = score_a_elo(min(score + 1.96 * err, 1.0 - 1e-9))
    return (
        f"  partidas en el PGN : {len(bloques)}  (repetidas: {repetidas})\n"
        f"  partidas UNICAS    : {n}  (+{wins} ={draws} -{losses})\n"
        f"  score real         : {score:.1%} +- {err:.1%}\n"
        f"  LLR real ({elo0:g},{elo1:g})   : {llr(wins, draws, losses, elo0, elo1):.2f}"
        f"   (limites tipicos [-2.94, 2.94])\n"
        f"  elo puntual        : {score_a_elo(score):+.0f}  "
        f"(IC 95%: {bajo:+.0f} .. {alto:+.0f})"
    )


def main() -> None:
    base = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "results_sprt"
    if not base.is_dir():
        raise SystemExit(f"No existe la carpeta {base}")
    hubo = False
    for carpeta in sorted(base.iterdir()):
        pgn = carpeta / "games.pgn"
        if not pgn.is_file():
            continue
        hubo = True
        elo0, elo1 = 0.0, 5.0
        veredicto = carpeta / "veredicto.txt"
        guardado = "(sin veredicto guardado)"
        if veredicto.is_file():
            texto = veredicto.read_text(encoding="utf-8")
            lineas = texto.splitlines()
            guardado = next((x for x in lineas if x.startswith("Partidas:")), guardado)
            m = re.search(r"elo <= ([-\d.]+), H1: elo >= ([-\d.]+)", texto)
            if m:
                elo0, elo1 = float(m.group(1)), float(m.group(2))
        print(carpeta.name)
        print(f"  veredicto guardado : {guardado}")
        print(auditar(pgn, elo0, elo1))
        print()
    if not hubo:
        print(f"No se encontro ningun games.pgn en {base}")


if __name__ == "__main__":
    main()
