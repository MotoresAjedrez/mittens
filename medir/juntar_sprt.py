#!/usr/bin/env python3
"""Suma los resultados de varios trabajadores de `sprt_diverso.py` y recalcula
el veredicto conjunto.

Sumar trabajadores solo es LEGITIMO si midieron lo mismo sobre aperturas
DISJUNTAS. Esta herramienta lo verifica antes de sumar y aborta si no se
cumple:

  * la firma (binarios, pesos, elo0/elo1/alpha/beta, nodos, banco) tiene que
    ser identica salvo el campo `apertura_inicial`;
  * ninguna huella de partida puede aparecer en dos trabajadores (si aparece,
    es la misma partida contada dos veces -- exactamente el bug del arnes
    viejo, solo que repartido entre procesos).

Uso:
  python3 medir/juntar_sprt.py results_sprt/NOMBRE_A results_sprt/NOMBRE_B ...
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sprt_diverso import calcular_llr, elo_y_error, limites_wald  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    dirs = [pathlib.Path(a).resolve() for a in sys.argv[1:]]

    firma_ref = None
    wins = draws = losses = 0
    huellas: set[str] = set()
    total_partidas = 0
    for d in dirs:
        estado = json.loads((d / "state.json").read_text(encoding="utf-8"))
        firma = dict(estado["firma"])
        tramo = firma.pop("apertura_inicial", None)
        if firma_ref is None:
            firma_ref = firma
        elif firma != firma_ref:
            raise SystemExit(
                f"ABORTADO: {d.name} no midio lo mismo que el primero "
                "(binarios, pesos, nodos o banco distintos)."
            )
        nuevas = set(estado.get("huellas", []))
        solapadas = nuevas & huellas
        if solapadas:
            raise SystemExit(
                f"ABORTADO: {d.name} comparte {len(solapadas)} partidas con otro "
                "trabajador; sumarlas contaria la misma evidencia dos veces."
            )
        huellas |= nuevas
        wins += int(estado["wins"])
        draws += int(estado["draws"])
        losses += int(estado["losses"])
        total_partidas += int(estado["hechas"])
        print(
            f"{d.name}: +{estado['wins']} ={estado['draws']} -{estado['losses']} "
            f"({estado['hechas']} partidas, tramo desde apertura {tramo})"
        )

    n = wins + draws + losses
    if n != total_partidas:
        raise SystemExit("Los W/D/L no suman las partidas jugadas; estado corrupto.")
    elo0 = firma_ref["elo0"]
    elo1 = firma_ref["elo1"]
    la, lb = limites_wald(firma_ref["alpha"], firma_ref["beta"])
    llr = calcular_llr(wins, draws, losses, elo0, elo1)
    elo, err = elo_y_error(wins, draws, losses)
    score = (wins + 0.5 * draws) / n

    print()
    print(f"CONJUNTO: {n} partidas (+{wins} ={draws} -{losses}), score={score:.2%}")
    print(f"Partidas DISTINTAS: {len(huellas)}/{n} ({len(huellas) / n:.1%})")
    print(f"Elo estimado: {elo:+.1f} +/- {err:.1f} (IC95%)")
    print(f"LLR: {llr:.4f} (limites [{la:.4f}, {lb:.4f}])")
    if llr >= lb:
        print(f"VEREDICTO: acepta H1 (elo >= {elo1})")
    elif llr <= la:
        print(f"VEREDICTO: acepta H0 (elo <= {elo0})")
    else:
        print("VEREDICTO: AMBIGUO, el LLR no cruzo ningun limite")


if __name__ == "__main__":
    main()
