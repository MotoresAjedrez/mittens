#!/usr/bin/env python3
"""Suma los resultados de varios trabajadores de sprt_real.py y da el
veredicto conjunto.

Los trabajadores lanzados con `apertura_inicial` distinto juegan sobre tramos
DISJUNTOS del libro, asi que sus partidas son independientes y sus W/D/L se
pueden sumar. Este script hace la suma y ademas:

  * VERIFICA que los tramos no se solapen (si se solapan, hay partidas
    repetidas entre trabajadores y el total no vale);
  * VERIFICA que todos los trabajadores comparen los mismos binarios, los
    mismos pesos y el mismo numero de nodos;
  * recalcula score, LLR y el intervalo de confianza del Elo sobre el total.

El LLR conjunto se reporta como un test de MUESTRA FIJA sobre el total
acumulado: es la lectura honesta cuando ningun trabajador corto por si mismo.
Un SPRT secuencial de verdad exige una regla de parada unica, asi que NO uses
este numero para "seguir jugando hasta que cruce"; usalo para leer que dice
la evidencia que ya tenes.

Uso:
  python3 tools/juntar_sprt.py results_sprt/nombreA results_sprt/nombreB ...
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sprt_real import compute_llr, sprt_bounds  # noqa: E402


def score_a_elo(score: float) -> float:
    if score <= 0.0 or score >= 1.0:
        return float("nan")
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    carpetas = [pathlib.Path(a) for a in sys.argv[1:]]

    total_w = total_d = total_l = 0
    comparacion = None
    tramos: list[tuple[int, int, str]] = []
    elo0 = elo1 = alpha = beta = None

    for carpeta in carpetas:
        estado_path = carpeta / "state.json"
        if not estado_path.is_file():
            raise SystemExit(f"Falta {estado_path}")
        estado = json.loads(estado_path.read_text(encoding="utf-8"))
        firma = estado["signature"]

        # Lo que TIENE que coincidir entre trabajadores para poder sumar.
        clave = (
            firma["candidate_sha256"],
            firma["candidate_weights_sha256"],
            firma["baseline_sha256"],
            firma["baseline_weights_sha256"],
            firma["nodes"],
            firma.get("openings_sha256"),
        )
        if comparacion is None:
            comparacion = clave
            elo0, elo1 = firma["elo0"], firma["elo1"]
            alpha, beta = firma["alpha"], firma["beta"]
        elif clave != comparacion:
            raise SystemExit(
                f"{carpeta.name} no compara lo mismo que los anteriores "
                "(binarios, pesos, nodos o libro distintos): NO se pueden sumar."
            )

        inicio = int(firma.get("opening_start", 0))
        completadas = int(estado["completed"])
        # Cada apertura da 2 partidas; el trabajador consumio ceil(n/2).
        aperturas_usadas = (completadas + 1) // 2
        tramos.append((inicio, inicio + aperturas_usadas, carpeta.name))

        total_w += int(estado["wins"])
        total_d += int(estado["draws"])
        total_l += int(estado["losses"])

    # Solape entre tramos = partidas repetidas entre trabajadores.
    tramos.sort()
    for (ini_a, fin_a, nom_a), (ini_b, fin_b, nom_b) in zip(tramos, tramos[1:]):
        if fin_a > ini_b:
            raise SystemExit(
                f"ABORTADO: los tramos de {nom_a} [{ini_a},{fin_a}) y {nom_b} "
                f"[{ini_b},{fin_b}) se SOLAPAN. Hay partidas repetidas entre "
                "trabajadores; el total no es evidencia independiente."
            )

    n = total_w + total_d + total_l
    if n == 0:
        raise SystemExit("Sin partidas.")
    score = (total_w + 0.5 * total_d) / n
    err = math.sqrt(score * (1.0 - score) / n) if 0.0 < score < 1.0 else 0.0
    la, lb = sprt_bounds(alpha, beta)
    llr = compute_llr(total_w, total_d, total_l, elo0, elo1)

    print(f"Trabajadores: {len(carpetas)}")
    for ini, fin, nom in tramos:
        print(f"  {nom:32s} aperturas [{ini}, {fin})")
    print(f"Total: {n} partidas  (+{total_w} ={total_d} -{total_l})")
    print(f"Score: {score:.2%} +- {err:.2%}")
    print(
        f"Elo puntual: {score_a_elo(score):+.1f}  "
        f"(IC 95%: {score_a_elo(max(score - 1.96 * err, 1e-9)):+.1f} .. "
        f"{score_a_elo(min(score + 1.96 * err, 1 - 1e-9)):+.1f})"
    )
    print(
        f"LLR de muestra fija (H0: elo<={elo0:g}, H1: elo>={elo1:g}): {llr:.3f}  "
        f"limites [{la:.2f}, {lb:.2f}]"
    )
    if llr >= lb:
        print("=> cruza el limite superior: la evidencia favorece H1")
    elif llr <= la:
        print("=> cruza el limite inferior: la evidencia favorece H0")
    else:
        print("=> NO cruza ningun limite: sin decidir, hacen falta mas partidas")


if __name__ == "__main__":
    main()
