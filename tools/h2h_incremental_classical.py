#!/usr/bin/env python3
"""Torneo aislado: acumulador clásico incremental vs producción Magic.

Uso:
  python3 tools/h2h_incremental_classical.py BINARIO_CANDIDATO NOMBRE [PARTIDAS=40] [MS=600]

Reutiliza el arnés de H2H ya auditado, pero fija explícitamente la base
Magic/Hindsight actual y los mismos pesos NNUE para ambos motores. No escribe
ni reinicia el bot de Lichess.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys


HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[1]
HARNESS = (
    HERE.parents[2]
    / "reckless-candidates"
    / "h2h_hindsight.py"
)

if not HARNESS.exists():
    raise SystemExit(f"No existe el arnés reutilizable: {HARNESS}")

spec = importlib.util.spec_from_file_location("h2h_base", HARNESS)
if spec is None or spec.loader is None:
    raise SystemExit("No se pudo cargar el arnés H2H")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Base de producción confirmada por SHA antes de cada torneo. Cambiar este
# valor fuerza un aborto seguro si alguien despliega otro binario durante el
# torneo, evitando comparar contra una referencia distinta sin avisar.
module.ROOT = REPO
module.BASE = pathlib.Path(
    os.environ.get(
        "MIMOTOR_BASE_BIN",
        str(pathlib.Path.home() / "mimotor-lichess-bot/engines/mimotor-tal-rust"),
    )
)
module.WEIGHTS = pathlib.Path(
    os.environ.get(
        "MIMOTOR_BASE_WEIGHTS",
        str(pathlib.Path.home() / "mimotor-lichess-bot/engines/nn_weights/pesos_flat_3m.bin"),
    )
)
module.EXPECTED_BASE_SHA256 = (
    "b45933e252009b73c0af1a032127ebc690712c8bbfda23b13afbbddf96c870ed"
)

if __name__ == "__main__":
    module.main()
