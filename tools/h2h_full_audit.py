#!/usr/bin/env python3
"""H2H aislado de la auditoría completa contra el binario desplegado."""

from __future__ import annotations

import importlib.util
import os
import pathlib

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[1]
HARNESS = HERE.parents[2] / "reckless-candidates" / "h2h_hindsight.py"
spec = importlib.util.spec_from_file_location("h2h_base", HARNESS)
if spec is None or spec.loader is None:
    raise SystemExit(f"No se pudo cargar {HARNESS}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.ROOT = REPO
# Rutas del binario/pesos DESPLEGADOS, configurables por entorno (son
# especificas de la maquina donde corre el bot real).
module.BASE = pathlib.Path(
    os.environ.get(
        "MITTENS_BASE_BIN",
        str(pathlib.Path.home() / "mimotor-lichess-bot/engines/mimotor-tal-rust"),
    )
)
module.WEIGHTS = pathlib.Path(
    os.environ.get(
        "MITTENS_BASE_WEIGHTS",
        str(pathlib.Path.home() / "mimotor-lichess-bot/engines/nn_weights/pesos_flat_3m.bin"),
    )
)
module.EXPECTED_BASE_SHA256 = (
    "151e9b4af65aea070168e330eacfcf0bc9e6f5ca808de8ee7513b4bea87fb755"
)

if __name__ == "__main__":
    module.main()
