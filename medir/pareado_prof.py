#!/usr/bin/env python3
"""Diferencia PAREADA de profundidad alcanzada (salidas de prof_a_nodos.py)."""
import math
import pathlib
import sys


def crudo(p: pathlib.Path) -> list[int]:
    for linea in p.read_text().splitlines():
        if linea.startswith("CRUDO "):
            return [int(x) for x in linea.split()[1:]]
    raise SystemExit(f"sin linea CRUDO en {p}")


if len(sys.argv) < 3:
    raise SystemExit(__doc__)
a = crudo(pathlib.Path(sys.argv[1]))
b = crudo(pathlib.Path(sys.argv[2]))
if len(a) != len(b):
    raise SystemExit(f"distinto n: {len(a)} vs {len(b)}")
d = [y - x for x, y in zip(a, b)]
n = len(d)
m = sum(d) / n
var = sum((x - m) ** 2 for x in d) / (n - 1)
se = math.sqrt(var / n)
print(f"posiciones : {n}")
print(f"prof base  : {sum(a)/n:.3f}")
print(f"prof cand  : {sum(b)/n:.3f}")
print(f"DELTA prof : {m:+.3f} plies  IC95 [{m-1.96*se:+.3f}, {m+1.96*se:+.3f}]")
print(f"reparto    : mas hondo {sum(1 for x in d if x>0)}, igual {sum(1 for x in d if x==0)}, menos hondo {sum(1 for x in d if x<0)}")
