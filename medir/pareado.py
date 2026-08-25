#!/usr/bin/env python3
"""Compara dos salidas CRUDO de medir/nodos_arbol.py posicion a posicion.

Razon geometrica pareada con IC95% (bootstrap-libre: t sobre log-razones).
Uso: pareado.py archivo_base.txt archivo_cand.txt
"""
import math
import pathlib
import sys


def crudo(p: pathlib.Path) -> list[int]:
    for linea in p.read_text().splitlines():
        if linea.startswith("CRUDO "):
            return [int(x) for x in linea.split()[1:]]
    raise SystemExit(f"sin linea CRUDO en {p}")


def main() -> None:
    a = crudo(pathlib.Path(sys.argv[1]))
    b = crudo(pathlib.Path(sys.argv[2]))
    if len(a) != len(b):
        raise SystemExit(f"distinto numero de posiciones: {len(a)} vs {len(b)}")
    logs = [math.log(y / x) for x, y in zip(a, b) if x > 0 and y > 0]
    n = len(logs)
    m = sum(logs) / n
    var = sum((l - m) ** 2 for l in logs) / (n - 1)
    se = math.sqrt(var / n)
    lo, hi = m - 1.96 * se, m + 1.96 * se
    print(f"posiciones : {n}")
    print(f"nodos base : {sum(a)}")
    print(f"nodos cand : {sum(b)}")
    print(f"total      : x{sum(b)/sum(a):.4f}  ({100*(sum(b)/sum(a)-1):+.2f}%)")
    print(
        f"PAREADO    : x{math.exp(m):.4f}  ({100*(math.exp(m)-1):+.2f}%)"
        f"  IC95 [x{math.exp(lo):.4f}, x{math.exp(hi):.4f}]"
        f" = [{100*(math.exp(lo)-1):+.2f}%, {100*(math.exp(hi)-1):+.2f}%]"
    )
    peor = max(range(n), key=lambda i: logs[i])
    mejor = min(range(n), key=lambda i: logs[i])
    print(f"peor pos   : #{peor} x{math.exp(logs[peor]):.2f}   mejor pos: #{mejor} x{math.exp(logs[mejor]):.2f}")


if __name__ == "__main__":
    main()
