#!/usr/bin/env python3
"""Mide el TAMANO DEL ARBOL (nodos a profundidad fija) en el regimen de PARTIDA.

Es el instrumento mas sensible de los tres de esta carpeta, y el que apunta
directo a la brecha conocida de Mittens (necesita 1,3-2,75x mas nodos que
Reckless para la misma profundidad nominal):

  - depth_en_partida.py   mide profundidad con nodos fijos. Es un ENTERO, asi
                          que hace falta muchisima muestra para ver un 1%.
  - orden_en_partida.py   mide el % de cortes en la 1ra jugada. Determinista y
                          fino, pero es un proxy indirecto del tamano del arbol.
  - nodos_en_partida.py   mide NODOS a profundidad fija. Determinista, continuo
                          y es el tamano del arbol en si.

Por que se puede hacer en el regimen de partida: `go depth N` de Mittens entra
por `search_time` (no por `search_fixed_depth`), o sea que SI pasa por
`decaer_history()` y SI reutiliza el mismo Searcher jugada tras jugada, igual
que una partida real. El bench, en cambio, crea un `Searcher::new` por posicion
y arranca con las tablas de historia en cero -- el regimen que no se juega, y
el que ya hizo aprobar un paquete que despues perdio -62 Elo en SPRT (ver el
comentario de `decaer_history` en search.rs).

Se reportan tres numeros por variante, porque el total solo puede ser dominado
por unas pocas posiciones que explotan:
  - nodos TOTALES (la suma cruda),
  - razon MEDIANA por posicion contra la base (robusta a outliers),
  - media GEOMETRICA de las razones (el promedio correcto de razones).
Todo es determinista: dos corridas del mismo binario dan lo mismo exacto.

Uso:
  python3 nodos_en_partida.py PESOS PGN PROFUNDIDAD N_PARTIDAS ETIQUETA=BINARIO[,VAR=VAL...] ...
"""

from __future__ import annotations

import math
import os
import statistics
import subprocess
import sys
import time

import chess
import chess.pgn


def leer_partidas(pgn_path: str, cuantas: int) -> list[list[str]]:
    partidas: list[list[str]] = []
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        while len(partidas) < cuantas:
            juego = chess.pgn.read_game(fh)
            if juego is None:
                break
            tablero = juego.board()
            jugadas = []
            for mv in juego.mainline_moves():
                jugadas.append(mv.uci())
                tablero.push(mv)
            if len(jugadas) >= 20:
                partidas.append(jugadas)
    return partidas


class Motor:
    def __init__(self, binario: str, entorno: dict[str, str], pesos: str):
        env = dict(os.environ)
        env.update(entorno)
        self.p = subprocess.Popen(
            [binario],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        self._cmd("uci", hasta="uciok")
        for opcion in (
            "setoption name Threads value 1",
            "setoption name Hash value 128",
            f"setoption name NNUEPath value {pesos}",
            "setoption name UseNNUE value true",
        ):
            self._send(opcion)
        self._cmd("isready", hasta="readyok")

    def _send(self, linea: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(linea + "\n")
        self.p.stdin.flush()

    def _cmd(self, linea: str, hasta: str) -> list[str]:
        self._send(linea)
        salida = []
        assert self.p.stdout is not None
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError(f"el motor murio esperando '{hasta}'")
            salida.append(l.rstrip())
            if l.startswith(hasta):
                return salida

    def nueva_partida(self) -> None:
        self._send("ucinewgame")
        self._cmd("isready", hasta="readyok")

    def buscar(self, jugadas: list[str], profundidad: int) -> int:
        """Nodos de la ULTIMA iteracion reportada (el total de la busqueda)."""
        self._send("position startpos moves " + " ".join(jugadas) if jugadas
                   else "position startpos")
        lineas = self._cmd(f"go depth {profundidad}", hasta="bestmove")
        nodos = 0
        for l in lineas:
            if l.startswith("info depth") and " nodes " in l:
                partes = l.split()
                try:
                    nodos = max(nodos, int(partes[partes.index("nodes") + 1]))
                except (ValueError, IndexError):
                    pass
        return nodos

    def cerrar(self) -> None:
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main() -> None:
    pesos, pgn_path, prof_s, n_partidas_s, *especificaciones = sys.argv[1:]
    profundidad = int(prof_s)
    partidas = leer_partidas(pgn_path, int(n_partidas_s))
    n_pos = sum(len(j) for j in partidas)
    print(
        f"{len(partidas)} partidas, {n_pos} posiciones, profundidad fija "
        f"{profundidad}\n",
        flush=True,
    )

    resultados = {}
    for espec in especificaciones:
        etiqueta, resto = espec.split("=", 1)
        trozos = resto.split(",")
        binario = trozos[0]
        entorno = dict(t.split("=", 1) for t in trozos[1:])
        motor = Motor(binario, entorno, pesos)
        nodos: list[int] = []
        t0 = time.monotonic()
        try:
            for jugadas in partidas:
                motor.nueva_partida()
                for i in range(len(jugadas)):
                    nodos.append(motor.buscar(jugadas[:i], profundidad))
        finally:
            motor.cerrar()
        dt = time.monotonic() - t0
        resultados[etiqueta] = (nodos, dt)
        print(
            f"{etiqueta:28s} nodos totales {sum(nodos):>12,} | "
            f"mediana por posicion {statistics.median(nodos):>9,.0f} | {dt:5.1f}s"
            f"  entorno={entorno}",
            flush=True,
        )

    etiquetas = list(resultados)
    if len(etiquetas) > 1:
        base = resultados[etiquetas[0]][0]
        total_base = sum(base)
        print(f"\nContra '{etiquetas[0]}' (menos nodos = arbol mas eficiente):")
        for et in etiquetas[1:]:
            cur = resultados[et][0]
            total = sum(cur)
            # Razones por posicion, ignorando las que la base dejo en 0 (no
            # deberia haber ninguna, pero un 0 arruinaria la geometrica).
            razones = [c / b for c, b in zip(cur, base) if b > 0 and c > 0]
            mediana = statistics.median(razones)
            geo = math.exp(statistics.fmean(math.log(r) for r in razones))
            mejor = sum(1 for r in razones if r < 1.0)
            peor = sum(1 for r in razones if r > 1.0)
            print(
                f"  {et:26s} total {100.0 * (total / total_base - 1.0):+7.2f}%"
                f" | mediana {100.0 * (mediana - 1.0):+6.2f}%"
                f" | geometrica {100.0 * (geo - 1.0):+6.2f}%"
                f" | menos nodos en {mejor}, mas en {peor}"
            )


if __name__ == "__main__":
    main()
