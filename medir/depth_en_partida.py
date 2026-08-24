#!/usr/bin/env python3
"""Mide la EFICIENCIA DEL ARBOL en el regimen de PARTIDA, no en el del bench.

Por que hace falta: `mittens bench` crea un `Searcher::new` por posicion y
llama a `search_fixed_depth`, que NO pasa por `decaer_history()`. O sea que el
bench mide con las tablas de historia en CERO y sin envejecimiento entre
jugadas. Una partida real usa el MISMO Searcher jugada tras jugada, entrando
por `search_time`, que SI envejece las tablas en cada "go". Son dos regimenes
distintos de la misma maquinaria, y cualquier cambio en las tablas de historia
se comporta distinto en cada uno.

Que mide: se reproducen partidas reales jugada por jugada mandando
`position startpos moves ...` + `go nodes N` al motor (con `ucinewgame` al
empezar cada partida, igual que hace sprt_real.py), y se anota la PROFUNDIDAD
alcanzada en cada jugada. A nodos FIJOS, mas profundidad con los mismos nodos
= arbol mas eficiente. No es Elo -- es un filtro barato para decidir que
merece un SPRT.

Uso:
  python3 depth_en_partida.py PESOS PGN NODOS N_PARTIDAS ETIQUETA=BINARIO[,VAR=VAL...] ...
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time

import chess
import chess.pgn


def leer_partidas(pgn_path: str, cuantas: int) -> list[list[str]]:
    """Devuelve listas de jugadas en UCI, una por partida."""
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
        import os

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

    def buscar(self, jugadas: list[str], nodos: int) -> tuple[int, int]:
        """Devuelve (profundidad final, seldepth final) de esta busqueda."""
        self._send("position startpos moves " + " ".join(jugadas) if jugadas
                   else "position startpos")
        lineas = self._cmd(f"go nodes {nodos}", hasta="bestmove")
        prof = 0
        seldep = 0
        for l in lineas:
            if l.startswith("info depth"):
                partes = l.split()
                try:
                    prof = max(prof, int(partes[partes.index("depth") + 1]))
                except (ValueError, IndexError):
                    pass
                if "seldepth" in partes:
                    try:
                        seldep = max(seldep, int(partes[partes.index("seldepth") + 1]))
                    except (ValueError, IndexError):
                        pass
        return prof, seldep

    def cerrar(self) -> None:
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main() -> None:
    pesos, pgn_path, nodos_s, n_partidas_s, *especificaciones = sys.argv[1:]
    nodos = int(nodos_s)
    partidas = leer_partidas(pgn_path, int(n_partidas_s))
    n_pos = sum(len(j) for j in partidas)
    print(f"{len(partidas)} partidas, {n_pos} posiciones, {nodos} nodos fijos por jugada\n")

    resultados = {}
    for espec in especificaciones:
        etiqueta, resto = espec.split("=", 1)
        trozos = resto.split(",")
        binario = trozos[0]
        entorno = dict(t.split("=", 1) for t in trozos[1:])
        motor = Motor(binario, entorno, pesos)
        profundidades: list[int] = []
        seldeps: list[int] = []
        t0 = time.monotonic()
        try:
            for jugadas in partidas:
                motor.nueva_partida()
                for i in range(len(jugadas)):
                    prof, seldep = motor.buscar(jugadas[:i], nodos)
                    profundidades.append(prof)
                    seldeps.append(seldep)
        finally:
            motor.cerrar()
        dt = time.monotonic() - t0
        media = statistics.mean(profundidades)
        resultados[etiqueta] = (media, statistics.mean(seldeps), profundidades, dt)
        print(
            f"{etiqueta:28s} profundidad media {media:6.3f} | seldepth media "
            f"{statistics.mean(seldeps):6.3f} | {dt:5.1f}s  entorno={entorno}"
        )

    # Comparacion emparejada POSICION POR POSICION contra la primera etiqueta:
    # cada posicion es la misma para los dos motores, asi que la diferencia
    # emparejada tiene mucho menos ruido que comparar las dos medias sueltas.
    etiquetas = list(resultados)
    if len(etiquetas) > 1:
        base_prof = resultados[etiquetas[0]][2]
        print(f"\nDiferencia emparejada contra '{etiquetas[0]}' (por posicion):")
        for et in etiquetas[1:]:
            prof = resultados[et][2]
            difs = [a - b for a, b in zip(prof, base_prof)]
            media = statistics.mean(difs)
            # Error estandar de la media de las diferencias emparejadas.
            ee = statistics.stdev(difs) / (len(difs) ** 0.5) if len(difs) > 1 else 0.0
            mejor = sum(1 for d in difs if d > 0)
            peor = sum(1 for d in difs if d < 0)
            print(
                f"  {et:26s} {media:+.4f} plies +-{ee:.4f} (1 sigma)  "
                f"| mas profundo en {mejor}, menos en {peor}, igual en "
                f"{len(difs) - mejor - peor}"
            )


if __name__ == "__main__":
    main()
