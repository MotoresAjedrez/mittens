#!/usr/bin/env python3
"""Mide la CALIDAD DEL ORDEN DE JUGADAS en el regimen de PARTIDA.

Metrica: porcentaje de cortes beta que caen en la PRIMERA jugada probada del
nodo. Es la metrica estandar de calidad de ordenamiento y, a diferencia del
conteo de nodos, no es ruidosa: con nodos FIJOS y las MISMAS posiciones para
todos los motores, el numero que sale es DETERMINISTA (no tiene error
estadistico; dos corridas del mismo binario dan exactamente lo mismo).

Por que hace falta, si `mittens bench` ya la imprime: el bench crea un
`Searcher::new` por posicion y llama a `search_fixed_depth`, o sea que mide
con las tablas de historia en CERO y sin envejecimiento entre jugadas. Una
partida real usa el MISMO Searcher jugada tras jugada, entrando por
`search_time`, que envejece las tablas en cada "go" y las deja calientes de la
jugada anterior. Cualquier cambio en las tablas de historia se comporta
distinto en cada regimen, y el que se juega es este. Ver el comentario de
`decaer_history` en search.rs y el paquete que el bench aprobo (-17% de nodos)
y el SPRT rechazo (-62 Elo).

Complementa a depth_en_partida.py: aquel mide la PROFUNDIDAD alcanzada (efecto
final sobre el arbol, poco sensible porque es un entero); este mide la calidad
del ORDEN directamente (mucho mas sensible, pero es un proxy: mejor orden no
garantiza mas Elo).

Requiere un binario compilado con el `info string orden` de lib.rs, que se
activa con MITTENS_ORDEN_INFO=1 (el script la pone sola).

Uso:
  python3 orden_en_partida.py PESOS PGN NODOS N_PARTIDAS ETIQUETA=BINARIO[,VAR=VAL...] ...
"""

from __future__ import annotations

import os
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
        env = dict(os.environ)
        env["MITTENS_ORDEN_INFO"] = "1"
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

    def buscar(self, jugadas: list[str], nodos: int) -> tuple[int, int, int]:
        """Devuelve (profundidad, cortes en 1ra ACUMULADOS, cortes beta ACUMULADOS)."""
        self._send("position startpos moves " + " ".join(jugadas) if jugadas
                   else "position startpos")
        lineas = self._cmd(f"go nodes {nodos}", hasta="bestmove")
        prof = 0
        primera = beta = 0
        for l in lineas:
            if l.startswith("info depth"):
                partes = l.split()
                try:
                    prof = max(prof, int(partes[partes.index("depth") + 1]))
                except (ValueError, IndexError):
                    pass
            elif l.startswith("info string orden "):
                partes = l.split()
                # info string orden primera P beta B
                primera = int(partes[4])
                beta = int(partes[6])
        return prof, primera, beta

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
    print(
        f"{len(partidas)} partidas, {n_pos} posiciones, {nodos} nodos fijos "
        f"por jugada\n",
        flush=True,
    )

    resultados = {}
    for espec in especificaciones:
        etiqueta, resto = espec.split("=", 1)
        trozos = resto.split(",")
        binario = trozos[0]
        entorno = dict(t.split("=", 1) for t in trozos[1:])
        motor = Motor(binario, entorno, pesos)
        profundidades: list[int] = []
        primera_total = 0
        beta_total = 0
        t0 = time.monotonic()
        try:
            for jugadas in partidas:
                motor.nueva_partida()
                # Los contadores del Searcher son ACUMULADOS y se tiran en
                # "ucinewgame": el ultimo valor visto en la partida es su
                # total. Se guarda el ultimo de CADA partida y se suma.
                ultima_primera = ultimo_beta = 0
                for i in range(len(jugadas)):
                    prof, primera, beta = motor.buscar(jugadas[:i], nodos)
                    profundidades.append(prof)
                    if beta >= ultimo_beta:  # monotono dentro de la partida
                        ultima_primera, ultimo_beta = primera, beta
                primera_total += ultima_primera
                beta_total += ultimo_beta
        finally:
            motor.cerrar()
        dt = time.monotonic() - t0
        if beta_total == 0:
            raise SystemExit(
                f"{etiqueta}: 0 cortes beta contados -- el binario no imprime "
                f"'info string orden' (falta MITTENS_ORDEN_INFO en lib.rs?)"
            )
        pct = 100.0 * primera_total / beta_total
        resultados[etiqueta] = (pct, primera_total, beta_total, profundidades, dt)
        print(
            f"{etiqueta:28s} orden(1ra) {pct:6.3f}%  ({primera_total}/{beta_total})"
            f" | prof media {statistics.mean(profundidades):6.3f} | {dt:5.1f}s"
            f"  entorno={entorno}",
            flush=True,
        )

    etiquetas = list(resultados)
    if len(etiquetas) > 1:
        base_pct = resultados[etiquetas[0]][0]
        base_prof = resultados[etiquetas[0]][3]
        print(f"\nDiferencia contra '{etiquetas[0]}':")
        for et in etiquetas[1:]:
            pct, _, _, prof, _ = resultados[et]
            difs = [a - b for a, b in zip(prof, base_prof)]
            media = statistics.mean(difs)
            ee = statistics.stdev(difs) / (len(difs) ** 0.5) if len(difs) > 1 else 0.0
            print(
                f"  {et:26s} orden {pct - base_pct:+.3f} puntos "
                f"(determinista) | profundidad {media:+.4f} +-{ee:.4f} plies"
            )


if __name__ == "__main__":
    main()
