#!/usr/bin/env python3
"""Genera el libro de aperturas UNICAS que usa sprt_real.py.

POR QUE EXISTE ESTE ARCHIVO
---------------------------
`sprt_real.py` tenia 20 aperturas cableadas y las recorria con
`opening = OPENINGS[(index // 2) % 20]`, `candidato_blancas = index % 2 == 0`.
Con los dos motores jugando a NODOS FIJOS, 1 hilo y `ucinewgame` entre
partidas, el juego es COMPLETAMENTE DETERMINISTA: solo existen 20 x 2 = 40
partidas posibles, y a partir de la partida 41 cada partida es una copia
exacta, jugada por jugada, de una anterior. El LLR seguia sumando "evidencia"
sobre partidas repetidas, o sea que todo veredicto por encima de ~40 partidas
era confianza fabricada.

Verificado sobre los resultados que ya estaban en el repo (results_sprt/):

    corrplexity_lmr      2843 partidas ->   40 unicas (una repetida 72 veces)
    ttpv_persistente     1562 partidas ->   40 unicas (una repetida 40 veces)
    corrhist_menores      362 partidas ->   40 unicas
    fable_solo_tt_gen     357 partidas ->   40 unicas
    finales_aplastantes   476 partidas ->   36 unicas

Es el mismo error que el proyecto ya documento dos veces (el bench que corria
sin NNUE, y el h2h con `apertura = i % 12` que jugaba 12 partidas unicas
repetidas 5 veces), esta vez dentro del arnes que se usa para ACEPTAR o
RECHAZAR cambios.

COMO SE ARMA EL LIBRO
---------------------
Recorrido en ANCHURA (BFS) del arbol COMPLETO del libro Polyglot
`performance.bin`, que ya vive en el repo. Se recogen todas las posiciones
unicas (por EPD) cuya profundidad cae en la banda [PLY_MIN, PLY_MAX].

`performance.bin` es un libro DEEP pero ANGOSTO (~93.000 entradas, pero 1-3
jugadas por posicion): a 8 medias jugadas el arbol entero tiene solo 116
posiciones distintas, asi que un solo nivel no alcanza para un SPRT. Tomando
la banda 8..16 salen ~2.300 aperturas -> ~4.600 partidas unicas, que si
alcanza. Cada apertura se juega DOS veces (una con cada color).

El recorrido es 100% determinista (jugadas ordenadas por peso descendente y
UCI ascendente), asi que el libro es reproducible; ademas se versiona el
`libro_sprt.epd` resultante y su sha256 entra en la firma del checkpoint de
sprt_real.py.

LIMITE CONOCIDO: al ser un libro angosto, algunas aperturas de la banda son
ancestro/descendiente de otras, asi que dos partidas pueden compartir parte
del comienzo. No son partidas repetidas (el detector de duplicados de
sprt_real.py las distinguiria), pero un libro mas ancho -- tipo UHO/Pohl, con
decenas de miles de lineas independientes -- seria mejor todavia. Ese es el
siguiente paso natural si algun dia hace falta un SPRT de mas de ~4.600
partidas.

Uso:  python3 tools/generar_libro_sprt.py [n_aperturas] [ply_min] [ply_max]
"""

from __future__ import annotations

import pathlib
import sys

import chess
import chess.polyglot

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIBRO_POLYGLOT = ROOT / "performance.bin"
SALIDA = ROOT / "libro_sprt.epd"
PLY_MIN = 8
PLY_MAX = 16
N_APERTURAS = 4000


def muestra_espaciada(items: list, tope: int) -> list:
    """Submuestra `tope` elementos con paso constante (conserva la variedad)."""
    if len(items) <= tope:
        return items
    paso = len(items) / tope
    return [items[int(i * paso)] for i in range(tope)]


def generar(n_aperturas: int, ply_min: int, ply_max: int) -> list[str]:
    cosecha: list[str] = []
    with chess.polyglot.open_reader(LIBRO_POLYGLOT) as libro:
        nivel = [chess.Board()]
        vistos: set[str] = set()
        for ply in range(1, ply_max + 1):
            siguiente: list[chess.Board] = []
            for tablero in nivel:
                entradas = list(libro.find_all(tablero))
                # Determinista: peso descendente, UCI ascendente para desempatar.
                entradas.sort(key=lambda e: (-e.weight, e.move.uci()))
                for entrada in entradas:
                    hijo = tablero.copy(stack=False)
                    hijo.push(entrada.move)
                    if hijo.is_game_over():
                        continue
                    epd = hijo.epd()
                    if epd in vistos:
                        continue
                    vistos.add(epd)
                    siguiente.append(hijo)
            if not siguiente:
                break
            if ply >= ply_min:
                cosecha.extend(t.fen() for t in siguiente)
            nivel = siguiente
    return muestra_espaciada(cosecha, n_aperturas)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_APERTURAS
    ply_min = int(sys.argv[2]) if len(sys.argv) > 2 else PLY_MIN
    ply_max = int(sys.argv[3]) if len(sys.argv) > 3 else PLY_MAX
    aperturas = generar(n, ply_min, ply_max)
    assert len(set(aperturas)) == len(aperturas), "el BFS produjo FENs repetidas"
    SALIDA.write_text("\n".join(aperturas) + "\n", encoding="utf-8")
    print(
        f"{len(aperturas)} aperturas unicas (plies {ply_min}..{ply_max}) -> {SALIDA}\n"
        f"=> tope duro de partidas unicas para sprt_real.py: {2 * len(aperturas)}"
    )
    if len(aperturas) < n:
        print(
            f"AVISO: se pidieron {n} y el arbol del libro solo da {len(aperturas)} "
            "posiciones distintas en esa banda."
        )


if __name__ == "__main__":
    main()
