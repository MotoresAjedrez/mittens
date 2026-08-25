#!/usr/bin/env python3
"""SPRT entre dos binarios UCI de Mittens, con aperturas DIVERSAS y control de
duplicados EN VIVO.

Uso:
  python3 medir/sprt_diverso.py CANDIDATO_BIN CANDIDATO_PESOS BASELINE_BIN \
      BASELINE_PESOS NOMBRE [elo0=0] [elo1=5] [alpha=0.05] [beta=0.05] \
      [nodos=25000] [max_partidas=2000] [apertura_inicial=0]

`apertura_inicial` desplaza el tramo del banco que se usa. Sirve para dos
cosas: (a) partir el mismo test en varios trabajadores paralelos sobre tramos
DISJUNTOS (sumar los W/D/L despues es legitimo porque no comparten ni una
apertura), y (b) confirmar un resultado sobre aperturas que NO se usaron en la
medicion de tanteo, para que el veredicto no dependa del subconjunto que ya
salio favorable.

  python3 medir/sprt_diverso.py --smoke-test     (autochequeo, sin motores)


POR QUE EXISTE ESTE ARNES
-------------------------
El arnes historico del proyecto (`sprt_real.py`) tenia un bug grave:

    OPENINGS = [ ...20 aperturas fijas... ]
    opening = OPENINGS[(index // 2) % len(OPENINGS)]
    candidate_white = index % 2 == 0

o sea, el ciclo completo son 40 partidas. Los dos motores son DETERMINISTAS a
nodos fijos y el arnes manda `ucinewgame` (que limpia TT e historia) antes de
cada partida: a partir de la partida 41 se repite EXACTAMENTE la misma
secuencia de resultados. El LLR sigue creciendo solo porque cuenta los mismos
40 resultados una y otra vez, asi que SIEMPRE termina cruzando un limite.
Cualquier veredicto con N > 40 de ese arnes es un artefacto, no evidencia.

Este arnes arregla las tres cosas que hacian falta:

  1. BANCO GRANDE Y BALANCEADO: >1000 aperturas distintas sacadas del libro
     Polyglot del repo con semilla fija, filtradas para que ninguna arranque
     ya decidida (ver `medir/generar_aperturas.py`). N aperturas x 2 colores
     = 2N partidas realmente independientes; el arnes recorta max_partidas a
     ese tope y avisa, en vez de repetir en silencio.

  2. DETECCION DE DUPLICADOS EN VIVO: se hashea la secuencia de jugadas de
     cada partida terminada. Cada linea imprime `distintas=N/M`. Si la
     fraccion de partidas distintas cae por debajo de 60% (con al menos 40
     partidas jugadas, para no disparar con ruido inicial), el arnes ABORTA:
     eso significa que el banco se agoto o que algo volvio a hacer
     deterministas las partidas, y seguir contando seria inflar el LLR igual
     que el arnes viejo.

  3. CHECKPOINT / REANUDACION: estado en `state.json` (incluye los hashes de
     partida, para que el conteo de duplicados sobreviva a una reanudacion) y
     firma sha256 de binarios+pesos+parametros, de modo que reanudar con otro
     binario aborta en vez de mezclar evidencia.

FORMULA DE LLR
--------------
GSPRT sobre el "elo normalizado", igual que fishtest (el framework de testing
de Stockfish); ver https://www.chessprogramming.org/SPRT y
fishtest/stats/LLRcalc.py (metodo de Michel Van den Bergh):

    s0, s1  = score esperado de elo0 y elo1 (formula logistica)
    LLR     = N * (s1 - s0) * (2*score_medio - s0 - s1) / (2 * varianza)

Limites de Wald:
    LA = ln(beta / (1 - alpha))      (acepta H0)
    LB = ln((1 - beta) / alpha)      (acepta H1)
"""

from __future__ import annotations

import atexit
import datetime
import hashlib
import json
import math
import os
import pathlib
import signal
import sys
import time

import chess
import chess.engine
import chess.pgn

RAIZ = pathlib.Path(__file__).resolve().parent.parent
BANCO = pathlib.Path(
    os.environ.get("MITTENS_BANCO_APERTURAS")
    or (pathlib.Path(__file__).resolve().parent / "aperturas.txt")
)
MAX_PLIES = 300

# Fraccion minima de partidas DISTINTAS. Por debajo de esto el arnes aborta:
# significa que se volvio a caer en el bug del arnes viejo (partidas repetidas
# inflando el LLR).
FRACCION_DISTINTAS_MIN = 0.60
PARTIDAS_ANTES_DE_VIGILAR = 40


def cargar_aperturas() -> list[str]:
    if not BANCO.exists():
        raise SystemExit(
            f"Falta el banco de aperturas {BANCO}.\n"
            "Generarlo con: python3 medir/generar_aperturas.py "
            "medir/aperturas.txt 1200 <MOTOR> <PESOS>"
        )
    fens = [l.strip() for l in BANCO.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(fens) < 200:
        raise SystemExit(f"{BANCO} tiene solo {len(fens)} aperturas; hacen falta >= 200.")
    if len(set(fens)) != len(fens):
        raise SystemExit(f"{BANCO} tiene FEN repetidos: el banco no sirve para medir.")
    return fens


_MOTORES_VIVOS: list[chess.engine.SimpleEngine] = []


def cerrar_motores() -> None:
    while _MOTORES_VIVOS:
        motor = _MOTORES_VIVOS.pop()
        try:
            motor.quit()
        except Exception:
            pass


def manejar_señal(signum: int, _frame: object) -> None:
    cerrar_motores()
    raise SystemExit(128 + signum)


for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_s, manejar_señal)

atexit.register(cerrar_motores)


# ---------------------------------------------------------------------------
# GSPRT
# ---------------------------------------------------------------------------

def elo_a_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def calcular_llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    n = wins + draws + losses
    if n == 0:
        return 0.0
    score = (wins + 0.5 * draws) / n
    varianza = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2
    ) / n
    if varianza < 1e-9:
        if n < 8:
            return 0.0
        varianza = 1.0 / n
    s0 = elo_a_score(elo0)
    s1 = elo_a_score(elo1)
    return n * (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * varianza)


def limites_wald(alpha: float, beta: float) -> tuple[float, float]:
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def elo_y_error(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Elo estimado e intervalo de confianza al 95% (en Elo), estilo cutechess."""
    n = wins + draws + losses
    if n == 0:
        return 0.0, float("inf")
    score = (wins + 0.5 * draws) / n
    if score <= 0.0 or score >= 1.0:
        return (float("inf") if score >= 1.0 else float("-inf")), float("inf")
    varianza = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2
    ) / n
    sigma = math.sqrt(varianza / n)
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    lo = max(1e-9, score - 1.96 * sigma)
    hi = min(1.0 - 1e-9, score + 1.96 * sigma)
    elo_lo = -400.0 * math.log10(1.0 / lo - 1.0)
    elo_hi = -400.0 * math.log10(1.0 / hi - 1.0)
    return elo, (elo_hi - elo_lo) / 2.0


# ---------------------------------------------------------------------------
# Autochequeo sin motores
# ---------------------------------------------------------------------------

def smoke_test() -> str:
    import random

    lineas = []
    elo0, elo1, alpha, beta = 0.0, 5.0, 0.05, 0.05
    la, lb = limites_wald(alpha, beta)
    lineas.append(f"Limites Wald: LA={la:.4f}, LB={lb:.4f}")

    w = d = l = 0
    serie = []
    for _ in range(200):
        w += 1
        serie.append(calcular_llr(w, d, l, elo0, elo1))
    assert serie[-1] > lb
    lineas.append(f"Puras victorias: LLR final {serie[-1]:.2f} (cruza LB) OK")

    w = d = l = 0
    for _ in range(200):
        l += 1
    assert calcular_llr(w, d, l, elo0, elo1) < la
    lineas.append("Puras derrotas: cruza LA OK")

    rng = random.Random(12345)
    w = d = l = 0
    llr_40 = None
    for i in range(1, 40001):
        r = rng.random()
        if r < 0.30:
            d += 1
        elif r < 0.65:
            w += 1
        else:
            l += 1
        if i == 40:
            llr_40 = calcular_llr(w, d, l, elo0, elo1)
    assert la < llr_40 < lb, "40 partidas NO deberian decidir a 0 elo real"
    lineas.append(f"Candidato a ~0 elo: LLR a las 40 partidas = {llr_40:.4f} (sin decidir) OK")

    fens = cargar_aperturas()
    lineas.append(f"Banco de aperturas: {len(fens)} FEN distintos en {BANCO.name}")
    assert len(set(fens)) == len(fens)

    # El bug del arnes viejo, reproducido: con 20 aperturas el indice de
    # apertura se repite a partir de la partida 41.
    viejo = [(i // 2) % 20 for i in range(120)]
    assert viejo[:40] == viejo[40:80] == viejo[80:120]
    lineas.append(
        "Reproducido el bug del arnes viejo: con 20 aperturas las partidas "
        "41-80 y 81-120 usan EXACTAMENTE la misma secuencia que las 1-40."
    )
    nuevo = [(i // 2) % len(fens) for i in range(120)]
    assert len(set(nuevo)) == 60
    lineas.append(f"Con este banco, las primeras 120 partidas usan 60 aperturas distintas.")

    lineas.append("AUTOCHEQUEO: OK.")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Motor de partidas
# ---------------------------------------------------------------------------

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def guardar_estado(path: pathlib.Path, estado: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def configurar(motor: chess.engine.SimpleEngine, pesos: pathlib.Path) -> None:
    pedido = {"Threads": 1, "Hash": 128, "NNUEPath": str(pesos), "UseNNUE": True}
    motor.configure({k: v for k, v in pedido.items() if k in motor.options})


def jugar_partida(
    cand: chess.engine.SimpleEngine,
    base: chess.engine.SimpleEngine,
    cand_blancas: bool,
    nodos: int,
    apertura: str,
    id_partida: object,
) -> tuple[float, chess.pgn.Game, str]:
    tablero = chess.Board(apertura)
    limite = chess.engine.Limit(nodes=nodos)
    jugadas: list[str] = []
    while not tablero.is_game_over(claim_draw=True) and tablero.ply() < MAX_PLIES:
        turno_cand = (tablero.turn == chess.WHITE) == cand_blancas
        res = (cand if turno_cand else base).play(tablero, limite, game=id_partida)
        tablero.push(res.move)
        jugadas.append(res.move.uci())

    final = tablero.outcome(claim_draw=True)
    if final is None or final.winner is None:
        puntos = 0.5
    else:
        puntos = float((final.winner == chess.WHITE) == cand_blancas)
    # Huella de la partida: apertura + secuencia completa de jugadas. Dos
    # partidas con la misma huella son la MISMA partida (el bug del arnes
    # viejo), no dos observaciones independientes.
    huella = hashlib.sha256((apertura + "|" + " ".join(jugadas)).encode()).hexdigest()[:16]
    return puntos, chess.pgn.Game.from_board(tablero), huella


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--smoke-test":
        print(smoke_test())
        return
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)

    aperturas = cargar_aperturas()

    cand_bin = pathlib.Path(sys.argv[1]).resolve()
    cand_pesos = pathlib.Path(sys.argv[2]).resolve()
    base_bin = pathlib.Path(sys.argv[3]).resolve()
    base_pesos = pathlib.Path(sys.argv[4]).resolve()
    nombre = sys.argv[5]
    elo0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    elo1 = float(sys.argv[7]) if len(sys.argv) > 7 else 5.0
    alpha = float(sys.argv[8]) if len(sys.argv) > 8 else 0.05
    beta = float(sys.argv[9]) if len(sys.argv) > 9 else 0.05
    nodos = int(sys.argv[10]) if len(sys.argv) > 10 else 25_000
    max_partidas = int(sys.argv[11]) if len(sys.argv) > 11 else 2_000
    apertura_inicial = int(sys.argv[12]) if len(sys.argv) > 12 else 0
    if not 0 <= apertura_inicial < len(aperturas):
        raise SystemExit(
            f"apertura_inicial={apertura_inicial} fuera del banco (0..{len(aperturas) - 1})"
        )

    for p in (cand_bin, cand_pesos, base_bin, base_pesos):
        if not p.exists():
            raise SystemExit(f"No existe: {p}")

    # El banco tiene `len(aperturas)` posiciones x 2 colores. Mas partidas que
    # eso volverian a repetir; se avisa y se recorta.
    tope_independiente = 2 * (len(aperturas) - apertura_inicial)
    if max_partidas > tope_independiente:
        print(
            f"AVISO: max_partidas={max_partidas} supera las {tope_independiente} "
            f"partidas independientes que quedan en el banco desde la apertura "
            f"{apertura_inicial} ({len(aperturas) - apertura_inicial} aperturas x 2 "
            f"colores). Se recorta a {tope_independiente}.",
            flush=True,
        )
        max_partidas = tope_independiente

    salida = RAIZ / "results_sprt" / nombre
    salida.mkdir(parents=True, exist_ok=True)
    pgn_path = salida / "games.pgn"
    estado_path = salida / "state.json"
    veredicto_path = salida / "veredicto.txt"

    la, lb = limites_wald(alpha, beta)

    firma = {
        "cand_sha256": sha256(cand_bin),
        "cand_pesos_sha256": sha256(cand_pesos),
        "base_sha256": sha256(base_bin),
        "base_pesos_sha256": sha256(base_pesos),
        "elo0": elo0,
        "elo1": elo1,
        "alpha": alpha,
        "beta": beta,
        "nodos": nodos,
        "aperturas_sha256": hashlib.sha256("\n".join(aperturas).encode()).hexdigest(),
        "apertura_inicial": apertura_inicial,
    }

    if estado_path.exists():
        estado = json.loads(estado_path.read_text(encoding="utf-8"))
        if estado.get("firma") != firma:
            raise SystemExit(
                "ABORTADO: binarios, pesos o parametros cambiaron desde el checkpoint. "
                "Usa otro NOMBRE o borra el resultado viejo tras auditarlo."
            )
        wins = int(estado["wins"])
        draws = int(estado["draws"])
        losses = int(estado["losses"])
        hechas = int(estado["hechas"])
        huellas = set(estado.get("huellas", []))
        if hechas and not pgn_path.exists():
            raise SystemExit("Hay checkpoint pero falta games.pgn; no se reanuda a ciegas.")
    else:
        wins = draws = losses = hechas = 0
        huellas = set()
        estado = {
            "firma": firma,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "hechas": 0,
            "huellas": [],
        }
        guardar_estado(estado_path, estado)

    def escribir_veredicto(razon: str, llr: float) -> None:
        n = wins + draws + losses
        frac = (wins + 0.5 * draws) / n if n else 0.0
        elo, err = elo_y_error(wins, draws, losses)
        texto = (
            f"{nombre}\n"
            f"Candidato : {cand_bin}\n"
            f"Baseline  : {base_bin}\n"
            f"Nodos/jugada: {nodos}, Threads=1 en ambos\n"
            f"Tramo del banco: aperturas {apertura_inicial}.."
            f"{apertura_inicial + (n + 1) // 2 - 1} de {len(aperturas)}\n"
            f"Partidas: {n} (+{wins} ={draws} -{losses}), score={frac:.2%}\n"
            f"Partidas DISTINTAS: {len(huellas)}/{n} "
            f"({(len(huellas) / n if n else 0):.1%})\n"
            f"Elo estimado: {elo:+.1f} +/- {err:.1f} (IC95%)\n"
            f"LLR final: {llr:.4f} (limites [{la:.4f}, {lb:.4f}])\n"
            f"H0: elo <= {elo0}, H1: elo >= {elo1}, alpha={alpha}, beta={beta}\n"
            f"Razon de corte: {razon}\n"
        )
        veredicto_path.write_text(texto, encoding="utf-8")
        print(texto, end="", flush=True)

    try:
        cand = chess.engine.SimpleEngine.popen_uci([str(cand_bin)])
        _MOTORES_VIVOS.append(cand)
        base = chess.engine.SimpleEngine.popen_uci([str(base_bin)])
        _MOTORES_VIVOS.append(base)
        configurar(cand, cand_pesos)
        configurar(base, base_pesos)
    except BaseException:
        cerrar_motores()
        raise

    razon = None
    llr = calcular_llr(wins, draws, losses, elo0, elo1)
    try:
        modo = "a" if hechas else "w"
        with pgn_path.open(modo, encoding="utf-8") as pgn:
            for indice in range(hechas, max_partidas):
                cand_blancas = indice % 2 == 0
                apertura = aperturas[apertura_inicial + (indice // 2)]
                t0 = time.monotonic()
                puntos, partida, huella = jugar_partida(
                    cand, base, cand_blancas, nodos, apertura, (nombre, indice)
                )
                partida.headers["Event"] = f"SPRT {nombre}"
                partida.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
                partida.headers["White"] = nombre if cand_blancas else "baseline"
                partida.headers["Black"] = "baseline" if cand_blancas else nombre
                partida.headers["FEN"] = apertura
                partida.headers["SetUp"] = "1"
                print(partida, file=pgn, end="\n\n")
                pgn.flush()

                huellas.add(huella)
                if puntos == 1.0:
                    wins += 1
                elif puntos == 0.5:
                    draws += 1
                else:
                    losses += 1
                hechas = indice + 1
                llr = calcular_llr(wins, draws, losses, elo0, elo1)
                frac_distintas = len(huellas) / hechas

                estado.update(
                    {
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "hechas": hechas,
                        "huellas": sorted(huellas),
                    }
                )
                guardar_estado(estado_path, estado)

                elo, err = elo_y_error(wins, draws, losses)
                print(
                    f"{hechas}: +{wins} ={draws} -{losses} "
                    f"({(wins + 0.5 * draws) / hechas:.1%}) "
                    f"distintas={len(huellas)}/{hechas} "
                    f"LLR={llr:.2f} [{la:.2f}, {lb:.2f}] "
                    f"elo={elo:+.1f}+/-{err:.1f} {time.monotonic() - t0:.1f}s",
                    flush=True,
                )

                if hechas >= PARTIDAS_ANTES_DE_VIGILAR and frac_distintas < FRACCION_DISTINTAS_MIN:
                    razon = (
                        f"ABORTADO POR DUPLICADOS: solo {len(huellas)}/{hechas} "
                        f"({frac_distintas:.1%}) partidas distintas, por debajo del "
                        f"minimo {FRACCION_DISTINTAS_MIN:.0%}. El LLR de aca en "
                        "adelante seria un artefacto (mismo bug que sprt_real.py)."
                    )
                    break

                if llr >= lb:
                    razon = (
                        f"LLR cruzo el limite superior ({llr:.4f} >= {lb:.4f}): "
                        f"acepta H1 (candidato >= {elo1} elo)"
                    )
                    break
                if llr <= la:
                    razon = (
                        f"LLR cruzo el limite inferior ({llr:.4f} <= {la:.4f}): "
                        f"acepta H0 (candidato <= {elo0} elo)"
                    )
                    break
            else:
                razon = f"Se alcanzo max_partidas={max_partidas} sin cruzar limites: AMBIGUO"
    finally:
        cerrar_motores()

    if razon is None:
        razon = f"Se alcanzo max_partidas={max_partidas} sin cruzar limites: AMBIGUO"
    escribir_veredicto(razon, llr)


if __name__ == "__main__":
    main()
