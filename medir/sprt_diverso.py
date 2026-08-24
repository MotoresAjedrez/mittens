#!/usr/bin/env python3
"""SPRT con banco de aperturas DIVERSO y detector de partidas duplicadas.

POR QUE EXISTE (bug encontrado el 24-ago-2026, invalida resultados viejos)
--------------------------------------------------------------------------
El harness anterior (sprt_real.py) juega con un banco FIJO de 20 aperturas,
cada una con los dos colores, y manda `ucinewgame` antes de cada partida.
Mittens es DETERMINISTA a nodos fijos y `ucinewgame` le tira la TT y el
Searcher enteros, asi que la partida numero 40 es EXACTAMENTE la partida
numero 0, jugada por jugada. El harness solo puede producir 40 partidas
distintas: todo lo que pase de ahi son repeticiones bit a bit.

Medido sobre los PGN que dejaron las corridas anteriores (md5 de la secuencia
de jugadas de cada partida):

  corrplexity_lmr ....... 2.843 partidas, 40 DISTINTAS (71,1x cada una)
  ttpv_persistente ...... 1.562 partidas, 40 DISTINTAS (39,0x)
  corrhist_menores_2ply ... 362 partidas, 40 DISTINTAS ( 9,1x)

Consecuencia: el tamano de muestra EFECTIVO de todos esos test es 40, no
miles. El LLR se calcula como si cada repeticion fuera informacion nueva, o
sea que esta inflado por el factor de repeticion (~71x en corrplexity). Los
dos candidatos que "cruzaron el limite superior" y se fusionaron a main
(corrplexity y ttpv) lo cruzaron con 40 partidas distintas de evidencia real.
Lo mismo vale para los rechazos.

QUE HACE ESTE HARNESS DISTINTO
------------------------------
1. Banco de aperturas GRANDE y determinista: parte de las mismas 20 lineas y
   las extiende con jugadas silenciosas elegidas por un RNG con semilla fija,
   filtrando con el motor baseline para quedarse solo con posiciones
   equilibradas (|score| <= umbral). El banco se guarda en JSON para que la
   corrida sea reproducible y para poder auditarlo.
2. DETECTOR DE DUPLICADOS incorporado: se guarda el md5 de la secuencia de
   jugadas de cada partida y la linea de progreso informa cuantas partidas
   DISTINTAS lleva. Si la fraccion de distintas cae por debajo de un piso, el
   harness ABORTA en vez de seguir inflando el LLR. Este fallo no puede volver
   a pasar en silencio.
3. La matematica del GSPRT es la misma que sprt_real.py (formula de fishtest);
   lo que cambia es de donde salen las partidas.

Uso:
  python3 sprt_diverso.py CAND CAND_PESOS BASE BASE_PESOS NOMBRE \
      [elo0=0] [elo1=5] [alpha=0.05] [beta=0.05] [nodos=50000] \
      [max_partidas=40000] [n_aperturas=400]
"""

from __future__ import annotations

import atexit
import datetime
import hashlib
import io
import json
import math
import pathlib
import random
import signal
import sys
import time

import chess
import chess.engine
import chess.pgn

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_PLIES = 300

# Semilla y parametros del banco: fijos, para que dos corridas del mismo
# tamano usen EXACTAMENTE el mismo banco.
SEMILLA = 20260824
PLIES_EXTRA = 4          # 2 jugadas de cada bando sobre la linea base
NODOS_FILTRO = 6000      # busqueda corta para juzgar el equilibrio
UMBRAL_CP = 100          # |score| maximo para aceptar una apertura
# Piso de diversidad: si menos del 60% de las partidas jugadas son distintas,
# algo volvio a colapsar el banco y no tiene sentido seguir.
PISO_DISTINTAS = 0.60
MIN_PARTIDAS_PARA_CHEQUEAR = 60

LINEAS_BASE = [
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
    "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6",
    "1. e4 c5 2. Nf3 d6 3. d4 cxd4",
    "1. e4 c5 2. Nf3 Nc6 3. d4 cxd4",
    "1. e4 c5 2. c3 d5 3. exd5 Qxd5",
    "1. e4 e6 2. d4 d5 3. Nc3 Nf6",
    "1. e4 c6 2. d4 d5 3. Nc3 dxe4",
    "1. e4 d5 2. exd5 Qxd5 3. Nc3 Qd8",
    "1. d4 d5 2. c4 e6 3. Nc3 Nf6",
    "1. d4 d5 2. c4 c6 3. Nf3 Nf6",
    "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7",
    "1. d4 Nf6 2. c4 e6 3. Nf3 b6",
    "1. d4 Nf6 2. c4 c5 3. d5 e6",
    "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6",
    "1. c4 c5 2. Nf3 Nf6 3. d4 cxd4",
    "1. Nf3 d5 2. g3 Nf6 3. Bg2 g6",
    "1. Nf3 Nf6 2. c4 g6 3. g3 Bg7",
    "1. g3 d5 2. Bg2 Nf6 3. Nf3 g6",
    "1. b3 e5 2. Bb2 Nc6 3. e3 Nf6",
    "1. f4 d5 2. Nf3 Nf6 3. e3 g6",
]

_LIVE: list[chess.engine.SimpleEngine] = []


def cerrar_motores() -> None:
    while _LIVE:
        try:
            _LIVE.pop().quit()
        except Exception:
            pass


def _shutdown(signum: int, _f: object) -> None:
    cerrar_motores()
    raise SystemExit(128 + signum)


for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_s, _shutdown)
atexit.register(cerrar_motores)


# --------------------------------------------------------------------------
# GSPRT (identico a sprt_real.py: formula de fishtest)
# --------------------------------------------------------------------------

def elo_a_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_a_elo(score: float) -> float:
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def llr(w: int, d: int, l: int, elo0: float, elo1: float) -> float:
    n = w + d + l
    if n == 0:
        return 0.0
    score = (w + 0.5 * d) / n
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * (0 - score) ** 2) / n
    if var < 1e-9:
        if n < 8:
            return 0.0
        var = 1.0 / n
    s0, s1 = elo_a_score(elo0), elo_a_score(elo1)
    return n * (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * var)


def limites(alpha: float, beta: float) -> tuple[float, float]:
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


# --------------------------------------------------------------------------
# Banco de aperturas
# --------------------------------------------------------------------------

def tablero_de_linea(pgn_linea: str) -> chess.Board:
    juego = chess.pgn.read_game(io.StringIO(pgn_linea))
    if juego is None:
        raise ValueError(pgn_linea)
    b = juego.board()
    for mv in juego.mainline_moves():
        b.push(mv)
    return b


def construir_banco(
    motor: chess.engine.SimpleEngine, cuantas: int, cache: pathlib.Path
) -> list[list[str]]:
    """Devuelve `cuantas` aperturas como listas de jugadas UCI desde startpos."""
    clave = {
        "cuantas": cuantas,
        "semilla": SEMILLA,
        "plies_extra": PLIES_EXTRA,
        "nodos_filtro": NODOS_FILTRO,
        "umbral_cp": UMBRAL_CP,
        "lineas": hashlib.sha256("\n".join(LINEAS_BASE).encode()).hexdigest(),
    }
    if cache.exists():
        guardado = json.loads(cache.read_text())
        if guardado.get("clave") == clave:
            return [list(a) for a in guardado["aperturas"]]

    rng = random.Random(SEMILLA)
    vistas: set[str] = set()
    banco: list[list[str]] = []
    intentos = 0
    limite_intentos = cuantas * 60
    while len(banco) < cuantas and intentos < limite_intentos:
        intentos += 1
        base = LINEAS_BASE[len(banco) % len(LINEAS_BASE)]
        b = tablero_de_linea(base)
        jugadas = [m.uci() for m in b.move_stack]
        ok = True
        for _ in range(PLIES_EXTRA):
            # Solo jugadas silenciosas: mantiene la posicion cerca del libro
            # y evita aperturas donde alguien ya regalo material.
            cands = [
                m
                for m in b.legal_moves
                if not b.is_capture(m) and m.promotion is None
            ]
            if not cands:
                ok = False
                break
            m = rng.choice(cands)
            b.push(m)
            jugadas.append(m.uci())
        if not ok or b.is_game_over() or b.is_check():
            continue
        firma = " ".join(jugadas)
        if firma in vistas:
            continue
        info = motor.analyse(b, chess.engine.Limit(nodes=NODOS_FILTRO))
        sc = info["score"].white()
        if sc.is_mate():
            continue
        if abs(sc.score()) > UMBRAL_CP:
            continue
        vistas.add(firma)
        banco.append(jugadas)
    if len(banco) < cuantas:
        print(
            f"AVISO: solo se consiguieron {len(banco)} aperturas de {cuantas} "
            f"pedidas en {intentos} intentos",
            flush=True,
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"clave": clave, "aperturas": banco}, indent=1))
    return banco


def jugar(
    cand: chess.engine.SimpleEngine,
    base: chess.engine.SimpleEngine,
    cand_blancas: bool,
    nodos: int,
    apertura: list[str],
    game_id: object,
) -> tuple[float, chess.pgn.Game]:
    b = chess.Board()
    for u in apertura:
        b.push(chess.Move.from_uci(u))
    lim = chess.engine.Limit(nodes=nodos)
    while not b.is_game_over(claim_draw=True) and b.ply() < MAX_PLIES:
        turno_cand = (b.turn == chess.WHITE) == cand_blancas
        r = (cand if turno_cand else base).play(b, lim, game=game_id)
        b.push(r.move)
    out = b.outcome(claim_draw=True)
    pts = 0.5 if (out is None or out.winner is None) else float(
        (out.winner == chess.WHITE) == cand_blancas
    )
    return pts, chess.pgn.Game.from_board(b)


def main() -> None:
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    cand_bin = pathlib.Path(sys.argv[1]).resolve()
    cand_w = pathlib.Path(sys.argv[2]).resolve()
    base_bin = pathlib.Path(sys.argv[3]).resolve()
    base_w = pathlib.Path(sys.argv[4]).resolve()
    nombre = sys.argv[5]
    elo0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    elo1 = float(sys.argv[7]) if len(sys.argv) > 7 else 5.0
    alpha = float(sys.argv[8]) if len(sys.argv) > 8 else 0.05
    beta = float(sys.argv[9]) if len(sys.argv) > 9 else 0.05
    nodos = int(sys.argv[10]) if len(sys.argv) > 10 else 50_000
    max_partidas = int(sys.argv[11]) if len(sys.argv) > 11 else 40_000
    n_aperturas = int(sys.argv[12]) if len(sys.argv) > 12 else 400

    for p in (cand_bin, cand_w, base_bin, base_w):
        if not p.exists():
            raise SystemExit(f"No existe: {p}")

    salida = ROOT / "results_sprt" / nombre
    salida.mkdir(parents=True, exist_ok=True)
    pgn_path = salida / "games.pgn"
    estado_path = salida / "state.json"
    la, lb = limites(alpha, beta)

    def sha256(p: pathlib.Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    firma_corrida = {
        "cand": sha256(cand_bin),
        "cand_pesos": sha256(cand_w),
        "base": sha256(base_bin),
        "base_pesos": sha256(base_w),
        "elo0": elo0,
        "elo1": elo1,
        "alpha": alpha,
        "beta": beta,
        "nodos": nodos,
        "n_aperturas": n_aperturas,
        "semilla": SEMILLA,
    }

    def opciones(e: chess.engine.SimpleEngine, w: pathlib.Path) -> None:
        pedido = {"Threads": 1, "Hash": 128, "NNUEPath": str(w), "UseNNUE": True}
        e.configure({k: v for k, v in pedido.items() if k in e.options})

    cand = chess.engine.SimpleEngine.popen_uci([str(cand_bin)])
    _LIVE.append(cand)
    base = chess.engine.SimpleEngine.popen_uci([str(base_bin)])
    _LIVE.append(base)
    opciones(cand, cand_w)
    opciones(base, base_w)

    banco = construir_banco(
        base, n_aperturas, ROOT / "medir" / f"aperturas_{n_aperturas}.json"
    )
    print(
        f"banco de aperturas: {len(banco)} distintas "
        f"(el harness viejo tenia 20 -> 40 partidas posibles)",
        flush=True,
    )

    # Reanudacion: mismo contrato que sprt_real.py -- si hay checkpoint y la
    # firma coincide (mismos binarios, pesos, parametros y banco), se continua;
    # si cambio algo, se aborta en vez de mezclar partidas de dos candidatos.
    w = d = l = hechas = 0
    firmas: set[str] = set()
    if estado_path.exists():
        guardado = json.loads(estado_path.read_text(encoding="utf-8"))
        if guardado.get("firma") != firma_corrida:
            raise SystemExit(
                "ABORTADO: el binario, los pesos o los parametros cambiaron "
                "desde el checkpoint. Usa otro nombre o borra el resultado."
            )
        w, d, l = guardado["wins"], guardado["draws"], guardado["losses"]
        hechas = guardado["completed"]
        firmas = set(guardado.get("firmas", []))
        if hechas and not pgn_path.exists():
            raise SystemExit("Hay checkpoint pero falta games.pgn; no se reanuda a ciegas.")

    def guardar_estado() -> None:
        tmp = estado_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "firma": firma_corrida,
                    "wins": w,
                    "draws": d,
                    "losses": l,
                    "completed": hechas,
                    "firmas": sorted(firmas),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        tmp.replace(estado_path)

    razon = None
    valor_llr = llr(w, d, l, elo0, elo1)
    try:
        with pgn_path.open("a" if hechas else "w", encoding="utf-8") as fh:
            for i in range(hechas, max_partidas):
                cand_blancas = i % 2 == 0
                apertura = banco[(i // 2) % len(banco)]
                t0 = time.monotonic()
                pts, juego = jugar(
                    cand, base, cand_blancas, nodos, apertura, (nombre, i)
                )
                juego.headers["Event"] = f"SPRT {nombre}"
                juego.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
                juego.headers["White"] = nombre if cand_blancas else "baseline"
                juego.headers["Black"] = "baseline" if cand_blancas else nombre
                print(juego, file=fh, end="\n\n")
                fh.flush()

                firmas.add(
                    hashlib.md5(
                        " ".join(m.uci() for m in juego.mainline_moves()).encode()
                    ).hexdigest()
                )
                if pts == 1.0:
                    w += 1
                elif pts == 0.5:
                    d += 1
                else:
                    l += 1
                hechas = i + 1
                n = w + d + l
                valor_llr = llr(w, d, l, elo0, elo1)
                guardar_estado()
                score = (w + 0.5 * d) / n
                print(
                    f"{n}: +{w} ={d} -{l}, score={100*score:.2f}%, "
                    f"Elo~{score_a_elo(score):+.1f}, LLR={valor_llr:.2f} "
                    f"[{la:.2f}, {lb:.2f}], distintas={len(firmas)}/{n}, "
                    f"{time.monotonic()-t0:.1f}s",
                    flush=True,
                )

                # TECHO DEL BANCO: con B aperturas y los dos colores hay 2*B
                # partidas distintas posibles; a partir de ahi se empiezan a
                # repetir igual que en el harness viejo, solo que mas tarde.
                # Se avisa UNA vez, al cruzarlo: el detector de abajo tarda
                # mucho mas en saltar porque mira la fraccion acumulada.
                if n == 2 * len(banco) + 1:
                    print(
                        f"AVISO: se agotaron las {2*len(banco)} partidas "
                        f"distintas que permite un banco de {len(banco)} "
                        f"aperturas. De aca en adelante se repiten: relanza con "
                        f"n_aperturas >= max_partidas/2 si necesitas mas.",
                        flush=True,
                    )

                # DETECTOR DE DUPLICADOS: el fallo que invalido los test viejos.
                if n >= MIN_PARTIDAS_PARA_CHEQUEAR and len(firmas) / n < PISO_DISTINTAS:
                    razon = (
                        f"ABORTADO: solo {len(firmas)} partidas distintas en {n} "
                        f"jugadas ({100*len(firmas)/n:.0f}% < "
                        f"{100*PISO_DISTINTAS:.0f}%). El banco de aperturas se "
                        f"agoto o los motores son deterministas: el LLR estaria "
                        f"inflado por las repeticiones."
                    )
                    break
                if valor_llr >= lb:
                    razon = f"LLR cruzo el limite superior: acepta H1 (>= {elo1} elo)"
                    break
                if valor_llr <= la:
                    razon = f"LLR cruzo el limite inferior: acepta H0 (<= {elo0} elo)"
                    break
            else:
                razon = f"max_partidas={max_partidas} sin cruzar limites: AMBIGUO"
    finally:
        cerrar_motores()

    n = w + d + l
    score = (w + 0.5 * d) / n if n else 0.0
    texto = (
        f"{nombre}\n"
        f"Partidas: {n} (+{w} ={d} -{l}), score={100*score:.2f}%, "
        f"Elo~{score_a_elo(score):+.1f}\n"
        f"Partidas DISTINTAS: {len(firmas)} de {n}\n"
        f"Aperturas del banco: {len(banco)}\n"
        f"LLR final: {valor_llr:.4f} (limites [{la:.4f}, {lb:.4f}])\n"
        f"H0: elo <= {elo0}, H1: elo >= {elo1}, alpha={alpha}, beta={beta}\n"
        f"Nodos fijos: {nodos}\n"
        f"Razon de corte: {razon}\n"
    )
    (salida / "veredicto.txt").write_text(texto, encoding="utf-8")
    print(texto, end="")


if __name__ == "__main__":
    main()
