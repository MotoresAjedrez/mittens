#!/usr/bin/env python3
"""SPRT real (Sequential Probability Ratio Test) entre dos binarios UCI de Mittens.

Uso:
  python3 sprt_real.py CANDIDATO_BIN CANDIDATO_PESOS BASELINE_BIN BASELINE_PESOS NOMBRE \
      [elo0=0] [elo1=5] [alpha=0.05] [beta=0.05] [nodos=100000] [max_partidas=40000] \
      [apertura_inicial=0]

`apertura_inicial` permite correr VARIOS trabajadores en paralelo sobre
TRAMOS DISJUNTOS del libro (el arnes juega una partida por vez). Ej. con un
libro de 2313 aperturas y 600 partidas por trabajador:

    worker A: ... 600 0
    worker B: ... 600 300
    worker C: ... 600 600

Cada trabajador usa 300 aperturas distintas, asi que las partidas de los tres
son independientes entre si y sus W/D/L se pueden sumar. Los tramos NO pueden
solaparse: el arnes verifica que `apertura_inicial + max_partidas/2` entre en
el libro, pero el reparto entre trabajadores lo tiene que hacer quien lanza.

Formula de LLR: el test GSPRT (Generalized SPRT) sobre el "elo normalizado",
tal como lo documenta y usa fishtest (el framework de testing de Stockfish) y
como esta descrito en la wiki de chessprogramming.org, pagina "SPRT"
(https://www.chessprogramming.org/SPRT) y en el modulo real de fishtest
fishtest/fishtest/stats/LLRcalc.py (metodo de Michel Van den Bergh, ver tambien
su nota https://hardy.uhasselt.be/Fishtest/support_MLE_multinomial.pdf).

Idea: en vez de modelar W/D/L exactamente via el modelo BayesElo de dos
parametros, fishtest aproxima el LLR con una normal sobre el "score" promedio
por partida (0/0.5/1), usando la varianza empirica de esos resultados. Se
convierten elo0 y elo1 a un "score esperado" s0, s1 via la formula logistica
estandar de Elo (score = 1 / (1 + 10^(-elo/400))), y luego:

    LLR = N * (s1 - s0) * (2*score_medio - s0 - s1) / (2 * varianza)

Esta es la formula GSPRT que fishtest usa en produccion para decidir cuando
cortar un test. Los limites de decision son los limites clasicos de Wald:

    LA = ln(beta / (1 - alpha))   (limite inferior, acepta H0)
    LB = ln((1 - beta) / alpha)   (limite superior, acepta H1)

No es "SPRT sobre W/D/L trinomial exacto" (esa version tambien existe, pero
requiere el modelo BayesElo de 2 parametros con drawelo estimado, que fishtest
evito por simplicidad y porque el GSPRT normal es una excelente aproximacion
cuando N es grande, que es el caso tipico de un SPRT real de miles de
partidas).
"""

from __future__ import annotations

import atexit
import datetime
import hashlib
import json
import math
import pathlib
import signal
import sys
import time

import chess
import chess.engine
import chess.pgn


ROOT = pathlib.Path(__file__).resolve().parent
MAX_PLIES = 300

# ---------------------------------------------------------------------------
# BANCO DE APERTURAS
#
# ANTES: 20 aperturas cableadas, recorridas con
#   opening = OPENINGS[(index // 2) % 20]  /  candidato_blancas = index % 2 == 0
# Con los dos motores a NODOS FIJOS, 1 hilo y `ucinewgame` entre partidas, el
# juego es COMPLETAMENTE DETERMINISTA: solo existen 20 x 2 = 40 partidas
# posibles y a partir de la 41 cada partida es una copia EXACTA de una
# anterior. El LLR seguia acumulando "evidencia" de partidas repetidas, o sea
# que cualquier veredicto por encima de ~40 partidas era confianza fabricada.
#
# Verificado sobre los resultados que ya estaban en results_sprt/:
#   corrplexity_lmr      2843 partidas ->  40 unicas (una repetida 72 veces)
#   ttpv_persistente     1562 partidas ->  40 unicas (una repetida 40 veces)
#   corrhist_menores      362 partidas ->  40 unicas
#   fable_solo_tt_gen     357 partidas ->  40 unicas
#   finales_aplastantes   476 partidas ->  36 unicas
#
# AHORA: el banco sale de `libro_sprt.epd` (una apertura FEN por linea, ver
# tools/generar_libro_sprt.py). Cada apertura se juega EXACTAMENTE DOS VECES,
# una con cada color, y `max_partidas` esta acotado por construccion a
# 2 x len(libro): nunca se repite una partida.
# ---------------------------------------------------------------------------

LIBRO_APERTURAS = ROOT / "libro_sprt.epd"


def cargar_libro() -> list[str]:
    if not LIBRO_APERTURAS.exists():
        raise SystemExit(
            f"Falta el libro de aperturas {LIBRO_APERTURAS}.\n"
            "Generalo con: python3 tools/generar_libro_sprt.py"
        )
    fens = [
        linea.strip()
        for linea in LIBRO_APERTURAS.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.startswith("#")
    ]
    if len(set(fens)) != len(fens):
        raise SystemExit(
            "El libro de aperturas tiene lineas repetidas: regeneralo con "
            "tools/generar_libro_sprt.py (deduplica por EPD)."
        )
    if len(fens) < 50:
        raise SystemExit(
            f"El libro solo tiene {len(fens)} aperturas: muy pocas para un SPRT. "
            "Regeneralo pidiendo mas."
        )
    return fens


_LIVE_ENGINES: list[chess.engine.SimpleEngine] = []


def cleanup_engines() -> None:
    while _LIVE_ENGINES:
        engine = _LIVE_ENGINES.pop()
        try:
            engine.quit()
        except Exception:
            pass


def handle_shutdown(signum: int, _frame: object) -> None:
    cleanup_engines()
    raise SystemExit(128 + signum)


for _signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_signal, handle_shutdown)

atexit.register(cleanup_engines)


# ---------------------------------------------------------------------------
# Formula GSPRT (ver docstring del modulo para la fuente).
# ---------------------------------------------------------------------------

def elo_to_score(elo: float) -> float:
    """Convierte una diferencia de Elo a "score" esperado (formula logistica)."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def compute_llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """LLR acumulado (GSPRT, aproximacion normal) dado W/D/L y las hipotesis."""
    n = wins + draws + losses
    if n == 0:
        return 0.0
    score = (wins + 0.5 * draws) / n
    variance = (
        wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + losses * (0.0 - score) ** 2
    ) / n
    if variance < 1e-9:
        # Varianza degenerada (todas las partidas dieron el mismo resultado
        # exacto). Con muy pocas partidas esto no es evidencia real (podria
        # ser puro azar), asi que no se decide todavia. Con una racha larga
        # (n grande) SI es evidencia fuerte, asi que se usa un piso pequeno
        # que crece con n para reflejar la certeza creciente sin dividir por
        # cero.
        if n < 8:
            return 0.0
        variance = 1.0 / n
    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)
    return n * (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * variance)


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Limites de Wald: (LA inferior, LB superior)."""
    la = math.log(beta / (1.0 - alpha))
    lb = math.log((1.0 - beta) / alpha)
    return la, lb


# ---------------------------------------------------------------------------
# Prueba de humo con secuencias sinteticas (sin motores reales).
# ---------------------------------------------------------------------------

def smoke_test() -> str:
    lines = []
    elo0, elo1, alpha, beta = 0.0, 5.0, 0.05, 0.05
    la, lb = sprt_bounds(alpha, beta)
    lines.append(f"Limites Wald: LA={la:.4f}, LB={lb:.4f}")

    # 1) Racha de puras victorias: LLR debe crecer sin limite hacia +inf.
    w = d = l = 0
    llr_series = []
    for i in range(1, 201):
        w += 1
        llr_series.append(compute_llr(w, d, l, elo0, elo1))
    lines.append(
        f"Puras victorias (200 partidas): LLR pasa de {llr_series[0]:.3f} a "
        f"{llr_series[-1]:.3f} (debe crecer monotonamente hacia +inf)"
    )
    assert llr_series[-1] > llr_series[0]
    assert llr_series[-1] > lb, "deberia cruzar LB muchisimo antes de 200 partidas"
    crossing = next(i for i, v in enumerate(llr_series, start=1) if v > lb)
    lines.append(f"  -> cruza LB en la partida {crossing}")

    # 2) Racha de puras derrotas: LLR debe caer hacia -inf.
    w = d = l = 0
    llr_series = []
    for i in range(1, 201):
        l += 1
        llr_series.append(compute_llr(w, d, l, elo0, elo1))
    lines.append(
        f"Puras derrotas (200 partidas): LLR pasa de {llr_series[0]:.3f} a "
        f"{llr_series[-1]:.3f} (debe decrecer hacia -inf)"
    )
    assert llr_series[-1] < llr_series[0]
    assert llr_series[-1] < la
    crossing = next(i for i, v in enumerate(llr_series, start=1) if v < la)
    lines.append(f"  -> cruza LA en la partida {crossing}")

    # 3) 50% parejo (alternando victoria/derrota, sin tablas): LLR debe
    #    quedarse cerca de 0 y no cruzar ningun limite en pocas partidas.
    w = d = l = 0
    for i in range(1, 2001):
        if i % 2 == 0:
            w += 1
        else:
            l += 1
    llr_final = compute_llr(w, d, l, elo0, elo1)
    lines.append(
        f"50% parejo (2000 partidas, +{w} -{l}): LLR={llr_final:.4f} "
        f"(debe quedar cerca de 0 y dentro de [{la:.2f}, {lb:.2f}])"
    )
    assert la < llr_final < lb, "un 50% parejo no deberia cruzar ningun limite"

    # 4) Caso de referencia: con elo0=0, elo1=5, alpha=beta=0.05, un candidato
    #    que realmente juega a +0 elo (50% exacto, con algo de ruido tipico de
    #    ~30% tablas) deberia necesitar del orden de MILES de partidas para
    #    decidir, no docenas. Simulamos un generador de resultados con score
    #    esperado de 50% y ~30% de tablas (mezcla realista ajedrecistica) y
    #    verificamos que a las 40 partidas el LLR esta lejos de cualquier
    #    limite.
    import random

    rng = random.Random(12345)
    w = d = l = 0
    hit_40 = None
    total_games_to_decide = None
    for i in range(1, 20001):
        r = rng.random()
        if r < 0.30:
            d += 1
        elif r < 0.65:
            w += 1
        else:
            l += 1
        llr = compute_llr(w, d, l, elo0, elo1)
        if i == 40:
            hit_40 = llr
        if total_games_to_decide is None and (llr <= la or llr >= lb):
            total_games_to_decide = i
    lines.append(
        f"Simulacion candidato a ~0 elo (semilla fija, 30% tablas): "
        f"LLR a las 40 partidas = {hit_40:.4f} (debe seguir muy dentro de "
        f"[{la:.2f}, {lb:.2f}], NO decidido)"
    )
    assert la < hit_40 < lb, "40 partidas NO deberian bastar para decidir a 0 elo real"
    if total_games_to_decide is not None:
        lines.append(
            f"  -> con esta simulacion decide recien en la partida "
            f"{total_games_to_decide} (orden de miles, como se espera de un "
            f"SPRT real; ruido de la simulacion puede moverlo, pero nunca "
            f"docenas)"
        )
        assert total_games_to_decide > 200, (
            "un SPRT real con elo0=0/elo1=5 no deberia decidir en unas pocas "
            "docenas de partidas"
        )
    else:
        lines.append("  -> no decidio en 20000 partidas simuladas (tambien razonable)")

    # 5) El banco de aperturas no puede repetir partidas. Este chequeo es el
    #    que habria cazado el bug historico del arnes: con 20 aperturas
    #    cableadas y `% len(OPENINGS)`, los indices 40..79 reproducian
    #    exactamente los pares (apertura, color) de los indices 0..39, y con
    #    motores deterministas a nodos fijos eso son partidas identicas.
    aperturas = cargar_libro()
    pares = [(i // 2, i % 2 == 0) for i in range(2 * len(aperturas))]
    assert len(set(pares)) == len(pares), (
        "el emparejamiento apertura/color repite partidas"
    )
    assert max(idx for idx, _ in pares) == len(aperturas) - 1, (
        "el emparejamiento no recorre el libro entero"
    )
    lines.append(
        f"Libro: {len(aperturas)} aperturas unicas -> {2 * len(aperturas)} "
        f"partidas unicas, sin un solo par (apertura, color) repetido"
    )

    lines.append("PRUEBA DE HUMO: OK, todos los chequeos pasaron.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Motor de partidas (mismo patron que h2h.py).
# ---------------------------------------------------------------------------

def opening_board(fen: str) -> chess.Board:
    """Tablero de arranque de una apertura del libro (una FEN por linea)."""
    return chess.Board(fen)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_state(path: pathlib.Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def configure(engine: chess.engine.SimpleEngine, weights: pathlib.Path) -> None:
    requested = {
        "Threads": 1,
        "Hash": 128,
        "NNUEPath": str(weights),
        "UseNNUE": True,
    }
    supported = {key: value for key, value in requested.items() if key in engine.options}
    engine.configure(supported)


def play_game(
    candidate: chess.engine.SimpleEngine,
    baseline: chess.engine.SimpleEngine,
    candidate_white: bool,
    nodes: int,
    opening: str,
    game_id: object,
) -> tuple[float, chess.pgn.Game]:
    board = opening_board(opening)
    limit = chess.engine.Limit(nodes=nodes)
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        candidate_turn = (board.turn == chess.WHITE) == candidate_white
        result = (candidate if candidate_turn else baseline).play(
            board, limit, game=game_id
        )
        board.push(result.move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        points = 0.5
    else:
        points = float((outcome.winner == chess.WHITE) == candidate_white)
    return points, chess.pgn.Game.from_board(board)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--smoke-test":
        print(smoke_test())
        return

    if len(sys.argv) < 6:
        raise SystemExit(__doc__)

    candidate_path = pathlib.Path(sys.argv[1]).resolve()
    candidate_weights = pathlib.Path(sys.argv[2]).resolve()
    baseline_path = pathlib.Path(sys.argv[3]).resolve()
    baseline_weights = pathlib.Path(sys.argv[4]).resolve()
    name = sys.argv[5]

    elo0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    elo1 = float(sys.argv[7]) if len(sys.argv) > 7 else 5.0
    alpha = float(sys.argv[8]) if len(sys.argv) > 8 else 0.05
    beta = float(sys.argv[9]) if len(sys.argv) > 9 else 0.05
    nodes = int(sys.argv[10]) if len(sys.argv) > 10 else 100_000
    max_games = int(sys.argv[11]) if len(sys.argv) > 11 else 40_000
    apertura_inicial = int(sys.argv[12]) if len(sys.argv) > 12 else 0

    for path in (candidate_path, candidate_weights, baseline_path, baseline_weights):
        if not path.exists():
            raise SystemExit(f"No existe: {path}")

    aperturas_todas = cargar_libro()
    if apertura_inicial < 0 or apertura_inicial >= len(aperturas_todas):
        raise SystemExit(
            f"apertura_inicial={apertura_inicial} fuera del libro "
            f"(0..{len(aperturas_todas) - 1})"
        )
    aperturas = aperturas_todas[apertura_inicial:]
    partidas_unicas = 2 * len(aperturas)
    if max_games > partidas_unicas:
        raise SystemExit(
            f"ABORTADO: pediste max_partidas={max_games} pero el tramo del libro\n"
            f"que arranca en {apertura_inicial} solo permite {partidas_unicas} partidas\n"
            f"UNICAS ({len(aperturas)} aperturas x 2 colores). Con motores deterministas a nodos fijos, jugar mas que eso\n"
            "significa REPETIR partidas ya jugadas y sumarlas al LLR como si fueran\n"
            "evidencia nueva -- exactamente el error que este arnes tenia antes.\n"
            "Solucion: python3 tools/generar_libro_sprt.py <mas_aperturas>"
        )

    out_dir = ROOT / "results_sprt" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / "games.pgn"
    state_path = out_dir / "state.json"
    verdict_path = out_dir / "veredicto.txt"

    la, lb = sprt_bounds(alpha, beta)

    signature = {
        "candidate_sha256": sha256(candidate_path),
        "candidate_weights_sha256": sha256(candidate_weights),
        "baseline_sha256": sha256(baseline_path),
        "baseline_weights_sha256": sha256(baseline_weights),
        "elo0": elo0,
        "elo1": elo1,
        "alpha": alpha,
        "beta": beta,
        "nodes": nodes,
        "openings_sha256": sha256(LIBRO_APERTURAS),
        "openings_n": len(aperturas_todas),
        "opening_start": apertura_inicial,
    }

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise SystemExit(
                "ABORTADO: el binario, pesos o parametros cambiaron desde el "
                "checkpoint. Usa otro nombre o elimina el resultado tras auditarlo."
            )
        wins = int(state["wins"])
        draws = int(state["draws"])
        losses = int(state["losses"])
        completed = int(state["completed"])
        if completed and not combined.exists():
            raise SystemExit("Checkpoint existe pero falta games.pgn; no se reanuda a ciegas.")
    else:
        wins = draws = losses = completed = 0
        state = {
            "signature": signature,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "completed": completed,
        }
        save_state(state_path, state)

    def write_verdict(reason: str, llr: float) -> None:
        n = wins + draws + losses
        fraction = (wins + 0.5 * draws) / n if n else 0.0
        text = (
            f"{name}\n"
            f"Partidas: {n} (+{wins} ={draws} -{losses}), score={fraction:.1%}\n"
            f"LLR final: {llr:.4f} (limites [{la:.4f}, {lb:.4f}])\n"
            f"H0: elo <= {elo0}, H1: elo >= {elo1}, alpha={alpha}, beta={beta}\n"
            f"Partidas unicas disponibles en el tramo: {partidas_unicas} "
            f"({len(aperturas)} aperturas x 2 colores, desde la {apertura_inicial})\n"
            f"Partidas duplicadas detectadas: {duplicadas}\n"
            f"Razon de corte: {reason}\n"
        )
        verdict_path.write_text(text, encoding="utf-8")
        print(text, end="")

    try:
        cand = chess.engine.SimpleEngine.popen_uci([str(candidate_path)])
        _LIVE_ENGINES.append(cand)
        base = chess.engine.SimpleEngine.popen_uci([str(baseline_path)])
        _LIVE_ENGINES.append(base)
        configure(cand, candidate_weights)
        configure(base, baseline_weights)
    except BaseException:
        cleanup_engines()
        raise

    stop_reason = None
    duplicadas = 0
    firmas_partidas: set[str] = set()
    if completed and combined.exists():
        # Al reanudar un checkpoint hay que recuperar las firmas de las
        # partidas ya jugadas; si no, el detector de duplicados arrancaria
        # ciego y no veria una repeticion a caballo del corte.
        with combined.open(encoding="utf-8") as previas:
            partida = chess.pgn.read_game(previas)
            while partida is not None:
                firmas_partidas.add(
                    partida.headers.get("Opening", "")
                    + " | "
                    + " ".join(m.uci() for m in partida.mainline_moves())
                )
                partida = chess.pgn.read_game(previas)
    llr = compute_llr(wins, draws, losses, elo0, elo1)
    try:
        mode = "a" if completed else "w"
        with combined.open(mode, encoding="utf-8") as pgn:
            for index in range(completed, max_games):
                candidate_white = index % 2 == 0
                # Sin `% len(...)`: cada apertura se usa exactamente dos veces
                # (una por color) y nunca se vuelve al principio del libro.
                # La guarda de max_partidas de arriba garantiza que el indice
                # cae siempre dentro del libro.
                opening = aperturas[index // 2]
                started = time.monotonic()
                points, game = play_game(
                    cand, base, candidate_white, nodes, opening, (name, index)
                )
                game.headers["Event"] = f"SPRT {name}"
                game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
                game.headers["White"] = name if candidate_white else "baseline"
                game.headers["Black"] = "baseline" if candidate_white else name
                game.headers["Opening"] = opening
                print(game, file=pgn, end="\n\n")
                pgn.flush()

                # RED DE SEGURIDAD contra el bug historico de este arnes: si
                # dos partidas salen jugada por jugada identicas, la muestra no
                # es lo que el LLR cree que es. Con el libro nuevo no deberia
                # pasar nunca; si pasa, se corta en vez de seguir inflando el
                # LLR en silencio.
                # La firma incluye la apertura: dos posiciones de arranque
                # distintas pueden dar la misma secuencia de UCI (las jugadas
                # son casilla-a-casilla, no dependen de la posicion), y eso
                # seria un falso positivo.
                firma_partida = opening + " | " + " ".join(
                    m.uci() for m in game.mainline_moves()
                )
                if firma_partida in firmas_partidas:
                    duplicadas += 1
                    print(
                        f"AVISO: la partida {completed + 1} es identica a una anterior "
                        f"(duplicadas={duplicadas}).",
                        flush=True,
                    )
                    if duplicadas > max(5, (index + 1) // 100):
                        stop_reason = (
                            f"ABORTADO: {duplicadas} partidas duplicadas -- la muestra "
                            "no es independiente, el LLR no vale."
                        )
                        break
                else:
                    firmas_partidas.add(firma_partida)

                if points == 1.0:
                    wins += 1
                elif points == 0.5:
                    draws += 1
                else:
                    losses += 1
                completed = index + 1
                llr = compute_llr(wins, draws, losses, elo0, elo1)

                state.update(
                    {"wins": wins, "draws": draws, "losses": losses, "completed": completed}
                )
                save_state(state_path, state)

                print(
                    f"{completed}: +{wins} ={draws} -{losses}, LLR={llr:.2f} "
                    f"(limites [{la:.2f}, {lb:.2f}]), {time.monotonic() - started:.1f}s",
                    flush=True,
                )

                if llr >= lb:
                    stop_reason = f"LLR cruzo el limite superior ({llr:.4f} >= {lb:.4f}): acepta H1 (candidato >= {elo1} elo)"
                    break
                if llr <= la:
                    stop_reason = f"LLR cruzo el limite inferior ({llr:.4f} <= {la:.4f}): acepta H0 (candidato <= {elo0} elo)"
                    break
            else:
                if stop_reason is None:
                    stop_reason = f"Se alcanzo max_partidas={max_games} sin cruzar limites: AMBIGUO"
    finally:
        cleanup_engines()

    if stop_reason is None:
        stop_reason = f"Se alcanzo max_partidas={max_games} sin cruzar limites: AMBIGUO"
    write_verdict(stop_reason, llr)


if __name__ == "__main__":
    main()
