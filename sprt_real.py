#!/usr/bin/env python3
"""SPRT real (Sequential Probability Ratio Test) entre dos binarios UCI de Mittens.

Uso:
  python3 sprt_real.py CANDIDATO_BIN CANDIDATO_PESOS BASELINE_BIN BASELINE_PESOS NOMBRE \
      [elo0=0] [elo1=5] [alpha=0.05] [beta=0.05] [nodos=100000] [max_partidas=40000]

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
import io
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

# Mismo banco de 20 aperturas que h2h.py (cada una se juega con colores
# invertidos), para no inventar un banco nuevo ni depender de rutas de h2h.py.
OPENINGS = [
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

    lines.append("PRUEBA DE HUMO: OK, todos los chequeos pasaron.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Motor de partidas (mismo patron que h2h.py).
# ---------------------------------------------------------------------------

def opening_board(pgn_line: str) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn_line))
    if game is None:
        raise ValueError(f"No se pudo leer apertura: {pgn_line}")
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


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

    for path in (candidate_path, candidate_weights, baseline_path, baseline_weights):
        if not path.exists():
            raise SystemExit(f"No existe: {path}")

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
        "openings_sha256": hashlib.sha256("\n".join(OPENINGS).encode()).hexdigest(),
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
    llr = compute_llr(wins, draws, losses, elo0, elo1)
    try:
        mode = "a" if completed else "w"
        with combined.open(mode, encoding="utf-8") as pgn:
            for index in range(completed, max_games):
                candidate_white = index % 2 == 0
                opening = OPENINGS[(index // 2) % len(OPENINGS)]
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
