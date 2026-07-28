#!/usr/bin/env python3
"""H2H Correction History contra pinlegal_v2 en produccion.

Uso:
  python3 h2h_current.py CANDIDATO NOMBRE [PARTIDAS=40] [MS=600]

Ambos motores reciben exactamente las mismas opciones y los mismos pesos
NNUE. No despliega ni modifica produccion.
"""

from __future__ import annotations

import datetime
import atexit
import hashlib
import io
import json
import pathlib
import signal
import sys
import time

import chess
import chess.engine
import chess.pgn


import os

ROOT = pathlib.Path(__file__).resolve().parent
# BASE/BASE_WEIGHTS apuntan al binario/pesos DESPLEGADOS (fuera de esta carpeta,
# en la instalacion real del bot) para comparar candidatos contra produccion.
# Configurables por entorno porque esa ruta es especifica de cada maquina.
BASE = pathlib.Path(
    os.environ.get(
        "MIMOTOR_BASE_BIN",
        str(pathlib.Path.home() / "mimotor-lichess-bot/engines/mimotor-tal-rust"),
    )
)
BASE_WEIGHTS = pathlib.Path(
    os.environ.get(
        "MIMOTOR_BASE_WEIGHTS",
        str(pathlib.Path.home() / "mimotor-lichess-bot/engines/nn_weights/pesos_flat_3m.bin"),
    )
)
ROOT_WEIGHTS = ROOT / "pesos_amenazas_prueba.bin"
MAX_PLIES = 300
EXPECTED_BASE_SHA256 = "205a580cd480eda70c0090e76f10aff36ca79fffcb5d388e546ebf4a7ed349b8"

# Veinte aperturas cortas y equilibradas. Cada una se juega dos veces con
# colores invertidos, de modo que un H2H de 40 partidas son 20 pares reales,
# no cuarenta repeticiones deterministas desde startpos.
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
    """Cierra todos los motores hijos, incluso durante una señal suave."""
    while _LIVE_ENGINES:
        engine = _LIVE_ENGINES.pop()
        try:
            engine.quit()
        except Exception:
            # Si el hijo ya murió, no dejar que el cierre genere otro fallo.
            pass


def handle_shutdown(signum: int, _frame: object) -> None:
    cleanup_engines()
    raise SystemExit(128 + signum)


for _signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_signal, handle_shutdown)


atexit.register(cleanup_engines)


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
        "Personalidad": "tal",
        "NNUEPath": str(weights),
        "UseNNUE": True,
    }
    supported = {key: value for key, value in requested.items() if key in engine.options}
    engine.configure(supported)


def play_game(
    candidate: chess.engine.SimpleEngine,
    baseline: chess.engine.SimpleEngine,
    candidate_white: bool,
    movetime_ms: int,
    opening: str,
    game_id: object,
) -> tuple[float, chess.pgn.Game]:
    board = opening_board(opening)
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)
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
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    candidate_path = pathlib.Path(sys.argv[1]).resolve()
    name = sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    movetime_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 600

    for path in (candidate_path, BASE, BASE_WEIGHTS, ROOT_WEIGHTS):
        if not path.exists():
            raise SystemExit(f"No existe: {path}")

    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / "games.pgn"
    state_path = out_dir / "state.json"

    base_sha = sha256(BASE)
    if base_sha != EXPECTED_BASE_SHA256:
        raise SystemExit(
            "ABORTADO: produccion cambio desde que se aislaron las candidatas. "
            f"Esperado {EXPECTED_BASE_SHA256}, actual {base_sha}."
        )
    signature = {
        "base_sha256": base_sha,
        "candidate_sha256": sha256(candidate_path),
        "candidate_weights_sha256": sha256(ROOT_WEIGHTS),
        "base_weights_sha256": sha256(BASE_WEIGHTS),
        "games": games,
        "movetime_ms": movetime_ms,
        "openings_sha256": hashlib.sha256("\n".join(OPENINGS).encode()).hexdigest(),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise SystemExit(
                "ABORTADO: el binario, pesos o protocolo cambiaron desde el checkpoint. "
                "Usa otro nombre de candidata o elimina el resultado solo tras auditarlo."
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

    try:
        cand = chess.engine.SimpleEngine.popen_uci([str(candidate_path)])
        _LIVE_ENGINES.append(cand)
        base = chess.engine.SimpleEngine.popen_uci([str(BASE)])
        _LIVE_ENGINES.append(base)
        configure(cand, ROOT_WEIGHTS)
        configure(base, BASE_WEIGHTS)
    except BaseException:
        cleanup_engines()
        raise

    try:
        mode = "a" if completed else "w"
        with combined.open(mode, encoding="utf-8") as pgn:
            for index in range(completed, games):
                candidate_white = index % 2 == 0
                opening = OPENINGS[(index // 2) % len(OPENINGS)]
                started = time.monotonic()
                points, game = play_game(
                    cand,
                    base,
                    candidate_white,
                    movetime_ms,
                    opening,
                    (name, index),
                )
                game.headers["Event"] = f"{name} vs production-29b304"
                game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
                game.headers["White"] = name if candidate_white else "production-29b304"
                game.headers["Black"] = "production-29b304" if candidate_white else name
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
                state.update(
                    {
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "completed": completed,
                    }
                )
                save_state(state_path, state)
                fraction = (wins + 0.5 * draws) / (index + 1)
                print(
                    f"{index + 1:02d}/{games}: +{wins} ={draws} -{losses} "
                    f"({fraction:.1%}), {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    finally:
        cleanup_engines()

    fraction = (wins + 0.5 * draws) / games
    verdict = "CONFIRMADA" if fraction > 0.55 else "DESCARTAR" if fraction < 0.45 else "AMBIGUA"
    summary = f"{name}: +{wins} ={draws} -{losses}, {fraction:.1%}, {verdict}\n"
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
