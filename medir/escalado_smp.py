#!/usr/bin/env python3
"""Bench de escalado SMP: profundidad alcanzada, nodos y tiempo real a tiempo fijo."""
import subprocess, sys, re, time

BIN = sys.argv[1]
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
HILOS = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['1','4','8'])]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2

POS = {
 "medio_abierto": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
 "medio_cerrado": "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2NP1NP1/PP3PBP/R1BQ1RK1 w - - 0 9",
 "tactico": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
}

def run(nh, fen, verbose=False):
    p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(f"uci\nsetoption name Threads value {nh}\nucinewgame\nisready\n"); p.stdin.flush()
    while True:
        l = p.stdout.readline()
        if l.startswith("readyok"): break
    p.stdin.write(f"position fen {fen}\n"); p.stdin.flush()
    t0 = time.time()
    p.stdin.write(f"go movetime {MT}\n"); p.stdin.flush()
    depth = 0; nodes = 0
    while True:
        line = p.stdout.readline()
        if not line: break
        if verbose and line.startswith("info") and "string" not in line: print("   ", line.rstrip())
        m = re.search(r"info depth (\d+)", line)
        if m: depth = max(depth, int(m.group(1)))
        m = re.search(r"nodes (\d+)", line)
        if m: nodes = max(nodes, int(m.group(1)))
        if line.startswith("bestmove"): break
    el = time.time()-t0
    p.stdin.write("quit\n"); p.stdin.flush()
    try: p.wait(timeout=10)
    except Exception: p.kill()
    return depth, nodes, el

if __name__ == "__main__":
    print(f"bin={BIN} movetime={MT}ms reps={REPS}")
    for nombre, fen in POS.items():
        for nh in HILOS:
            rs = [run(nh, fen) for _ in range(REPS)]
            d = "/".join(str(r[0]) for r in rs)
            n = "/".join(f"{r[1]/1e6:.2f}M" for r in rs)
            t = "/".join(f"{r[2]:.2f}s" for r in rs)
            nps = sum(r[1] for r in rs)/sum(r[2] for r in rs)/1000
            print(f"{nombre:15s} {nh}h  depth={d:10s} nodos={n:16s} t={t:14s} NPS={nps:.0f}k")
        sys.stdout.flush()
