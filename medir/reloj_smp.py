#!/usr/bin/env python3
"""Cuanto RELOJ REAL consume una jugada segun el numero de hilos.

Con `go wtime/btime` el motor calcula un presupuesto objetivo y un techo
duro; el corte blando decide cuando parar. Si cada hilo de Lazy SMP corre su
propio corte blando y el resultado se espera con join() de TODOS, el tiempo
de pared es el MAXIMO de N decisiones -> el motor gasta mas reloj cuanto mas
hilos, sin buscar mas profundo.
"""
import subprocess, sys, re, time, statistics

BIN = sys.argv[1]
RELOJ = int(sys.argv[2]) if len(sys.argv) > 2 else 180000
HILOS = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['1','4','8'])]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2

POS = [
 "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
 "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2NP1NP1/PP3PBP/R1BQ1RK1 w - - 0 9",
 "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
 "rnbq1rk1/pp2bppp/4pn2/2pp4/2PP4/1P2PN2/PB1N1PPP/R2QKB1R w KQ - 0 8",
 "2rq1rk1/pp1bppbp/3p1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 12",
 "r2q1rk1/1b2bppp/p2ppn2/1p6/3NPP2/1BN5/PPP3PP/R1BQ1R1K w - - 0 12",
]

def run(nh, fen):
    p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(f"uci\nsetoption name Threads value {nh}\nucinewgame\nisready\n"); p.stdin.flush()
    while not p.stdout.readline().startswith("readyok"): pass
    p.stdin.write(f"position fen {fen}\n"); p.stdin.flush()
    t0 = time.time()
    p.stdin.write(f"go wtime {RELOJ} btime {RELOJ}\n"); p.stdin.flush()
    depth = 0; nodes = 0
    while True:
        line = p.stdout.readline()
        if not line: break
        m = re.search(r"info depth (\d+)", line)
        if m: depth = max(depth, int(m.group(1)))
        m = re.search(r"nodes (\d+)", line)
        if m: nodes = max(nodes, int(m.group(1)))
        if line.startswith("bestmove"): break
    el = time.time()-t0
    p.stdin.write("quit\n"); p.stdin.flush()
    try: p.wait(timeout=10)
    except Exception: p.kill()
    return el, depth, nodes

print(f"bin={BIN} reloj={RELOJ}ms (objetivo teorico ~{RELOJ/30/1000:.1f}s/jugada) reps={REPS}")
for nh in HILOS:
    ts = []; ds = []
    for fen in POS:
        for _ in range(REPS):
            el, d, n = run(nh, fen)
            ts.append(el); ds.append(d)
    print(f"{nh}h  t_media={statistics.mean(ts):5.2f}s  t_mediana={statistics.median(ts):5.2f}s  "
          f"t_max={max(ts):5.2f}s  prof_media={statistics.mean(ds):5.2f}")
    sys.stdout.flush()
