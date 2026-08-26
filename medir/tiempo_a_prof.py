#!/usr/bin/env python3
"""Escalado SMP medido como TIEMPO PARA ALCANZAR UNA PROFUNDIDAD FIJA.

Metrica limpia: 'go depth N' con Threads=1/4/8 y se mide el reloj de pared
hasta bestmove. Un Lazy SMP sano baja ese tiempo al subir hilos.
"""
import subprocess, sys, re, time

BIN = sys.argv[1]
PROF = int(sys.argv[2]) if len(sys.argv) > 2 else 14
HILOS = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['1','4','8'])]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3

POS = {
 "medio_abierto": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
 "medio_cerrado": "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2NP1NP1/PP3PBP/R1BQ1RK1 w - - 0 9",
 "tactico": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
 "final": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}

def run(nh, fen, prof):
    p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(f"uci\nsetoption name Threads value {nh}\nucinewgame\nisready\n"); p.stdin.flush()
    while not p.stdout.readline().startswith("readyok"): pass
    p.stdin.write(f"position fen {fen}\n"); p.stdin.flush()
    t0 = time.time()
    p.stdin.write(f"go depth {prof}\n"); p.stdin.flush()
    nodes = 0; mv = None
    while True:
        line = p.stdout.readline()
        if not line: break
        m = re.search(r"nodes (\d+)", line)
        if m: nodes = max(nodes, int(m.group(1)))
        if line.startswith("bestmove"):
            mv = line.split()[1]; break
    el = time.time()-t0
    p.stdin.write("quit\n"); p.stdin.flush()
    try: p.wait(timeout=10)
    except Exception: p.kill()
    return el, nodes, mv

print(f"bin={BIN} depth={PROF} reps={REPS}")
tot = {h: 0.0 for h in HILOS}
for nombre, fen in POS.items():
    base = None
    for nh in HILOS:
        rs = [run(nh, fen, PROF) for _ in range(REPS)]
        t = min(r[0] for r in rs)   # mejor de N: menos ruido de la maquina
        n = sorted(r[1] for r in rs)[len(rs)//2]
        tot[nh] += t
        if base is None: base = t
        print(f"{nombre:15s} {nh}h  t_min={t:6.2f}s  speedup={base/t:4.2f}x  nodos_med={n/1e6:7.2f}M  mv={rs[0][2]}")
    sys.stdout.flush()
print("--- total tiempo (suma de posiciones) ---")
b = tot[HILOS[0]]
for h in HILOS:
    print(f"{h}h  {tot[h]:6.2f}s  speedup={b/tot[h]:4.2f}x")
