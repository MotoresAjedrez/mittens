#!/usr/bin/env python3
"""Gauntlet libro vs main: TODAS las partidas desde startpos, sin aperturas
externas -- el libro debe poder jugar. Alterna colores.

Uso: gauntlet_libro.py CAND_BIN MAIN_BIN [PARTIDAS] [MS]
"""
import chess, chess.engine, chess.pgn, sys, io, collections, math

CAND=sys.argv[1]; MAIN=sys.argv[2]
N=int(sys.argv[3]) if len(sys.argv)>3 else 160
MS=int(sys.argv[4]) if len(sys.argv)>4 else 150
MAX_PLIES=300
OPTS={"Threads":1,"Hash":64}

def abrir(p):
    e=chess.engine.SimpleEngine.popen_uci(p)
    e.configure(dict(OPTS))
    return e

cand=abrir(CAND); base=abrir(MAIN)
lim=chess.engine.Limit(time=MS/1000.0)
W=D=L=0
try:
    for i in range(N):
        cand_blancas = (i % 2 == 0)
        b=chess.Board()
        while not b.is_game_over(claim_draw=True) and b.ply()<MAX_PLIES:
            eng = cand if ((b.turn==chess.WHITE)==cand_blancas) else base
            try:
                r=eng.play(b, lim)
            except Exception as ex:
                print("ERROR motor:",ex); raise
            if r.move is None: break
            b.push(r.move)
        res=b.result(claim_draw=True)
        if res=="1-0": pt = 1 if cand_blancas else 0
        elif res=="0-1": pt = 0 if cand_blancas else 1
        else: pt=0.5
        if pt==1: W+=1
        elif pt==0.5: D+=1
        else: L+=1
        sc=(W+0.5*D)/(i+1)
        print(f"[{i+1}/{N}] {res} plies={b.ply()} cand={'B' if cand_blancas else 'N'}  W-D-L={W}-{D}-{L}  {sc*100:.1f}%", flush=True)
finally:
    cand.quit(); base.quit()
n=W+D+L; sc=(W+0.5*D)/n
# error estandar por resultado de partida
p=[W/n,D/n,L/n]; media=sc
var=sum(pi*(xi-media)**2 for pi,xi in zip(p,[1,0.5,0]))
se=math.sqrt(var/n)
print(f"\nFINAL W-D-L = {W}-{D}-{L}  ({n} partidas)  score {sc*100:.1f}%  +-{1.96*se*100:.1f}% (95%)")
