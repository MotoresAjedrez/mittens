import chess, chess.engine, json, sys
ENG=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 100
DEPTH=int(sys.argv[3]) if len(sys.argv)>3 else 16
lineas=json.load(open('/Users/Tavito/mi-motor-rust-agentes/ronda4_libro/analisis_libro/lineas.json'))[:N]
eng=chess.engine.SimpleEngine.popen_uci(ENG)
eng.configure({"Threads":1,"Hash":128,"OwnBook":False})
res=[]
for i,l in enumerate(lineas):
    b=chess.Board(l['fen'])
    info=eng.analyse(b, chess.engine.Limit(depth=DEPTH))
    sc=info['score'].white().score(mate_score=100000)
    # el lado que ELIGIO la ultima jugada del libro es el que acaba de mover
    lado_ultimo = not b.turn   # True=blancas
    cp_para_ese_lado = sc if lado_ultimo==chess.WHITE else -sc
    res.append({'p':l['p'],'d':l['d'],'ucis':l['ucis'],'fen':l['fen'],
                'cp_blancas':sc,'cp_ultimo':cp_para_ese_lado,
                'lado_ultimo':'blancas' if lado_ultimo else 'negras'})
    print('%3d/%d p=%.4f d=%2d  cpB=%+5d  cpUltimo(%s)=%+5d'%(i+1,N,l['p'],l['d'],sc,res[-1]['lado_ultimo'],cp_para_ese_lado), flush=True)
eng.quit()
json.dump(res, open('/Users/Tavito/mi-motor-rust-agentes/ronda4_libro/analisis_libro/evals.json','w'), indent=1)
