import chess, chess.polyglot, sys, json
BOOK='/Users/Tavito/mi-motor-rust-produccion/performance.bin'
r=chess.polyglot.open_reader(BOOK)
MAXD=40
lineas=[]  # (prob, san_list, uci_list, fen, profundidad, terminal?)
prof_hist={}
sys.setrecursionlimit(10000)
LIMITE=200000
cnt=0
def walk(board, prob, ucis, depth):
    global cnt
    cnt+=1
    if cnt>LIMITE: return
    ents=list(r.find_all(board))
    if not ents or depth>=MAXD:
        prof_hist[depth]=prof_hist.get(depth,0)+1
        lineas.append((prob, list(ucis), board.fen(), depth))
        return
    tot=sum(max(e.weight,1) for e in ents)
    for e in ents:
        p=prob*max(e.weight,1)/tot
        if p < 1e-5:   # poda de ramas irrelevantes
            continue
        board.push(e.move); ucis.append(e.move.uci())
        walk(board, p, ucis, depth+1)
        ucis.pop(); board.pop()
walk(chess.Board(), 1.0, [], 0)
print('nodos visitados',cnt,'hojas',len(lineas))
prof=[d for _,_,_,d in lineas]
import statistics
print('prof hoja: min %d max %d media %.1f mediana %d'%(min(prof),max(prof),statistics.mean(prof),statistics.median(prof)))
probs=sorted(prof_hist.items())
print('histograma prof:',probs)
lineas.sort(key=lambda x:-x[0])
print('prob total cubierta %.4f'%sum(l[0] for l in lineas))
json.dump([{'p':l[0],'ucis':l[1],'fen':l[2],'d':l[3]} for l in lineas[:400]], open('/Users/Tavito/mi-motor-rust-agentes/ronda4_libro/analisis_libro/lineas.json','w'))
for l in lineas[:15]:
    print('%.5f d=%d %s'%(l[0],l[3],' '.join(l[1])))
