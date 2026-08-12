# Comprueba que la clave Polyglot del motor coincide con la de python-chess:
# recorre lineas del libro y verifica que el motor devuelve SIEMPRE una jugada
# que python-chess tambien tiene en el libro para esa posicion.
import chess, chess.engine, chess.polyglot, random, sys
BOOK='/Users/Tavito/mi-motor-rust-produccion/performance.bin'
ENG=sys.argv[1]
r=chess.polyglot.open_reader(BOOK)
eng=chess.engine.SimpleEngine.popen_uci(ENG)
eng.configure({"Threads":1,"Hash":64,"BookPath":BOOK,"OwnBook":True})
random.seed(7)
ok=fallo=fuera=0
consultas=0
for partida in range(60):
    b=chess.Board()
    for ply in range(40):
        ents=list(r.find_all(b))
        if not ents: break
        esperadas={e.move.uci() for e in ents}
        res=eng.play(b, chess.engine.Limit(depth=1))
        consultas+=1
        if res.move.uci() in esperadas: ok+=1
        else:
            fallo+=1
            if fallo<6: print('MISMATCH', b.fen(), 'motor',res.move.uci(),'libro',esperadas)
        b.push(random.choice(ents).move)
print('consultas',consultas,'coinciden',ok,'discrepan',fallo)
# caso al paso y enroque explicitos
for fen in ["rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"]:
    b=chess.Board(fen)
    ents=list(r.find_all(b))
    print(fen[:40],'entradas libro:',[e.move.uci() for e in ents])
eng.quit()
