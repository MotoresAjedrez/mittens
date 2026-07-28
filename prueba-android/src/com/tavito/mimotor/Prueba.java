package com.tavito.mimotor;

import java.io.File;

/**
 * Prueba de humo dentro del runtime real de Android (ART).
 *
 * OJO: el motor redirige el file descriptor 1 (stdout) del proceso hacia
 * su pipe interno, asi que TODO lo que imprimamos despues de nativeNew()
 * tiene que ir a System.err (fd 2), que queda intacto.
 */
public class Prueba {

    static StringBuilder registro = new StringBuilder();

    static void log(String s) {
        System.err.println("[PRUEBA] " + s);
        registro.append(s).append('\n');
    }

    public static void main(String[] args) throws Exception {
        String rutaLib = "/data/local/tmp/mimotor/libmimotor_core.so";
        String rutaPesos = "/data/local/tmp/mimotor/pesos_amenazas_prueba.bin";

        log("cargando " + rutaLib);
        System.load(rutaLib);
        log("System.load OK");

        long h = MimotorNative.nativeNew();
        log("nativeNew -> handle=" + h);
        if (h == 0) {
            log("FALLO: handle nulo");
            System.exit(2);
        }

        // ---- 1) handshake UCI ----
        MimotorNative.nativeEnviar(h, "uci");
        boolean vioId = false, vioUciok = false;
        for (int i = 0; i < 200; i++) {
            String l = MimotorNative.nativeLeerLineaEsperando(h, 3000);
            if (l == null) break;
            log("<< " + l);
            if (l.startsWith("id name MiMotor Tal")) vioId = true;
            if (l.equals("uciok")) { vioUciok = true; break; }
        }
        log("vioId=" + vioId + " vioUciok=" + vioUciok);

        // ---- 2) isready ----
        MimotorNative.nativeEnviar(h, "isready");
        boolean vioReady = false;
        for (int i = 0; i < 50; i++) {
            String l = MimotorNative.nativeLeerLineaEsperando(h, 3000);
            if (l == null) break;
            log("<< " + l);
            if (l.equals("readyok")) { vioReady = true; break; }
        }
        log("vioReady=" + vioReady);

        // ---- 3) lectura NO bloqueante: debe devolver null si no hay nada ----
        String vacio = MimotorNative.nativeLeerLinea(h);
        log("nativeLeerLinea sin datos pendientes -> " + vacio);

        // ---- 4) cargar pesos NNUE reales ----
        boolean hayPesos = new File(rutaPesos).exists();
        log("pesos presentes: " + hayPesos + " (" + rutaPesos + ")");
        if (hayPesos) {
            MimotorNative.nativeEnviar(h, "setoption name NNUEPath value " + rutaPesos);
            MimotorNative.nativeEnviar(h, "setoption name UseNNUE value true");
            MimotorNative.nativeEnviar(h, "isready");
            for (int i = 0; i < 80; i++) {
                String l = MimotorNative.nativeLeerLineaEsperando(h, 8000);
                if (l == null) break;
                log("<< " + l);
                if (l.equals("readyok")) break;
            }
        }

        // ---- 5) busqueda real ----
        MimotorNative.nativeEnviar(h, "position startpos");
        MimotorNative.nativeEnviar(h, "go movetime 1500");
        String bestmove = null;
        long t0 = System.currentTimeMillis();
        for (int i = 0; i < 2000; i++) {
            String l = MimotorNative.nativeLeerLineaEsperando(h, 10000);
            if (l == null) break;
            log("<< " + l);
            if (l.startsWith("bestmove")) { bestmove = l; break; }
        }
        long ms = System.currentTimeMillis() - t0;
        log("bestmove=" + bestmove + " en " + ms + " ms");

        // ---- 6) segunda busqueda desde otra posicion (con movimientos) ----
        MimotorNative.nativeEnviar(h, "position startpos moves e2e4 e7e5 g1f3");
        MimotorNative.nativeEnviar(h, "go movetime 1000");
        String bestmove2 = null;
        for (int i = 0; i < 2000; i++) {
            String l = MimotorNative.nativeLeerLineaEsperando(h, 10000);
            if (l == null) break;
            if (l.startsWith("bestmove")) { bestmove2 = l; log("<< " + l); break; }
        }
        log("bestmove2=" + bestmove2);

        // ---- 7) "stop" corta una busqueda infinita ----
        MimotorNative.nativeEnviar(h, "setoption name Threads value 4");
        MimotorNative.nativeEnviar(h, "position startpos moves e2e4 c7c5");
        MimotorNative.nativeEnviar(h, "go infinite");
        Thread.sleep(1200);
        MimotorNative.nativeEnviar(h, "stop");
        String bestmove3 = null;
        for (int i = 0; i < 3000; i++) {
            String l = MimotorNative.nativeLeerLineaEsperando(h, 10000);
            if (l == null) break;
            if (l.startsWith("bestmove")) { bestmove3 = l; log("<< " + l); break; }
        }
        log("stop+4 hilos -> bestmove3=" + bestmove3);

        boolean ok = bestmove3 != null && vioId && vioUciok && vioReady
                && bestmove != null && !bestmove.contains("(none)")
                && bestmove2 != null;
        log(ok ? "RESULTADO: TODO OK" : "RESULTADO: FALLO");
        System.err.flush();
        // Salir sin esperar a los hilos del motor.
        Runtime.getRuntime().halt(ok ? 0 : 1);
    }
}
