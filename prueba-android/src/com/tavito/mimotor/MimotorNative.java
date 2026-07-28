package com.tavito.mimotor;

/**
 * Declaracion Java equivalente a la clase Kotlin que usara la app.
 * Sirve para probar dentro del runtime real de Android (ART) que los
 * simbolos JNI Java_com_tavito_mimotor_MimotorNative_* se resuelven.
 */
public class MimotorNative {
    public static native long nativeNew();
    public static native void nativeEnviar(long handle, String comando);
    public static native String nativeLeerLinea(long handle);
    public static native String nativeLeerLineaEsperando(long handle, long timeoutMs);
    public static native void nativeLiberar(long handle);
}
