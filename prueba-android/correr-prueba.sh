#!/bin/bash
# Prueba de humo del puente JNI dentro del runtime real de Android (ART).
#
# Compila las clases Java, las convierte a DEX con d8, empuja el .so + los
# pesos NNUE al emulador y ejecuta la prueba con app_process (el mismo
# runtime ART que usa cualquier app, asi que la resolucion de los simbolos
# JNI Java_com_tavito_mimotor_MimotorNative_* es la de verdad).
set -e

AQUI="$(cd "$(dirname "$0")" && pwd)"
RAIZ="$(dirname "$AQUI")"
SDK="${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk}"
export PATH="$JAVA_HOME/bin:$SDK/platform-tools:$PATH"
D8="$SDK/build-tools/34.0.0/d8"

# ABI del dispositivo/emulador conectado -> target de Rust
ABI="$(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
case "$ABI" in
  arm64-v8a)   TARGET=aarch64-linux-android ;;
  armeabi-v7a) TARGET=armv7-linux-androideabi ;;
  x86_64)      TARGET=x86_64-linux-android ;;
  *) echo "ABI no soportada: $ABI"; exit 1 ;;
esac
echo "ABI del dispositivo: $ABI -> $TARGET"

rm -rf "$AQUI/build"; mkdir -p "$AQUI/build/clases"
javac --release 11 -d "$AQUI/build/clases" \
  "$AQUI/src/com/tavito/mimotor/MimotorNative.java" \
  "$AQUI/src/com/tavito/mimotor/Prueba.java" 2>&1 | grep -v "bootstrap class path" || true

"$D8" --min-api 24 --output "$AQUI/build" \
  $(find "$AQUI/build/clases" -name '*.class')

adb shell mkdir -p /data/local/tmp/mimotor
adb push "$RAIZ/target/$TARGET/release/libmimotor_core.so" /data/local/tmp/mimotor/ >/dev/null
adb push "$RAIZ/pesos_amenazas_prueba.bin" /data/local/tmp/mimotor/ >/dev/null
adb push "$AQUI/build/classes.dex" /data/local/tmp/mimotor/prueba.dex >/dev/null
echo "archivos empujados"

adb shell "cd /data/local/tmp/mimotor && CLASSPATH=/data/local/tmp/mimotor/prueba.dex app_process /data/local/tmp/mimotor com.tavito.mimotor.Prueba"
echo "codigo de salida: $?"
