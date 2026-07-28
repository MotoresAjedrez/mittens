#!/bin/bash
# Crea (si hace falta) y arranca un emulador arm64-v8a API 34, nativo en
# Apple Silicon (sin traduccion), y espera a que termine de bootear.
set -e
SDK="${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}"
export ANDROID_SDK_ROOT="$SDK" ANDROID_HOME="$SDK"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk}"
export PATH="$JAVA_HOME/bin:$SDK/platform-tools:$SDK/emulator:$SDK/cmdline-tools/latest/bin:$PATH"
AVD=mimotor34

if ! "$SDK/emulator/emulator" -list-avds | grep -qx "$AVD"; then
  echo "creando AVD $AVD"
  echo no | avdmanager create avd -n "$AVD" -k "system-images;android-34;google_apis;arm64-v8a" -d pixel_6
fi

if ! adb devices | grep -q emulator; then
  echo "arrancando emulador..."
  nohup "$SDK/emulator/emulator" -avd "$AVD" -no-window -no-audio -no-snapshot -gpu swiftshader_indirect \
    > /tmp/emulador-mimotor.log 2>&1 &
fi

adb wait-for-device
echo "dispositivo visible, esperando boot completo..."
until [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do sleep 3; done
adb shell getprop ro.product.cpu.abi
echo "EMULADOR LISTO"
