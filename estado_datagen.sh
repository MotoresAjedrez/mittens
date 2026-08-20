#!/bin/bash
# Avance del datagen distribuido: posiciones generadas por dispositivo.
set -u
DEST=${DEST:-$HOME/datagen_wdl}
TOTAL=0

for D in $(adb devices | awk '/\tdevice$/{print $1}'); do
    MODELO=$(adb -s "$D" shell getprop ro.product.model | tr -d '\r')
    VIVO=$(adb -s "$D" shell "pgrep -f 'mittens datagen' | wc -l" 2>/dev/null | tr -d '\r')
    BYTES=$(adb -s "$D" shell "cat /data/local/tmp/dg.data.parte* 2>/dev/null | wc -c" 2>/dev/null | tr -d '\r')
    BYTES=${BYTES:-0}
    POS=$((BYTES / 32))
    TOTAL=$((TOTAL + POS))
    TEMP=$(adb -s "$D" shell "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null" | tr -d '\r')
    printf "  %-12s %10d pos   hilos vivos:%-3s temp:%s\n" "$MODELO" "$POS" "${VIVO:-?}" "${TEMP:-?}"
done

MACB=$(cat "$DEST"/mac.data* 2>/dev/null | wc -c | tr -d ' ')
MACB=${MACB:-0}
MACPOS=$((MACB / 32))
TOTAL=$((TOTAL + MACPOS))
MACVIVO=$(pgrep -f "mittens datagen" | wc -l | tr -d ' ')
printf "  %-12s %10d pos   procesos vivos:%s\n" "Mac" "$MACPOS" "$MACVIVO"

echo "  ----------------------------------------"
printf "  %-12s %10d posiciones\n" "TOTAL" "$TOTAL"
