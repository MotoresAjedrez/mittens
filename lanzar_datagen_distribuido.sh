#!/bin/bash
# Genera dataset de self-play con RESULTADO REAL de partida, repartido entre
# la Mac y los telefonos Android por USB.
#
# POR QUE ESTE DATASET: los 6 datasets que ya tiene el proyecto traen el campo
# `result` SINTETIZADO por umbral desde el score (verificado byte a byte el
# 2026-08-19: solapamiento exactamente cero entre clases). O sea que nunca se
# entreno con desenlaces reales y `wdl` tuvo que quedar en 0.0. Este dataset
# si trae resultados de partidas jugadas de verdad.
#
# SEMILLA DISTINTA POR DISPOSITIVO: es obligatorio. `datagen` es determinista
# por semilla -- se midio que dos telefonos con la misma semilla producen
# exactamente las mismas 1815 posiciones. Dentro de un dispositivo cada hilo
# ya se separa solo (semilla ^ (h+1)*0x9E3779B97F4A7C15), pero entre
# dispositivos hay que separarlos a mano o se genera basura triplicada.

set -u

PROF=${PROF:-8}                 # profundidad de busqueda por jugada
PARTIDAS_TEL=${PARTIDAS_TEL:-40000}
PARTIDAS_MAC=${PARTIDAS_MAC:-40000}
HILOS_TEL=${HILOS_TEL:-6}
HILOS_MAC=${HILOS_MAC:-8}
DEST=${DEST:-$HOME/datagen_wdl}

REPO=/Users/Tavito/mi-motor-rust-produccion
mkdir -p "$DEST"

echo "=== Lanzando datagen distribuido (profundidad $PROF) ==="

# --- Telefonos -------------------------------------------------------------
# SIN nice. Medido el 2026-08-19 en el S24 Ultra con 6 hilos y la MISMA semilla
# (4784 vs 4781 posiciones, o sea trabajo identico): nice 19 -> 281 pos/s,
# nice 0 -> 341 pos/s. **21% mas rapido sin nice.** Android mete las tareas de
# baja prioridad en un cgroup restringido que les limita la frecuencia: bajo
# nice 19 los dos telefonos corrian a ~57% de su reloj maximo.
# En los telefonos el nice no protege NADA (no hay nada mas corriendo ahi);
# solo tiene sentido en la Mac, donde hay un entrenamiento que cuidar.
#
# stay_on_while_plugged_in 3 = no se duerme mientras este enchufado (si se
# duerme, Android congela el proceso y se pierde el avance).
i=0
for D in $(adb devices | awk '/\tdevice$/{print $1}'); do
    i=$((i+1))
    MODELO=$(adb -s "$D" shell getprop ro.product.model | tr -d '\r')
    SEMILLA=$((1000 + i * 7919))
    echo "  [$MODELO] $HILOS_TEL hilos, semilla $SEMILLA"
    adb -s "$D" shell settings put global stay_on_while_plugged_in 3 >/dev/null 2>&1
    adb -s "$D" shell "rm -f /data/local/tmp/dg.data*"
    adb -s "$D" shell "nohup /data/local/tmp/mittens datagen \
        --salida /data/local/tmp/dg.data \
        --partidas $PARTIDAS_TEL --prof $PROF --hilos $HILOS_TEL \
        --semilla $SEMILLA > /data/local/tmp/dg.log 2>&1 &" >/dev/null 2>&1
done

# --- Mac -------------------------------------------------------------------
# nice 19 tambien aca: hay un entrenamiento de la red 1536 corriendo y NO se
# lo debe frenar. El datagen solo debe comer lo que sobra.
echo "  [Mac] $HILOS_MAC hilos, semilla 99991"
rm -f "$DEST"/mac.data*
nohup nice -n 19 "$REPO/target/release/mittens" datagen \
    --salida "$DEST/mac.data" \
    --partidas "$PARTIDAS_MAC" --prof "$PROF" --hilos "$HILOS_MAC" \
    --semilla 99991 > "$DEST/mac.log" 2>&1 &

echo
echo "Lanzado. Para ver avance:   bash $REPO/estado_datagen.sh"
echo "Para recoger de telefonos:  bash $REPO/recoger_datagen.sh"
