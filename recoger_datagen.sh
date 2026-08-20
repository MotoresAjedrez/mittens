#!/bin/bash
# Recoge las partes de los telefonos, las junta con las de la Mac y VERIFICA
# que el resultado sea real antes de dar el dataset por bueno.
set -u
DEST=${DEST:-$HOME/datagen_wdl}
mkdir -p "$DEST"

for D in $(adb devices | awk '/\tdevice$/{print $1}'); do
    MODELO=$(adb -s "$D" shell getprop ro.product.model | tr -d '\r')
    echo "Bajando de $MODELO..."
    adb -s "$D" shell "cat /data/local/tmp/dg.data.parte* > /data/local/tmp/dg_junto.data" 2>/dev/null
    adb -s "$D" pull /data/local/tmp/dg_junto.data "$DEST/${MODELO}.data" 2>&1 | tail -1
done

echo "Juntando todo..."
cat "$DEST"/*.data "$DEST"/mac.data.parte* > "$DEST/selfplay_wdl.data" 2>/dev/null

# VERIFICACION OBLIGATORIA: si el resultado fuese sintetico (umbral sobre el
# score) las clases NO se solaparian, que es justo el defecto de los 6
# datasets viejos. Con partidas reales tienen que solaparse mucho.
python3 - "$DEST/selfplay_wdl.data" <<'PY'
import struct, sys, collections
ruta = sys.argv[1]
rng = {0: [9999, -9999], 1: [9999, -9999], 2: [9999, -9999]}
cnt = collections.Counter()
with open(ruta, 'rb') as f:
    for _ in range(500000):
        b = f.read(32)
        if len(b) < 32:
            break
        sc = struct.unpack_from('<h', b, 24)[0]
        r = b[26]
        if r > 2:
            continue
        rng[r][0] = min(rng[r][0], sc)
        rng[r][1] = max(rng[r][1], sc)
        cnt[r] += 1

import os
print(f"\n{os.path.getsize(ruta)//32} posiciones totales")
for r, n in ((0, 'derrota'), (1, 'tablas'), (2, 'victoria')):
    print(f"  result={r} ({n:8s}) {cnt[r]:8d}  score de {rng[r][0]:6d} a {rng[r][1]:6d}")

solapa = rng[0][1] > rng[1][0] and rng[2][0] < rng[1][1]
print("\n  -> RESULTADOS REALES (las clases se solapan)" if solapa
      else "\n  -> CUIDADO: parecen SINTETICOS, no usar para wdl>0")
PY
