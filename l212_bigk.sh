#!/bin/sh
# L212 -- extend the pool-drop sweep to k=16/20, where the l204 model says the
# wall cut reaches the ~10% that would put NET past the rank-3 threshold.
# Runs after L211 so the two do not contend.
set -u
R=/c/ICCAD_ml/ship_final
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd "$R" || exit 1
while ! grep -q "L211_DONE" l211_price.out 2>/dev/null; do sleep 20; done
echo "=== L212 larger k  $(date -u +%FT%TZ) ==="
cd "$R/iccad2026contest" || exit 1
for k in 16 20; do
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_L211_DROP="../l211_drop_k${k}.json" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
    -o "../results_L211_k${k}.json" > "../l211_k${k}.log" 2>&1
  echo "k${k} exit=$?  table loaded=$(grep -c 'drop table loaded' "../l211_k${k}.log")"
done
cd "$R" || exit 1
"$PY" -u l212_curve.py
echo L212_DONE
