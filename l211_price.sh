#!/bin/sh
# L211 -- price the per-block-count pool drop. Runs AFTER the L210 chain so it
# does not contend; quality is contention-free but the machine is not.
#
# Baseline is results_L209_det1.json (route A off, no drop). Each arm differs
# from it ONLY by the drop table, so the delta is the quality the wall buys.
set -u
R=/c/ICCAD_ml/ship_final
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd "$R" || exit 1
while ! grep -q "L210_CHAIN_DONE" l210_chain.out 2>/dev/null; do sleep 30; done
echo "=== L211 pricing the pool drop  $(date -u +%FT%TZ) ==="
cd "$R/iccad2026contest" || exit 1
# k=0 control: the probe with NO drop table must reproduce the shipped result
# bit-for-bit, or the probe itself is the thing being measured.
env ICCAD_ADAPTIVE_CORES=48 "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_l205probe.py -o ../results_L211_k0.json \
  > ../l211_k0.log 2>&1
echo "k0 exit=$?"
for k in 3 8 12; do
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_L211_DROP="../l211_drop_k${k}.json" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
    -o "../results_L211_k${k}.json" > "../l211_k${k}.log" 2>&1
  echo "k${k} exit=$?  table loaded=$(grep -c 'drop table loaded' "../l211_k${k}.log")"
done
cd "$R" || exit 1
"$PY" -u l211_score.py
echo L211_DONE
