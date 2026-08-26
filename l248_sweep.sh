#!/bin/sh
# L248 -- the lens-D size curve, on both sides of the pool's max->sum crossover.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l248.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
echo "=== L248 lens-D size sweep  $(date -u +%FT%TZ) ==="
for I in 1 2 3; do
  for K in 0 6 12 20 32; do
    if [ "$K" = 0 ]; then set --; else set -- ICCAD_LENSD_POOL=1 ICCAD_LENSD_K=$K; fi
    rm -f "../l248_prof_k${K}_${I}.txt"
    env ICCAD_ADAPTIVE_CORES=48 "$@" ICCAD_PROFILE_TIMEOUT=600 \
        ICCAD_PROFILE_TIMING="../l248_prof_k${K}_${I}.txt" \
      "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l248probe.py \
      -o "../results_L248_k${K}_${I}.json" > "../l248_k${K}_${I}.log" 2>&1
    echo "  K=$K rep$I exit=$? profiles/heavy=$(awk '$1>100{c[$1]++} END{n=0;for(k in c){n=c[k]};print n}' "../l248_prof_k${K}_${I}.txt" 2>/dev/null)"
  done
done
echo L248_SWEEP_DONE $(date -u +%FT%TZ)
