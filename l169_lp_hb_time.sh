#!/bin/sh
# L169 -- the LP-baseline arm's real cost. The lens measured whole-eval wall
# (504s -> 1707s, "partly contended", and a clean 972s = 1.93x), which cannot
# separate the LP from CPU contention -- the same differencing error that made
# every earlier LP cost figure wrong. ICCAD_LP_TIMING reports the seconds spent
# INSIDE _shape_lp, per case, in-process. That is the number that prices.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_LP_TIMING=1 "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../_probe_lp_hb.py \
    -o "../results_L169_${tag}.json" > "../l169_${tag}.log" 2>&1
  echo "$tag exit=$?"
}
run ctrl
run arm ICCAD_LP_HB_PRED=0.2994
echo L169_DONE
