#!/bin/sh
# L159b -- measure the 2nd LP pass WITHIN one run, so no cross-run drift enters.
# Plus 3 reps each of band/l147, whose difference (the tangent's cost) genuinely
# needs two arms and therefore needs per-case medians to beat the drift.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_LP_TIMING=1 "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L159b_${tag}.json" > "../l159b_${tag}.log" 2>&1
  echo "$tag exit=$?"
}
run k2pp ICCAD_SHAPE_LP_ITERS=2
for i in 1 2 3; do
  run band$i ICCAD_SHAPE_LP_L147=0
  run l147$i ICCAD_SHAPE_LP_ITERS=1
done
echo L159B_DONE
