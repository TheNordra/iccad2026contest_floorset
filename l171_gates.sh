#!/bin/sh
# L171 gates -- the n-set became a per-case DEPTH MAP built from the published
# medians instead of a fit. Same three checks as L160.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l171_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l171_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L171_${tag}.json" > "../l171_${tag}.log" 2>&1
  echo "$tag exit=$?"
}
run det1    ICCAD_SHAPE_LP_NOOP=1
run det2    ICCAD_SHAPE_LP_NOOP=2
run l147off ICCAD_SHAPE_LP_L147=0
echo L171_GATES_DONE
