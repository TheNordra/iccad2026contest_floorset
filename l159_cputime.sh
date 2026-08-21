#!/bin/sh
# L159 -- re-measure the LP's cost in CPU TIME, because the teammate's L140
# proved that whole-eval wall differencing (how L147, L154 and L157 were all
# priced) is wrong by 2.4x on a dev box.
#
# Four arms, all at 48c. Each prints one [lptime] line per case with the CPU
# time spent INSIDE _shape_lp, which is where both mechanisms' cost lives:
#   band  pre-L147 band, k=1            the LP baseline
#   l147  L147 tangent,  k=1            band + tangent rows
#   ship  L147 + the n-set depth gate   what this branch ships
#   k2    L147 + k=2 everywhere         the teammate's L140 shape, on our base
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_LP_TIMING=1 "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L159_${tag}.json" > "../l159_${tag}.log" 2>&1
  echo "$tag exit=$?  lptime lines=$(grep -c '^\[lptime\]' "../l159_${tag}.log")"
}
run band ICCAD_SHAPE_LP_L147=0
run l147 ICCAD_SHAPE_LP_ITERS=1
run ship ICCAD_SHAPE_LP_NOOP=1
run k2   ICCAD_SHAPE_LP_ITERS=2
echo L159_DONE
