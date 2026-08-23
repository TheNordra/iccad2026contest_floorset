#!/usr/bin/env bash
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BASE="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () { tag="$1"; shift; env ICCAD_ADAPTIVE_CORES=48 $BASE "$@" "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_constructive.py -o "../results_L149_${tag}.json" > "../l149_${tag}.log" 2>&1; echo "$tag exit=$?"; }
# the REFINE question, asked correctly: lift L137's cap (which subsumed the band-cut)
run hint6  ICCAD_ADAPTIVE_REFINE=0 ICCAD_HINT_REFINE=6
run hint12 ICCAD_ADAPTIVE_REFINE=0 ICCAD_HINT_REFINE=12
# min-of-3 timing for the shippable candidate, arms interleaved, box exclusive
for rep in 1 2 3; do
  run "t${rep}_base"
  run "t${rep}_lp2" ICCAD_SHAPE_LP_ITERS=2
done
echo CHAIN_DONE
