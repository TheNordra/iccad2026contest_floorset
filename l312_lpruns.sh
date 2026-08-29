#!/bin/bash
# Measure the LP directly (ICCAD_LP_TIMING=1) instead of by differencing walls.
cd /c/ICCAD_ml/ship_final/iccad2026contest
PY="/c/Users/.01/anaconda3/envs/floorset/python.exe"
run () { tag=$1; shift; echo "=== $tag ==="
  env "$@" ICCAD_LP_TIMING=1 ICCAD_ADAPTIVE_CORES=48 \
    $PY -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py -o ../$tag.json \
    2> ../$tag.log | tr '\r' '\n' | grep -E "Total Score|Feasible:"
  grep -c lptime ../$tag.log; }
run l312_ship  ICCAD_X=0
run l312_g0    ICCAD_LP_GATE=0
run l312_g0k2  ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=2
echo LPRUNS_DONE
