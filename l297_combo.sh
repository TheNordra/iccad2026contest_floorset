#!/bin/bash
cd /c/ICCAD_ml/ship_final/iccad2026contest
PY="/c/Users/.01/anaconda3/envs/floorset/python.exe"
run () { tag=$1; shift; echo "=== $tag : $* ==="; env "$@" ICCAD_ADAPTIVE_CORES=48 \
    $PY -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py -o ../$tag.json 2>&1 \
    | tr '\r' '\n' | grep -E "Total Score|Feasible:|Avg Cost|Avg Runtime"; }
P=/c/ICCAD_ml/ship_final/constructive_l296.exe
run l297_ship  ICCAD_X=0
run l297_g0k2  ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=2
run l297_g0k4  ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=4
run l297_gf    ICCAD_CONSTRUCTIVE_BIN=$P ICCAD_LS_GF=150000
run l297_ship2 ICCAD_X=0
echo COMBO_DONE
