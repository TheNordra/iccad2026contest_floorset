#!/usr/bin/env bash
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BASE="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () { tag="$1"; shift; env ICCAD_ADAPTIVE_CORES=48 $BASE "$@" "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_constructive.py -o "../results_L150_${tag}.json" > "../l150_${tag}.log" 2>&1; echo "$tag exit=$?"; }
run g115  ICCAD_SHAPE_LP_G_BIG=1.15 ICCAD_SHAPE_LP_TOL_BIG=0.0046
run r13b  ICCAD_SHAPE_LP_R_BIG=1.3
run r12b  ICCAD_SHAPE_LP_R_BIG=1.2
echo L150_DONE
