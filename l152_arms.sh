#!/usr/bin/env bash
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive_l152.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () { tag="$1"; shift; env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" $LP "$@" \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o "../results_L152_${tag}.json" > "../l152_${tag}.log" 2>&1; echo "$tag exit=$?"; }
run ctrl                                     # probe binary, flags off -> must equal the base
run r2   ICCAD_BND_FRAME_ITEM=1
run r3   ICCAD_BND_SNAP_BEST=1
run both ICCAD_BND_FRAME_ITEM=1 ICCAD_BND_SNAP_BEST=1
echo L152_DONE
