#!/bin/bash
# L301c -- the remaining arms on the SAME instrument. Mixing a wall-differenced
# dt for one arm with an LP-clocked dt for another is exactly the basis error
# this ledger keeps paying for (L287 1.1, L165), so every row in the final table
# has to come from one clock.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () { tag="$1"; shift
  echo "=== $tag ==="; date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" ICCAD_LP_TIMING=1 "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../l301b_${tag}.json" > "../l301b_${tag}.log" 2>&1
  echo "    exit=$?"; grep -E "Total Score|^Feasible" "../l301b_${tag}.log"
  echo "    lptime lines: $(grep -o '\[lptime\]' "../l301b_${tag}.log" | wc -l)"; }
run gate0 ICCAD_LP_GATE=0
run lp2   ICCAD_SHAPE_LP_ITERS=2
run both  ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=2
echo L301C_DONE
