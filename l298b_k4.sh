#!/bin/bash
# L298b -- both4 was only ever measured against the stale baseline, where its
# dt read 74.65 s. The block correction cut `both`'s dt from 47.71 to 28.67 s,
# so that number is certainly wrong too. Re-measure it bracketed rather than
# leave a stale figure standing in the ledger.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () { tag="$1"; shift
  echo "=== $tag ==="; date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../l298_${tag}.json" > "../l298_${tag}.log" 2>&1
  echo "    exit=$?"; grep -E "Total Score|^Feasible" "../l298_${tag}.log"; }
run both4 ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=4
run ship4
echo L298B_DONE
