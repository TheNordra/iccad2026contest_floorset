#!/bin/bash
# L294b -- determinism repeat of the gate0 arm (handoff 2026-08-29 §3.2 item 2:
# "two identical runs must agree bit-for-bit"). The shipped arm already did,
# 100/100 on cost AND positions, in l294_lpgate.sh.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
rm -f ../l294_gate0_r2_stats.txt
env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="C:/ICCAD_ml/ship_final/constructive.exe" \
    ICCAD_SHAPE_LP_STATS=../l294_gate0_r2_stats.txt ICCAD_LP_GATE=0 \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o ../l294_gate0_r2.json > ../l294_gate0_r2.log 2>&1
echo "exit=$?"; grep -E "Total Score|^Feasible" ../l294_gate0_r2.log
