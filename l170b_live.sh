#!/bin/sh
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
env ICCAD_ADAPTIVE_CORES=48 ICCAD_LP_HB_PRED=0.2994 "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_constructive.py -o ../results_L170_live.json \
  > ../l170_live.log 2>&1
echo "exit=$?"
