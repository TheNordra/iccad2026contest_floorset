#!/bin/sh
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_L147=0 "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_constructive.py -o ../results_L172_l147off.json \
  > ../l172_l147off.log 2>&1
echo "kill switch arm exit=$?"
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_NOOP=1 "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_constructive.py -o ../results_L172_det.json \
  > ../l172_det.log 2>&1
echo "default arm exit=$?"
