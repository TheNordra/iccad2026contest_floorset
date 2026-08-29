#!/bin/bash
cd /c/ICCAD_ml/ship_final/iccad2026contest
PY="/c/Users/.01/anaconda3/envs/floorset/python.exe"
for c in 0.0067 0.0337 0.337 1.0; do
  echo "=== ICCAD_LS_C=$c ==="
  ICCAD_LS_C=$c ICCAD_ADAPTIVE_CORES=48 \
    ICCAD_CONSTRUCTIVE_BIN=/c/ICCAD_ml/ship_final/constructive_l296.exe \
    $PY -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py -o ../l296_c$c.json 2>&1 \
    | tr '\r' '\n' | grep -E "Total Score|Feasible:|Avg Cost"
done
echo "SWEEP_DONE"
