#!/bin/sh
# L163d -- the fallback end to end, on the real evaluator.
# _noscipy/ blocks the system scipy, so the code must find vendor/ instead and
# the score must come back to the LP-enabled value 1.191977686767963. If it
# lands on 1.260246745790688 the fallback did nothing.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
PYTHONPATH="/c/ICCAD_ml/ship_final/_noscipy" \
  env ICCAD_ADAPTIVE_CORES=48 "$PY" -u iccad2026_evaluate.py \
    --evaluate ../optimizer_constructive.py -o ../results_L163_vendor.json \
    > ../l163_vendor.log 2>&1
echo "exit=$?"
grep -E "Tests:|Feasible:" ../l163_vendor.log | head -2
grep "scipy:" ../l163_vendor.log | head -2
echo L163D_DONE
