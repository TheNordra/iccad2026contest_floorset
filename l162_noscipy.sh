#!/bin/sh
# L162 -- what ACTUALLY happens if the grader has no scipy?
# The empty requirements.txt is sanctioned by the official Beta guidelines S2
# (Case A: the env provides numpy/torch/scipy/numba/tqdm/shapely/threadpoolctl),
# but the Beta evaluation REPORT S2(a) says the opposite -- "Do not assume any
# package beyond the Python standard library is available" -- and names scipy.
# Our whole LP lane (+2.54%) depends on scipy. CLAUDE.md claims a missing scipy
# degrades gracefully to the shipped band rather than crashing. That claim has
# never been tested, so test it: block scipy and run the real evaluator.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
PYTHONPATH="/c/ICCAD_ml/ship_final/_noscipy" \
  env ICCAD_ADAPTIVE_CORES=48 "$PY" -u iccad2026_evaluate.py \
    --evaluate ../optimizer_constructive.py -o ../results_L162_noscipy.json \
    > ../l162_noscipy.log 2>&1
echo "exit=$?"
grep -E "Tests:|Feasible:|Avg Cost" ../l162_noscipy.log | head -3
echo "  scipy-related stderr lines: $(grep -ci 'scipy\|shape LP' ../l162_noscipy.log)"
echo L162_DONE
