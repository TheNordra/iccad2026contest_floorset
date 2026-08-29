#!/bin/bash
# L312 -- per-case gates for the RF-SAFE _L196_LPGATE (83/100).
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () { tag="$1"; shift
  echo "=== $tag ==="; date +"    start %H:%M:%S"
  env ICCAD_CONSTRUCTIVE_BIN="$BIN" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../l312_${tag}.json" > "../l312_${tag}.log" 2>&1
  echo "    exit=$?"; grep -E "Total Score|Feasible" "../l312_${tag}.log" | tail -2
  grep -ci "SA fallback" "../l312_${tag}.log" | sed 's/^/    SA-fallback lines: /'; }
run rfsafe_c48 ICCAD_ADAPTIVE_CORES=48
run rfsafe_def
echo L312_GATES_DONE
