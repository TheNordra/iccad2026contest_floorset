#!/bin/bash
# L294 -- measure ICCAD_LP_GATE=0 on the IN-SET 100 (handoff 2026-08-29 §3.1,
# "the highest-value open item").
#
# The gate switches the shape LP OFF for 29 block counts carrying 44.2 % of the
# graded weight, including 8 of the heavy 20. It exists FOR RUNTIME -- the exact
# quantity L287/L291 showed was being over-charged 33x -- so its cost has to be
# re-measured under the corrected pricing.
#
# SANDWICH ORDER (ship, gate0, ship) so dt is differenced against a same-session
# baseline and the two ship runs also give the wall-clock noise floor.
#
# HARD LIVENESS GATE. `_shape_lp_maybe` never raises by design (handoff trap 5),
# so a dead flag is indistinguishable from a decision not to act. The stats file
# gets one line per LP execution: the shipped arm must show exactly 71 distinct
# block counts, the gate0 arm exactly 100. Checked in l294_gate.py afterwards.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1

run () {           # tag  extra-env...
  tag="$1"; shift
  rm -f "../l294_${tag}_stats.txt"
  echo "=== $tag : $* ==="
  date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" \
      ICCAD_SHAPE_LP_STATS="../l294_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../l294_${tag}.json" > "../l294_${tag}.log" 2>&1
  echo "    exit=$?"
  date +"    end   %H:%M:%S"
  grep -E "Total Score|^Feasible|Avg Runtime" "../l294_${tag}.log"
  echo "    stats lines: $(wc -l < "../l294_${tag}_stats.txt" 2>/dev/null || echo 0)"
}

run ship
run gate0 ICCAD_LP_GATE=0
run ship_r2
echo L294_DONE
