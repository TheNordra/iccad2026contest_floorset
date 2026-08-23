#!/bin/sh
# L157 round 2. Round 1 found the gate cannot fire on this box: it is stated in
# absolute seconds against the grader's median and this box runs ~9x slower per
# case, so 0/100 cases fell inside the budget. Three arms:
#   k1b   -- kill switch, AFTER the coupling + speed-knob edits. Must still be
#            bit-identical to results_L154_catchoff.json.
#   notan -- tangent OFF, default depth. Proves the coupling: col 4 all 1, i.e.
#            a package with no ICCAD_* set does NOT run the unmeasured arm.
#   gateS -- tangent ON, ICCAD_SHAPE_LP_DEPTH_S=7.75, the scale that reproduces
#            the priced 75/100 selection FRACTION. Exercises the mechanism end
#            to end. Does NOT reproduce the grader's per-case ordering.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BASE="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; base="$2"; shift 2
  rm -f "../l157_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 $base ICCAD_SHAPE_LP_STATS="../l157_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L157_${tag}.json" > "../l157_${tag}.log" 2>&1
  echo "$tag exit=$?"
}
run k1b   "$BASE" ICCAD_SHAPE_LP_DEPTH2=0
run notan "ICCAD_SHAPE_LP=1" ICCAD_SHAPE_LP_NOOP=1
run gateS "$BASE" ICCAD_SHAPE_LP_DEPTH_S=7.75
echo L157_GATES2_DONE
