#!/bin/bash
# L275 -- re-price the L250-L274 arc's main arms on the IN-SET 100, the shape the
# score is actually computed on. Every arm in that arc was priced on the OOS heavy
# band only (40 cases, n>=101). L274 found l269p1/p2 flip sign here. This asks
# whether that is specific to L269 or systematic.
#
# Only flags NO profile sets itself are used, so ambient env really reaches the
# binary. ICCAD_FRAME_SCALES is deliberately NOT tested this way: 44 of 55 profile
# dicts set it themselves and `env.update(env_over)` makes the profile win, so an
# ambient ladder is a silent no-op (handoff trap #2).
cd /c/ICCAD_ml/ship_final/iccad2026contest
PY=/c/Users/.01/anaconda3/envs/floorset/python.exe
run () {  # name  binary  extra-env...
  local name=$1; shift; local bin=$1; shift
  echo "=== $name ==="
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="C:/ICCAD_ml/ship_final/$bin" "$@" \
    $PY iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o ../results_L275_${name}.json 2>&1 | grep -E "Total Score|Feasible|Avg Runtime"
}
run adapt   constructive_l270.exe ICCAD_L267=1
run nosize  constructive_l270.exe ICCAD_L268=4
run l271sng constructive_l273.exe ICCAD_L271=6
echo ALL_DONE
