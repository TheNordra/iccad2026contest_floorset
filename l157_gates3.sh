#!/bin/sh
# L158 round 3 -- the L147 tangent is now a CODE DEFAULT, so the three arms
# that matter are about what an UNCONFIGURED run does.
#
#   l147off  ICCAD_SHAPE_LP_L147=0 -> must be bit-identical to the pre-L147
#            shipped band (results_L157_notan.json, round 2's tangent-off arm).
#   default  nothing set at all    -> must be bit-identical to
#            results_L154_catchoff.json, i.e. the code default reproduces the
#            arm every L147/L154/L157 number was measured on. The depth gate is
#            active here and correctly inert on this box at S=1.
#   defaultS default + S=7.75      -> must be bit-identical to round 2's
#            results_L157_gateS.json, which set all four flags EXPLICITLY.
#            Same mechanism, reached two different ways.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l157_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l157_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L157_${tag}.json" > "../l157_${tag}.log" 2>&1
  echo "$tag exit=$?"
}
run l147off  ICCAD_SHAPE_LP_L147=0
run default  ICCAD_SHAPE_LP_NOOP=1
run defaultS ICCAD_SHAPE_LP_DEPTH_S=7.75
echo L157_GATES3_DONE
