#!/bin/sh
# L157 in-set gates. Three arms, all with the L147 tangent flags ($BASE), so
# each has a committed bit-for-bit reference:
#   depthoff -> must equal results_L147_r15g.json  (kill switch = shipped k=1)
#   k2       -> must equal results_L148_lp2.json   (ungated k=2 arm unchanged)
#   gated    -> the ship. Quality between the two, stats col 4 mixed 1 and 2.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BASE="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l157_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 $BASE ICCAD_SHAPE_LP_STATS="../l157_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L157_${tag}.json" > "../l157_${tag}.log" 2>&1
  echo "$tag exit=$?"
}
run depthoff ICCAD_SHAPE_LP_DEPTH2=0
run k2       ICCAD_SHAPE_LP_ITERS=2
run gated    ICCAD_SHAPE_LP_NOOP=1
echo L157_GATES_DONE
