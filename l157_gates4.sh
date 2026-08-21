#!/bin/sh
# L158 round 4 -- the depth gate is now DETERMINISTIC (n-set), so the run that
# matters most is the one that proves it.
#
#   det1 / det2  the SAME unconfigured run, twice. Must be bit-identical.
#                Round 3 measured the clock form deciding 5 block counts
#                differently between two runs and moving 4 cases; that is what
#                broke make_submission verify and l113_ship_gate G4.
#   l147off2     kill switch, must still be the pre-L147 band bit-for-bit.
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
run det1     ICCAD_SHAPE_LP_NOOP=1
run det2     ICCAD_SHAPE_LP_NOOP=2
run l147off2 ICCAD_SHAPE_LP_L147=0
echo L157_GATES4_DONE
