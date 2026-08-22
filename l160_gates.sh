#!/bin/sh
# L160 gates -- the n-set widened 75 -> 89 after f was measured at 2.71.
#   det1/det2  two unconfigured runs, must be bit-identical (determinism)
#   l147off    kill switch, must still be the pre-L147 band bit-for-bit
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l160_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l160_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L160_${tag}.json" > "../l160_${tag}.log" 2>&1
  echo "$tag exit=$?"
}
run det1    ICCAD_SHAPE_LP_NOOP=1
run det2    ICCAD_SHAPE_LP_NOOP=2
run l147off ICCAD_SHAPE_LP_L147=0
echo L160_GATES_DONE
