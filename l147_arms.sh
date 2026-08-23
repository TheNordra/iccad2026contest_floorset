#!/usr/bin/env bash
# L147 Gate 2 -- in-set official eval, one arm at a time, 48-core pool shape.
# Sequential on purpose: each eval already saturates the box, and a profile that
# times out under contention drops OUT OF THE POOL, which changes the QUALITY
# result, not just the wall. (l147 plan, "concurrency rules".)
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1

run () {                       # run <tag> <env assignments...>
  tag="$1"; shift
  rm -f "../l147_${tag}_stats.txt"
  echo "=== $tag : $* ==="
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l147_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L147_${tag}.json" > "../l147_${tag}.log" 2>&1
  echo "$tag exit=$?"
}

# the R->1 invariant: shapes must freeze, so the result must be the flag-off one
run r1freeze ICCAD_SHAPE_LP_R=1.0000001
run r12      ICCAD_SHAPE_LP_R=1.2 ICCAD_SHAPE_LP_PRICE=1.0
run r13      ICCAD_SHAPE_LP_R=1.3 ICCAD_SHAPE_LP_PRICE=1.0
run r15g     ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0
# Gate 1 re-run on the FINAL file state: the kept-rate counter was added after
# the first flag-off eval started, so this closes the "edited while measuring"
# hole. Must be bit-identical to results_L136_48c_anchor.json again.
run ctrl2    ICCAD_SHAPE_LP_STATSDUMMY=1
echo ARMS_DONE
