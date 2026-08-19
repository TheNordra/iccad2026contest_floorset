#!/usr/bin/env bash
# L148 -- re-scan the ledger's runtime-killed mechanisms with the measured medians.
#
# Base is the config we are actually proposing to ship: L137 defaults (already on
# in the tree) + L147's tangent cut. Every arm is that base plus ONE knob, so the
# delta is attributable.
#
# Sequential: each eval saturates the box, and a profile that times out under
# contention drops out of the pool, which changes QUALITY, not just wall.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BASE="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1

run () {
  tag="$1"; shift
  echo "=== $tag : $* ==="
  env ICCAD_ADAPTIVE_CORES=48 $BASE "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L148_${tag}.json" > "../l148_${tag}.log" 2>&1
  echo "$tag exit=$?"
}

# LP depth ON TOP of the tangent cut -- L122 showed R=1.5 dominates k=2..12
# standalone, but the COMPOSITION has never been measured.
run lp2   ICCAD_SHAPE_LP_ITERS=2
run lp3   ICCAD_SHAPE_LP_ITERS=3
# REFINE band-cut restore (M49/M50). Priced RED on a 2.4x ratio; quality on the
# current tree was never measured -- a 20-case pilot even read the wrong sign.
run refine ICCAD_ADAPTIVE_REFINE=0
# ICCAD_REFRAME: in the ledger as "dead code kept gated off", QUALITY NEVER
# MEASURED. Cheapest unknown on the list.
run reframe ICCAD_REFRAME=1
# Everything restored: full 41-profile pool + full REFINE. Upper bound on the
# whole pruning family (M41/M42/M45), which only ever existed to buy runtime.
run pool0 ICCAD_ADAPTIVE_POOL=0
echo RESCAN_DONE
