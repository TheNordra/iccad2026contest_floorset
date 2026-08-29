#!/bin/bash
# L301d -- repeats, so each distinct WORK UNIT gets min-of-N.
#
# The same-work identity caught it: gate0 and ship do bit-identical work on the
# 71, yet the LP clock read 10.70 s vs 14.33 s (-25.3 %), while the three
# independent measurements of k=2-on-the-71 (lp2 / mix / both) agree to 2.8 %.
# So one k=1-on-71 observation is bad and there are only two of them. CLAUDE.md:
# "量時間要用 min-of-N".
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () { tag="$1"; shift
  echo "=== $tag ==="; date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" ICCAD_LP_TIMING=1 "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../l301b_${tag}.json" > "../l301b_${tag}.log" 2>&1
  echo "    exit=$? ; $(grep -o '\[lptime\]' "../l301b_${tag}.log" | wc -l) lptime lines"
  grep -E "Total Score" "../l301b_${tag}.log"; }
run ship_r2
run gate0_r2 ICCAD_LP_GATE=0
run ship_r3
run lp2_r2 ICCAD_SHAPE_LP_ITERS=2
echo L301D_DONE
