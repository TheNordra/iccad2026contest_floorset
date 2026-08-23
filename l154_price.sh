#!/usr/bin/env bash
# L154 price. Two questions, both min-of-3 with the arms INTERLEAVED:
#
#  A) what does the retry actually cost, on the cases that actually retry?
#     cases 10 (n=31) and 21 (n=42) -- CATCH off vs on.
#  B) what would it cost if a BIG case rejected, which is what happens on
#     Linux (case 96, n=117) and on both OOS samples (n=116 / n=119)?
#     The retry is one shipped-band LP solve, so that is LP-off vs shipped band
#     on cases 96 (n=117) and 92 (n=113).
#
# Whole-run avg_runtime is disqualified for this: the two in-set runs read L154
# as 0.37s/case FASTER than L147, which is impossible for strictly-added work.
# A 2-case effect cannot be seen through a 2.8% p50 / 8.9% max whole-run spread.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
: > ../l154_price.txt

one () {   # one <tag> <case> <env...>
  tag="$1"; case="$2"; shift 2
  env ICCAD_ADAPTIVE_CORES=48 "$@" "$PY" -u iccad2026_evaluate.py \
    --evaluate ../optimizer_constructive.py --test-id "$case" \
    -o "../_l154_p.json" > /dev/null 2>&1
  rt=$("$PY" -c "import json;print(json.load(open('../_l154_p.json'))['test_results'][0]['runtime_seconds'])")
  echo "$tag case=$case solve=$rt" | tee -a ../l154_price.txt
}

for rep in 1 2 3; do
  echo "--- rep $rep ---"
  for c in 10 21; do
    one A_off "$c" $LP
    one A_on  "$c" $LP ICCAD_SHAPE_LP_CATCH=1
  done
  for c in 96 92; do
    one B_lpoff "$c" ICCAD_SHAPE_LP=0
    one B_band  "$c"
  done
done
echo L154_PRICE_DONE
