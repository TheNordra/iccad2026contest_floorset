#!/usr/bin/env bash
# L147 Gate 3 -- the deployed price of the chosen arm, measured properly.
#
# min-of-3, control and arm INTERLEAVED, one eval at a time, nothing else on the
# box. Single-shot timing is disqualified here: L122 read 1.95x single-shot where
# min-of-3 read 2.33x, and L137 measured a 19.6% run-to-run spread on the
# BASELINE's own wall -- larger than the effect being measured.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1

# never start while another eval is still holding the box
while ! grep -q LPOFF_DONE l147_lpoff.log 2>/dev/null; do sleep 15; done

cd iccad2026contest || exit 1
one () {                      # one <tag> <env...>
  tag="$1"; shift
  env ICCAD_ADAPTIVE_CORES=48 "$@" "$PY" -u iccad2026_evaluate.py \
    --evaluate ../optimizer_constructive.py -o "../results_L147_${tag}.json" \
    > "../l147_${tag}.log" 2>&1
  echo "  $tag done exit=$?"
}
for rep in 1 2 3; do
  echo "=== rep $rep ==="
  one "t${rep}_ctrl"
  one "t${rep}_r15g" ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 \
                     ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0
done
echo GATE3_DONE
