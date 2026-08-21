#!/usr/bin/env bash
# L154 gates A/B -- in-set 48c. A: CATCH off must be bit-identical to the L147
# arm (the refactor is a no-op). B: CATCH on, and the tier field says which tier
# kept each case.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final || exit 1
rm -f l154_stats_off.txt l154_stats_on.txt
cd iccad2026contest || exit 1
echo "=== A: CATCH off (must equal results_L147_on_L137.json bit-for-bit) ==="
env ICCAD_ADAPTIVE_CORES=48 $LP ICCAD_SHAPE_LP_STATS=../l154_stats_off.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o ../results_L154_catchoff.json > ../l154_catchoff.log 2>&1
echo "  A exit=$?"
echo "=== B: CATCH on ==="
env ICCAD_ADAPTIVE_CORES=48 $LP ICCAD_SHAPE_LP_CATCH=1 ICCAD_SHAPE_LP_STATS=../l154_stats_on.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o ../results_L154_catchon.json > ../l154_catchon.log 2>&1
echo "  B exit=$?"
grep -inE "fallback|unavailable|all profiles failed|\[constructive\]" ../l154_catchoff.log ../l154_catchon.log | head -5
echo L154_INSET_DONE
