#!/usr/bin/env bash
# L154 OOS. OFF side = the L147 arm; ON side = + ICCAD_SHAPE_LP_CATCH=1.
#
# The s1 OFF arm is re-run FRESH even though Gate A proved the refactor
# bit-identical through the official evaluator: the OOS corpus goes through
# m77_oos_probe, a different driver with different multiprocessing, and
# HANDOFF_2026-08-20 §4.4 is about exactly this -- a base measured on one path
# and an arm on another. If the fresh OFF arm is bit-equal to l151_oos_s1_on.json
# then l151_oos_s2_on.json is reusable as the s2 OFF side; if not, s2 OFF gets
# re-run too.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final || exit 1
rm -f l154_oos_stats_*.txt

echo "=== s1 OFF (fresh, bit-check against l151_oos_s1_on.json) ==="
env $LP ICCAD_SHAPE_LP_STATS=l154_oos_stats_s1off.txt \
  $PY -u l140_oos_soft_audit.py run --sample s1 --cores 48 \
  --out l154_oos_s1_off.json > l154_s1_off.log 2>&1; echo "  exit=$?"

echo "=== s1 ON (band-catch) ==="
env $LP ICCAD_SHAPE_LP_CATCH=1 ICCAD_SHAPE_LP_STATS=l154_oos_stats_s1on.txt \
  $PY -u l140_oos_soft_audit.py run --sample s1 --cores 48 \
  --out l154_oos_s1_on.json > l154_s1_on.log 2>&1; echo "  exit=$?"

echo "=== s2 ON (band-catch) ==="
env $LP ICCAD_SHAPE_LP_CATCH=1 ICCAD_SHAPE_LP_STATS=l154_oos_stats_s2on.txt \
  $PY -u l140_oos_soft_audit.py run --sample s2 --cores 48 \
  --out l154_oos_s2_on.json > l154_s2_on.log 2>&1; echo "  exit=$?"

echo L154_OOS_DONE
