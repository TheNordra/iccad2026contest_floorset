#!/bin/sh
# L165 -- the OOS k=3 arms, so the per-case depth map can be scored against
# MEASURED quality instead of arm-mixed estimates.
#
# Same driver, same two disjoint 240-case samples, same $LP flag string as
# l157_oos.sh -- byte for byte -- with ICCAD_SHAPE_LP_ITERS=3 as the only
# difference. An explicit ITERS is ungated by design (_shape_lp_depth), so
# this is a FLAT k=3 arm, which is what a depth map needs to mix from.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  echo "=== $S at ICCAD_SHAPE_LP_ITERS=3 ==="
  env $LP ICCAD_SHAPE_LP_ITERS=3 ICCAD_SHAPE_LP_STATS=l165_oos_stats_${S}.txt \
    "$PY" -u l140_oos_soft_audit.py run --sample $S --cores 48 \
    --out l165_oos_${S}_k3.json > l165_${S}_k3.log 2>&1
  echo "  exit=$?"
  grep -inE "fallback|unavailable|all profiles failed|\[constructive\]" \
    l165_${S}_k3.log | head -3
  echo "  scipy marker: $(grep -c 'scipy\] source=' l165_${S}_k3.log)"
done
echo L165_OOS_DONE
