#!/usr/bin/env bash
# L157 OOS -- the deciding measurement for selective LP depth.
#
# L157 priced the gate at NET +0.231%~+0.411% against a 0.30% bar using in-set
# quality, i.e. UNDECIDED. The unknown is how much of k=2's quality survives out
# of sample; L147 transferred at 86%, which would put the brackets at
# +0.199%/+0.353% -- still straddling. So it has to be measured.
#
# Only the k=2 arms are run. The k=1 side already exists and is bit-verified:
#   s1 OFF  l151_oos_s1_on.json  (== l154_oos_s1_off.json, 240/240 cost AND
#                                 positions -- L154 §2 proved the refactor was a
#                                 no-op through this very driver)
#   s2 OFF  l151_oos_s2_on.json
# Re-running them would only add timing noise to a comparison that is exact.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final || exit 1

for S in s1 s2; do
  echo "=== $S at ICCAD_SHAPE_LP_ITERS=2 ==="
  env $LP ICCAD_SHAPE_LP_ITERS=2 ICCAD_SHAPE_LP_STATS=l157_oos_stats_${S}.txt \
    "$PY" -u l140_oos_soft_audit.py run --sample $S --cores 48 \
    --out l157_oos_${S}_k2.json > l157_${S}_k2.log 2>&1
  echo "  exit=$?"
  grep -inE "fallback|unavailable|all profiles failed|\[constructive\]" \
    l157_${S}_k2.log | head -3
done
echo L157_OOS_DONE
