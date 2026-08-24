#!/bin/sh
# L167 -- does the 32-vector probe tier transfer out of sample?
#
# In set it is +0.4519% on the current tree (control reproduces
# results_L165_det1.json bit-for-bit, so the arm is a clean A/B). That is not
# the question. This project has measured in-sample-to-OOS transfer as low as
# 5% (M76) and 15-25% (L127); the one time it transferred well was M80, whose
# vectors were also drawn without in-sample fitting -- which is exactly the
# property these 32 have, being uniformly random.
#
# Same driver, same two disjoint 240-case samples, same $LP string as
# l157_oos.sh / l165_oos_k3.sh. ICCAD_LENSD_POOL=1 is the only difference.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  echo "=== $S with ICCAD_LENSD_POOL=1 ==="
  env $LP ICCAD_LENSD_POOL=1 ICCAD_SHAPE_LP_STATS=l167_oos_stats_${S}.txt \
    "$PY" -u l140_oos_soft_audit.py run --sample $S --cores 48 \
    --out l167_oos_${S}_pool.json > l167_${S}_pool.log 2>&1
  echo "  exit=$?"
  grep -icE "fallback|unavailable|all profiles failed|\[constructive\]" l167_${S}_pool.log
done
echo L167_OOS_DONE
