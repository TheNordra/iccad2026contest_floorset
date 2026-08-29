#!/bin/sh
# L170 OOS -- liveness已在 in-set 確認（4/100 相同 = LIVE，+0.0772%）。
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  echo "=== $S ICCAD_LP_HB_PRED=0.2994 ==="
  env $LP ICCAD_LP_HB_PRED=0.2994 \
    "$PY" -u l140_oos_soft_audit.py run --sample $S --cores 48 \
    --out l170_oos_${S}_hb.json > l170_${S}_hb.log 2>&1
  echo "  exit=$?  size=$(stat -c%s l170_oos_${S}_hb.json 2>/dev/null || echo MISSING)"
done
echo L170_OOS_DONE
