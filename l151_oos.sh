#!/usr/bin/env bash
# L151 -- OOS for the COMBINED config. The OFF side must be re-run: the earlier
# l140_oos_*_c48.json baselines were taken on the L136 tree, and this tree now
# carries L137 defaults ON, so reusing them would price L137+L147 together and
# call it L147.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final || exit 1
for s in s1 s2; do
  $PY -u l140_oos_soft_audit.py run --sample "$s" --cores 48 \
      --out "l151_oos_${s}_off.json" > "l151_${s}_off.log" 2>&1; echo "$s off exit=$?"
  env $LP $PY -u l140_oos_soft_audit.py run --sample "$s" --cores 48 \
      --out "l151_oos_${s}_on.json"  > "l151_${s}_on.log"  2>&1; echo "$s on  exit=$?"
done
echo L151_DONE
