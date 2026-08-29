#!/bin/bash
# L312 -- RF-SAFE on BOTH held-out samples. L275's rule: an arm must be positive
# on s1 AND s2 before it is a candidate. ship and gate0 are already cached from
# L295/L299, so only rfsafe's 240 solves per sample actually run; gate0 rides
# along free and gives the "how much of the full ungate did we keep" column.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  echo "===== $S ====="; date +"start %H:%M:%S"
  "$PY" -u l287_transfer.py --sample $S --arms ship,gate0,rfsafe \
      > "l312_oos_${S}.log" 2>&1
  echo "exit=$?"; date +"end   %H:%M:%S"
  tail -14 "l312_oos_${S}.log"
done
echo L312_OOS_DONE
