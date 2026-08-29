#!/bin/bash
# L295 -- gate0 on BOTH held-out samples. L275's rule: an arm must be positive
# on s1 AND s2 before it is a candidate. `ship` is already cached for both, so
# only the new arm's 240 solves per sample actually run.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  echo "===== $S ====="
  date +"start %H:%M:%S"
  "$PY" -u l287_transfer.py --sample $S --arms ship,lp2,gate0 \
      > "l295_${S}.log" 2>&1
  echo "exit=$?"
  date +"end   %H:%M:%S"
  tail -14 "l295_${S}.log"
done
echo L295_DONE
