#!/bin/bash
# L299 -- the mix arm on BOTH held-out samples, END TO END.
# Its s1/s2 values so far are arm-mixed. L296 proved that mixing is exact here
# (the disjointness holds bit-for-bit on both samples), but exact-by-argument is
# not the same as run -- and mixing cannot see an interaction it does not model.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  echo "===== $S ====="; date +"start %H:%M:%S"
  "$PY" -u l287_transfer.py --sample $S --arms ship,lp2,gate0,both,mix \
      > "l299_${S}.log" 2>&1
  echo "exit=$?"; date +"end   %H:%M:%S"
  tail -12 "l299_${S}.log"
done
echo L299_DONE
