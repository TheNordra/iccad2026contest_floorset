#!/bin/sh
# L226 -- full re-verification of the k=8 + REFINE=2 package.
#   1. 10 in-set arms (adds refine4, the L223 kill switch)
#   2. verdict with G11: the kill switch must reproduce the pre-L223 package
#      bit-for-bit AND the default must move exactly the 1 measured case, which
#      must lie above n=100 -- a band that leaked into the mid band shows up
#      only in that third check.
#   3. the five Linux lanes
set -u
R=/c/ICCAD_ml/ship_final
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd "$R" || exit 1
while ! grep -qE "L225_STAGED|NOT IMPLEMENTING|ABORT" l225_implement.out 2>/dev/null; do sleep 30; done
if ! grep -q "L225_STAGED" l225_implement.out 2>/dev/null; then
  echo "L226 ABORT: L225 did not stage (see l225_implement.out)"; exit 1
fi
echo "=== L226 in-set gates  $(date -u +%FT%TZ) ==="
sh l226_gates.sh
echo; echo "=== L226 verdict ==="
"$PY" -u l199_verdict.py L226 2>&1 | tee l226_verdict.out
echo; echo "=== L226 Linux lanes  $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l207_wsl_final.sh' \
  > l226_wsl.log 2>&1
echo "L226_WSL_RC=$?"
grep -E "LANE|PASS|FAIL|liveness|LP gate|DETERMIN|GATE|budget|md5|ABORT|REFINE|POOLDROP" l226_wsl.log | tail -40
echo "L226_CHAIN_DONE $(date -u +%FT%TZ)"
