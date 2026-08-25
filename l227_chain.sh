#!/bin/sh
# L227 -- full re-verification of the REFINE=2 package (pool drop OFF).
#
# L224's uncontended timing reversed the L225 plan: stacked on REFINE=2 the
# pool drop is worth +0.203pp of wall against -0.2989pp of measured OOS
# quality, so it came off. The contended measurement disagreed because dropping
# 8 of 51 processes ALSO relieves scheduler contention on this 32-core box --
# relief the 48-core grader does not have.
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
echo "=== L227 in-set gates  $(date -u +%FT%TZ) ==="
sh l227_gates.sh
echo; echo "=== L227 verdict ==="
"$PY" -u l199_verdict.py L227 2>&1 | tee l227_verdict.out
echo; echo "=== L227 Linux lanes  $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l207_wsl_final.sh' \
  > l227_wsl.log 2>&1
echo "L227_WSL_RC=$?"
grep -E "LANE|PASS|FAIL|liveness|LP gate|DETERMIN|GATE|budget|md5|ABORT|REFINE|POOLDROP" l227_wsl.log | tail -40
echo "L227_CHAIN_DONE $(date -u +%FT%TZ)"
