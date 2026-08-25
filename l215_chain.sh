#!/bin/sh
# L215 -- full re-verification of the pool-drop package (route A off + L211/L213).
#   1. in-set gates, 9 arms (adds pooldropoff)
#   2. verdict, now with G10 -- the kill switch must reproduce L209 bit-for-bit
#      AND the default must move exactly the 12 cases the drop was measured on.
#      A table that never loaded passes the first and fails the second; a table
#      that dropped everything passes the second and fails the first.
#   3. the five Linux lanes against the new tar
set -u
R=/c/ICCAD_ml/ship_final
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd "$R" || exit 1
echo "=== L214 in-set gates, pool drop ON  $(date -u +%FT%TZ) ==="
sh l214_gates.sh
echo; echo "=== L214 verdict ==="
"$PY" -u l199_verdict.py L214 2>&1 | tee l214_verdict.out
echo; echo "=== L215 Linux lanes  $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l207_wsl_final.sh' \
  > l215_wsl.log 2>&1
echo "L215_WSL_RC=$?"
grep -E "LANE|PASS|FAIL|liveness|LP gate|DETERMIN|GATE|budget|md5|ABORT|pool drop|POOLDROP" l215_wsl.log | tail -40
echo "L215_CHAIN_DONE $(date -u +%FT%TZ)"
