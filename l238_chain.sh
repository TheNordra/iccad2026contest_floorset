#!/bin/sh
# L238 -- full re-verification of the L234 + L235 package.
set -u
R=/c/ICCAD_ml/ship_final
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd "$R" || exit 1
export PYTHONIOENCODING=utf-8
echo "=== L238 in-set gates  $(date -u +%FT%TZ) ==="
sh l238_gates.sh
echo; echo "=== L238 verdict ==="
"$PY" -u l238_verdict.py 2>&1 | tee l238_verdict.out
echo; echo "=== L238 Linux lanes  $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l238_wsl_final.sh' \
  > l238_wsl.log 2>&1
echo "L238_WSL_RC=$?"
grep -E "LANE|PASS|FAIL|liveness|LP gate|DETERMIN|GATE|budget|md5|ABORT|REFINE|L235|hits" l238_wsl.log | tail -45
echo "L238_CHAIN_DONE $(date -u +%FT%TZ)"
