#!/bin/sh
set -u
cd /c/ICCAD_ml/ship_final || exit 1
echo "=== L238 Linux lanes (rerun)  $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l238_wsl_final.sh' \
  > l238_wsl.log 2>&1
echo "L238_WSL_RC=$?"
grep -E "LANE|PASS|FAIL|liveness|LP gate|DETERMIN|GATE|budget|md5|ABORT|REFINE|L235|hits|depth map" l238_wsl.log | tail -45
echo L238_WSL_DONE $(date -u +%FT%TZ)
