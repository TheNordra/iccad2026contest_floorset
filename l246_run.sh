#!/bin/sh
set -u
cd /c/ICCAD_ml/ship_final || exit 1
echo "=== L246 Linux lanes on the NO-VENDOR package  $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l246_wsl_D.sh' \
  > l246_wsl.log 2>&1
echo "L246_WSL_RC=$?"
grep -E "LANE|PASS|FAIL|liveness|LP gate|DETERMIN|GATE|md5|ABORT|REFINE|vendor|requirements|depth map|L235" l246_wsl.log | tail -40
echo L246_DONE_ALL $(date -u +%FT%TZ)
