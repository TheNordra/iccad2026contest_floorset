#!/bin/sh
cd /c/ICCAD_ml/ship_final || exit 1
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l244b_wsl.sh' \
  > l244b_wsl.log 2>&1
echo "L244B_WSL_RC=$?"
grep -E "D tar|op_wrapper|vendor entries|WSL scipy|D1_RC|D2_RC|weighted cost|feasible|worth|PASS|FAIL|LINUX-VERIFY|LP liveness" l244b_wsl.log | tail -30
echo L244B_RUN_DONE
