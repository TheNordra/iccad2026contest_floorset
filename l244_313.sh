#!/bin/sh
cd /c/ICCAD_ml/ship_final || exit 1
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l163_test313.sh' 2>&1
echo L244_DONE
