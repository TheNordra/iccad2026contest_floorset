#!/bin/sh
# L210 -- the full re-verification chain for the route-A-off package.
#   1. in-set gates, 8 arms (L209)
#   2. verdict with G9: every arm must reproduce its L199 counterpart
#      BIT-FOR-BIT. Route A is verified result-neutral, so the change is
#      supposed to move wall and nothing else; a result difference would mean
#      it never was neutral, which matters far more than the wall saving.
#   3. the five Linux lanes against the new tar
set -u
R=/c/ICCAD_ml/ship_final
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd "$R" || exit 1
echo "=== L209 in-set gates, route A OFF  $(date -u +%FT%TZ) ==="
sh l209_gates.sh
echo; echo "=== L209 verdict (G9 = bit-identical to L199) ==="
"$PY" -u l199_verdict.py L209 L199 2>&1 | tee l209_verdict.out
echo; echo "=== L210 Linux lanes  $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l207_wsl_final.sh' \
  > l210_wsl.log 2>&1
echo "L210_WSL_RC=$?"
grep -E "LANE|PASS|FAIL|liveness|LP gate|DETERMIN|GATE|budget|md5|ABORT|route A off" l210_wsl.log | tail -40
echo "L210_CHAIN_DONE $(date -u +%FT%TZ)"
