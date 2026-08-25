#!/bin/sh
# L200 driver -- wait for the L199 in-set gates, score them, then run the five
# Linux lanes without a gap. Detached (nohup) so it survives the caller.
#
# The verdict is scored and KEPT either way, but the lanes launch regardless:
# a FAIL here is worth seeing next to the Linux result, not instead of it.
set -u
R=/c/ICCAD_ml/ship_final
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd "$R" || exit 1

while ! grep -q "L199_GATES_DONE" l199_gates.out 2>/dev/null; do sleep 15; done
echo "=== L199 gates finished $(date -u +%FT%TZ) ==="
cat l199_gates.out

echo; echo "=== L199 verdict ==="
"$PY" -u l199_verdict.py 2>&1 | tee l199_verdict.out
echo "VERDICT_RC=$?"

echo; echo "=== L200 Linux lanes launching $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l200_wsl_verify.sh' \
  > l200_wsl_verify.log 2>&1
echo "L200_WSL_RC=$?"
tail -40 l200_wsl_verify.log
echo "L200_DRIVE_DONE $(date -u +%FT%TZ)"
