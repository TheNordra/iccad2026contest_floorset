#!/bin/sh
# L208 -- the final Linux verify, chained after L206's restage.
#
# L200's lanes ran against the PRE-instrument tar. L206 proves the instrument
# is inert and restages, so the tar changes identity even though the results do
# not. The package that ships must be the package that was verified, so the
# five lanes run once more against the final tar -- with the two stale
# thresholds l200 inherited corrected (see l207_wsl_final.sh's header).
set -u
R=/c/ICCAD_ml/ship_final
cd "$R" || exit 1
while ! grep -q "L206_CHAIN_DONE" l206_chain.out 2>/dev/null; do sleep 20; done
if ! grep -q "^PASS$" _l206_inert.flag 2>/dev/null; then
  echo "L208 ABORT: the instrument was not proven inert; not verifying a tar"
  echo "            that was never restaged."; exit 1
fi
echo "=== L208 final Linux verify $(date -u +%FT%TZ) ==="
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
  wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l207_wsl_final.sh' \
  > l207_wsl_final.log 2>&1
echo "L207_WSL_RC=$?"
grep -E "LANE|PASS|FAIL|liveness|LP gate|DETERMIN|GATE|budget|md5|ABORT" \
  l207_wsl_final.log | tail -40
echo "L208_CHAIN_DONE $(date -u +%FT%TZ)"
