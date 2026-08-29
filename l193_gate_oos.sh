#!/bin/sh
# L193 -- the OOS half the gated-thin candidate is missing.
#
# The gate is a per-n rule, not an implemented mechanism, so there is nothing to
# run: arm-mixing gives the exact answer (the ledger records mixing flat arms as
# reproducing a really-run gated arm 100/100 on cost AND positions). The gate
# picks, per block count, between "thin pool, LP off" and "thin pool, LP k=1".
#
# ALREADY MEASURED (L192): thin pool + LP k=1 on both samples,
#   l192_s1_thin.json  l192_s2_thin.json
# MISSING: thin pool with the LP OFF, same two samples. That is this script.
#
# Same flags as the L192 thin arm except ICCAD_SHAPE_LP=0, so the pair differs
# in exactly one thing and the mix is clean.
#
# ⚠️ What this can and cannot settle. The RF side of the candidate is already
# measured (real walls, calibration-free transport) and does not depend on the
# corpus. What is in-set and therefore under test is WHICH CASES the LP helps --
# a per-case selection rule, which is the form that has failed out of sample
# most often in this ledger (L127 tally fitting at 15-25% transfer; today the
# twins, L171, and the thin pool all moved against their in-set reading).
# Expect the OOS number to come back smaller than +2.551%.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l193.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 ICCAD_SHAPE_LP=0 \
      ICCAD_M80_TIER=0 ICCAD_M124_TWIN=0 ICCAD_HINT_MODE=0 \
    "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
    --out "l193_${S}_thinoff.json" > "l193_${S}_thinoff.log" 2>&1
  echo "$S/thinoff exit=$?  SAfallback=$(grep -c 'SA fallback' "l193_${S}_thinoff.log")  $(grep -E 'feasible|weighted cost' "l193_${S}_thinoff.log" | tr -s ' ' | tr '\n' ' ')"
done
echo L193_GATE_OOS_DONE
