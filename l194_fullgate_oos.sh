#!/bin/sh
# L194 -- measure the LP gate's capture efficiency on the FULL pool.
#
# L193 measured it on the THIN pool: the gate fires on 45% of block counts and
# collects 40% of the LP's OOS quality, i.e. an efficiency of 0.889. The
# full-pool candidate (best upside: rank 2 if route A delivers, still ahead of
# beta if it does not) was scored by APPLYING that 0.889 to a gate that fires on
# only 30% of block counts. That is an assumption, not a measurement, and it is
# the last one holding the recommendation up.
#
# It could plausibly go either way. The full pool's own wall has already eaten
# most of the budget, so the 30% that survive the gate are the cases with the
# MOST slack -- which are not necessarily the cases the LP helps most. If the
# LP's value is concentrated on the heavy, low-slack cases, the full-pool gate
# will capture far less than 30% and the candidate collapses.
#
# ALREADY MEASURED: full pool + LP k=1, both samples -> l192_{s}_full.json
# MISSING: full pool with the LP OFF -> this script.
# Then arm-mix, which the ledger records as exact.
#
# Same flags as the L192 full arm except ICCAD_SHAPE_LP=0, so the pair differs
# in exactly one thing.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l194.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 ICCAD_SHAPE_LP=0 \
    "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
    --out "l194_${S}_fulloff.json" > "l194_${S}_fulloff.log" 2>&1
  echo "$S/fulloff exit=$?  SAfallback=$(grep -c 'SA fallback' "l194_${S}_fulloff.log")  $(grep -E 'feasible|weighted cost' "l194_${S}_fulloff.log" | tr -s ' ' | tr '\n' ' ')"
done
echo L194_FULLGATE_OOS_DONE
