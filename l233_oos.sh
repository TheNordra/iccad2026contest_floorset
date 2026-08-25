#!/bin/sh
# L233 -- the mid band OUT OF SAMPLE, which is the half that decides it.
#
# In set the mid-band cut is free (L231 rep-1: -0.005% at mid=3, +0.000% at
# mid=2). It was free in set for the HEAVY band too -- +0.0400%, 2 of 100 cases
# moving -- and out of sample that flipped to -0.1788%. M50/M74 derived the 6
# STRICTLY SELECTION-PRESERVING, so going below it changes selections BY
# CONSTRUCTION and the in-set null was never going to survive. The in-set corpus
# has exactly ONE case per block count; the OOS samples carry 2-4 different
# floorplans at each, which is where a changed selection can actually lose.
#
# BASELINE ALREADY EXISTS: l223_{s}_r2.json is heavy=2 with the probe's own
# mid=6 default -- i.e. the shipped configuration -- run on 2026-08-25. Only
# the treated arm needs running, and it must carry BOTH bands explicitly
# through L219_REFINE_TABLE so a table that failed to load reads as a no-op
# rather than as a win.
#
# Cheapest-first: mid=2 is the biggest wall cut and dominates mid=3 if it
# survives. mid=3 runs only if mid=2's OOS cost eats the RF gain.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l233.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
cd "$R" || exit 1
while [ -d "$R/.l230.lock" ] || [ -d "$R/.l231.lock" ] || [ -d "$R/.l232.lock" ]; do
  sleep 30
done
echo "=== L233 mid-band OOS  $(date -u +%FT%TZ) ==="
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 ICCAD_OOS_OPT=optimizer_l205probe \
      L219_REFINE_TABLE=l231_mid2.json \
    "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
    --out "l233_${S}_mid2.json" > "l233_${S}_mid2.log" 2>&1
  rc=$?
  M=$(grep -c 'optimizer module -> optimizer_l205probe' "l233_${S}_mid2.log")
  T=$(grep -c 'refine table loaded' "l233_${S}_mid2.log")
  echo "$S/mid2 exit=$rc probe_module=$M refine_table=$T SAfallback=$(grep -c 'SA fallback' "l233_${S}_mid2.log")"
  if [ "$M" -eq 0 ] || [ "$T" -eq 0 ]; then
    echo "  !! ABORT: the arm did not carry the probe module AND the table"
    exit 1
  fi
done
echo L233_OOS_DONE $(date -u +%FT%TZ)
