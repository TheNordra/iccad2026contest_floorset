#!/bin/sh
# L219 -- price the OTHER runtime knob: make the slowest profiles cheaper
# instead of dropping them.
#
# REFINE_ITERS is a C++ post-processing budget the wrapper sets per band
# (_M49_REFINE_BAND: 6 for 60<n<=100, 4 for n>100). M49 derived those strictly
# selection-preserving, i.e. it took the free part and stopped, and the ledger
# then recorded "do not stack more wall cuts, the floor is saturated". That
# premise no longer holds: 45 of 100 cases sit ABOVE the RF floor, and there
# cutting REFINE collects real RF.
#
# Four arms on the heavy band only (n>100, which carries the deficit): the
# shipped 4, then 3, 2, 1. Durations are instrumented so the wall side is
# measured, not modelled from a constant.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l219.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -qE "L217_DONE|ABORT" l217_measure.out 2>/dev/null; do sleep 30; done
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
echo "=== L219 REFINE sweep on n>100  $(date -u +%FT%TZ) ==="
for R in 4 3 2 1; do
  rm -f "../l219_prof_r${R}.txt"
  env ICCAD_ADAPTIVE_CORES=48 L219_REFINE_HEAVY=$R \
      ICCAD_PROFILE_TIMING="../l219_prof_r${R}.txt" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
    -o "../results_L219_r${R}.json" > "../l219_r${R}.log" 2>&1
  echo "REFINE=$R exit=$?  records=$(wc -l < "../l219_prof_r${R}.txt" 2>/dev/null || echo 0)"
done
cd /c/ICCAD_ml/ship_final || exit 1
"$PY" -u l219_score.py
echo L219_DONE
