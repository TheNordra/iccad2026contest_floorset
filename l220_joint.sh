#!/bin/sh
# L220 -- joint re-derivation of the two runtime knobs.
#
# They are NOT additive by assumption: both shorten the same max-setter, and on
# the quality side dropping profiles changes which one wins while lowering
# REFINE changes what each one produces. Per-n composition across BLOCK COUNTS
# was verified exact (0/100 differ); composition across KNOBS is a different
# claim and gets tested before it is used.
#
# Stage 1: three joint arms. If q(k,R) == q(k,ship) + q(0,R) - q(0,ship) to
#          within noise on all 100 block counts, the marginals compose and the
#          joint table can be built from the 6+4 arms already measured.
# Stage 2: build the joint table and measure it.
#
# If additivity FAILS the marginals are useless for mixing and the honest move
# is the full 6x4 grid (24 arms, ~66 min), which l220_grid.sh does.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l220.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -qE "L219_DONE|ABORT" l219_refine.out 2>/dev/null; do sleep 30; done
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
echo "=== L220 additivity probe  $(date -u +%FT%TZ) ==="
for KR in 8:2 16:2 8:1; do
  K=${KR%%:*}; R=${KR##*:}
  env ICCAD_ADAPTIVE_CORES=48 L211_DROP_TABLE="../l211_drop_k${K}.json" \
      L219_REFINE_HEAVY=$R ICCAD_PROFILE_TIMING="../l220_prof_k${K}r${R}.txt" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
    -o "../results_L220_k${K}r${R}.json" > "../l220_k${K}r${R}.log" 2>&1
  rc=$?; D=$(grep -c 'drop table loaded' "../l220_k${K}r${R}.log")
  echo "k=$K R=$R exit=$rc  drop_table=$D  records=$(wc -l < "../l220_prof_k${K}r${R}.txt" 2>/dev/null || echo 0)"
  [ "$D" -eq 0 ] && { echo "  !! ABORT: drop table not loaded"; exit 1; }
done
cd /c/ICCAD_ml/ship_final || exit 1
"$PY" -u l220_additivity.py
echo L220_DONE
