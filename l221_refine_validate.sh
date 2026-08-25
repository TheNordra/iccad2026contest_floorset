#!/bin/sh
# L221 -- validate the REFINE finding before believing it.
#
# L219 measured REFINE 4->2 on n>100 as a ~20% profile-wall cut at roughly zero
# in-set quality cost (2 of 100 cases moved, net slightly better). Verified
# consistent: 81% of the 1020 heavy-band profiles got faster, median ratio
# 0.801, while the untouched n<=100 control band sat at 1.021 with 45% faster.
#
# Two things stand between that and a ship decision, and this project has been
# burned by both:
#
#   TIMING   one run pair. The control band read -2.9% where it must read 0, so
#            this box's run-to-run wall noise is ~3%. Repeat and take the pair
#            that agrees; the ledger's rule is min-of-N, never a single timing.
#   QUALITY  in-set only, and 2 movers is a small footprint to extrapolate from.
#            Every offline advantage in this ledger shrank or reversed OOS, and
#            M49 chose REFINE=4 by a STRICT SELECTION-PRESERVING derivation --
#            going below it changes selections by construction, so "2 movers in
#            set" is exactly the kind of number that grows out of sample.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l221.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -qE "L220_DONE|ABORT" l220_joint.out 2>/dev/null; do sleep 30; done
echo "=== L221 timing repeat  $(date -u +%FT%TZ) ==="
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
for R in 4 2; do
  rm -f "../l221_prof_r${R}.txt"
  env ICCAD_ADAPTIVE_CORES=48 L219_REFINE_HEAVY=$R \
      ICCAD_PROFILE_TIMING="../l221_prof_r${R}.txt" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
    -o "../results_L221_r${R}.json" > "../l221_r${R}.log" 2>&1
  echo "repeat REFINE=$R exit=$?  records=$(wc -l < "../l221_prof_r${R}.txt" 2>/dev/null || echo 0)"
done
cd /c/ICCAD_ml/ship_final || exit 1
echo; echo "=== L221 OOS quality, REFINE=2 on n>100  $(date -u +%FT%TZ) ==="
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 ICCAD_OOS_OPT=optimizer_l205probe L219_REFINE_HEAVY=2 \
    "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
    --out "l221_${S}_r2.json" > "l221_${S}_r2.log" 2>&1
  rc=$?; M=$(grep -c 'optimizer module -> optimizer_l205probe' "l221_${S}_r2.log")
  echo "$S/R2 exit=$rc  probe_module=$M  SAfallback=$(grep -c 'SA fallback' "l221_${S}_r2.log")"
  [ "$M" -eq 0 ] && { echo "  !! ABORT: probe module not used"; exit 1; }
done
"$PY" -u l221_score.py
echo L221_DONE
