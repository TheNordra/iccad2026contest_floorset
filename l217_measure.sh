#!/bin/sh
# L217 -- measure the targeted per-block-count drop shape.
#
# The shipped table drops 8 profiles at EVERY block count. That is the wrong
# shape for the same reason L203 found the LP gate's family was: the objective
# is separable in n, and RF's derivative is ZERO on the floor -- so dropping at
# a case already on the floor buys exactly nothing and costs quality. 54 of 100
# block counts are on the floor in the shipped configuration.
#
# The shape here maximises, per block count, the EXACT RF gain minus a quality
# price that is linear in k and calibrated globally (lam x2, the conservative
# end). No per-n quality fitting -- that is what the 2.41x OOS amplification
# exists to punish. Robust: NET stays +2.14..+2.31% across a 6x range of lam.
#
# Modelled at NET +2.190% vs the shipped +1.692%. Measured here.
#
# The OOS baselines already exist (l213_s{1,2}_base.json, no drop), so only the
# treated arm needs running.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l217.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
echo "=== L217 in set  $(date -u +%FT%TZ) ==="
env ICCAD_ADAPTIVE_CORES=48 L211_DROP_TABLE=../l217_drop_targeted.json \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
  -o ../results_L217_inset.json > ../l217_inset.log 2>&1
echo "inset exit=$?  table=$(grep -c 'drop table loaded' ../l217_inset.log)"
cd /c/ICCAD_ml/ship_final || exit 1
echo; echo "=== L217 OOS  $(date -u +%FT%TZ) ==="
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 ICCAD_OOS_OPT=optimizer_l205probe \
      L211_DROP_TABLE=l217_drop_targeted.json \
    "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
    --out "l217_${S}_tgt.json" > "l217_${S}_tgt.log" 2>&1
  rc=$?; T=$(grep -c 'drop table loaded' "l217_${S}_tgt.log")
  M=$(grep -c 'optimizer module -> optimizer_l205probe' "l217_${S}_tgt.log")
  echo "$S/tgt exit=$rc  probe_module=$M  drop_table=$T  SAfallback=$(grep -c 'SA fallback' "l217_${S}_tgt.log")"
  [ "$M" -eq 0 ] && { echo "  !! ABORT: probe module not used"; exit 1; }
  [ "$T" -eq 0 ] && { echo "  !! ABORT: drop table not loaded -- this arm measured the baseline"; exit 1; }
done
"$PY" -u l217_score.py
echo L217_DONE
