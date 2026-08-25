#!/bin/sh
# L222 -- validate the JOINT candidate the same way L221 validates REFINE alone.
#
# Measured in set, single-run timing: k=8 + REFINE=2 reads NET +4.233%
# (graded 0.88737) against an r2 threshold of 0.88819 -- past it by 0.09%,
# while the per-n wall measurement carries ~1pp of NET noise. That is a
# candidate, not a rank.
#
# Two more timing repeats (min-of-N, the ledger's rule) and both OOS samples.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l222.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -qE "L221_DONE|ABORT" l221_validate.out 2>/dev/null; do sleep 30; done
echo "=== L222 joint timing repeats  $(date -u +%FT%TZ) ==="
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
for I in 2 3; do
  rm -f "../l222_prof_k8r2_${I}.txt"
  env ICCAD_ADAPTIVE_CORES=48 L211_DROP_TABLE=../l211_drop_k8.json \
      L219_REFINE_HEAVY=2 ICCAD_PROFILE_TIMING="../l222_prof_k8r2_${I}.txt" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
    -o "../results_L222_k8r2_${I}.json" > "../l222_k8r2_${I}.log" 2>&1
  rc=$?; D=$(grep -c 'drop table loaded' "../l222_k8r2_${I}.log")
  echo "repeat $I exit=$rc  drop_table=$D  records=$(wc -l < "../l222_prof_k8r2_${I}.txt" 2>/dev/null || echo 0)"
  [ "$D" -eq 0 ] && { echo "  !! ABORT: drop table not loaded"; exit 1; }
done
# a matching pair of REFINE=4 repeats, so the ratio is min-of-N on BOTH sides
for I in 2 3; do
  rm -f "../l222_prof_r4_${I}.txt"
  env ICCAD_ADAPTIVE_CORES=48 L219_REFINE_HEAVY=4 \
      ICCAD_PROFILE_TIMING="../l222_prof_r4_${I}.txt" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
    -o "../results_L222_r4_${I}.json" > "../l222_r4_${I}.log" 2>&1
  echo "baseline repeat $I exit=$?  records=$(wc -l < "../l222_prof_r4_${I}.txt" 2>/dev/null || echo 0)"
done
cd /c/ICCAD_ml/ship_final || exit 1
echo; echo "=== L222 OOS, k=8 + REFINE=2  $(date -u +%FT%TZ) ==="
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 ICCAD_OOS_OPT=optimizer_l205probe \
      L211_DROP_TABLE=l211_drop_k8.json L219_REFINE_HEAVY=2 \
    "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
    --out "l222_${S}_k8r2.json" > "l222_${S}_k8r2.log" 2>&1
  rc=$?; M=$(grep -c 'optimizer module -> optimizer_l205probe' "l222_${S}_k8r2.log")
  D=$(grep -c 'drop table loaded' "l222_${S}_k8r2.log")
  echo "$S exit=$rc  probe_module=$M  drop_table=$D  SAfallback=$(grep -c 'SA fallback' "l222_${S}_k8r2.log")"
  { [ "$M" -eq 0 ] || [ "$D" -eq 0 ]; } && { echo "  !! ABORT: arm did not carry both knobs"; exit 1; }
done
echo L222_DONE
