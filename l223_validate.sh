#!/bin/sh
# L223 -- validate the two REFINE candidates, cheapest-first and short-circuiting.
#
# Replaces L221+L222, which ran the WEAKER candidate's OOS first. If the joint
# arm survives OOS it dominates REFINE-alone (+4.233% vs +3.667%), so
# REFINE-alone's 70 minutes are only worth spending when the joint fails.
#
# Order: all timing first (short, and it is the part that needs an unloaded
# box), then the joint OOS, then REFINE-alone OOS only if the joint fell over.
#
# Timing is min-of-N on BOTH sides of the ratio -- the baseline is re-run the
# same number of times, because a ratio built from one noisy numerator and one
# noisy denominator is twice as noisy as either. The control band (n<=100, where
# REFINE is untouched) read -2.9% on a single pair, which is this box's noise
# floor and the thing these repeats exist to average away.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l223.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
cd "$R/iccad2026contest" || exit 1
echo "=== L223 timing, 3 repeats per arm  $(date -u +%FT%TZ) ==="
for I in 1 2 3; do
  for ARM in r4 r2 k8r2; do
    case $ARM in
      r4)   ENVX="L219_REFINE_HEAVY=4" ;;
      r2)   ENVX="L219_REFINE_HEAVY=2" ;;
      k8r2) ENVX="L219_REFINE_HEAVY=2 L211_DROP_TABLE=../l211_drop_k8.json" ;;
    esac
    rm -f "../l223_prof_${ARM}_${I}.txt"
    env ICCAD_ADAPTIVE_CORES=48 $ENVX \
        ICCAD_PROFILE_TIMING="../l223_prof_${ARM}_${I}.txt" \
      "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
      -o "../results_L223_${ARM}_${I}.json" > "../l223_${ARM}_${I}.log" 2>&1
    echo "  $ARM rep$I exit=$?  records=$(wc -l < "../l223_prof_${ARM}_${I}.txt" 2>/dev/null || echo 0)"
  done
done
cd "$R" || exit 1
"$PY" -u l223_timing.py | tee l223_timing.out
echo; echo "=== L223 OOS: the JOINT arm first  $(date -u +%FT%TZ) ==="
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 ICCAD_OOS_OPT=optimizer_l205probe \
      L211_DROP_TABLE=l211_drop_k8.json L219_REFINE_HEAVY=2 \
    "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
    --out "l223_${S}_k8r2.json" > "l223_${S}_k8r2.log" 2>&1
  rc=$?; M=$(grep -c 'optimizer module -> optimizer_l205probe' "l223_${S}_k8r2.log")
  D=$(grep -c 'drop table loaded' "l223_${S}_k8r2.log")
  echo "$S/k8r2 exit=$rc  probe_module=$M  drop_table=$D  SAfallback=$(grep -c 'SA fallback' "l223_${S}_k8r2.log")"
  { [ "$M" -eq 0 ] || [ "$D" -eq 0 ]; } && { echo "  !! ABORT: arm did not carry both knobs"; exit 1; }
done
echo; echo "=== L223 OOS: REFINE alone, only as the fallback  $(date -u +%FT%TZ) ==="
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 ICCAD_OOS_OPT=optimizer_l205probe L219_REFINE_HEAVY=2 \
    "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
    --out "l223_${S}_r2.json" > "l223_${S}_r2.log" 2>&1
  echo "$S/r2 exit=$?  probe_module=$(grep -c 'optimizer module -> optimizer_l205probe' "l223_${S}_r2.log")"
done
echo L223_DONE
