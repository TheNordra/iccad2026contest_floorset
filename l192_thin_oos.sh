#!/bin/sh
# L192 -- does the thin pool's in-set quality deficit hold OUT of sample?
#
# L189: thin pool + LP k=1 is the ONLY configuration that beats beta without
# betting on route A (+0.268% neutral, rank 2 if route A delivers). It gets
# there by giving up 1.390% of IN-SET quality (+6.450% -> +5.060%) to save
# 7.4 grader-seconds.
#
# That 1.390% is the number under test. In-set has misled us twice today:
#   * the L124 twins moved 0 of 100 in-set cases and are worth +0.67% OOS
#   * L171 read +0.0772% on the old depth map and -0.0512% on the new one
# and the thin pool turns OFF three mechanisms at once, including the twins --
# whose entire value is known to live out of sample. So the in-set deficit is
# very likely an UNDER-estimate of what thinning really costs.
#
# TWO ARMS, one variable each way, LP at k=1 in BOTH so this measures the POOL:
#   full  = the shipped pool (51 profiles)
#   thin  = ICCAD_M80_TIER=0 ICCAD_M124_TWIN=0 ICCAD_HINT_MODE=0 (35 profiles)
#
# ICCAD_ROUTE_A=0 in both, for wall only -- route A is verified result-neutral
# (bit-identical single cases; L177 det1 vs det2 matched 100/100 on cost AND
# positions with it live) and costs 2.9x on this 16-physical-core box.
#
# ICCAD_SHAPE_LP_DEPTH2=0 in both: k=1 strictly dominates the depth map in both
# route-A scenarios (L189), so the depth map is not what is being decided here.
#
# LOCKFILE: four measurements were destroyed today by two agents sharing this
# box (see _quarantine/).
set -u
LOCK=/c/ICCAD_ml/ship_final/.l192.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  for ARM in full thin; do
    if [ "$ARM" = thin ]; then
      EX="ICCAD_M80_TIER=0 ICCAD_M124_TWIN=0 ICCAD_HINT_MODE=0"
    else
      EX="ICCAD_L192_FULL=1"
    fi
    env ICCAD_ROUTE_A=0 ICCAD_SHAPE_LP_DEPTH2=0 $EX \
      "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
      --out "l192_${S}_${ARM}.json" > "l192_${S}_${ARM}.log" 2>&1
    echo "$S/$ARM exit=$?  SAfallback=$(grep -c 'SA fallback' "l192_${S}_${ARM}.log")  $(grep -E 'feasible|weighted cost' "l192_${S}_${ARM}.log" | tr -s ' ' | tr '\n' ' ')"
  done
done
echo L192_THIN_OOS_DONE
