#!/bin/sh
# L171 -- re-run of the L170 hb-baseline OOS, after the first attempt was
# destroyed by a race (see _quarantine/README.txt: two copies of
# l170_oos_hb.sh ran at once, 30+ constructive.exe, 132/240 cases fell back
# to python SA, weighted cost 9.900044 against a healthy 1.4169).
#
# LOCKFILE: this is the fix for that failure mode. A second launch exits.
# LIVENESS: no separate in-set arm. The control is the arm-mixed shipped
# depth map (l147 k=1 / l157 k=2 / l165 k=3), so if this run comes back
# identical to the mix, the flag was a no-op and l171_score.py will say so.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l171.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy of this script is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  echo "=== $S with ICCAD_LP_HB_PRED=0.2994 (gated depth, shipped default) ==="
  env $LP ICCAD_LP_HB_PRED=0.2994 ICCAD_SHAPE_LP_STATS=l171_oos_stats_${S}.txt \
    "$PY" -u l140_oos_soft_audit.py run --sample $S --cores 48 \
    --out l171_oos_${S}_hb.json > l171_${S}_hb.log 2>&1
  echo "  exit=$?"
  echo "  SA fallbacks: $(grep -c 'SA fallback' l171_${S}_hb.log)  (MUST be 0)"
  echo "  scipy marker: $(grep -c 'scipy] source=' l171_${S}_hb.log)  (MUST be 1)"
  grep -E "feasible|weighted cost" l171_${S}_hb.log | head -2
done
echo L171_OOS_DONE
