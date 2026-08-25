#!/bin/sh
# L201 -- the decisive arm for G2, the one the stale anchor cannot settle.
#
# results_L165_l147off.json was produced before the LP gate existed, when
# ICCAD_SHAPE_LP_L147=0 ran the LP on all 100 cases. Under L196 it runs on 63,
# so the flat bit-compare reads 63/100 on a package where nothing is wrong.
# Measured: the 37 that differ are EXACTLY the 37 the gate drops (both
# directions empty), so the hatch itself is intact -- but "exactly explains the
# difference" is an argument, not a gate.
#
# This arm removes the argument. With L147 off the depth is k=1 in BOTH trees
# (_shape_lp_depth gates depth>=2 on tangent_on, not on the map), so
# ICCAD_SHAPE_LP_L147=0 + ICCAD_LP_GATE=0 IS the L165 configuration and must
# reproduce that anchor 100/100. Anything less is a real regression in the
# escape hatch, not a bookkeeping artefact.
#
# Run AFTER the Linux lanes -- it competes for the same cores.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l201.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
tag=l147off_gateoff
rm -f "../l199_${tag}_stats.txt"
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l199_${tag}_stats.txt" \
    ICCAD_SHAPE_LP_L147=0 ICCAD_LP_GATE=0 \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o "../results_L199_${tag}.json" > "../l199_${tag}.log" 2>&1
rc=$?
n=$(wc -l < "../l199_${tag}_stats.txt" 2>/dev/null || echo 0)
echo "$tag exit=$rc  LPran=${n} (expect 100)  SAfallback=$(grep -c 'SA fallback' "../l199_${tag}.log")  scipy=$(grep -c 'scipy] source=' "../l199_${tag}.log")"
cd /c/ICCAD_ml/ship_final || exit 1
echo; echo "=== L199 verdict, final (G2c present) ==="
"$PY" -u l199_verdict.py 2>&1 | tee l199_verdict_final.out
echo L201_DONE
