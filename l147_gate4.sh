#!/usr/bin/env bash
# L147 Gate 4 -- OOS 240 x 2 at the grader's 48-core pool shape, chained after
# Gate 3's timing runs so the box is never shared with a wall measurement.
#
# Harness is l140_oos_soft_audit.py, NOT l137_oos_ab.py: the latter captures
# only ICCAD_HINT_* around m77_oos_probe's import-time strip of every ICCAD_*,
# so ICCAD_SHAPE_LP_* would be silently dropped and both arms would come out
# byte-identical -- a clean, plausible, completely empty A/B.
#
# The OFF side already exists: l140_oos_s{1,2}_c48.json were produced by this
# same harness on this tree with the flags off, and Gate 1 proved flags-off is
# bit-identical. So only the ON side has to run.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1

while ! grep -q GATE3_DONE l147_gate3.log 2>/dev/null; do sleep 30; done
echo "=== Gate 3 evals finished; pricing ==="
"$PY" -u l147_price.py --quality 2.4995 --arm r15g > l147_gate3_price.txt 2>&1
cat l147_gate3_price.txt

echo "=== Gate 4: OOS s1/s2, arm ON ==="
for s in s1 s2; do
  ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 \
  ICCAD_SHAPE_LP_PRICE=1.0 \
    "$PY" -u l140_oos_soft_audit.py run --sample "$s" --cores 48 \
      --out "l147_oos_${s}_r15g.json" > "l147_oos_${s}.log" 2>&1
  echo "  $s done exit=$?"
done
echo GATE4_DONE
