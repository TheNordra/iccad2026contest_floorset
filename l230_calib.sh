#!/bin/sh
# L230 -- re-measure the LP-gate inputs ON THE POST-REFINE TREE, in ONE batch.
#
# WHY THIS EXISTS. HANDOFF_2026-08-26 §4 claims a rebuilt _L196_LPGATE is worth
# +0.483pp, and l228_gate_new.txt carries the table -- but NO script on disk
# derives it. The derivation was inline. This file makes it reproducible, and
# it fixes a units problem the inline version could not have avoided:
#
#   l203_marginal_gate.py builds POOL[n] from _l181_cur.json (LP off) and the
#   box->grader map k from _l181_m73.json, BOTH from the 2026-08-24 batch. The
#   2026-08-25 L227 arms run 17.5% slower than that batch in the n<=100 control
#   band -- a band where NOTHING changed. Mixing an Aug-25 pool time with an
#   Aug-24 k therefore carries a ~17% systematic error straight into a
#   threshold test.  The fix is the ledger's own rule: take the RATIO ON THE
#   SAME BOX IN THE SAME BATCH and let the machine factor cancel.
#
#     POOL_new[n] = POOL_old[n] * (A[n]/B[n])      A = REFINE 2, B = REFINE 4
#     DT_new[n]   = DT_old[n]   * (C[n]-A[n])/(D[n]-B[n])
#
# min-of-3 on BOTH sides of every ratio (per-case noise on this box is ~17%).
# The n<=100 band is the control: REFINE is untouched there, so A/B must read
# 1.00 and any drift is the estimator's own bias.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l230.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: $LOCK exists"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  env ICCAD_ADAPTIVE_CORES=48 "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L230_${tag}.json" > "../l230_${tag}.log" 2>&1
  echo "  $tag exit=$?  SAfallback=$(grep -c 'SA fallback' "../l230_${tag}.log")"
}
echo "=== L230 calibration batch  $(date -u +%FT%TZ)  nproc=$(nproc) ==="
for i in 1 2 3; do
  echo "-- rep $i  $(date -u +%TZ)"
  run "A$i"                                ICCAD_SHAPE_LP=0
  run "B$i" ICCAD_L223_REFINE_HEAVY=4      ICCAD_SHAPE_LP=0
  run "C$i"                                ICCAD_LP_GATE=0
  run "D$i" ICCAD_L223_REFINE_HEAVY=4      ICCAD_LP_GATE=0
done
echo "L230_CALIB_DONE $(date -u +%FT%TZ)"
