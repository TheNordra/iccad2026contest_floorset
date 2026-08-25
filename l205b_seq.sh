#!/bin/sh
# L205b -- the same imbalance, measured UNCONTENDED, which removes the one bias
# that could still rescue route A.
#
# L205 measured D_max/D_mean with all 51 profiles running at once on this box's
# 32 logical cores (1.6x oversubscribed). The grader runs 51 on 48 (1.06x).
# Under heavy oversubscription short profiles finish early and hand their share
# to the long ones, so the measured ratio is COMPRESSED toward 1 -- and the
# verdict only needs the true median to rise from 1.377 to 1.530 to flip. That
# is an 11% gap, well inside what a compression bias could hide.
#
# ICCAD_PROF_SEQ=1 runs the profiles one at a time. The durations are then a
# pure property of the workload, with no scheduler in the way, and the ratio
# they give is the one that transports to a box that is barely oversubscribed.
#
# Slower by construction: the case wall becomes the SUM of 51 profiles instead
# of their max. ~45-75 minutes for all 100. Run it once; the ratio is what we
# want, the wall of this run means nothing.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l205.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
rm -f ../l205b_prof_seq.txt
env ICCAD_ADAPTIVE_CORES=48 ICCAD_ROUTE_A=0 ICCAD_PROF_SEQ=1 \
    ICCAD_PROFILE_TIMING=../l205b_prof_seq.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
  -o ../results_L205b_seq.json > ../l205b_seq.log 2>&1
echo "seq exit=$?  records=$(wc -l < ../l205b_prof_seq.txt 2>/dev/null || echo 0) (expect 5100)"
cd /c/ICCAD_ml/ship_final || exit 1
# Sequential vs parallel must not change the ANSWER -- only the timing. If it
# does, the probe is perturbing what it measures and neither run is usable.
"$PY" - <<'PYX'
import json
J = lambda f: {r["test_id"]: r for r in json.load(open(f))["test_results"]}
try:
    a, b = J("results_L199_det1.json"), J("results_L205b_seq.json")
except Exception as e:
    print("   sequential-identity cross-check skipped:", e); raise SystemExit(0)
ids = sorted(set(a) & set(b))
c = sum(1 for i in ids if a[i]["cost"] == b[i]["cost"])
p = sum(1 for i in ids if a[i]["positions"] == b[i]["positions"])
print("   SEQUENTIAL == SHIPPED RESULT: cost {}/{}  positions {}/{}   {}"
      .format(c, len(ids), p, len(ids),
              "PASS" if c == len(ids) == p else "FAIL"))
PYX
echo; echo "=== UNCONTENDED imbalance ==="
"$PY" -u l205b_compare.py
echo L205B_DONE
