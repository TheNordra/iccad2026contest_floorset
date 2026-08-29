#!/bin/sh
# L172 in-set gates -- the LP depth map was rebuilt on the 2026-08-23 medians.
#
# Same three checks as L165 (determinism x2, L147 kill switch) plus a fourth
# that L165 did not need: an explicit k=1 ANCHOR run on THIS tree, so the
# quality delta is measured against something produced in the same session
# rather than against a stale committed arm. That is the whole reason this
# rebuild exists -- a stale reference is what made the old map look like a
# gain when it had become a loss.
#
# LOCKFILE: a second copy exits. Two concurrent copies of the L170 script and
# 30+ constructive.exe are what destroyed the previous OOS attempt
# (_quarantine/README.txt); nothing on this box may run two of these at once.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l172.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l172_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l172_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L172_${tag}.json" > "../l172_${tag}.log" 2>&1
  echo "$tag exit=$?  SAfallback=$(grep -c 'SA fallback' "../l172_${tag}.log")  scipy=$(grep -c 'scipy] source=' "../l172_${tag}.log")"
}
run det1    ICCAD_SHAPE_LP_NOOP=1
run det2    ICCAD_SHAPE_LP_NOOP=2
run k1      ICCAD_SHAPE_LP_DEPTH2=0
run l147off ICCAD_SHAPE_LP_L147=0
echo L172_GATES_DONE
