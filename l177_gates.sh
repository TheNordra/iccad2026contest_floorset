#!/bin/sh
# L177 -- in-set gates for the MERGED tree: L172's x0.90 depth map plus L171's
# shape-LP hpwl_baseline predictor. Neither has ever been gated together, and
# they interact: L171 changes the LP's objective, L172 changes how many LP
# passes each case gets, so the combination is not the sum of the parts.
#
# Five arms. The last one is the reason this is not just a re-run of L172's:
#
#   det1 / det2   determinism, cost AND positions, 100/100
#   k1            ICCAD_SHAPE_LP_DEPTH2=0 -- the depth kill switch, so G4's
#                 quality delta is measured against an anchor from THIS tree
#                 rather than a committed arm from a tree that no longer exists
#   l147off       ICCAD_SHAPE_LP_L147=0 -- must reproduce the pre-L147 band
#                 bit-for-bit; the escape hatch has to still work
#   hboff         ICCAD_LP_HB_PRED=0 -- isolates L171's contribution ON THE NEW
#                 MAP. L171 was measured on the OLD map, which ran 66 cases at
#                 k=3; the new one runs 22. Fewer passes for it to act on, so
#                 its +0.0772% in-set does NOT carry over unexamined.
#
# LOCKFILE. Four separate measurements were destroyed today by two agents
# running on this box at once (see _quarantine/). Nothing starts if another
# copy holds the lock.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l177.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l177_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l177_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L177_${tag}.json" > "../l177_${tag}.log" 2>&1
  rc=$?
  echo "$tag exit=$rc  SAfallback=$(grep -c 'SA fallback' "../l177_${tag}.log")  scipy=$(grep -c 'scipy] source=' "../l177_${tag}.log")"
}
run det1    ICCAD_SHAPE_LP_NOOP=1
run det2    ICCAD_SHAPE_LP_NOOP=2
run k1      ICCAD_SHAPE_LP_DEPTH2=0
run l147off ICCAD_SHAPE_LP_L147=0
run hboff   ICCAD_LP_HB_PRED=0
echo L177_GATES_DONE
