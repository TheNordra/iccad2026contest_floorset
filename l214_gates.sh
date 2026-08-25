#!/bin/sh
# L214 -- in-set gates for the L196 tree WITH ROUTE A OFF + the L211/L213 pool drop
#
# The pool drop DOES change results (that is the whole point), so G9's
# bit-equality against L209 no longer applies; G10 replaces it with the pair
# that can actually fail: the kill-switch arm must reproduce L209 bit-for-bit
# AND the default must differ from it on exactly the 12 cases the drop was
# measured to move. That is the point: the change is supposed to
# move wall and nothing else, and a result difference would mean route A was
# never result-neutral in the first place. Scored by
#   l199_verdict.py L214 L199
# which adds G9 = every arm identical to the L199 anchor.
#
# Original header follows.
# L199 -- in-set gates for the L196 tree: _L157_DEPTH flattened to all-1s plus
# the per-case LP gate _L196_LPGATE (s=1.2, fires on 63 of 100 block counts).
#
# Seven arms. Three of them exist only because L196 changed WHAT the gates are
# allowed to assume:
#
#   det1 / det2   determinism, cost AND positions, 100/100. Both are the plain
#                 default -- L177 distinguished them with ICCAD_SHAPE_LP_NOOP,
#                 a variable that DOES NOT EXIST in the tree (grep: 0 hits).
#                 Harmless there, but this file will not hang a gate on a
#                 phantom flag; the arms differ by tag only, which is what a
#                 determinism test actually wants.
#   gateoff       ICCAD_LP_GATE=0 -- the kill switch AND the pre-L196 anchor.
#                 Its stats line count must be 100 where the default's is 63.
#                 That line count IS the liveness proof: ICCAD_SHAPE_LP_STATS
#                 only writes when the LP actually executes, so a table that
#                 silently kept old values would pass determinism and the kill
#                 switch while changing nothing -- the failure this project
#                 records most often.
#   k1            ICCAD_SHAPE_LP_DEPTH2=0. Under the OLD map this measured the
#                 depth map's quality. Under all-1s it must be BIT-IDENTICAL to
#                 det1, because _shape_lp_depth() returns 1 pass either way.
#                 The check flips from "how much did depth buy" to "is the map
#                 really flat" -- same arm, opposite assertion.
#   l147off       ICCAD_SHAPE_LP_L147=0 -- must reproduce the committed
#                 pre-L147 band bit-for-bit; the escape hatch has to still work.
#   hboff         ICCAD_LP_HB_PRED=0 -- isolates L171 in THIS configuration.
#                 Its sign already flipped once when the map changed underneath
#                 it (+0.0772% on the old map, -0.0512% on L172's).
#   lpoff         ICCAD_SHAPE_LP=0 -- no LP at all. 0 stats lines, and the
#                 total in-set value of the gated LP.
#
# LOCKFILE. Four separate measurements were destroyed by two agents running on
# this box at once (see _quarantine/). Nothing starts if another copy holds it.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l214.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l214_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l214_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L214_${tag}.json" > "../l214_${tag}.log" 2>&1
  rc=$?
  n=$(wc -l < "../l214_${tag}_stats.txt" 2>/dev/null || echo 0)
  echo "$tag exit=$rc  LPran=${n}  SAfallback=$(grep -c 'SA fallback' "../l214_${tag}.log")  scipy=$(grep -c 'scipy] source=' "../l214_${tag}.log")"
}
run det1
run det2
run gateoff ICCAD_LP_GATE=0
run k1      ICCAD_SHAPE_LP_DEPTH2=0
run l147off ICCAD_SHAPE_LP_L147=0
run hboff   ICCAD_LP_HB_PRED=0
run lpoff   ICCAD_SHAPE_LP=0
run l147off_gateoff ICCAD_SHAPE_LP_L147=0 ICCAD_LP_GATE=0
run pooldropoff ICCAD_L211_POOLDROP=0
echo L214_GATES_DONE
