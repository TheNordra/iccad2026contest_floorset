#!/bin/sh
# L238 -- in-set gates for the L234 + L235 package.
#
# Changes since the L227 suite, and the arm each one needs:
#   L231  _M49_REFINE_BAND mid band 6 -> 2   -> arm `refinemid6` (kill switch)
#   L234  _L196_LPGATE + 8 block counts, 0 removed  -> G3 reads the table
#   L235  the LP row-construction rewrite, which must be INVISIBLE -> that is
#         not gated here at all; it was gated by whole-portfolio bit identity
#         against results_L237_base.json before the tree was staged, which is a
#         strictly stronger test than anything a suite of arms can do.
#
# Both REFINE kill switches are exercised, and each must move cases ONLY inside
# its own band. A band that leaked into the other one passes determinism, passes
# the kill switch as a pair, and shows up only in that third check.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l238.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l238_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l238_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L238_${tag}.json" > "../l238_${tag}.log" 2>&1
  rc=$?
  n=$(wc -l < "../l238_${tag}_stats.txt" 2>/dev/null || echo 0)
  echo "$tag exit=$rc  LPran=${n}  SAfallback=$(grep -c 'SA fallback' "../l238_${tag}.log")  scipy=$(grep -c 'scipy] source=' "../l238_${tag}.log")"
}
run det1
run det2
run gateoff    ICCAD_LP_GATE=0
run k1         ICCAD_SHAPE_LP_DEPTH2=0
run l147off    ICCAD_SHAPE_LP_L147=0
run hboff      ICCAD_LP_HB_PRED=0
run lpoff      ICCAD_SHAPE_LP=0
run refine4    ICCAD_L223_REFINE_HEAVY=4
run refinemid6 ICCAD_L231_REFINE_MID=6
run pooldropon ICCAD_L211_POOLDROP=1
echo L238_GATES_DONE
