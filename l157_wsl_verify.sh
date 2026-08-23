#!/usr/bin/env bash
# L157 -- the Linux verify of selective LP depth.
# Run INSIDE WSL:  wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l157_wsl_verify.sh
#
# Two differences from l153_wsl_verify.sh, both deliberate:
#
#  * the CONTROL is the L147 config at k=1, not the shipped band. L157 is an
#    increment on top of L147 and every number it was priced with had the
#    tangent rows in play, so a run judged against the shipped band would
#    credit L157 with L147's +2.5% and say nothing about the depth gate.
#
#  * LANE 3 runs UNGATED k=2. The gate is stated in absolute seconds against
#    the grader's median, so on any box slower than the grader it does not
#    fire -- Windows measured 0/100 at S=1. Lane 3 sidesteps that entirely:
#    it runs the MOST LP this change can ever cause, which is exactly where a
#    cross-platform problem would show up first. It needs no speed knob and
#    no assumption about this box's speed. Lane 4 then exercises the gate
#    itself at a scale derived from THIS box's own control-lane timings.
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l157
TAR=$R/build_submission/cadc1075.tar.gz
LP147="--env ICCAD_SHAPE_LP_R=1.5 --env ICCAD_SHAPE_LP_G=1.10 \
       --env ICCAD_SHAPE_LP_TOL=0.006 --env ICCAD_SHAPE_LP_PRICE=1.0"
cd "$R" || exit 1
mkdir -p "$L117_WORK"

echo "== L157 Linux verify  $(date -u +%FT%TZ)  nproc=$(nproc)  $(uname -r)"
"$V" -c "import sys,scipy,shapely;print('   py',sys.version.split()[0],'sp',scipy.__version__,'shapely',shapely.__version__)"
echo "   tar: $(md5sum $TAR | cut -c1-32)"
echo "   ELF in tar: $(tar xzOf $TAR cadc1075/bin/constructive_linux | md5sum | cut -c1-32)"
echo "   L157 in tar: $(tar xzOf $TAR cadc1075/op_src.py | grep -c _depth_affordable) hits on _depth_affordable"

echo; echo "########## LANE 1 -- 48c, LP off: the Linux pre-LP base ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_lpoff --no-judge \
     --base results_L153_lpoff_L137.json --env ICCAD_SHAPE_LP=0
echo "LANE1_RC=$?"
BASE=$L117_WORK/t_lpoff/cadc1075/results_l117_t_lpoff.json

echo; echo "########## LANE 2 -- 48c, L147 at k=1: the CONTROL + kill switch ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_k1 --no-judge --stats \
     --base "$BASE" $LP147 --env ICCAD_SHAPE_LP_DEPTH2=0
echo "LANE2_RC=$?"
CTRL=$L117_WORK/t_k1/cadc1075/results_l117_t_k1.json

echo; echo "########## BUDGET -- measured on Linux, same tree, same box ##########"
"$V" l153_budget.py "$BASE" "$CTRL" | tee "$R/l157_budget_linux.txt"
B=$(grep '^BUDGET' "$R/l157_budget_linux.txt" | awk '{print $2}')
echo "   budget = $B"

echo; echo "########## LANE 3 -- 48c, UNGATED k=2: the deepest LP this can cause ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_k2 --stats \
     --base "$BASE" --ctrl "$CTRL" --win results_L157_k2.json \
     --budget "${B:-0}" --live-min 0.15 $LP147 --env ICCAD_SHAPE_LP_ITERS=2
echo "LANE3_RC=$?"

echo; echo "########## S -- derived from THIS box's control lane, not Windows' ##########"
S=$("$V" - "$CTRL" <<'PY'
import json, sys
A, B, THR = 0.0196, 1.168, 0.3046
r = json.load(open(sys.argv[1]))["test_results"]
rat = sorted(x["runtime_seconds"] / (THR * A * (x["block_count"] ** B)) for x in r)
print(f"{rat[int(0.75*len(rat))-1]:.2f}")
PY
)
echo "   S = $S  (the scale at which 75% of THIS box's cases fall inside the budget)"

echo; echo "########## LANE 4 -- 48c, THE L157 GATE at S=$S ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_gate --stats \
     --base "$BASE" --ctrl "$CTRL" --win results_L157_gateS.json \
     --budget "${B:-0}" --live-min 0.15 $LP147 --env "ICCAD_SHAPE_LP_DEPTH_S=$S"
echo "LANE4_RC=$?"

echo; echo "########## GATE LIVENESS -- col 4 = LP passes actually spent ##########"
for t in t_k1 t_k2 t_gate; do
  S4=$L117_WORK/$t/lp_stats.txt
  if [ -f "$S4" ]; then
    echo "   $t  $(awk '{h[$4]++} END {for (k in h) printf "passes=%s:%d  ", k, h[k]}' "$S4")"
  else
    echo "   $t  NO STATS FILE -- cannot prove the LP ran at all"
  fi
done

echo; echo "L157_WSL_DONE $(date -u +%FT%TZ)"
