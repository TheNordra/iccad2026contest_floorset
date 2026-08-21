#!/usr/bin/env bash
# L157 -- the Linux verify of selective LP depth.
# Run INSIDE WSL:  wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l157_wsl_verify.sh
#
# Same shape as l153_wsl_verify.sh, with one difference that matters: the
# CONTROL arm is the L147 config at k=1, not the shipped band. L157 is an
# increment ON TOP of L147 and every number it was priced with had the tangent
# flags on, so a run judged against the shipped band would credit L157 with
# L147's +2.5% and tell us nothing about the depth gate.
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
echo "   tar: $(md5sum $TAR)"
echo "   ELF in tar: $(tar xzOf $TAR cadc1075/bin/constructive_linux | md5sum)"

echo; echo "########## LANE 1 -- 48c, LP off: the Linux pre-LP base ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_lpoff --no-judge \
     --base results_L153_lpoff_L137.json --env ICCAD_SHAPE_LP=0
echo "LANE1_RC=$?"
BASE=$L117_WORK/t_lpoff/cadc1075/results_l117_t_lpoff.json

echo; echo "########## LANE 2 -- 48c, L147 at k=1: the CONTROL ##########"
# ICCAD_SHAPE_LP_DEPTH2=0 is L157's kill switch. This lane is therefore also
# the cross-platform check that the kill switch really restores k=1.
"$V" l117_linux_verify.py final48 "$TAR" --tag t_k1 --no-judge --stats \
     --base "$BASE" $LP147 --env ICCAD_SHAPE_LP_DEPTH2=0
echo "LANE2_RC=$?"
CTRL=$L117_WORK/t_k1/cadc1075/results_l117_t_k1.json

echo; echo "########## BUDGET -- measured on Linux, same tree, same box ##########"
"$V" l153_budget.py "$BASE" "$CTRL" | tee "$R/l157_budget_linux.txt"
B=$(grep '^BUDGET' "$R/l157_budget_linux.txt" | awk '{print $2}')
echo "   budget = $B"

echo; echo "########## LANE 3 -- 48c, L147 + THE L157 DEPTH GATE ##########"
# live-min 0.15: L157's in-set quality is ~+0.29%, an order below L147's
# +2.5%, so L153's --live-min 2.0 would reject a perfectly live run. The
# liveness that actually matters here is col 4 of the stats file carrying
# BOTH 1 and 2 -- checked below, and not satisfiable by a dropped flag.
"$V" l117_linux_verify.py final48 "$TAR" --tag t_gate --stats \
     --base "$BASE" --ctrl "$CTRL" --win results_L157_gated.json \
     --budget "${B:-0}" --live-min 0.15 $LP147
echo "LANE3_RC=$?"

echo; echo "########## GATE LIVENESS -- did the depth gate actually gate? ##########"
for t in t_k1 t_gate; do
  S=$L117_WORK/$t/lp_stats.txt          # t3() writes WORK/<tag>/lp_stats.txt
  if [ -f "$S" ]; then
    echo "   $t  $(awk '{h[$4]++} END {for (k in h) printf "passes=%s:%d  ", k, h[k]}' "$S")"
  else
    echo "   $t  NO STATS FILE -- cannot prove the gate ran"
  fi
done

echo; echo "L157_WSL_DONE $(date -u +%FT%TZ)"
