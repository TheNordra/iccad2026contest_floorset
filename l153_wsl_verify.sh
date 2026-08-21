#!/usr/bin/env bash
# L153 -- the Linux verify of the L147 config (HANDOFF_2026-08-20 §5.1).
# Run INSIDE WSL:  wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l153_wsl_verify.sh
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l153
TAR=$R/build_submission/cadc1075.tar.gz
cd "$R" || exit 1
mkdir -p "$L117_WORK"

echo "== L153 Linux verify  $(date -u +%FT%TZ)  nproc=$(nproc)  $(uname -r)"
"$V" -c "import sys,numpy,scipy,torch,shapely;print('   py',sys.version.split()[0],'np',numpy.__version__,'sp',scipy.__version__,'torch',torch.__version__,'shapely',shapely.__version__)"
echo "   tar: $(md5sum $TAR)"
echo "   ELF in tar: $(tar xzOf $TAR cadc1075/bin/constructive_linux | md5sum)"

echo; echo "########## LANE 1 -- default cores, bundled-ELF bit lane ##########"
"$V" l117_linux_verify.py final "$TAR" --tag t_default \
     --anchor results_L153_default_L137.json
echo "LANE1_RC=$?"

echo; echo "########## LANE 2 -- 48c, LP off: the Linux pre-LP base ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_lpoff --no-judge \
     --base results_L153_lpoff_L137.json --env ICCAD_SHAPE_LP=0
echo "LANE2_RC=$?"
BASE=$L117_WORK/t_lpoff/cadc1075/results_l117_t_lpoff.json

echo; echo "########## LANE 3 -- 48c, shipped band: the control arm ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_ctrl --no-judge --stats \
     --base "$BASE"
echo "LANE3_RC=$?"
CTRL=$L117_WORK/t_ctrl/cadc1075/results_l117_t_ctrl.json

echo; echo "########## BUDGET -- measured on Linux, same tree, same box ##########"
"$V" l153_budget.py "$BASE" "$CTRL" | tee "$R/l153_budget_linux.txt"
B=$(grep '^BUDGET' "$R/l153_budget_linux.txt" | awk '{print $2}')
echo "   budget = $B"

echo; echo "########## LANE 4 -- 48c, THE L147 CONFIG ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_arm --stats \
     --base "$BASE" --ctrl "$CTRL" --win results_L147_on_L137.json \
     --budget "${B:-0}" --live-min 2.0 \
     --env ICCAD_SHAPE_LP_R=1.5 --env ICCAD_SHAPE_LP_G=1.10 \
     --env ICCAD_SHAPE_LP_TOL=0.006 --env ICCAD_SHAPE_LP_PRICE=1.0
echo "LANE4_RC=$?"

echo; echo "########## LANE 5 -- t4, corrupt ELF must fall through to g++ ##########"
"$V" l117_linux_verify.py t4 "$TAR" --anchor results_L153_default_L137.json
echo "LANE5_RC=$?"

echo; echo "L153_WSL_DONE $(date -u +%FT%TZ)"
