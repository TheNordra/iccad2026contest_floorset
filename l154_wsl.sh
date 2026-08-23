#!/usr/bin/env bash
# L154 on Linux -- the corpus the mechanism was designed for. L153 measured
# case 96 (n=117) rejected there and kept on Windows, and that one case was
# 107% of the Windows/Linux spread. Both arms run through the SAME packaged
# driver so the base cannot drift (HANDOFF_2026-08-20 §4.4).
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l154
TAR=$R/build_submission/cadc1075.tar.gz
cd "$R" || exit 1
mkdir -p "$L117_WORK"
LPB="--base results_L153_linux_lpoff.json"
FLAGS="--env ICCAD_SHAPE_LP_R=1.5 --env ICCAD_SHAPE_LP_G=1.10 --env ICCAD_SHAPE_LP_TOL=0.006 --env ICCAD_SHAPE_LP_PRICE=1.0"

echo "== L154 linux  $(date -u +%FT%TZ)  tar $(md5sum $TAR | cut -c1-32)"

echo; echo "###### OFF -- L147 arm, re-run on the patched source ######"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_off --no-judge --stats $LPB $FLAGS
echo "OFF_RC=$?"

echo; echo "###### ON -- band-catch ######"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_on --no-judge --stats $LPB $FLAGS \
     --env ICCAD_SHAPE_LP_CATCH=1
echo "ON_RC=$?"

cp $L117_WORK/t_off/cadc1075/results_l117_t_off.json $R/results_L154_linux_off.json
cp $L117_WORK/t_on/cadc1075/results_l117_t_on.json   $R/results_L154_linux_on.json
cp $L117_WORK/t_off/lp_stats.txt $R/l154_linux_stats_off.txt
cp $L117_WORK/t_on/lp_stats.txt  $R/l154_linux_stats_on.txt
echo L154_WSL_DONE
