#!/usr/bin/env bash
# L300 -- the Linux lane for the `mix` arm.
# Run INSIDE WSL:  wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l313_wsl_rfsafe.sh
#
# RF-SAFE is WRAPPER-ONLY: verified here, not taken on trust -- constructive.cpp
# md5 e2c7b2f4 and bin/constructive_linux md5 bc991207 are BYTE-IDENTICAL to D,
# and _L196_LPGATE goes 71 -> 83 with exactly [38,40,56,76,79,81,94,95,107,108,114,120]
# newly on and nothing lost. So no ELF rebuild is involved.
# (original note) `mix` is WRAPPER-ONLY: `build_submission.RFSAFE` is the shipped package with
# `_L196_LPGATE` -> all 1s and `_L157_DEPTH` -> 2 on the old 1-set, and the SAME
# ELF (verified byte-equal). So no ELF rebuild is involved and the glibc-2.43
# floor problem does not arise -- that risk only exists if the ELF is rebuilt here.
#
# The 48c lane is NOT bit-reproducible across platforms (Win scipy 1.15.3 vs this
# Ubuntu 1.18.0 land on different optima of the same degenerate LP), so it is
# judged on judge48()'s shipping invariants, with the per-case regression budget
# MEASURED from the already-deployed band on the same base.
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l313
SHIP=$R/build_submission.D/cadc1075.tar.gz
SAFE=$R/build_submission.RFSAFE/cadc1075.tar.gz
cd "$R" || exit 1
mkdir -p "$L117_WORK"

echo "== L313 Linux verify (RF-SAFE)  $(date -u +%FT%TZ)  nproc=$(nproc)  $(uname -r)"
"$V" -c "import sys,numpy,scipy,shapely;print('   py',sys.version.split()[0],'np',numpy.__version__,'sp',scipy.__version__,'shapely',shapely.__version__)"
echo "   ship tar ELF: $(tar xzOf $SHIP cadc1075/bin/constructive_linux | md5sum | cut -c1-32)"
echo "   safe tar ELF: $(tar xzOf $SAFE  cadc1075/bin/constructive_linux | md5sum | cut -c1-32)"
echo "   ship wrapper: $(tar xzOf $SHIP cadc1075/op_wrapper.py | md5sum | cut -c1-32)"
echo "   mix  wrapper: $(tar xzOf $SAFE  cadc1075/op_wrapper.py | md5sum | cut -c1-32)"

echo; echo "########## LANE 1a -- SHIP, default cores, bundled-ELF bit lane ##########"
"$V" l117_linux_verify.py final "$SHIP" --tag d_ship --anchor l300_win32_ship.json
echo "LANE1a_RC=$?"

echo; echo "########## LANE 1b -- MIX, default cores: must be INERT below the gate ##########"
"$V" l117_linux_verify.py final "$SAFE" --tag d_rfsafe --anchor l300_win32_ship.json
echo "LANE1b_RC=$?"

echo; echo "########## LANE 2 -- 48c, LP off: the Linux pre-LP base ##########"
"$V" l117_linux_verify.py final48 "$SHIP" --tag t_lpoff --no-judge \
     --base l285_lp_off.json --env ICCAD_SHAPE_LP=0
echo "LANE2_RC=$?"
BASE=$L117_WORK/t_lpoff/cadc1075/results_l117_t_lpoff.json

echo; echo "########## LANE 3 -- 48c, SHIPPED band: the control arm ##########"
"$V" l117_linux_verify.py final48 "$SHIP" --tag t_ctrl --no-judge --stats --base "$BASE"
echo "LANE3_RC=$?"
CTRL=$L117_WORK/t_ctrl/cadc1075/results_l117_t_ctrl.json

echo; echo "########## BUDGET -- measured on Linux, same box, same base ##########"
"$V" l153_budget.py "$BASE" "$CTRL" | tee "$R/l300_budget_linux.txt"
B=$(grep '^BUDGET' "$R/l300_budget_linux.txt" | awk '{print $2}')
echo "   budget = $B"

echo; echo "########## LANE 4 -- 48c, THE MIX PACKAGE, judged ##########"
# live-min 0.5: RF-SAFE is +0.904 % over the shipped band in set, so a floor of 0.5
# is far above what a dropped-flag run could produce (0.0) and safely below the
# expected value.
"$V" l117_linux_verify.py final48 "$SAFE" --tag t_rfsafe --stats \
     --base "$BASE" --ctrl "$CTRL" --win l313_win48_rfsafe.json --budget "${B:-0}" --live-min 0.5
echo "LANE4_RC=$?"

echo; echo "########## LANE 5 -- t4 on the MIX tar: corrupt ELF must fall through ##########"
"$V" l117_linux_verify.py t4 "$SAFE" --anchor l300_win32_ship.json
echo "LANE5_RC=$?"

echo; echo "L313_WSL_DONE $(date -u +%FT%TZ)"
