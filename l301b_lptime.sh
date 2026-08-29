#!/bin/bash
# L301b -- measure the LP's cost DIRECTLY instead of differencing two ~140 s walls.
#
# WHY. The L301 block's control failed: `mix` does bit-identical work in both
# blocks (its total is 1.195229398 in each and every same-work gate passes) yet
# its dt read 35.36 s here against 22.34 s in the L298 block, and `mix3` read
# CHEAPER than `mix` while doing strictly more passes. The box was loaded.
#
# `ICCAD_LP_TIMING=1` prints per case `cpu` and `wall` for the LP itself, taken
# inside the process with time.process_time(). L159 built it for exactly this:
# "51 portfolio subprocesses run concurrently with everything else on the box;
# process_time() counts only this process's own CPU, and the LP runs
# synchronously in the main process, so it is the right clock." Contention
# cannot inflate it.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {  # tag  optimizer
  tag="$1"; opt="$2"
  echo "=== $tag ==="; date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" ICCAD_LP_TIMING=1 \
    "$PY" -u iccad2026_evaluate.py --evaluate "../$opt" \
    -o "../l301b_${tag}.json" > "../l301b_${tag}.log" 2>&1
  echo "    exit=$?"
  grep -E "Total Score|^Feasible" "../l301b_${tag}.log"
  echo "    lptime lines: $(grep -c '^\[lptime\]' "../l301b_${tag}.log")"
}
run ship  optimizer_constructive.py
run mix   l296_mix_optimizer.py
run mix3  l301_mix3_optimizer.py
run mix4  l301_mix4_optimizer.py
echo L301B_DONE
