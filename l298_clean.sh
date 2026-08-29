#!/bin/bash
# L298 -- re-measure dt for mix and both in ONE contiguous block.
#
# WHY.  The dt vectors above were differenced against l294_ship*.json, three
# hours older. G6 proved `mix` does bit-identical work to gate0 on the 29 and to
# lp2 on the 71, so those halves MUST cost the same -- and they read -1.27 s and
# -1.29 s cheaper. That is box drift, and it is consistent, so it is a bias in
# dt, not scatter. `both` is worse: on the 71, where it is bit-equal to lp2, it
# reads +7.93 s for identical work, and its two repeats differ by 12.5 s of wall.
#
# The handoff's own lesson, on exactly this: "the earlier k=4 figure used the
# L276-era dt; re-measured back to back it is 27.57 s. Use the fresh number."
#
# ship, mix, both, ship -- one block, the baseline on both sides of the arms.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {  # tag  optimizer  extra-env...
  tag="$1"; shift; opt="$1"; shift
  rm -f "../l298_${tag}_stats.txt"
  echo "=== $tag ($opt) ==="; date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" \
      ICCAD_SHAPE_LP_STATS="../l298_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate "../$opt" \
    -o "../l298_${tag}.json" > "../l298_${tag}.log" 2>&1
  echo "    exit=$?"; date +"    end   %H:%M:%S"
  grep -E "Total Score|^Feasible" "../l298_${tag}.log"
}
run ship  optimizer_constructive.py
run mix   l296_mix_optimizer.py
run both  optimizer_constructive.py ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=2
run ship2 optimizer_constructive.py
run gate0 optimizer_constructive.py ICCAD_LP_GATE=0
run ship3 optimizer_constructive.py
echo L298_DONE
