#!/bin/bash
# L301 -- deepen ONLY on the 71 the L196 gate already allows.
#
# `both4` already showed uniform depth 4 is RED, but it spends those passes on
# the 29 HEAVY cases, where the measured f is 1.62 -- the most expensive seconds
# on the table (L308). This shape spends them on the light and mid cases, where
# f is 2.66-4.67. Same mechanism, different band, and L296 showed the band is
# what decides the price.
#
# Contiguous block with `mix` re-run as a CONTROL: its dt must reproduce 22.34 s,
# which is the drift check the stale-baseline episode taught us to carry.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {  # tag  optimizer
  tag="$1"; opt="$2"
  rm -f "../l301_${tag}_stats.txt"
  echo "=== $tag ($opt) ==="; date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" \
      ICCAD_SHAPE_LP_STATS="../l301_${tag}_stats.txt" \
    "$PY" -u iccad2026_evaluate.py --evaluate "../$opt" \
    -o "../l301_${tag}.json" > "../l301_${tag}.log" 2>&1
  echo "    exit=$?"; date +"    end   %H:%M:%S"
  grep -E "Total Score|^Feasible" "../l301_${tag}.log"
  echo "    LP on $(cut -d' ' -f1 "../l301_${tag}_stats.txt" | sort -un | wc -l) distinct n; passes: $(cut -d' ' -f4 "../l301_${tag}_stats.txt" | sort | uniq -c | tr '\n' ' ')"
}
run ship  optimizer_constructive.py
run mix3  l301_mix3_optimizer.py
run mix4  l301_mix4_optimizer.py
run ship2 optimizer_constructive.py
run mix   l296_mix_optimizer.py
run ship3 optimizer_constructive.py
echo L301_DONE
