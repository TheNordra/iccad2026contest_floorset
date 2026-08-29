#!/bin/bash
# L297 -- MEASURE the mix arm end to end instead of trusting the arm-mixing.
#
# G5 priced it exactly (the G0 bit-equalities prove the two knobs act on
# disjoint case sets, so mixing is exact) -- but arm-mixing cannot see an
# interaction it does not model, and this project's rule is to run the thing.
# `l296_mix_optimizer.py` is a COPY of the tree wrapper with two table defaults
# changed and nothing else; the tree is untouched.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
for tag in mix mix_r2; do
  rm -f "../l297_${tag}_stats.txt"
  echo "=== $tag ==="; date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" \
      ICCAD_SHAPE_LP_STATS="../l297_${tag}_stats.txt" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../l296_mix_optimizer.py \
    -o "../l297_${tag}.json" > "../l297_${tag}.log" 2>&1
  echo "    exit=$?"; date +"    end   %H:%M:%S"
  grep -E "Total Score|^Feasible" "../l297_${tag}.log"
  echo "    LP ran on $(cut -d' ' -f1 "../l297_${tag}_stats.txt" | sort -un | wc -l) distinct n"
  echo "    passes spent: $(cut -d' ' -f4 "../l297_${tag}_stats.txt" | sort | uniq -c | tr '\n' ' ')"
done
echo L297_DONE
