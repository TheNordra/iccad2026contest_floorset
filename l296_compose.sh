#!/bin/bash
# L296 -- does the LP gate compose with LP depth?
#
# The two knobs are INDEPENDENT: `_lp_gate_ok` decides whether the LP runs on a
# case at all (71 -> 100 block counts), `_shape_lp_depth` decides how many
# passes where it does.  So `both` is not the sum -- it also buys a SECOND pass
# on the 29 cases gate0 newly admits, which neither arm alone can see.  It could
# be super-additive (that extra term) or sub-additive (the second pass has
# nothing left to take after the first).
#
# Phase 1, in-set: `both` x2 (determinism + dt), plus `both4` as the adjacent
# depth point -- l293's frontier put k=4 above k=2 for the GATED case, so
# whether k=2 is even the right partner is part of the question.
# Phase 2, OOS: the composition on s1 and s2, L275's rule.
#
# dt is differenced against l294_ship{,_r2}.json, the same baseline the gate0
# and k=2 numbers already use.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"

run () {           # tag  extra-env...
  tag="$1"; shift
  cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
  rm -f "../l296_${tag}_stats.txt"
  echo "=== $tag : $* ==="; date +"    start %H:%M:%S"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" \
      ICCAD_SHAPE_LP_STATS="../l296_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../l296_${tag}.json" > "../l296_${tag}.log" 2>&1
  echo "    exit=$?"; date +"    end   %H:%M:%S"
  grep -E "Total Score|^Feasible" "../l296_${tag}.log"
  echo "    LP ran on $(cut -d' ' -f1 "../l296_${tag}_stats.txt" | sort -un | wc -l) distinct n"
  echo "    passes spent: $(cut -d' ' -f4 "../l296_${tag}_stats.txt" | sort | uniq -c | tr '\n' ' ')"
}

run both    ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=2
run both_r2 ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=2
run both4   ICCAD_LP_GATE=0 ICCAD_SHAPE_LP_ITERS=4
echo PHASE1_DONE

cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  echo "===== OOS $S ====="; date +"start %H:%M:%S"
  "$PY" -u l287_transfer.py --sample $S --arms ship,lp2,gate0,both \
      > "l296_${S}.log" 2>&1
  echo "exit=$?"; date +"end   %H:%M:%S"
  tail -12 "l296_${S}.log"
done
echo L296_DONE
