#!/bin/bash
# L276 -- two questions, one experiment, on the GRADED shape (in-set 100 @48c).
#
# Q1 (hpwl diagnostic). The constraint-graph LP minimises exact HPWL *inside a
#    fixed topology*. So the hpwl_gap that deeper LP CANNOT remove is topology,
#    not placement. That is the decisive test of whether the hpwl line has any
#    non-topology headroom left.
# Q2 (shipping). _L157_DEPTH is a WRAPPER constant -- changing it needs no C++
#    change and no Linux ELF rebuild, which is the only class of change that can
#    safely ship right now. ICCAD_SHAPE_LP_ITERS is ungated and read from the
#    process env by _shape_lp_depth(), so ambient really reaches it.
cd /c/ICCAD_ml/ship_final/iccad2026contest
PY=/c/Users/.01/anaconda3/envs/floorset/python.exe
for k in 2 4; do
  echo "=== SHAPE_LP_ITERS=$k ==="
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_ITERS=$k \
      ICCAD_CONSTRUCTIVE_BIN="C:/ICCAD_ml/ship_final/constructive.exe" \
    $PY iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o ../results_L276_k$k.json 2>&1 | grep -E "Total Score|Feasible|Avg Runtime"
done
echo ALL_DONE
