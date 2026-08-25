#!/bin/sh
# L231 -- price the MID band (60 < n <= 100), the other stale REFINE constant,
#         plus the two post-processing budgets nobody has ever priced downward.
#
# HANDOFF_2026-08-26 §7: after REFINE 4->2 on the heavy band, 67.3% of the
# REMAINING 0.745pp RF deficit sits in 60<n<=100, where 25 of 40 cases are still
# ABOVE the floor and the band runs REFINE=6 (M50, M74: 8->6). That 6 came from
# the same strictly-selection-preserving derivation as the heavy band's 4 -- it
# took the half that costs no quality and stopped -- on a tree that has since
# taken L131/L136/L147/L124/L223.
#
# Every arm carries the SHIPPED heavy band (=2) via L219_REFINE_TABLE, so mid=6
# IS the shipped configuration and the baseline; l231_score.py hard-gates that
# by requiring m6 to reproduce results_L227_det1.json 100/100 on cost.
#
# arm pc: _M49/_M50 only ever moved REFINE. The ledger records
# PUSH_PASSES/COMPACT_ITERS as bit-identical no-ops UPWARD (both loops
# early-break) and has never priced them DOWNWARD. Screened globally here; if
# the sign is right it becomes a band like REFINE.
#
# 3 reps for min-of-3 on BOTH sides of every ratio; n<=60 is the control band.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l231.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
while [ -d "$R/.l230.lock" ]; do sleep 30; done
cd "$R/iccad2026contest" || exit 1
echo "=== L231 mid-band sweep  $(date -u +%FT%TZ) ==="
for I in 1 2 3; do
  for M in 6 4 3 2 pc; do
    if [ "$M" = pc ]; then
      TBL=../l231_mid6.json
      set -- ICCAD_PUSH_PASSES=2 ICCAD_COMPACT_ITERS=2
    else
      TBL=../l231_mid${M}.json
      set --
    fi
    rm -f "../l231_prof_m${M}_${I}.txt"
    env ICCAD_ADAPTIVE_CORES=48 L219_REFINE_TABLE="$TBL" "$@" \
        ICCAD_PROFILE_TIMING="../l231_prof_m${M}_${I}.txt" \
      "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
      -o "../results_L231_m${M}_${I}.json" > "../l231_m${M}_${I}.log" 2>&1
    rc=$?
    echo "  arm=$M rep$I exit=$rc records=$(wc -l < "../l231_prof_m${M}_${I}.txt" 2>/dev/null || echo 0) table=$(grep -c 'refine table loaded' "../l231_m${M}_${I}.log")"
  done
done
echo L231_SWEEP_DONE $(date -u +%FT%TZ)
