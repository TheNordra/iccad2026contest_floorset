#!/bin/sh
# L229 -- G2's anchor, one configuration further out.
#
# results_L165_l147off.json predates BOTH the LP gate and the L223 REFINE band.
# L216 already split G2 so the gate no longer confuses it; now the band does.
# Measured: G2c differs from the anchor on exactly [113, 115], which is exactly
# the pair REFINE 4->2 moves. The hatch is intact; the anchor is two mechanisms
# behind.
#
# The arm that settles it is the anchor's OWN configuration: L147 off, LP gate
# off, REFINE back to 4. That is precisely L165, so it must reproduce it 100/100.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l229.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -qE "L227_CHAIN_DONE|ABORT" l227_chain.out 2>/dev/null; do sleep 30; done
echo "=== L229 G2 anchor arm  $(date -u +%FT%TZ) ==="
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
tag=l147off_gateoff_r4
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l227_${tag}_stats.txt" \
    ICCAD_SHAPE_LP_L147=0 ICCAD_LP_GATE=0 ICCAD_L223_REFINE_HEAVY=4 \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o "../results_L227_${tag}.json" > "../l227_${tag}.log" 2>&1
echo "$tag exit=$?  LPran=$(wc -l < "../l227_${tag}_stats.txt" 2>/dev/null || echo 0) (expect 100)"
cd /c/ICCAD_ml/ship_final || exit 1
"$PY" - <<'PYX'
import json
J = lambda f: {r["block_count"]: r for r in json.load(open(f))["test_results"]}
a = J("results_L227_l147off_gateoff_r4.json"); b = J("results_L165_l147off.json")
ids = sorted(set(a) & set(b))
c = sum(1 for n in ids if a[n]["cost"] == b[n]["cost"])
print("G2c'' L147 + gateoff + REFINE=4 vs results_L165_l147off.json: {}/{}   {}"
      .format(c, len(ids), "PASS" if c == len(ids) else "FAIL"))
print("      (the hatch in the anchor's OWN configuration: no gate, no L223 band)")
PYX
echo L229_DONE
