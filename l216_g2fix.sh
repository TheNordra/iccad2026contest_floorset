#!/bin/sh
# L216 -- restore G2's anchor to a comparison it can actually support.
#
# results_L165_l147off.json was produced with the full 51-profile pool. The
# L211/L213 drop removes 8 per block count, so the portfolio winner can change
# for reasons that have nothing to do with the L147 escape hatch, and G2a/G2c
# read FAIL on a package where the hatch is fine. Same shape as the stale
# anchors already corrected this session, one layer further out.
#
# The fix is an arm in the anchor's OWN pool configuration: L147 off, LP gate
# off, AND pool drop off. That is exactly the L165 configuration, so it must
# reproduce it 100/100. Anything less is a real regression in the hatch.
#
# G2b needs no fix -- it compares two arms that both carry the drop.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l216.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -q "L215_CHAIN_DONE" l215_chain.out 2>/dev/null; do sleep 20; done
echo "=== L216 G2 anchor arm  $(date -u +%FT%TZ) ==="
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
tag=l147off_gateoff_nodrop
rm -f "../l214_${tag}_stats.txt"
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS="../l214_${tag}_stats.txt" \
    ICCAD_SHAPE_LP_L147=0 ICCAD_LP_GATE=0 ICCAD_L211_POOLDROP=0 \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o "../results_L214_${tag}.json" > "../l214_${tag}.log" 2>&1
echo "$tag exit=$?  LPran=$(wc -l < "../l214_${tag}_stats.txt" 2>/dev/null || echo 0) (expect 100)"
cd /c/ICCAD_ml/ship_final || exit 1
"$PY" - <<'PYX'
import json
J = lambda f: {r["test_id"]: r for r in json.load(open(f))["test_results"]}
a = J("results_L214_l147off_gateoff_nodrop.json")
b = J("results_L165_l147off.json")
ids = sorted(set(a) & set(b))
c = sum(1 for i in ids if a[i]["cost"] == b[i]["cost"])
print("G2c' L147+gateoff+nodrop vs results_L165_l147off.json: {}/{}   {}"
      .format(c, len(ids), "PASS" if c == len(ids) else "FAIL"))
print("     (the hatch tested in the anchor's own pool configuration; the")
print("      L214 g2c arm carries the pool drop and cannot be compared to it)")
PYX
echo L216_DONE
