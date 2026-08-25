#!/bin/sh
# L237 -- ship the L235 LP rewrite, gated on BIT IDENTITY of the whole portfolio.
#
# The rewrite is Python-only: same rows, same order, same floats, so the LP hands
# HiGHS an identical program and the package must produce IDENTICAL results. That
# makes the gate the strongest one this project has -- equality, not a quality
# measurement -- and it needs no OOS runs at all.
#
#   1. anchor   run the L234 tree as it stands  -> results_L237_base.json
#   2. patch    l235_patch.py --inplace
#   3. gate     re-run the same arm -> must be 100/100 identical on cost AND
#               positions. Anything else and the rewrite is reverted, full stop.
#   4. gate2    the standalone LP A/B over all 100 cases (objective, layout hash,
#               rows-by-origin, kept/dropped, calls, hard_ok)
#
# The LP-gate widening that the speedup pays for is NOT done here -- it moves
# results by design and would destroy the bit-identity gate above. It goes in
# separately, after this passes.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l237.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
cd "$R" || exit 1
export PYTHONIOENCODING=utf-8

echo "=== L237 anchor run (L234 tree)  $(date -u +%FT%TZ) ==="
cp optimizer_constructive.py _l237_pre.py
cd "$R/iccad2026contest" || exit 1
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS=../l237_base_stats.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o ../results_L237_base.json > ../l237_base.log 2>&1
echo "  anchor exit=$?  LPran=$(wc -l < ../l237_base_stats.txt 2>/dev/null || echo 0)  SAfallback=$(grep -c 'SA fallback' ../l237_base.log)"
cd "$R" || exit 1

echo; echo "=== L237 applying the L235 rewrite IN PLACE ==="
"$PY" l235_patch.py --inplace || { echo "patch failed"; exit 1; }
"$PY" -m py_compile optimizer_constructive.py || {
  echo "!! does not compile -- reverting"; cp _l237_pre.py optimizer_constructive.py; exit 1; }

echo; echo "=== L237 gate 1: whole-portfolio bit identity ==="
cd "$R/iccad2026contest" || exit 1
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS=../l237_post_stats.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o ../results_L237_post.json > ../l237_post.log 2>&1
echo "  post exit=$?  LPran=$(wc -l < ../l237_post_stats.txt 2>/dev/null || echo 0)  SAfallback=$(grep -c 'SA fallback' ../l237_post.log)"
cd "$R" || exit 1
"$PY" - <<'PYX'
import json, sys
def load(f):
    return {r["block_count"]: r for r in json.load(open(f))["test_results"]}
a, b = load("results_L237_base.json"), load("results_L237_post.json")
cost = sum(1 for n in a if n in b and a[n]["cost"] == b[n]["cost"])
pos = sum(1 for n in a if n in b and a[n]["positions"] == b[n]["positions"])
feas = sum(1 for n in b if b[n]["is_feasible"])
print(f"  cost {cost}/{len(a)}   positions {pos}/{len(a)}   feasible {feas}/{len(b)}")
if cost != len(a) or pos != len(a):
    bad = [n for n in sorted(a) if n in b and (a[n]["cost"] != b[n]["cost"]
           or a[n]["positions"] != b[n]["positions"])]
    print("  !! NOT BIT-IDENTICAL on", bad[:15])
    sys.exit(1)
print("  GATE 1 PASS -- the rewrite is invisible to the package, as required")
PYX
[ $? -eq 0 ] || { echo "!! reverting optimizer_constructive.py"; cp _l237_pre.py optimizer_constructive.py; exit 1; }

echo; echo "=== L237 gate 2: the shipped file IS the module the A/B passed on ==="
# l235_lpbench.py ab already ran optimizer_l235lp against optimizer_constructive
# over all 100 cases and matched objective, layout hash, rows-by-origin,
# kept/dropped, calls and hard_ok. That evidence only transfers if the file now
# in the tree is that same module, so check it rather than re-run it.
"$PY" - <<'PYX'
import pathlib, sys
a = pathlib.Path("optimizer_constructive.py").read_text(encoding="utf-8")
b = pathlib.Path("optimizer_l235lp.py").read_text(encoding="utf-8")
hdr_end = b.index('"""\n', b.index("L235 PROBE COPY")) + 4
if a != b[hdr_end:]:
    print("  !! the in-place patch differs from the A/B'd module")
    sys.exit(1)
print("  GATE 2 PASS -- byte-identical to optimizer_l235lp.py (minus its header)")
PYX
[ $? -eq 0 ] || { echo "!! reverting"; cp _l237_pre.py optimizer_constructive.py; exit 1; }
echo
"$PY" -u make_submission.py stage 2>&1 | tail -3
echo "   op_wrapper: $(md5sum build_submission/cadc1075/op_wrapper.py | cut -c1-32)"
echo L237_DONE $(date -u +%FT%TZ)
