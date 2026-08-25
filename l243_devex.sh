#!/bin/sh
# L243 -- decide `devex` with a measurement instead of an argument.
#
# L241, 100 cases min-of-3: the whole LP is 1.072x faster (solver 1.120x), which
# on top of L235's 1.170x would make the LP 1.254x and is worth roughly
# +0.15pp of RF. But it moves 66/100 LAYOUTS and the LP objective on 4 of them
# by 1e-3..4e-2 -- build counts 2/1 and 4/6, i.e. a different degenerate vertex
# breaks a different cluster and hence freezes a different unit set. The moves
# go BOTH ways (case 10 +3.4% worse, case 32 -4.2% better), so this is a
# variance-increasing change with ~0 expected quality effect, which is the
# shape route A had when it was turned off.
#
# So it does not get shipped on the RF number. It gets an OOS pass, and the bar
# is the one this ledger always uses: BOTH samples non-negative.
#
#   0. the knob OFF must reproduce the staged package bit-for-bit
#   1. in set, knob ON
#   2. OOS baseline (knob off) and OOS devex, both samples
set -u
LOCK=/c/ICCAD_ml/ship_final/.l243.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
export PYTHONIOENCODING=utf-8

echo "=== L243 gate 0: the knob OFF is a no-op  $(date -u +%FT%TZ) ==="
cd "$R/iccad2026contest" || exit 1
env ICCAD_ADAPTIVE_CORES=48 "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_constructive.py -o ../results_L243_off.json \
  > ../l243_off.log 2>&1
echo "  off exit=$?  SAfallback=$(grep -c 'SA fallback' ../l243_off.log)"
env ICCAD_ADAPTIVE_CORES=48 ICCAD_LP_EDGE_WEIGHT=devex "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_constructive.py -o ../results_L243_devex.json \
  > ../l243_devex.log 2>&1
echo "  devex exit=$?  SAfallback=$(grep -c 'SA fallback' ../l243_devex.log)"
cd "$R" || exit 1
"$PY" - <<'PYX'
import json, math, sys
L = lambda f: {r["block_count"]: r for r in json.load(open(f))["test_results"]}
ref, off, dvx = (L("results_L237_post.json"), L("results_L243_off.json"),
                 L("results_L243_devex.json"))
c = sum(1 for n in ref if ref[n]["cost"] == off[n]["cost"])
p = sum(1 for n in ref if ref[n]["positions"] == off[n]["positions"])
print(f"GATE 0  knob off vs the staged package: cost {c}/100  positions {p}/100"
      f"   {'PASS' if c == p == 100 else 'FAIL'}")
if c != 100 or p != 100:
    sys.exit(1)
W = lambda n: math.exp(n / 12.0)
SW = sum(W(n) for n in off)
q0 = sum(W(n) * off[n]["cost"] for n in off) / SW
q1 = sum(W(n) * dvx[n]["cost"] for n in dvx) / SW
mv = [n for n in off if off[n]["cost"] != dvx[n]["cost"]]
worse = [n for n in mv if dvx[n]["cost"] > off[n]["cost"]]
feas = sum(1 for n in dvx if dvx[n]["is_feasible"])
print(f"IN SET  devex {100*(q0-q1)/q0:+.4f}%   moved {len(mv)}/100  "
      f"worse {len(worse)}  feasible {feas}/100")
print("        (in set is one case per block count -- the sign here is not the"
      " verdict, the OOS pass below is)")
PYX
[ $? -eq 0 ] || { echo "!! the knob is not a no-op when unset -- STOP"; exit 1; }

echo; echo "=== L243 OOS: baseline then devex, both samples  $(date -u +%FT%TZ) ==="
for S in s1 s2; do
  env ICCAD_ROUTE_A=0 "$PY" -u l140_oos_soft_audit.py run --sample "$S" \
    --cores 48 --out "l243_${S}_base.json" > "l243_${S}_base.log" 2>&1
  echo "$S/base  exit=$?  SAfallback=$(grep -c 'SA fallback' "l243_${S}_base.log")"
  env ICCAD_ROUTE_A=0 ICCAD_LP_EDGE_WEIGHT=devex "$PY" -u l140_oos_soft_audit.py \
    run --sample "$S" --cores 48 --out "l243_${S}_devex.json" \
    > "l243_${S}_devex.log" 2>&1
  echo "$S/devex exit=$?  SAfallback=$(grep -c 'SA fallback' "l243_${S}_devex.log")"
done
"$PY" -u l243_score.py | tee l243_score.out
echo L243_DONE $(date -u +%FT%TZ)
