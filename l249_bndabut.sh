#!/bin/sh
# L249 -- ICCAD_BND_ABUT as a GLOBAL OVERLAY on the portfolio. Never measured.
#
# WHY IT IS WORTH A RUN. L144 screened this knob and the first verdict was RED
# -- then its own adversarial verifier REFUTED that verdict, because the screen
# had run on specs[0:16], which is 0.0126% of the weighted score. Re-run at full
# strength on all 240 OOS cases it reads solo +0.379% / +0.503%, the sign
# survives jackknifing out ANY single case (min +0.259%), and a placebo
# perturbation of the same branch reads -3.66% -- so it is mechanism-specific.
#
# And it is the only quality mechanism in the ledger that costs NO WALL:
# measured runtime ON/OFF 0.996x and 1.003x. It adds no profile, so it pays
# neither the C++ pool nor L167's ~71 ms serial proxy tax. Everything else on
# the quality axis has to buy its way past one of those.
#
# What was never measured is THIS form. L144 screened it as a TWIN (append an
# ON-variant profile and let the proxy arbitrate), which transfers at 1/7. The
# global overlay -- every profile gets the flag -- was only ever measured SOLO,
# on one profile at a time, with the caveat "the portfolio may absorb the gain
# because profile diversity alone already removes 51% of boundary violations".
# That caveat is a hypothesis. This measures it.
#
# PRECONDITION ALREADY CHECKED: constructive.cpp has not been touched since
# 2026-08-19, so constructive_l144v1.cpp is branched from the CURRENT shipping
# source and its numbers transfer. (The ledger's repeated failure is the
# opposite: a probe binary anchored to a placer that has since moved.)
#
#   G0  off-path: the probe binary with the flag UNSET must reproduce the
#       shipped package 100/100 on cost AND positions. Without this, any delta
#       below could be the 101 other changed lines rather than the mechanism.
#   1   in-set quality of the overlay
#
# ICCAD_PROFILE_TIMEOUT=600 because this runs alongside the L248 sweep: an
# oversubscribed box stretches profiles past the 120 s default and they are
# SILENTLY dropped, which would read as a quality change. SA-fallback count is
# asserted to be 0 for the same reason.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l249.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
BIN="$R/constructive_l144v1.exe"
export PYTHONIOENCODING=utf-8
cd "$R/iccad2026contest" || exit 1
echo "=== L249  $(date -u +%FT%TZ) ==="
echo "  probe binary: $(md5sum "$BIN" | cut -c1-32)"

for ARM in offpath abut; do
  if [ "$ARM" = offpath ]; then set --; else set -- ICCAD_BND_ABUT=1; fi
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" \
      ICCAD_PROFILE_TIMEOUT=600 "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L249_${ARM}.json" > "../l249_${ARM}.log" 2>&1
  echo "  $ARM exit=$?  SAfallback=$(grep -c 'SA fallback' "../l249_${ARM}.log")  allfail=$(grep -c 'all profiles failed' "../l249_${ARM}.log")"
done
cd "$R" || exit 1
"$PY" - <<'PYX'
import json, math, sys
L = lambda f: {r["block_count"]: r for r in json.load(open(f))["test_results"]}
ref, off, on = (L("results_L237_post.json"), L("results_L249_offpath.json"),
                L("results_L249_abut.json"))
c = sum(1 for n in ref if ref[n]["cost"] == off[n]["cost"])
p = sum(1 for n in ref if ref[n]["positions"] == off[n]["positions"])
print(f"G0 off-path  cost {c}/100  positions {p}/100   "
      f"{'PASS' if c == p == 100 else 'FAIL'}")
if c != 100 or p != 100:
    print("   the probe binary is not off-path; any delta below is contaminated")
    sys.exit(1)
W = lambda n: math.exp(n / 12.0)
SW = sum(W(n) for n in off)
q0 = sum(W(n) * off[n]["cost"] for n in off) / SW
q1 = sum(W(n) * on[n]["cost"] for n in on) / SW
mv = [n for n in off if off[n]["cost"] != on[n]["cost"]]
worse = [n for n in mv if on[n]["cost"] > off[n]["cost"]]
feas = sum(1 for n in on if on[n]["is_feasible"])
print(f"OVERLAY on the portfolio: {100*(q0-q1)/q0:+.4f}%   moved {len(mv)}/100  "
      f"worse {len(worse)}  feasible {feas}/100")
for lo, hi in ((20, 60), (60, 100), (100, 121)):
    ns = [n for n in off if lo < n <= hi]
    d = 100 * sum(W(n) * (off[n]["cost"] - on[n]["cost"]) for n in ns) / \
        sum(W(n) * off[n]["cost"] for n in off)
    m = sum(1 for n in ns if off[n]["cost"] != on[n]["cost"])
    print(f"   {lo+1:>3}-{hi:<3} contributes {d:+.4f}%   moved {m}/{len(ns)}")
print()
print("L144 measured this SOLO at +0.379%/+0.503% on 240 cases. If the portfolio")
print("absorbs it, this reads near 0 and the axis closes. If it survives, it is")
print("the only quality gain in the ledger that costs no wall.")
PYX
echo L249_DONE $(date -u +%FT%TZ)
