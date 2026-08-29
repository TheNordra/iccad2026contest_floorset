"""L296: does the LP gate compose with LP depth, and is the sum the right model?

The two knobs are independent by construction:

    _lp_gate_ok(n)      -> whether the LP runs on this case at all (71 -> 100)
    _shape_lp_depth()   -> how many passes where it does          (1 -> k)

which makes an EXACT prediction, and the prediction is a gate rather than a
hope.  Split the 100 cases by the gate table:

    on the 71 gated-ON n :  ship == gate0   and   lp2 == both     (bit-for-bit)
    on the 29 gated-OFF n:  ship == lp2                           (bit-for-bit)

If those hold, the composition decomposes with no residual:

    both - ship  =  [lp2 - ship, on the 71]        <- depth, where the LP ran
                 +  [gate0 - ship, on the 29]      <- the gate's first pass
                 +  [both - gate0, on the 29]      <- the CROSS TERM: a second
                                                      pass on cases neither arm
                                                      alone can reach
The cross term is the whole question.  It is what makes `both` more than the
sum, or -- if the first pass already took everything -- less.
"""
import json
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402
import math                                                       # noqa: E402

F, SCALE, GRADER_BETA, THRESH_S = 3.17, 0.8679, 52.0712, 64.1
SHIPS = ["l294_ship.json", "l294_ship_r2.json"]
ARMS = {"ship": SHIPS, "gate0": ["l294_gate0.json", "l294_gate0_r2.json"],
        "lp2": ["l290_inset_lp2.json"], "both": ["l296_both.json",
        "l296_both_r2.json"], "both4": ["l296_both4.json"]}
W = lambda n: math.exp(n / 12.0)


def cs(f):
    return {r["test_id"]: r for r in json.load(open(DIR / f))["test_results"]}


def gate_map():
    t = (DIR / "optimizer_constructive.py").read_text(errors="replace")
    i = t.index("_L196_LPGATE = {")
    return eval(t[i + len("_L196_LPGATE = "):t.index("}", i) + 1])


G = gate_map()
OFF = {n for n, v in G.items() if not v}
A = {a: cs(fs[0]) for a, fs in ARMS.items() if (DIR / fs[0]).exists()}
missing = [a for a in ARMS if a not in A]
if missing:
    print("not measured yet: %s" % missing)
ids = sorted(A["ship"])
sw = sum(W(A["ship"][i]["block_count"]) for i in ids)


def tot(a, sub=None):
    s = [i for i in ids if sub is None or (A["ship"][i]["block_count"] in OFF) == sub]
    return sum(W(A["ship"][i]["block_count"]) * A[a][i]["cost"] for i in s) / sw


def eq(x, y, sub):
    s = [i for i in ids if (A["ship"][i]["block_count"] in OFF) == sub]
    n = sum(1 for i in s if A[x][i]["cost"] == A[y][i]["cost"]
            and A[x][i].get("positions") == A[y][i].get("positions"))
    return n, len(s)


print("== G0: the mechanism's own predictions, bit-for-bit ==")
for x, y, sub, lbl in (("ship", "gate0", False, "on the 71 gated-ON  n"),
                       ("lp2", "both", False, "on the 71 gated-ON  n"),
                       ("ship", "lp2", True, "on the 29 gated-OFF n")):
    if x in A and y in A:
        n, m = eq(x, y, sub)
        print("   %-6s == %-6s %s : %3d/%3d   %s"
              % (x, y, lbl, n, m, "PASS" if n == m else "FAIL"))
if "both" in A and (DIR / ARMS["both"][1]).exists():
    b2 = cs(ARMS["both"][1])
    n = sum(1 for i in ids if A["both"][i]["cost"] == b2[i]["cost"]
            and A["both"][i].get("positions") == b2[i].get("positions"))
    print("   both determinism (cost AND positions)        : %3d/100   %s"
          % (n, "PASS" if n == 100 else "FAIL"))

print("\n== G1: totals and feasibility ==")
for a in ARMS:
    if a not in A:
        continue
    j = json.load(open(DIR / ARMS[a][0]))
    print("   %-6s %.9f   feasible %3d/100   local %.2f s"
          % (a, j["total_score"],
             sum(1 for t in j["test_results"] if t["is_feasible"]),
             sum(t.get("runtime_seconds", 0.0) for t in j["test_results"])))

print("\n== G2: is the composition the sum? ==")
ts = tot("ship")
# NOTE two conventions are in play in this ledger and they differ by ~2 %
# relative: l276_price.quality_pct divides by the BASE total, l287_transfer.py
# divides by the ARM total.  Everything priced (l276, l293, L294's NET) uses the
# base-denominator form, so that is what is used here.  G3 below already does.
g = {a: 100.0 * (ts - tot(a)) / ts for a in A}
for a in ("gate0", "lp2", "both", "both4"):
    if a in g:
        print("   %-6s %+8.4f %%" % (a, g[a]))
if "both" in g:
    naive = g["gate0"] + g["lp2"]
    print("   %-6s %+8.4f %%   <- naive sum, gate0 + lp2" % ("sum", naive))
    print("   residual (both - sum) : %+.4f pp  => %s"
          % (g["both"] - naive,
             "SUPER-additive" if g["both"] > naive + 1e-4 else
             "sub-additive" if g["both"] < naive - 1e-4 else "exactly additive"))

print("\n== G3: the exact decomposition, on the gate's own split ==")
if "both" in A:
    on = 100.0 * (tot("ship", False) - tot("lp2", False)) / ts
    off1 = 100.0 * (tot("ship", True) - tot("gate0", True)) / ts
    off2 = 100.0 * (tot("gate0", True) - tot("both", True)) / ts
    print("   depth on the 71 the LP already ran on : %+.4f pp" % on)
    print("   the gate's first pass on the 29       : %+.4f pp" % off1)
    print("   CROSS TERM, a 2nd pass on those 29    : %+.4f pp" % off2)
    print("   ---------------------------------------------")
    print("   total                                  %+.4f pp  (measured %+.4f)"
          % (on + off1 + off2, 100.0 * (ts - tot("both")) / ts))

print("\n== G4: priced ==")
rows = [dict(x, t=x["t"] * SCALE) for x in P.load()]
base = P.total(rows)
tl = {A["ship"][i]["block_count"]:
      statistics.mean(cs(f)[i]["runtime_seconds"] for f in SHIPS) for i in ids}
print("   %-7s %9s %9s %10s %10s %10s %8s"
      % ("arm", "quality", "dt s", "RF@3.17", "NET@3.17", "NET ratio", "grader"))
for a in ("gate0", "lp2", "both", "both4"):
    if a not in A:
        continue
    ds = [P.dt_by_n(s, f) for s in SHIPS for f in ARMS[a] if (DIR / f).exists()]
    dt = {n: statistics.mean(d[n] for d in ds) for n in ds[0]}
    rf = 100.0 * (base - P.total(rows, lambda r: dt.get(r["n"], 0.0) / F)) / base
    frac = {n: dt[n] / tl[n] for n in dt}
    of = lambda r: r["t"] * frac.get(r["n"], 0.0)
    rr = 100.0 * (base - P.total(rows, of)) / base
    gs = GRADER_BETA * SCALE + sum(max(0.0, of(r)) for r in rows)
    print("   %-7s %+8.4f%% %9.2f %+9.4f%% %+9.4f%% %+9.4f%% %7.1fs%s"
          % (a, g[a], sum(dt.values()), rf, g[a] + rf, g[a] + rr, gs,
             "" if gs < THRESH_S else "  <-- OVER"))
print("   NET ratio = same-box ratio pricing, f cancels (the conservative one)")

# ---- G5: the composition that the G0 bit-equalities let us price EXACTLY ----
# `both` applies depth 2 EVERYWHERE, including on the 29 heavy cases the gate
# newly admits -- and that is where the time goes.  But G0 proved the two knobs
# act on DISJOINT case sets:
#     on the 71 gated-ON  n : gate0 == ship  (the gate changes nothing there)
#     on the 29 gated-OFF n : lp2   == ship  (depth changes nothing there)
# So "LP everywhere, second pass only where the L196 table said it is
# affordable" is exactly additive in BOTH quality and time, and can be priced
# from the arms already measured with no new run.  Arm-mixing is exact here for
# the same reason L172/L196 relied on it: the selection sees only block count,
# never a case's cost.
#
# ⚠️ It is NOT reachable by env var.  ICCAD_SHAPE_LP_ITERS is ungated by
# design, so realising this needs `_L196_LPGATE` -> all 1s AND `_L157_DEPTH`
# -> 2 on the old 1-set, 1 elsewhere.  Two code defaults, not a flag.
print("\n== G5: 'LP everywhere at k=1, second pass only on the old 71' ==")
if "both" in A:
    dg = [P.dt_by_n(s, f) for s in SHIPS for f in ARMS["gate0"]]
    dl = [P.dt_by_n(s, f) for s in SHIPS for f in ARMS["lp2"]]
    dtg = {n: statistics.mean(d[n] for d in dg) for n in dg[0]}
    dtl = {n: statistics.mean(d[n] for d in dl) for n in dl[0]}
    dtb = {n: statistics.mean(d[n] for d in
                              [P.dt_by_n(s, f) for s in SHIPS
                               for f in ARMS["both"]]) for n in dtg}
    mix = {n: (dtg[n] if n in OFF else dtl[n]) for n in dtg}
    print("   leakage check (should be ~noise):")
    print("      gate0's dt on the 71 it does not touch : %+.2f s" %
          sum(v for n, v in dtg.items() if n not in OFF))
    print("      lp2's   dt on the 29 it does not touch : %+.2f s" %
          sum(v for n, v in dtl.items() if n in OFF))
    print("   dt: gate0 %.2f + lp2 %.2f = %.2f s mixed, vs `both` %.2f s "
          "(cross term %+.2f s)"
          % (sum(v for n, v in dtg.items() if n in OFF),
             sum(v for n, v in dtl.items() if n not in OFF),
             sum(mix.values()), sum(dtb.values()),
             sum(dtb.values()) - sum(mix.values())))
    qmix = g["gate0"] + g["lp2"]
    rf = 100.0 * (base - P.total(rows, lambda r: mix.get(r["n"], 0.0) / F)) / base
    frac = {n: mix[n] / tl[n] for n in mix}
    of = lambda r: r["t"] * frac.get(r["n"], 0.0)
    rr = 100.0 * (base - P.total(rows, of)) / base
    gs = GRADER_BETA * SCALE + sum(max(0.0, of(r)) for r in rows)
    print("   %-7s %+8.4f%% %9.2f %+9.4f%% %+9.4f%% %+9.4f%% %7.1fs"
          % ("mix", qmix, sum(mix.values()), rf, qmix + rf, qmix + rr, gs))
    print("\n   value per local second, where the time goes:")
    for lbl, q_, d_ in (("gate0 1st pass on the 29", g["gate0"],
                         sum(v for n, v in dtg.items() if n in OFF)),
                        ("k=2 2nd pass on the 71", g["lp2"],
                         sum(v for n, v in dtl.items() if n not in OFF)),
                        ("k=2 2nd pass on the 29 (cross)",
                         g["both"] - g["gate0"] - g["lp2"],
                         sum(dtb.values()) - sum(mix.values()))):
        print("      %-32s %+7.4f pp / %6.2f s = %+.4f pp/s"
              % (lbl, q_, d_, q_ / d_ if d_ > 1e-9 else float("nan")))
