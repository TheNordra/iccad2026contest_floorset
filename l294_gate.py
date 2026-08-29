"""L294: what does ICCAD_LP_GATE=0 actually cost and buy, in set?

The handoff's §3.1 open item.  `_L196_LPGATE` switches the shape LP OFF on 29
block counts carrying 44.2 % of the graded weight (8 of the heavy 20).  L205
recorded `G6 gate cost -3.3998 % vs the ungated LP` -- i.e. UNGATING is worth
+3.40 % of in-set quality, more than the whole LP was worth at the time.  The
gate exists purely to buy runtime, and L287/L291 showed runtime was being
over-charged 33x, so the trade has to be re-decided.

Measured as a SANDWICH (ship, gate0, ship) so dt is differenced against a
same-session baseline and the two ship runs give the noise floor.

Priced exactly as `l293_frontier.py` does: f = 3.17 (l172_depthmap.py:39,
measured L161), baseline runtime vector scaled 0.8679 (L285).
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402

F = 3.17
SCALE = 0.8679
GRADER_BETA = 52.0712
THRESH_S = 64.1
SHIP, GATE0, SHIP2 = "l294_ship.json", "l294_gate0.json", "l294_ship_r2.json"
ANCHOR = 1.2263251265


def rd(f):
    return json.load(open(DIR / f))


def cases(f):
    return {r["test_id"]: r for r in rd(f)["test_results"]}


def gate_map():
    txt = (DIR / "optimizer_constructive.py").read_text(errors="replace")
    i = txt.index("_L196_LPGATE = {")
    return eval(txt[i + len("_L196_LPGATE = "):txt.index("}", i) + 1])


def stat_ns(tag):
    p = DIR / ("l294_%s_stats.txt" % tag)
    if not p.exists():
        return Counter()
    return Counter(int(l.split()[0]) for l in p.read_text().split("\n") if l.strip())


G = gate_map()
OFF = {n for n, v in G.items() if not v}

# ---- G0: liveness.  A dead flag looks exactly like a decision not to act. ----
print("== G0 liveness: did the flag actually reach the LP? ==")
ok = True
for tag, want in (("ship", 71), ("gate0", 100), ("ship_r2", 71)):
    c = stat_ns(tag)
    good = len(c) == want
    ok &= good
    print("   %-8s LP ran on %3d distinct block counts (want %3d)  %s"
          % (tag, len(c), want, "PASS" if good else "FAIL"))
ran0, rans = set(stat_ns("gate0")), set(stat_ns("ship"))
print("   gate0 \ ship = %d block counts; equals the gated-off set: %s"
      % (len(ran0 - rans), (ran0 - rans) == OFF))
print("   %s\n" % ("ALL PASS" if ok and (ran0 - rans) == OFF else "*** FAILED ***"))

# ---- G1: anchor + determinism -------------------------------------------
a, b, c0 = cases(SHIP), cases(SHIP2), cases(GATE0)
same = sum(1 for i in a if a[i]["cost"] == b[i]["cost"])
posn = sum(1 for i in a if a[i].get("positions") == b[i].get("positions"))
print("== G1 anchor and determinism ==")
for lbl, f in (("ship", SHIP), ("ship_r2", SHIP2), ("gate0", GATE0)):
    j = rd(f)
    print("   %-8s total %.9f  feasible %3d/100  local %.2f s"
          % (lbl, j["total_score"],
             sum(1 for t in j["test_results"] if t["is_feasible"]),
             sum(t.get("runtime_seconds", 0.0) for t in j["test_results"])))
print("   ship reproduces the shipped anchor %.10f : %s"
      % (ANCHOR, abs(rd(SHIP)["total_score"] - ANCHOR) < 5e-10))
print("   ship vs ship_r2  cost %d/100  positions %d/100\n" % (same, posn))

# ---- G2: quality ---------------------------------------------------------
q = P.quality_pct(SHIP, GATE0)
print("== G2 quality, in-set 100, official evaluator ==")
print("   weighted quality delta, gate0 vs ship : %+.4f %%" % q)
mv = [i for i in a if a[i]["cost"] != c0[i]["cost"]]
bt = [i for i in mv if c0[i]["cost"] < a[i]["cost"]]
print("   %d cases moved (%d better / %d worse); all on gated-off n: %s"
      % (len(mv), len(bt), len(mv) - len(bt),
         all(a[i]["block_count"] in OFF for i in mv)))
print("   in-set totals  ship %.9f -> gate0 %.9f"
      % (rd(SHIP)["total_score"], rd(GATE0)["total_score"]))

# ---- G3: dt, differenced against BOTH ship runs --------------------------
d1, d2 = P.dt_by_n(SHIP, GATE0), P.dt_by_n(SHIP2, GATE0)
dt = {n: (d1[n] + d2[n]) / 2.0 for n in d1}
noise = P.dt_by_n(SHIP, SHIP2)
print("\n== G3 dt, local seconds ==")
for lbl, d in (("vs ship", d1), ("vs ship_r2", d2), ("mean", dt),
               ("ship vs ship_r2 (noise)", noise)):
    v = sorted(d.values())
    print("   %-24s sum %+8.2f s   p50 %+.3f s   max %+.3f s"
          % (lbl, sum(v), v[len(v) // 2], v[-1]))
von = sorted(dt[n] for n in dt if n in OFF)
print("   of which on the 29 gated-off n : sum %+.2f s (%.0f %% of the total)"
      % (sum(von), 100 * sum(von) / sum(dt.values())))

# ---- G4: RF price, exactly as l293_frontier ------------------------------
print("\n== G4 priced, f = %.2f, baseline scaled %.4f (l293_frontier) ==" % (F, SCALE))
fb = statistics.mean(dt.values())
rows = [dict(x, t=x["t"] * SCALE) for x in P.load()]
base = P.total(rows)
print("   %-12s %10s %10s %10s %10s" % ("f", "RF", "NET", "grader s", "verdict"))
for f in (1.00, 2.71, 3.17):
    rf = 100.0 * (base - P.total(rows, lambda r: dt.get(r["n"], fb) / f)) / base
    gs = GRADER_BETA * SCALE + sum(dt.values()) / f
    print("   %-12.2f %+9.4f%% %+9.4f%% %9.1fs %10s%s"
          % (f, rf, q + rf, gs, "GREEN" if q + rf > 0 else "RED",
             "" if gs < THRESH_S else "   <-- OVER the rank-2 threshold"))
lo, hi = 0.05, 200.0
for _ in range(80):
    m = (lo + hi) / 2
    r = 100.0 * (base - P.total(rows, lambda r: dt.get(r["n"], fb) / m)) / base
    if q + r > 0:
        hi = m
    else:
        lo = m
print("   break-even f = %.3f   (the grader must run the LP this much faster"
      " than this box)" % hi)
print("   shipped sits at %.1f s; rank-2 threshold %.1f s"
      % (GRADER_BETA * SCALE, THRESH_S))

# ---- G5: the partial-ungate frontier -------------------------------------
# If the full ungate is too expensive, the gate is not all-or-nothing: it is a
# per-block-count table.  Rank the 29 gated-off cases by graded value per local
# second and walk the prefix, pricing each prefix exactly as G4 does.  This is
# in-set only, so it is a SHAPE, not a candidate -- picking a cut point on the
# corpus it is scored on is the over-fit L275 warns about.
import math                                                      # noqa: E402

print("\n== G5 partial-ungate frontier (in-set shape only, NOT a pick) ==")
W = lambda n: math.exp(n / 12.0)
sw = sum(W(a[i]["block_count"]) for i in a)
tb = sum(W(a[i]["block_count"]) * a[i]["cost"] for i in a) / sw
val = []
for i in a:
    n = a[i]["block_count"]
    if n not in OFF:
        continue
    dq = 100.0 * W(n) * (a[i]["cost"] - c0[i]["cost"]) / sw / tb   # quality pp
    val.append((n, dq, dt.get(n, 0.0)))
val.sort(key=lambda x: -(x[1] / x[2] if x[2] > 1e-9 else (1e9 if x[1] > 0 else -1e9)))
print("   %-4s %10s %9s %12s" % ("n", "quality pp", "dt s", "pp per s"))
for n, dq, d in val:
    print("   %-4d %+9.4f %9.2f %12s"
          % (n, dq, d, "%+.4f" % (dq / d) if d > 1e-9 else "inf"))

print("\n   %-6s %10s %10s %10s %10s %10s"
      % ("keep", "quality", "dt s", "RF", "NET", "grader s"))
for k in (0, 5, 10, 15, 20, 25, 29):
    sub = {n: d for n, dq, d in val[:k]}
    qq = sum(dq for _n, dq, _d in val[:k])
    fb2 = statistics.mean(sub.values()) if sub else 0.0
    rf = 100.0 * (base - P.total(rows, lambda r: sub.get(r["n"], 0.0) / F)) / base
    gs = GRADER_BETA * SCALE + sum(sub.values()) / F
    print("   %-6d %+9.4f%% %9.2f %+9.4f%% %+9.4f%% %9.1fs%s"
          % (k, qq, sum(sub.values()), rf, qq + rf, gs,
             "" if gs < THRESH_S else "   <-- OVER"))
