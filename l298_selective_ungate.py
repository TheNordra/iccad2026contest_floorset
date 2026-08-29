"""L298 -- the LP gate is per-block-count, so it does not have to be all-or-nothing.

`ICCAD_LP_GATE=0` (L294) is worth -2.2282 % in-set / -2.5062 % projected on the
graded corpus, but its RF bill is +0.9726 % -- 13x LP k=4's bill for 40 % LESS
added time.  The reason is distribution, not volume: L295 measured the per-case
slack to the RF floor and it is only 0.054 s on case 112 and 0.19 s on case 101,
while gate0 spends 1.54 s and 0.18 s of LOCAL time there (0.49 / 0.06 grader s).

So the same question L157 asked about depth applies here: the floor is per case,
therefore the gate should be too.  `_L196_LPGATE` is a dict in
`optimizer_constructive.py` -- ungating a subset is a WRAPPER-ONLY change.

Two subsets are priced, and they differ in how much they are allowed to know:

  RF-SAFE   ungate n iff the added grader time fits inside that case's slack to
            the RF floor.  Uses NO quality information, so there is nothing to
            over-fit; this is the L157 shape.
  GREEDY    ungate in descending quality-per-RF order.  Reported because it is
            the upper bound, and flagged because selecting block counts by their
            in-set quality delta is the project's five-times-burned
            case-idiosyncratic-winner shape.
"""
import json, math, statistics, sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402
import l296_project as J                                          # noqa: E402

# L310: F is the machine-speed factor. 3.17 was the value in force when this
# was written; L308 later MEASURED it at 2.38-2.84 (band-dependent), and F
# enters the slack-fit test, so the RF-SAFE subset is a function of it.
# Default unchanged so every number this file has already reported still
# reproduces; override with `l298_selective_ungate.py <F>` or $ICCAD_F.
import os as _os
F = float(_os.environ.get("ICCAD_F") or
          (sys.argv[1] if len(sys.argv) > 1 else 3.17))
SHIP_S = J.SHIP_S
LB1 = 0.8586322662042342
BASE, ARM = "l294_ship.json", "l294_gate0.json"


def gate_map():
    txt = (DIR / "optimizer_constructive.py").read_text(errors="replace")
    i = txt.index("_L196_LPGATE = {")
    return eval(txt[i + len("_L196_LPGATE = "):txt.index("}", i) + 1])


G = gate_map()
OFF = sorted(n for n, v in G.items() if not v)
b = {r["test_id"]: r for r in json.load(open(DIR / BASE))["test_results"]}
a = {r["test_id"]: r for r in json.load(open(DIR / ARM))["test_results"]}
byn_b = {r["block_count"]: r for r in b.values()}
byn_a = {r["block_count"]: r for r in a.values()}
dt = P.dt_by_n(DIR / BASE, DIR / ARM)

rows = J.graded()
gr = {r["n"]: r for r in rows}
TH = 0.7 ** (1 / 0.3)
meds = {x["n"]: x["med"] for x in P.load()}
tship = {x["n"]: x["t"] * SHIP_S for x in P.load()}
slack = {n: TH * meds[n] - tship[n] for n in meds}

print("gated-OFF block counts: %d of %d  (%s)" % (len(OFF), len(G), OFF))
print()
print("  n   in-set dq   dt_local  dt_grader   slack   fits   graded weight")
W = sum(r["w"] for r in rows)
items = []
for n in OFF:
    if n not in byn_b or n not in byn_a:
        continue
    w = math.exp(n / 12.0)
    dq = (byn_a[n]["cost"] - byn_b[n]["cost"]) * w / W          # weighted cost delta, in-set
    d = max(0.0, dt.get(n, 0.0))
    dg = d / F
    fits = dg <= slack.get(n, 0.0)
    items.append(dict(n=n, dq=dq, d=d, dg=dg, sl=slack.get(n, 0.0), fits=fits, w=w / W))
    if n >= 90:
        print("  %3d  %+10.6f  %8.2fs  %8.3fs  %6.3fs   %-5s  %6.3f%%"
              % (n, dq, d, dg, slack.get(n, 0.0), "yes" if fits else "NO", 100 * w / W))


def evaluate(sel, label):
    """quality: scale the FULL gate0 g/phi by the share of in-set gain the subset
       carries; RF: only the selected block counts pay."""
    g_all, phi_all, _, _ = J.summarise(DIR / BASE, DIR / ARM)
    tot_dq = sum(i["dq"] for i in items)
    got = sum(i["dq"] for i in sel)
    frac = got / tot_dq if tot_dq else 0.0
    g = 1.0 + (g_all - 1.0) * frac
    phi = 1.0 + (phi_all - 1.0) * frac
    t0, t1 = J.project(g, phi, rows=rows)
    S = {i["n"] for i in sel}
    R = [dict(x, t=x["t"] * SHIP_S) for x in P.load()]
    b0 = P.total(R)
    b1 = P.total(R, lambda x: (dt.get(x["n"], 0.0) / F) if x["n"] in S else 0.0)
    rf = (b1 - b0) / b0
    tot = t1 * (1 + rf)
    gs = 52.0712 * SHIP_S + sum(max(0.0, dt.get(i["n"], 0.0)) for i in sel) / F
    print("  %-26s  n=%2d  qshare %5.1f%%  quality %+7.4f%%  RF %+7.4f%%"
          "  -> %.6f  (%.1f s)  vs r1 %+7.4f%% %s"
          % (label, len(sel), 100 * frac, 100 * (t1 / t0 - 1), 100 * rf, tot, gs,
             100 * (LB1 / tot - 1), "<== BEATS RANK 1" if tot < LB1 else ""))
    return tot


print()
print("== subsets ==")
evaluate(items, "ALL 29 (= ICCAD_LP_GATE=0)")
evaluate([i for i in items if i["fits"]], "RF-SAFE (dt fits slack)")
gain = sorted([i for i in items if i["dq"] < 0], key=lambda i: (i["dq"] / (i["dg"] + 1e-9)))
for k in (4, 8, 12, 16, 20):
    evaluate(gain[:k], "GREEDY top-%d by dq/dt" % k)
evaluate([i for i in items if i["dq"] < 0], "every n that helps in-set")
