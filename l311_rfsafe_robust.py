"""L311 -- is L298's RF-SAFE subset actually free when f is not what we assumed?

L298 built the "free" arm: ungate a block count only if the added grader time
fits inside that case's own slack to the RF floor. Its RF bill is +0.0000% BY
CONSTRUCTION -- but only at the f used to pick the subset. L298 used F = 3.17;
L308 later MEASURED f at 2.38-2.84. A subset chosen at 3.17 is choosing on the
assumption that the grader is 33 % faster than the low end of the measurement.

So the number that matters is not "is it free" (it is, tautologically) but

    pick the subset at F_sel, then pay for it at F_pay < F_sel.

That is the same shape as every other robustness question in this tree: an
in-sample selection evaluated out-of-sample. Here the "sample" is a machine
speed rather than a corpus.

Run:  <python> l311_rfsafe_robust.py
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402
import l296_project as J                                          # noqa: E402

SHIP_S = J.SHIP_S
LB1 = 0.8586322662042342                       # rank-1 total, 08-23 board
LB2 = 0.888187391                              # rank-2 total
BASE, ARM = "l294_ship.json", "l294_gate0.json"
TH = 0.7 ** (1 / 0.3)


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
W = sum(r["w"] for r in rows)
meds = {x["n"]: x["med"] for x in P.load()}
tship = {x["n"]: x["t"] * SHIP_S for x in P.load()}
slack = {n: TH * meds[n] - tship[n] for n in meds}

items = []
for n in OFF:
    if n not in byn_b or n not in byn_a:
        continue
    w = math.exp(n / 12.0)
    items.append(dict(n=n,
                      dq=(byn_a[n]["cost"] - byn_b[n]["cost"]) * w / W,
                      d=max(0.0, dt.get(n, 0.0))))
TOT_DQ = sum(i["dq"] for i in items)
G_ALL, PHI_ALL, _, _ = J.summarise(DIR / BASE, DIR / ARM)


def select(f_sel):
    return [i for i in items if (i["d"] / f_sel) <= slack.get(i["n"], 0.0)]


def evaluate(sel, f_pay):
    frac = (sum(i["dq"] for i in sel) / TOT_DQ) if TOT_DQ else 0.0
    t0, t1 = J.project(1.0 + (G_ALL - 1.0) * frac,
                       1.0 + (PHI_ALL - 1.0) * frac, rows=rows)
    S = {i["n"] for i in sel}
    R = [dict(x, t=x["t"] * SHIP_S) for x in P.load()]
    b0 = P.total(R)
    b1 = P.total(R, lambda x: (dt.get(x["n"], 0.0) / f_pay) if x["n"] in S else 0.0)
    rf = (b1 - b0) / b0
    return dict(n=len(sel), frac=frac, q=100 * (t1 / t0 - 1), rf=100 * rf,
                tot=t1 * (1 + rf))


def rank(t):
    return 1 if t < LB1 else 2 if t < LB2 else 3


F_GRID = [3.17, 2.84, 2.61, 2.38]
print("L311 -- RF-SAFE cross-pricing.  Rows = f used to PICK the subset,")
print("        columns = f we actually get.  L308 measured f = 2.38-2.84;")
print("        3.17 is the value L298 used.\n")
hdr = f"{'pick at':>8} {'n':>3} | " + " | ".join(f"pay at {f:>5.2f}" for f in F_GRID)
print(hdr)
print("-" * len(hdr))
for fs in F_GRID:
    sel = select(fs)
    cells = []
    for fp in F_GRID:
        e = evaluate(sel, fp)
        cells.append(f"RF {e['rf']:+6.4f}% r{rank(e['tot'])}")
    print(f"{fs:>8.2f} {len(sel):>3} | " + " | ".join(f"{c:>16}" for c in cells))

print("\nquality carried by each subset (independent of f_pay):")
for fs in F_GRID:
    e = evaluate(select(fs), fs)
    print(f"  pick at {fs:.2f}:  n={e['n']:>2}  qshare {100*e['frac']:>5.1f}%  "
          f"quality {e['q']:+.4f}%  total {e['tot']:.6f}  rank {rank(e['tot'])}")

print(f"""
READING IT
  The diagonal is L298's claim and it is exactly +0.0000% every time -- that is
  the construction, not evidence. The cells BELOW the diagonal are the real
  test: a subset picked assuming a fast grader, paid for on a slow one.
  If those stay at +0.0000% the arm is genuinely f-robust; if they do not, then
  "free" was a statement about our assumption, not about the package.

  Compare against the full ungate (all {len(items)} block counts):""")
for fp in F_GRID:
    e = evaluate(items, fp)
    print(f"    ALL, paid at f={fp:.2f}:  RF {e['rf']:+.4f}%  total {e['tot']:.6f}  "
          f"rank {rank(e['tot'])}")
