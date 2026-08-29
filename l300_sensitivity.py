"""L300 -- how much of "we pass rank 1" survives the assumptions.

The candidate (ICCAD_LP_GATE=0 + ICCAD_SHAPE_LP_ITERS=2) projects to 0.8580-0.8584
against rank 1's 0.858632.  The margin is 0.02-0.07 %, which is smaller than every
input's uncertainty, so the only honest presentation is the grid.

Three inputs move it:
  f          local LP second -> grader second.  l172_depthmap.py:39 carries 3.17;
             L160 measured 2.71.  Lower f = bigger RF bill.
  transfer   the fraction of the in-set component deltas that appears on the
             hidden set.  L287 measured 93 % for the package as a whole.
  baseline   A (flat -4.97 % for the shipped package) or B (per component x the
             0.97248 no kill switch can revert).
"""
import json, math, statistics, sys
from pathlib import Path
DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P
import l299_project2 as J

R = J.graded()
Rows = [dict(x, t=x["t"] * J.SHIP_S) for x in P.load()]
b0 = P.total(Rows)
gh0, ga0, ph0 = J.factors(J.M73, "l285_lp_on.json")
RANK1 = J.LB[1]

ARMS = [("gate0            ", "l294_ship.json", "l294_gate0.json"),
        ("gate0 + LP k=2   ", "l297_ship2.json", "l297_g0k2.json"),
        ("gate0 + LP k=2 *c", "l297_ship.json", "l297_g0k2.json"),
        ("LP k=2 alone     ", "l294_ship.json", "l290_inset_lp2.json")]

print("rank 1 = %.6f.  Cell = projected graded total; ** = beats rank 1." % RANK1)
for label, base, arm in ARMS:
    gh, ga, ph = J.factors(base, arm)
    dt = P.dt_by_n(DIR / base, DIR / arm)
    fb = statistics.mean(dt.values()) if dt else 0.0
    print()
    print("  %s   (in-set %+.4f %%)" % (label,
          100 * (json.load(open(DIR / arm))["total_score"] /
                 json.load(open(DIR / base))["total_score"] - 1)))
    print("            f=  2.20     2.71     3.17     4.00   |  grader s at f=3.17")
    for tr in (1.00, 0.93, 0.85, 0.75):
        row = []
        for f in (2.20, 2.71, 3.17, 4.00):
            rf = (P.total(Rows, lambda x: max(0.0, dt.get(x["n"], fb)) / f) - b0) / b0
            a = J.total(R, 1 + (gh - 1) * tr, 1 + (ga - 1) * tr, 1 + (ph - 1) * tr,
                        J.DQ_FLAT) * (1 + rf)
            b = J.total(R, gh0 * (1 + (gh - 1) * tr), ga0 * (1 + (ga - 1) * tr),
                        ph0 * (1 + (ph - 1) * tr)) * J.CODE * (1 + rf)
            row.append("%.5f%s" % (min(a, b), "**" if max(a, b) < RANK1 else
                                   ("*" if min(a, b) < RANK1 else " ")))
        gs = 52.0712 * J.SHIP_S + sum(max(0.0, d) for d in dt.values()) / 3.17
        print("  transfer %3.0f%%  %s  |  %.1f s" % (100 * tr, "  ".join(row), gs))
    print("            (cell shows the better of baselines A/B; ** = BOTH beat rank 1)")
