"""L301 -- price the gate0 + LP k=2 candidate against RANK 1 with f CANCELLED.

`l294_final.py` §(b) is the right pricing method and it is not mine: express the
added LP time as a fraction of that case's local wall on THIS box, then apply the
fraction to the grader's OWN measured per-case time.  The machine factor divides
out; the only external input is our own grader runtime vector.  Its control (LP
k=2 reproduced to 0.014 pp) is what makes it trustworthy.

This applies the same method to the combined arm and then carries the result
through L296's corpus projection to the leaderboard.
"""
import json, math, statistics, sys
from pathlib import Path
DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P
import l299_project2 as J

SCALE, GRADER_BETA = J.SHIP_S, 52.0712
RANK1 = J.LB[1]
R = J.graded()
rows = [dict(x, t=x["t"] * SCALE) for x in P.load()]
base = P.total(rows)
gh0, ga0, ph0 = J.factors(J.M73, "l285_lp_on.json")


def cases(f):
    return {r["test_id"]: r for r in json.load(open(DIR / f))["test_results"]}


def price(ships, arms, label):
    ds = [P.dt_by_n(s, g) for s in ships for g in arms]
    dt = {n: statistics.mean(d[n] for d in ds) for n in ds[0]}
    tl = {}
    c0 = cases(ships[0])
    for i in c0:
        n = c0[i]["block_count"]
        tl[n] = statistics.mean(cases(s)[i]["runtime_seconds"] for s in ships)
    frac = {n: dt[n] / tl[n] for n in dt}
    of = lambda r: r["t"] * frac.get(r["n"], 0.0)
    rf = (P.total(rows, of) - base) / base
    gadd = sum(max(0.0, of(r)) for r in rows)
    gh = ga = ph = 0.0
    n = 0
    for s in ships:
        for g in arms:
            a, b, c = J.factors(s, g)
            gh += a; ga += b; ph += c; n += 1
    gh, ga, ph = gh / n, ga / n, ph / n
    tA = J.total(R, gh, ga, ph, J.DQ_FLAT) * (1 + rf)
    tB = J.total(R, gh0 * gh, ga0 * ga, ph0 * ph) * J.CODE * (1 + rf)
    ins = statistics.mean(json.load(open(DIR / g))["total_score"] /
                          json.load(open(DIR / s))["total_score"] - 1
                          for s in ships for g in arms)
    print("  %-22s in-set %+7.4f%% | dt local %+6.2f s -> grader %+5.2f s "
          "(implied f %.2f) | RF %+7.4f%% | A %.6f  B %.6f | %s"
          % (label, 100 * ins, sum(dt.values()), gadd,
             sum(dt.values()) / gadd if gadd > 0 else float("inf"), 100 * rf,
             tA, tB,
             "BEATS RANK 1 on both" if max(tA, tB) < RANK1 else
             ("beats on one" if min(tA, tB) < RANK1 else "short")))
    return tA, tB


print("rank 1 = %.6f   shipped projects to A %.6f / B %.6f"
      % (RANK1, J.total(R, 1, 1, 1, J.DQ_FLAT), J.total(R, gh0, ga0, ph0) * J.CODE))
print()
price(["l294_ship.json", "l294_ship_r2.json"],
      ["l294_gate0.json", "l294_gate0_r2.json"], "gate0 (control: L294)")
price(["l297_ship.json", "l297_ship2.json"],
      ["l297_g0k2.json", "l297_g0k2_r2.json"], "gate0 + LP k=2")
price(["l294_ship.json"], ["l290_inset_lp2.json"], "LP k=2 alone (control)")
