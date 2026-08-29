"""L297 -- price any measured arm against RANK 1, not against the 0.30 % ship bar.

Combines three things the project already has but has never put together:

  * L296's corpus projection: the arm's geometry factor `g` and violation factor
    `phi`, applied to the GRADED corpus's own per-case (hgap, agap, V/N_soft).
    The in-set understates violation-trading mechanisms ~3x (in-set weighted
    vrel 0.01407, graded 0.04252).
  * the corrected RF pricing: dt measured locally, divided by f = 3.17
    (l172_depthmap.py:39, measured L161), added to the SHIPPED runtime vector
    (beta x 0.8679, L285) -- the two corrections of L287 §2.
  * the leaderboard as the target.

  <python> l297_rank1_price.py base.json arm.json[,arm.json...]
"""
import json, math, statistics, sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402
import l296_project as J                                          # noqa: E402

F = 3.17
SHIP_S = J.SHIP_S
LB = {1: 0.8586322662042342, 2: 0.888187391, 3: 0.8993286931994098,
      4: 0.9265861161320369, 5: 0.9507093062865333}
GRADER_BETA = 52.0712


def price(base, arm, rows):
    g, phi, _, _ = J.summarise(DIR / base, DIR / arm)
    t0, t1 = J.project(g, phi, rows=rows)                   # quality only, RF from base
    dt = P.dt_by_n(DIR / base, DIR / arm)
    fb = statistics.mean(dt.values()) if dt else 0.0
    # RF: same population, scaled to the shipped runtime vector, dt in grader seconds
    R = [dict(x, t=x["t"] * SHIP_S) for x in P.load()]
    b = P.total(R)
    s = P.total(R, lambda x: max(0.0, dt.get(x["n"], fb)) / F)
    rf = (s - b) / b                                        # >0 = costs
    tot = t1 * (1 + rf)
    gs = GRADER_BETA * SHIP_S + sum(max(0.0, d) for d in dt.values()) / F
    ins = 100 * (json.load(open(DIR / arm))["total_score"] /
                 json.load(open(DIR / base))["total_score"] - 1)
    return dict(g=g, phi=phi, ins=ins, q=100 * (t1 / t0 - 1), rf=100 * rf,
                tot=tot, t0=t0, gs=gs,
                feas=sum(1 for r in json.load(open(DIR / arm))["test_results"]
                         if r["is_feasible"]))


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "l294_ship.json"
    arms = sys.argv[2:] or ["l294_gate0.json"]
    rows = J.graded()
    t0, _ = J.project(1.0, 1.0, rows=rows)
    print("shipped, projected onto the graded corpus : %.6f   (%.1f grader s, rank %d)"
          % (t0, GRADER_BETA * SHIP_S, sum(1 for v in LB.values() if v < t0) + 1))
    print("targets: r1 %.6f   r2 %.6f   r3 %.6f" % (LB[1], LB[2], LB[3]))
    print()
    print("  %-22s %8s %8s | %9s %9s %9s | %9s %8s %6s %s"
          % ("arm", "g", "phi", "in-set", "quality", "RF", "graded", "grader", "feas", "rank"))
    for a in arms:
        if not (DIR / a).exists():
            print("  %-22s (missing)" % a); continue
        r = price(base, a, rows)
        rank = sum(1 for v in LB.values() if v < r["tot"]) + 1
        print("  %-22s %8.5f %8.5f | %+8.4f%% %+8.4f%% %+8.4f%% | %9.6f %7.1fs %5d   %d%s"
              % (a.replace(".json", ""), r["g"], r["phi"], r["ins"], r["q"], r["rf"],
                 r["tot"], r["gs"], r["feas"], rank,
                 "   <== BEATS RANK 1" if r["tot"] < LB[1] else ""))
    print()
    print("  f = %.2f, shipped runtime vector = beta x %.4f, quality projected onto"
          % (F, SHIP_S))
    print("  the graded corpus's own (hgap, agap, V/N_soft) distribution.")
