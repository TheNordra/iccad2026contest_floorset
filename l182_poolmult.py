"""L182 - the depth map assumed our pool costs t_beta. It costs 1.155 x t_beta.

L172 built `_L157_DEPTH` from

    budget(n) = 0.3046 * M(n) * s_med  -  t_beta(n)  -  dt_tangent(n)/f

with `t_beta(n)` = the grader's own per-case runtime for M73. That is our
runtime only if our pool costs what M73's pool cost. L181 measures it, route A
off, on one box, 100 cases:

    M73      112.77 s      weighted cost 1.281457
    current  130.21 s      weighted cost 1.260247        1.155x wall, +1.66% quality

So every case starts with LESS slack than the map assumed, and the map should be
re-derived against `P * t_beta(n)` with P the pool multiplier.

WHAT P ACTUALLY IS, honestly: 1.155 is measured with ROUTE A OFF. The shipped
package runs route A ON at >=40 detected cores, and route A has never run on the
grader -- beta was M73, which does not have it. L110/L111 PROJECTED -32.2% wall
at 48 real cores; on this box's 16 physical cores it costs 2.9x. So the true P
is somewhere in [~0.78, 1.155] and cannot be pinned from here. This grid prices
the map across that whole range instead of picking a point.

The quality side is unchanged: arm-mixed from the committed flat k=1/2/3 OOS
arms, which do not depend on how fast the pool got there.
"""
import json
from collections import Counter
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M

DIR = Path(__file__).parent


def build_P(rows, dtan, dpass, near, P, s_med, f=M.F):
    out = {}
    for r in rows:
        n = r["n"]
        budget = L.THR * r["med"] * s_med - P * r["t"] - dtan.get(near(n), 0.0) / f
        k = 1
        for kk in (2, 3):
            if (kk - 1) * dpass.get(near(n), 0.0) / f <= budget:
                k = kk
        out[n] = k
    return out


def rf_P(rows, dmap, dtan, dpass, near, P, s_true, f=M.F):
    num = den = 0.0
    for r in rows:
        n = r["n"]
        k = dmap.get(n, 1)
        t = P * r["t"] + (dtan.get(near(n), 0.0)
                          + (k - 1) * dpass.get(near(n), 0.0)) / f
        num += r["w"] * r["q"] * max(0.7, (t / (r["med"] * s_true)) ** 0.3)
        den += r["w"]
    return num / den


def main():
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    x090 = {int(x): v for x, v in
            json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    k1 = {n: 1 for n in x090}
    print(__doc__)
    print("=" * 78)

    print("how the SHIPPED x0.90 map prices as the pool multiplier moves")
    print("(RF is vs k=1 at the SAME P, so it isolates the map, not the pool)\n")
    print("{:>8}{:>12}{:>12}{:>12}{:>12}"
          .format("P", "RF @med x1.0", "@x0.90", "@x0.80", "NET @x0.90"))
    q = (M.quality(x090, "s1")[0] + M.quality(x090, "s2")[0]) / 2
    for P in (0.80, 0.90, 1.00, 1.155, 1.30):
        cells = ""
        for s in (1.00, 0.90, 0.80):
            b = rf_P(rows, k1, dtan, dpass, near, P, s)
            r = rf_P(rows, x090, dtan, dpass, near, P, s)
            cells += "{:>+11.4f}%".format(100 * (b - r) / b)
        b90 = rf_P(rows, k1, dtan, dpass, near, P, 0.90)
        r90 = rf_P(rows, x090, dtan, dpass, near, P, 0.90)
        print("{:>8.3f}{}{:>+11.4f}%".format(
            P, cells, q + 100 * (b90 - r90) / b90))

    print("\nmaps re-derived with the pool multiplier folded in (s_med = 0.90)")
    print("{:>8}{:>22}{:>10}{:>10}{:>12}"
          .format("P", "depths", "qual s1", "qual s2", "NET @P,x0.90"))
    for P in (0.80, 0.90, 1.00, 1.155, 1.30):
        m = build_P(rows, dtan, dpass, near, P, 0.90)
        q1 = M.quality(m, "s1")[0]
        q2 = M.quality(m, "s2")[0]
        b = rf_P(rows, k1, dtan, dpass, near, P, 0.90)
        r = rf_P(rows, m, dtan, dpass, near, P, 0.90)
        same = "  == shipped" if m == x090 else ""
        print("{:>8.3f}{:>22}{:>+9.4f}%{:>+9.4f}%{:>+11.4f}%{}"
              .format(P, str(dict(sorted(Counter(m.values()).items()))),
                      q1, q2, (q1 + q2) / 2 + 100 * (b - r) / b, same))

    print("\nAlso: what the POOL itself costs, priced on the new medians.")
    print("This is the +1.66% of quality's own bill, and it is separate from")
    print("the depth map -- no map choice can recover it.")
    b = rf_P(rows, k1, dtan, dpass, near, 1.0, 1.0)
    for P in (0.90, 1.00, 1.155, 1.30):
        r = rf_P(rows, k1, dtan, dpass, near, P, 1.0)
        print("   pool at {:.3f}x t_beta   RF {:+.4f}%".format(
            P, 100 * (b - r) / b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
