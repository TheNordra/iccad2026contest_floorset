"""L172b - the decision grid: which depth map survives being wrong about the
final medians?

We do not know the FINAL round's per-case medians. We know two published
tables for the same 100 hidden cases, four days apart, and every one of the
100 entries FELL between them (p50 x0.742). So the question a map has to
answer is not "what is optimal on the newest table" but "what does it cost me
if the table I built on is off by s".

BUILD on medians = new_table * s_build      (what we assume when we ship)
PRICE on medians = new_table * s_true       (what the graders actually use)

The shipped `_L157_DEPTH` was built on the OLD table, which is new_table
divided by 0.742, i.e. s_build ~ 1.35.

CAVEAT, and it runs the same way for every row: t_final is modelled as
t_beta + (dt_tangent + (k-1)*dt_pass)/f. t_beta is the beta package, which
had NO shape LP at all, so the FIRST LP pass's seconds are missing from every
number here. Including them would push every case closer to the RF edge and
make the deep maps look worse still, so this table is conservative in the
direction of the conclusion, not against it.
"""
import math
from collections import Counter

import l146_rf_price as L
import l172_depthmap as M


def rf_on(rows, dmap, dtan, dpass, near, s_true, f=M.F):
    num = den = 0.0
    for r in rows:
        n = r["n"]
        k = dmap.get(n, 1)
        t = r["t"] + (dtan.get(near(n), 0.0)
                      + (k - 1) * dpass.get(near(n), 0.0)) / f
        med = r["med"] * s_true
        num += r["w"] * r["q"] * max(0.7, (t / med) ** 0.3)
        den += r["w"]
    return num / den


def main():
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    k1 = {n: 1 for n in M.SHIPPED}
    S_TRUE = (1.30, 1.15, 1.00, 0.90, 0.80)

    maps = [("k=1 (kill switch)", k1), ("SHIPPED _L157_DEPTH", M.SHIPPED)]
    for sb in (1.15, 1.00, 0.90, 0.80):
        maps.append(("build on x%.2f" % sb,
                     M.build(rows, dtan, dpass, near, scale=sb)))

    q = {}
    for lbl, m in maps:
        q[lbl] = (M.quality(m, "s1")[0], M.quality(m, "s2")[0])

    print(__doc__)
    print("=" * 78)
    print("quality is arm-mixed from the committed flat k=1/2/3 OOS arms")
    print("(240 disjoint cases each; mixing reproduces a real gated run "
          "100/100 on cost AND positions)\n")
    hdr = "{:<21}{:>18}{:>9}{:>9}".format("map", "depths", "qual s1", "qual s2")
    hdr += "".join("{:>10}".format("x%.2f" % s) for s in S_TRUE)
    print(hdr)
    print("{:<21}{:>18}{:>9}{:>9}".format("", "", "", "")
          + "".join("{:>10}".format("NET") for _ in S_TRUE))
    print("-" * len(hdr))
    for lbl, m in maps:
        base = {s: rf_on(rows, k1, dtan, dpass, near, s) for s in S_TRUE}
        cells = ""
        for s in S_TRUE:
            rf = rf_on(rows, m, dtan, dpass, near, s)
            drf = 100 * (base[s] - rf) / base[s]
            net = (q[lbl][0] + q[lbl][1]) / 2 + drf
            cells += "{:>+9.3f}%".format(net)
        print("{:<21}{:>18}{:>+8.3f}%{:>+8.3f}%{}"
              .format(lbl, str(dict(sorted(Counter(m.values()).items()))),
                      q[lbl][0], q[lbl][1], cells))
    print("\nNET = mean(OOS s1, s2 quality) + RF delta vs the k=1 anchor,")
    print("both in % of the weighted total, positive = better.")
    print("\ns_true = 1.00 is the newest published table; the shipped map was")
    print("built at s_build ~ 1.35 and every column below x1.15 punishes it.")

    print("\n=== worst case over the plausible range x0.80..x1.15 ===")
    for lbl, m in maps:
        worst = min(
            (q[lbl][0] + q[lbl][1]) / 2
            + 100 * (rf_on(rows, k1, dtan, dpass, near, s)
                     - rf_on(rows, m, dtan, dpass, near, s))
            / rf_on(rows, k1, dtan, dpass, near, s)
            for s in (1.15, 1.00, 0.90, 0.80))
        print("  {:<21} worst NET {:>+8.3f}%".format(lbl, worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
