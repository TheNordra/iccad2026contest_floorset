"""L172i - the affordability rule has a blind spot: cases already past the floor.

`_depth_ok` asks "can this case absorb dt and STAY on the RF floor". For a case
whose own runtime already exceeds 0.3046*median the answer is no, forever, so
it gets k=1 -- but the floor is already lost there and the marginal cost of a
pass is just the derivative of R^0.3, which is small. Meanwhile those cases are
not rare: at s_build 0.90 there are 30 of them, and they hold a large share of
the quality the deep map was buying.

The rule under test is FIXED, not fitted -- "cases the affordability rule
cannot serve get depth D anyway" -- so measuring it on both disjoint OOS
samples is a test, not a fit.
"""
import json
from collections import Counter
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M
import l172_grid as G

DIR = Path(__file__).parent


def main():
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    k1 = {n: 1 for n in M.SHIPPED}
    cap = {int(k): v for k, v in
           json.load(open(DIR / "l172_depthmap_x090.json")).items()}

    # which n the affordability rule could not serve at s_build 0.90
    starved = []
    for r in rows:
        n = r["n"]
        if L.THR * r["med"] * 0.90 - r["t"] - dtan.get(near(n), 0.0) / M.F < 0:
            starved.append(n)
    print(__doc__)
    print("=" * 76)
    print("cases the rule cannot serve at s_build 0.90: {}".format(len(starved)))
    print("   n = {}".format(sorted(starved)))
    print("   they carry {:.1f}% of the beta corpus weight"
          .format(100 * sum(r["w"] for r in rows if r["n"] in starved)
                  / sum(r["w"] for r in rows)))

    print("\n{:>28}{:>19}{:>10}{:>10}{:>11}{:>11}"
          .format("variant", "depths", "qual s1", "qual s2",
                  "NET @x0.90", "NET @x0.80"))
    variants = [("x0.90 map (shipped now)", cap)]
    for D in (2, 3):
        m = dict(cap)
        for n in starved:
            m[n] = D
        variants.append(("starved cases -> k={}".format(D), m))
    m = dict(cap)
    for n in starved:
        if n > 100:
            m[n] = 3
    variants.append(("starved & n>100 -> k=3", m))
    variants.append(("k=3 everywhere", {n: 3 for n in cap}))

    for lbl, m in variants:
        q1, _, w1 = M.quality(m, "s1")[0], 0, M.quality(m, "s1")[2]
        q2, w2 = M.quality(m, "s2")[0], M.quality(m, "s2")[2]
        cells = ""
        for s in (0.90, 0.80):
            b = G.rf_on(rows, k1, dtan, dpass, near, s)
            r = G.rf_on(rows, m, dtan, dpass, near, s)
            cells += "{:>+10.4f}%".format((q1 + q2) / 2 + 100 * (b - r) / b)
        print("{:>28}{:>19}{:>+9.4f}%{:>+9.4f}%{}"
              .format(lbl, str(dict(sorted(Counter(m.values()).items()))),
                      q1, q2, cells))
    print("\nBoth OOS columns are disjoint 240-case samples and neither was")
    print("used to choose the rule, so agreement between them is the evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
