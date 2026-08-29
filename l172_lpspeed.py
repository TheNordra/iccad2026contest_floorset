"""L172h - L155 priced an LP speedup at 0.0000%. That verdict was table-bound.

L155's finding was "LP speed is worth nothing in itself, it is only the key to
k=2", and L157/L165 then bought the depth anyway because the 2026-08-19 medians
made it free. On the 2026-08-23 table it is NOT free: the deep map costs
-1.67% of RF, and the quality it was buying (+1.02% s1 / +0.88% s2) is real and
now unreachable. That makes LP speed worth something again -- possibly the
single largest remaining lever, because the quality has already been MEASURED.

For a speedup factor X applied to the tangent rows and every LP pass, this
rebuilds the affordability map at s_build = 0.90 with the cheaper passes and
prices the result. No new quality is assumed anywhere: every row reuses the
committed flat k=1/2/3 OOS arms, whose quality does not depend on how fast the
solver got there.

⚠️ The ceiling is the k<=3 arms we hold. A row cannot exceed "k=3 everywhere",
which is +1.13% / +0.94%; deeper arms have never been run.
"""
import json
from collections import Counter
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M
import l172_grid as G

DIR = Path(__file__).parent


def build_fast(rows, dtan, dpass, near, X, scale=0.90, f=M.F):
    out = {}
    for r in rows:
        n = r["n"]
        budget = L.THR * r["med"] * scale - r["t"] - dtan.get(near(n), 0.0) / (f * X)
        k = 1
        for kk in (2, 3):
            if (kk - 1) * dpass.get(near(n), 0.0) / (f * X) <= budget:
                k = kk
        out[n] = k
    return out


def rf_fast(rows, dmap, dtan, dpass, near, X, s_true, f=M.F):
    num = den = 0.0
    for r in rows:
        n = r["n"]
        k = dmap.get(n, 1)
        t = r["t"] + (dtan.get(near(n), 0.0)
                      + (k - 1) * dpass.get(near(n), 0.0)) / (f * X)
        num += r["w"] * r["q"] * max(0.7, (t / (r["med"] * s_true)) ** 0.3)
        den += r["w"]
    return num / den


def main():
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    k1 = {n: 1 for n in M.SHIPPED}
    k3 = {n: 3 for n in M.SHIPPED}
    print(__doc__)
    print("=" * 78)
    q3 = (M.quality(k3, "s1")[0], M.quality(k3, "s2")[0])
    print("ceiling, k=3 on every case: quality s1 {:+.4f}%  s2 {:+.4f}%"
          .format(*q3))
    print("what ships today (x0.90 map, X=1): quality s1 +0.4153%  s2 +0.4452%")
    print("\n{:>7}{:>20}{:>10}{:>10}{:>11}{:>11}"
          .format("LP X", "depths", "qual s1", "qual s2", "NET @x0.90",
                  "NET @x0.80"))
    for X in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 1e9):
        m = build_fast(rows, dtan, dpass, near, X)
        q1 = M.quality(m, "s1")[0]
        q2 = M.quality(m, "s2")[0]
        cells = ""
        for s in (0.90, 0.80):
            b = rf_fast(rows, k1, dtan, dpass, near, X, s)
            r = rf_fast(rows, m, dtan, dpass, near, X, s)
            cells += "{:>+10.4f}%".format((q1 + q2) / 2 + 100 * (b - r) / b)
        lbl = "free" if X > 1e8 else "{:.2f}x".format(X)
        print("{:>7}{:>20}{:>+9.4f}%{:>+9.4f}%{}"
              .format(lbl, str(dict(sorted(Counter(m.values()).items()))),
                      q1, q2, cells))
    print("\nThe X=free row is the most an infinitely fast LP could buy with")
    print("the arms we hold: it is the k=3-everywhere ceiling minus nothing.")
    print("Read the gap between X=1 and X=2 as what L155's shelved speedups")
    print("are worth NOW, and compare it against L155's own 0.0000%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
