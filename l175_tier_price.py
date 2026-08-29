"""L175 - re-price the M80 knob-cloud tier's K against a MEASURED wall.

M80 ships K=8 knob-cloud profiles. K was chosen out of sample on quality, and
its wall was priced at drf +0.19% (s1) / +0.37% (s2) using `m67e_rf48.py`'s
kappa -- which L159 records as a back-solve from one aggregate, wrong per case
by up to 8x -- under the premise that at 48 cores the pool is max-setter bound
so extra profiles are ~free. L173 measures the wall as LINEAR in profile count,
so that premise is gone and K has never been priced against a real cost.

Everything here is a DELTA against the shipped K=8, so no absolute calibration
of the stack's quality is needed and none is assumed:

    dScore(K) = [quality(K) - quality(8)]  +  [RF(K) - RF(8)]

  quality(K)  MEASURED OOS on two disjoint 240-case samples,
              results_M80_oos_s{1,2}_c48.json `curve`
  RF(K)       from the per-case marginal wall of one M80 profile,
              (w_cur - w_nom80)/8, both full 100-case runs on ONE box, then
              carried to the grader by  t_beta(n) * w(n)/w_M73(n) * k
              -- f cancels, see l173_final.py.

⚠️ The quality curve was measured on the M77-era tree (its K=0 total is
1.5559), not today's. Its SHAPE is what is used -- the marginal value of the
k-th profile -- and the ledger records that shape as stable across samples
(both curves put the elbow at 8). If the shape has moved, this is wrong, and
the honest fix is re-running the K sweep, not re-weighting it here.
"""
import json
import math
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M

DIR = Path(__file__).parent
SCAN_CORES = (4, 8, 16)


def fit(pts):
    xs = [1.0 / c for c in pts]
    ys = [pts[c] for c in pts]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def scan(fn):
    out = {}
    txt = (DIR / fn).read_text(errors="ignore").replace("\x00", "")
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 4 and p[0].isdigit() and p[1].isdigit() \
                and int(p[0]) in SCAN_CORES:
            try:
                out.setdefault(int(p[1]), {})[int(p[0])] = \
                    float(p[2].replace(",", ""))
            except ValueError:
                pass
    return out


def walls(fn):
    return {r["block_count"]: r["runtime_seconds"]
            for r in json.load(open(DIR / fn))["test_results"]}


def kfactor():
    sc, sm = scan("l173_cores.out"), scan("l173_cores_m73.out")
    ks = []
    for case in sorted(set(sc) & set(sm)):
        if len(sc[case]) < 3 or len(sm[case]) < 3:
            continue
        ac, bc = fit(sc[case])
        am, bm = fit(sm[case])
        ks.append(((ac + bc / 48) / (am + bm / 48))
                  / ((ac + bc / 16) / (am + bm / 16)))
    return sum(ks) / len(ks)


def main():
    print(__doc__)
    print("=" * 76)
    try:
        wc, wm, wn = (walls("_l173p_cur.json"), walls("_l173p_m73.json"),
                      walls("_l173p_nom80.json"))
    except FileNotFoundError as e:
        print("missing arm: {}".format(e))
        return 1
    k = kfactor()
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    x090 = {int(x): v for x, v in
            json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    W = sum(r["w"] for r in rows)
    curve = {}
    for s in ("s1", "s2"):
        d = json.load(open(DIR / "results_M80_oos_{}_c48.json".format(s)))
        curve[s] = {e["K"]: e["quality"] for e in d["curve"]}

    print("k (16 real cores -> 48) = {:.3f}".format(k))
    print("marginal wall of ONE M80 profile, (w_cur - w_nom80)/8, this box:")
    d8 = {n: (wc[n] - wn[n]) / 8.0 for n in wc if n in wn}
    v = sorted(d8.values())
    print("   p10 {:.3f}s  p50 {:.3f}s  p90 {:.3f}s  (negative = noise)"
          .format(v[10], v[50], v[90]))

    def total(K, dmap):
        num = 0.0
        for r in rows:
            n = r["n"]
            if n not in wn or n not in wm or wm[n] <= 0:
                t = r["t"]
            else:
                pool = (wn[n] + K * d8.get(n, 0.0)) / wm[n] * r["t"] * k
                lp = (dtan.get(near(n), 0.0)
                      + (dmap.get(n, 1) - 1) * dpass.get(near(n), 0.0)) \
                    * r["t"] / wm[n]
                t = pool + lp
            num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        return num / W

    base = total(8, x090)
    print("\n{:>4}{:>12}{:>11}{:>11}{:>11}{:>11}"
          .format("K", "graded", "dRF vs K=8", "dQual s1", "dQual s2",
                  "dScore mean"))
    for K in range(0, 9):
        t = total(K, x090)
        drf = 100 * (base - t) / base
        q1 = curve["s1"][K] - curve["s1"][8]
        q2 = curve["s2"][K] - curve["s2"][8]
        d = (q1 + q2) / 2 + drf
        mark = "   <= ships" if K == 8 else ("   <== best" if False else "")
        print("{:>4}{:>12.6f}{:>+10.4f}%{:>+10.4f}%{:>+10.4f}%{:>+10.4f}%{}"
              .format(K, t, drf, q1, q2, d, mark))
    print("\ndScore is relative to the shipped K=8; positive means better than")
    print("what ships. Quality deltas are OOS on two disjoint 240-case samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
