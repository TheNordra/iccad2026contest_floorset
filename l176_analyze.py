"""L176 - what the pool regression since M73 actually costs, and what K should be.

Calibration-free. `f` never appears, because both walls are measured on ONE box
under ONE configuration and only their RATIO is used:

    t_current_grader(n)  =  t_beta(n) * [ w_cur(n) / w_m73(n) ] * k

  t_beta(n)   M73's runtime AS MEASURED BY THE GRADER (beta_evaluation_results)
  w_*(n)      _l176_{cur,m73,nom80}.json -- three full 100-case runs, LP OFF,
              exclusive box, ICCAD_ADAPTIVE_CORES=48
  k           carries the ratio from this box's 16 real cores to the grader's
              48, from the a + b/C core scans. k is a RATIO OF RATIOS, so the
              ~4% contention inflation in those scans (measured: the same
              4c/n=120 point reads 36.902s clean and 38.416s contended) largely
              cancels. The k=1.00 row below shows the sensitivity.

And the marginal wall of ONE M80 knob profile is (w_cur - w_nom80)/8 per case,
which turns K into a priceable knob against the OOS quality curve already
committed in results_M80_oos_s{1,2}_c48.json:

    K       0      1      2      3      4      5      6      7      8
    s1   0.000  0.299  1.219  1.453  1.626  1.712  1.775  1.878  2.073
    s2   0.000  0.101  1.096  1.230  1.465  1.583  1.599  1.671  1.920

K=2 is already 57-59% of what K=8 buys. That shape only matters if profiles
cost wall -- which is exactly what this measures.
"""
import json
import math
from collections import Counter
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M

DIR = Path(__file__).parent
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333)]
SCAN_CORES = (4, 8, 16)


def fit(pts):
    xs = [1.0 / c for c in pts]
    ys = [pts[c] for c in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def scan(fn):
    out = {}
    try:
        txt = (DIR / fn).read_text(errors="ignore").replace("\x00", "")
    except FileNotFoundError:
        return out
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


def walls(tag):
    return {r["block_count"]: r["runtime_seconds"]
            for r in json.load(open(DIR / "_l176_{}.json".format(tag)))["test_results"]}


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
    return sum(ks) / len(ks) if ks else 1.0


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t) + 1


def main():
    print(__doc__)
    print("=" * 78)
    wc, wm, wn = walls("cur"), walls("m73"), walls("nom80")
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    x090 = {int(x): v for x, v in
            json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    W = sum(r["w"] for r in rows)
    curve = {}
    for s in ("s1", "s2"):
        d = json.load(open(DIR / "results_M80_oos_{}_c48.json".format(s)))
        curve[s] = {e["K"]: e["quality"] for e in d["curve"]}

    print("scored wall on this box, LP off, 100 cases:")
    for tag, w in (("M73 (35 profiles)", wm), ("nom80 (43)", wn),
                   ("current (51)", wc)):
        print("   {:<22} {:8.2f}s".format(tag, sum(w.values())))
    rat = sorted(wc[n] / wm[n] for n in wc if wm.get(n))
    print("\nper-n ratio current/M73: min {:.2f}  p10 {:.2f}  p50 {:.2f}"
          "  p90 {:.2f}  max {:.2f}".format(rat[0], rat[10], rat[50],
                                            rat[90], rat[-1]))
    for lo, hi in ((21, 60), (61, 100), (101, 120)):
        seg = [wc[n] / wm[n] for n in wc if lo <= n <= hi and wm.get(n)]
        print("   n {:3d}-{:3d}   p50 {:.2f}x".format(
            lo, hi, sorted(seg)[len(seg) // 2]))

    kf = kfactor()
    d8 = {n: (wc[n] - wn[n]) / 8.0 for n in wc if n in wn}

    def t_of(n, tbeta, K, dmap, k):
        if n not in wn or not wm.get(n):
            return tbeta
        pool = (wn[n] + K * d8.get(n, 0.0)) / wm[n] * tbeta * k
        lp = (dtan.get(near(n), 0.0)
              + (dmap.get(n, 1) - 1) * dpass.get(near(n), 0.0)) * tbeta / wm[n]
        return pool + lp

    def score(K, dmap, k, q=0.0):
        num = 0.0
        for r in rows:
            t = t_of(r["n"], r["t"], K, dmap, k)
            num += r["w"] * r["q"] * (1 - q / 100.0) \
                * max(0.7, (t / r["med"]) ** 0.3)
        return num / W

    beta = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in rows) / W
    print("\n=== where we land on runtime alone (quality NOT credited) ===")
    print("M73, what was graded:  wall {:.1f}s   graded {:.5f}   rank {}"
          .format(sum(r["t"] for r in rows), beta, rank_of(beta)))
    for k, lbl in ((kf, "k={:.3f} (fitted)".format(kf)), (1.0, "k=1.00")):
        s = score(8, x090, k)
        wall = sum(t_of(r["n"], r["t"], 8, x090, k) for r in rows)
        print("current K=8, {:<18} wall {:.1f}s ({:.2f}x)  graded {:.5f}  rank {}"
              .format(lbl, wall, wall / sum(r["t"] for r in rows), s, rank_of(s)))

    print("\n=== M80's K, re-priced. dScore is vs the shipped K=8 ===")
    base = score(8, x090, kf)
    print("{:>4}{:>11}{:>12}{:>11}{:>11}{:>12}{:>7}"
          .format("K", "wall", "dRF vs K=8", "dQual s1", "dQual s2",
                  "dScore mean", "rank"))
    for K in range(0, 9):
        s = score(K, x090, kf)
        drf = 100 * (base - s) / base
        q1 = curve["s1"][K] - curve["s1"][8]
        q2 = curve["s2"][K] - curve["s2"][8]
        d = (q1 + q2) / 2 + drf
        wall = sum(t_of(r["n"], r["t"], K, x090, kf) for r in rows)
        print("{:>4}{:>10.1f}s{:>+11.4f}%{:>+10.4f}%{:>+10.4f}%{:>+11.4f}%{:>7}"
              .format(K, wall, drf, q1, q2, d, rank_of(s)))
    print("\ndQual is OOS on two disjoint 240-case samples. Positive dScore")
    print("means better than what ships. The quality of everything OTHER than")
    print("M80 is not in these numbers -- it is identical in every row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
