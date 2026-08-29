"""L173g - the calibration-free projection of the CURRENT package's graded score.

NO cross-box constant is used. `f` does not appear. The transfer is

    t_current_grader(n)  =  t_beta(n) * [ w_cur(n) / w_M73(n) ] * k

  t_beta(n)   M73's runtime AS MEASURED BY THE GRADER (beta_evaluation_results)
  w_cur(n)    current tree, this box, LP off, 100 cases      _l173p_cur.json
  w_M73(n)    git 7f38893's wrapper, same box, same flags    _l173p_m73.json
  k           carries the ratio from this box's 16 REAL cores to the grader's
              48, from the a + b/C fits in l173_cores{,_m73}.out

Because both walls are measured on one box under one configuration, every
machine-speed constant cancels. This replaces the first attempt, which divided
Windows walls by an f calibrated on WSL and printed a 4.27x that was an
artefact of mixing boxes -- WSL runs the SAME configuration 3.0-4.1x slower
than Windows here.

⚠️ The LP is added back through the same per-n ratio as the pool. The LP is
purely serial and the pool is not, so if the grader's per-core speed is lower
than this box's, that UNDERSTATES the LP's share. It is the smaller term.
"""
import json
import math
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
                out.setdefault(int(p[1]), {})[int(p[0])] = float(p[2].replace(",", ""))
            except ValueError:
                pass
    return out


def walls(fn):
    return {r["block_count"]: r["runtime_seconds"]
            for r in json.load(open(DIR / fn))["test_results"]}


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t) + 1


def main():
    print(__doc__)
    print("=" * 76)
    sc, sm = scan("l173_cores.out"), scan("l173_cores_m73.out")
    ks = []
    for case in sorted(set(sc) & set(sm)):
        if len(sc[case]) < 3 or len(sm[case]) < 3:
            continue
        ac, bc = fit(sc[case])
        am, bm = fit(sm[case])
        r16 = (ac + bc / 16) / (am + bm / 16)
        r48 = (ac + bc / 48) / (am + bm / 48)
        ks.append(r48 / r16)
        print("case {}   current {:.3f}+{:.1f}/C   M73 {:.3f}+{:.1f}/C"
              .format(case, ac, bc, am, bm))
        print("          ratio at 16 real cores {:.3f}x -> at 48 {:.3f}x"
              "   (k = {:.3f})".format(r16, r48, r48 / r16))
    if not ks:
        print("core scans incomplete")
        return 1
    k = sum(ks) / len(ks)
    print("\nk (16 real cores -> 48) = {:.3f}".format(k))

    try:
        wc, wm = walls("_l173p_cur.json"), walls("_l173p_m73.json")
    except FileNotFoundError:
        print("\nthe paired full runs are not finished yet "
              "(_l173p_cur.json / _l173p_m73.json)")
        return 1

    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    x090 = {int(x): v for x, v in
            json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    W = sum(r["w"] for r in rows)

    rat = sorted(wc[n] / wm[n] for n in wc if n in wm and wm[n] > 0)
    print("per-n pool ratio current/M73 on this box, 16 real cores:")
    print("   min {:.2f}  p10 {:.2f}  p50 {:.2f}  p90 {:.2f}  max {:.2f}"
          .format(rat[0], rat[10], rat[50], rat[90], rat[-1]))

    def t_of(r, dmap):
        n = r["n"]
        pool = r["t"] * (wc[n] / wm[n]) * k if n in wc and wm.get(n) else r["t"]
        lp = (dtan.get(near(n), 0.0)
              + (dmap.get(n, 1) - 1) * dpass.get(near(n), 0.0)) \
            * (r["t"] / wm[n] if wm.get(n) else 0.0)
        return pool + lp

    def score(ts, q):
        return sum(r["w"] * r["q"] * (1 - q / 100.0)
                   * max(0.7, (ts[r["i"]] / r["med"]) ** 0.3)
                   for r in rows) / W

    print("\n{:>28}{:>11}{:>9}{:>11}{:>10}{:>8}"
          .format("configuration", "our wall", "vs beta", "off floor",
                  "graded", "rank"))
    base = {r["i"]: r["t"] for r in rows}
    print("{:>28}{:>10.1f}s{:>8.2f}x{:>8}/100{:>10.5f}{:>8}"
          .format("M73 = what was graded", sum(base.values()), 1.0,
                  sum(1 for r in rows if base[r["i"]] > L.THR * r["med"]),
                  score(base, 0.0), rank_of(score(base, 0.0))))
    for lbl, dmap, q in (("current pool, k=1, q=0", {n: 1 for n in x090}, 0.0),
                         ("current + x0.90 map, q=0", x090, 0.0),
                         ("current + x0.90, q=+3.0%", x090, 3.0),
                         ("current + x0.90, q=+4.0%", x090, 4.0)):
        ts = {r["i"]: t_of(r, dmap) for r in rows}
        s = score(ts, q)
        print("{:>28}{:>10.1f}s{:>8.2f}x{:>8}/100{:>10.5f}{:>8}"
              .format(lbl, sum(ts.values()),
                      sum(ts.values()) / sum(base.values()),
                      sum(1 for r in rows if ts[r["i"]] > L.THR * r["med"]),
                      s, rank_of(s)))
    print("\nthresholds 2026-08-23:  " + "  ".join(
        "r{} {:.5f}".format(a, b) for a, b in RANKS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
