"""L178 - can the 1.19 configuration actually be shipped? Quality AND runtime.

1.1945 is a LOCAL score with RuntimeFactor forced to 1.0. The graded score is
quality x RF, and L176 measured the RF side as 3.84x M73's wall. This puts the
two halves in one place.

QUALITY, all on this box, 48 cores, in-set 100:
    M73 proxy, LP off      1.281457     _l176_m73.json
    current pool, LP off   1.260247     _l176_cur.json      pool side  +1.66%
    current + LP           1.1945       l171_det1.log       LP side    +5.20%
                                                            together   +6.79%

The M73 proxy runs TODAY's constructive.exe, so it already contains L136's
FRAME_EPS fix. Against the ACTUAL beta package at 48 cores (1.295547821428148,
CLAUDE.md) the total quality gain is +7.80%.

TRANSFER TO THE HIDDEN SET. The beta package scored local 1.295548 at 48c and
hidden raw 1.3206649447461247, i.e. the hidden set is 1.0194x harder. Applying
that one factor is the only extrapolation here, and it is the same factor for
every row, so it cannot change the ORDER of the rows.

RUNTIME. t(n) = t_beta(n) * w_arm(n)/w_m73(n) * k + LP(n) * t_beta(n)/w_m73(n),
k = 0.935 from the a + b/C core fits. f cancels; see l176_analyze.py.
"""
import json
import math
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M

DIR = Path(__file__).parent
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333)]
K_CORES = 0.935
HIDDEN = 1.3206649447461247 / 1.295547821428148     # 1.0194
BETA_LOCAL_48 = 1.295547821428148


def walls(tag):
    return {r["block_count"]: r["runtime_seconds"] for r in
            json.load(open(DIR / "_l176_{}.json".format(tag)))["test_results"]}


def local(tag):
    d = json.load(open(DIR / "_l176_{}.json".format(tag)))["test_results"]
    w = lambda r: math.exp(r["block_count"] / 12.0)               # noqa: E731
    return sum(w(r) * r["cost"] for r in d) / sum(w(r) for r in d)


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t) + 1


def main():
    print(__doc__)
    print("=" * 78)
    wm, wc, wn = walls("m73"), walls("cur"), walls("nom80")
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    x090 = {int(x): v for x, v in
            json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    W = sum(r["w"] for r in rows)
    LP_LOCAL = 1.1945                      # current pool + LP, l171_det1
    pool_only = local("cur")               # 1.260247
    lp_gain = pool_only / LP_LOCAL         # what the LP multiplies quality by

    def graded(w_arm, local_score, lp_on):
        """local_score is this arm's in-set quality at RF=1."""
        raw_h = local_score * HIDDEN
        scale = raw_h / (BETA_LOCAL_48 * HIDDEN)   # vs beta's own hidden raw
        num = 0.0
        for r in rows:
            n = r["n"]
            if not wm.get(n):
                t = r["t"]
            else:
                t = w_arm[n] / wm[n] * r["t"] * K_CORES
                if lp_on:
                    t += (dtan.get(near(n), 0.0)
                          + (x090.get(n, 1) - 1) * dpass.get(near(n), 0.0)) \
                        * r["t"] / wm[n]
            num += r["w"] * r["q"] * scale * max(0.7, (t / r["med"]) ** 0.3)
        return num / W, sum(w_arm[n] / wm[n] * r["t"] * K_CORES
                            for r in rows if wm.get(r["n"])
                            for n in [r["n"]])

    print("the LP multiplies in-set quality by {:.4f} (1.260247 -> 1.1945)\n"
          .format(lp_gain))
    print("{:<34}{:>10}{:>10}{:>11}{:>9}{:>7}"
          .format("configuration", "local", "hidden raw", "our wall",
                  "graded", "rank"))

    beta_g = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
                 for r in rows) / W
    print("{:<34}{:>10.4f}{:>10.4f}{:>10.1f}s{:>9.5f}{:>7}"
          .format("M73 = the beta package, GRADED", BETA_LOCAL_48,
                  BETA_LOCAL_48 * HIDDEN, sum(r["t"] for r in rows),
                  beta_g, rank_of(beta_g)))

    arms = [
        ("current tree, K=8 + LP  (SHIPS)", wc, LP_LOCAL, True),
        ("current tree, K=8, LP off", wc, pool_only, False),
        ("M80 off (K=0) + LP", wn, local("nom80") / lp_gain, True),
        ("M73 pool + our LP  (hypothetical)", wm, local("m73") / lp_gain, True),
    ]
    for lbl, w_arm, loc, lp_on in arms:
        g, _ = graded(w_arm, loc, lp_on)
        wall = 0.0
        for r in rows:
            n = r["n"]
            if not wm.get(n):
                wall += r["t"]
                continue
            t = w_arm[n] / wm[n] * r["t"] * K_CORES
            if lp_on:
                t += (dtan.get(near(n), 0.0)
                      + (x090.get(n, 1) - 1) * dpass.get(near(n), 0.0)) \
                    * r["t"] / wm[n]
            wall += t
        print("{:<34}{:>10.4f}{:>10.4f}{:>10.1f}s{:>9.5f}{:>7}"
              .format(lbl, loc, loc * HIDDEN, wall, g, rank_of(g)))

    print("\nthresholds 2026-08-23:  " + "  ".join(
        "r{} {:.5f}".format(a, b) for a, b in RANKS))
    print("\nThe last row is the prize and it is NOT a measured arm: it assumes")
    print("the LP's quality multiplier carries onto M73's pool unchanged. The")
    print("LP legalises shapes the pool produced, so that is plausible but")
    print("unmeasured -- it needs one real run to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
