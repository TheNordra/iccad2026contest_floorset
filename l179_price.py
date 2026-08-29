"""L179b - price every pool configuration as "this pool + our LP", on the
2026-08-23 medians. Calibration-free: f cancels in the ratio.

    t(n)  =  t_beta(n) * w_arm(n)/w_m73(n) * k  +  LP(n) * t_beta(n)/w_m73(n)
    raw   =  local_arm * 1.0194 / 1.0550          <- LP multiplier, see below
    graded=  sum_i w_i * q_i * (raw/raw_beta) * max(0.7, (t_i/M_i)^0.3) / sum w

  t_beta(n)   M73's runtime AS MEASURED BY THE GRADER
  w_arm(n)    100-case runs on an exclusive box, LP OFF, ICCAD_ADAPTIVE_CORES=48
  k = 0.935   16 real cores -> 48, from the a + b/C fits
  1.0194      local -> hidden, from beta itself (1.3206649 / 1.2955478).
              One factor, identical in every row, so it cannot reorder them.

⚠️ THE ONE UNMEASURED STEP. Each arm's quality is measured with the LP OFF. The
LP multiplies in-set quality by 1.0550 ON THE CURRENT POOL (1.260247 -> 1.1945)
and that multiplier is assumed to carry to the other pools. The LP legalises
shapes the pool produced, so it is plausible, but it is an assumption -- the
winning row needs one real LP-on run before anything ships. If the multiplier is
smaller on a thinner pool, every row except the current one gets worse
together, so the ORDER is more robust than the absolute values.
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
HIDDEN = 1.3206649447461247 / 1.295547821428148
LP_MULT = 1.260247 / 1.1945          # 1.0550
ARMS = [("m73", "_l176_m73.json", "M73 pool (= the beta package)"),
        ("cur", "_l176_cur.json", "current: M80+twins+L137 (SHIPS)"),
        ("nom80", "_l176_nom80.json", "  - M80 tier"),
        ("notwin", "_l179_notwin.json", "  - M80 - L124 twins"),
        ("nohint", "_l179_nohint.json", "  - M80 - twins - L137 hint"),
        ("nom71", "_l179_nom71.json", "  - M80 - twins - L137 - M71")]


def load(fn):
    p = DIR / fn
    if not p.exists():
        return None
    d = json.load(open(p))["test_results"]
    w = lambda r: math.exp(r["block_count"] / 12.0)               # noqa: E731
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            sum(w(r) * r["cost"] for r in d) / sum(w(r) for r in d),
            sum(r["runtime_seconds"] for r in d),
            sum(1 for r in d if r.get("feasible", True)))


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t) + 1


def main():
    print(__doc__)
    print("=" * 80)
    wm = load("_l176_m73.json")[0]
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    x090 = {int(x): v for x, v in
            json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    W = sum(r["w"] for r in rows)
    beta_g = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
                 for r in rows) / W
    raw_beta = 1.295547821428148 * HIDDEN

    print("{:<34}{:>9}{:>9}{:>10}{:>10}{:>9}{:>6}"
          .format("pool configuration", "wall_box", "local", "raw_h",
                  "our wall", "graded", "rank"))
    print("{:<34}{:>9}{:>9.4f}{:>10.4f}{:>9.1f}s{:>9.5f}{:>6}"
          .format("BETA, as actually graded", "-", 1.295548, raw_beta,
                  sum(r["t"] for r in rows), beta_g, rank_of(beta_g)))
    for tag, fn, lbl in ARMS:
        got = load(fn)
        if not got:
            print("{:<34}{:>9}".format(lbl, "pending"))
            continue
        w_arm, loc_lpoff, box, fe = got
        loc = loc_lpoff / LP_MULT
        scale = loc * HIDDEN / raw_beta
        num = wall = 0.0
        for r in rows:
            n = r["n"]
            if not wm.get(n) or n not in w_arm:
                t = r["t"]
            else:
                t = w_arm[n] / wm[n] * r["t"] * K_CORES \
                    + (dtan.get(near(n), 0.0)
                       + (x090.get(n, 1) - 1) * dpass.get(near(n), 0.0)) \
                    * r["t"] / wm[n]
            wall += t
            num += r["w"] * r["q"] * scale * max(0.7, (t / r["med"]) ** 0.3)
        g = num / W
        print("{:<34}{:>8.1f}s{:>9.4f}{:>10.4f}{:>9.1f}s{:>9.5f}{:>6}{}"
              .format(lbl, box, loc, loc * HIDDEN, wall, g, rank_of(g),
                      "" if fe == 100 else "  !! feasible {}/100".format(fe)))
    print("\nthresholds 2026-08-23:  " + "  ".join(
        "r{} {:.5f}".format(a, b) for a, b in RANKS))
    print("\n'local' is with the LP credited at x{:.4f}; 'wall_box' is the raw"
          .format(LP_MULT))
    print("100-case scored wall on this box with the LP OFF, which is what was")
    print("measured. 'our wall' is the projected grader wall WITH the LP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
