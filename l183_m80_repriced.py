"""L183 - M80's K, re-priced against a route-A-free wall and its MARGINAL quality.

M80 ships K=8 knob-cloud profiles at a recorded OOS NET of +1.786% / +1.909%.
Two things make that number unsafe now:

1. ITS WALL WAS PRICED WITH `m67e_rf48.py`'s kappa, which L159 records as a
   back-solve from one aggregate, wrong per case by up to 8x -- and under a
   premise ("48 cores are max-setter bound, so extra profiles are ~free") that
   L173 showed does not hold. L181 measures the wall directly, route A OFF, one
   box, 100 cases:

       nom80 (43 profiles)  109.96 s      cur (51 profiles)  130.21 s

2. ITS QUALITY WAS MEASURED WITHOUT THE L124 TWINS IN THE POOL. The OOS curve's
   K=0 total is 1.5559, an M77-era tree. CLAUDE.md's own M76 lesson is that the
   escape tier and tier-5 are SUBSTITUTES -- "the same points cannot be counted
   twice" -- and the L124 twins arrived after. Measured on the current tree with
   the twins present, M80's marginal quality is:

       nom80  weighted cost 1.262895   ->   cur 1.260247   =  +0.2098%

   against the +2.07% / +1.92% the standalone OOS curve credits it with.

So this prices M80 at its MARGINAL quality against its MEASURED wall.

⚠️ THE HONEST LIMITS, and they are serious:
  * +0.2098% is IN-SET (100 cases we have tuned on) and with the LP OFF.
    The OOS 240-case equivalent has not been run -- that is the measurement
    that should settle this, at ~35 min per sample.
  * The wall is route-A-OFF. The shipped package runs route A ON, and its
    behaviour on 48 real cores is unmeasured, so P is a range not a point.
  * In-set the DIRECTION is unambiguous (M80 costs 18.4% of pool wall for
    0.21% of quality); the magnitude is not.
"""
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M

# l181, route A off, exclusive box, LP off, 100 cases
W_M73, W_NOM80, W_CUR = 112.77, 109.96, 130.21
Q_NOM80, Q_CUR = 1.262895, 1.260247


def rf_at(rows, P, s_true=1.0):
    num = den = 0.0
    for r in rows:
        num += r["w"] * r["q"] * max(0.7, (P * r["t"] / (r["med"] * s_true)) ** 0.3)
        den += r["w"]
    return num / den


def main():
    rows = M.rows_new()
    print(__doc__)
    print("=" * 78)
    p_cur, p_nom80 = W_CUR / W_M73, W_NOM80 / W_M73
    dq = 100 * (Q_NOM80 - Q_CUR) / Q_NOM80
    print("pool multiplier vs M73:   with M80 {:.3f}x   without {:.3f}x"
          .format(p_cur, p_nom80))
    print("M80's marginal quality on the CURRENT pool: {:+.4f}%  (in set, LP off)"
          .format(dq))
    print("the standalone OOS curve credits it with:   +2.0729% / +1.9200%")

    print("\n{:>10}{:>14}{:>14}{:>12}"
          .format("medians", "RF with M80", "RF without", "M80 NET"))
    for s in (1.00, 0.90, 0.80):
        b = rf_at(rows, 1.0, s)
        a = rf_at(rows, p_cur, s)
        c = rf_at(rows, p_nom80, s)
        rf_cur = 100 * (b - a) / b
        rf_nom = 100 * (b - c) / b
        print("{:>9.2f}x{:>13.4f}%{:>13.4f}%{:>+11.4f}%"
              .format(s, rf_cur, rf_nom, dq + (rf_cur - rf_nom)))

    print("\nRF columns are vs the M73 pool at the same medians, so the")
    print("difference between them is what M80 alone costs. M80 NET is that")
    print("cost plus its MARGINAL quality, not its standalone quality.")
    print("\nIf instead M80's standalone OOS quality (+2.00% mean) were the")
    print("right number, the same wall would give:")
    for s in (1.00, 0.90):
        b = rf_at(rows, 1.0, s)
        a, c = rf_at(rows, p_cur, s), rf_at(rows, p_nom80, s)
        print("   medians x{:.2f}   NET {:+.4f}%"
              .format(s, 2.00 + (100 * (b - a) / b - 100 * (b - c) / b)))
    print("\nSo the verdict turns entirely on WHICH quality number is right,")
    print("and that is exactly what an OOS run of cur vs nom80 would settle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
