"""L189 - the full cost/quality ladder, RF and quality separated.

Every rung is measured on ONE box, route A OFF, 100 cases, and transported to
the grader by the calibration-free ratio  t_beta(n) * w(n)/w_m73(n)  -- so `f`
never appears and no cross-box constant is used.

RF is applied PER CASE, not as a uniform multiplier. That matters: the earlier
uniform read of the pool growth (1.155x -> -0.69%) understates it, because the
growth is not spread evenly and exp(n/12) puts 71% of the weight above n=105.
The per-case column is the honest one.

⚠️ route A has never run on the grader. Both columns are shown: neutral (it
does nothing) and 0.68x (it delivers the -32.2% L110/L111 projected at 48 real
cores). Nothing here can narrow that.
"""
import json
import math

import l172_depthmap as M

RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333),
         (6, 0.9552271810705998), (7, 0.9638548902636931),
         (8, 0.9795094024339005), (9, 1.0006127694878413),
         (10, 1.0598302507627029)]


def rank_of(t):
    # beta's own score IS the r4 threshold; <= keeps it at 4 rather than 5
    return sum(1 for _, x in RANKS if x < t - 1e-12) + 1


def ld(f):
    d = json.load(open(f))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


def main():
    wm, qm = ld("_l181_m73.json")
    arms = [("M73 pool, no LP  (= beta)", "_l181_m73.json"),
            ("+ pool work, no LP", "_l181_cur.json"),
            ("+ LP at k=1", "_l189_k1.json"),
            ("+ LP, x0.90 depth map", "_l188_ours_full.json"),
            ("thin pool + LP k=1", "_l191_thinpool_lp.json")]
    rows = M.rows_new()
    W = sum(r["w"] for r in rows)
    wq = lambda q: (sum(math.exp(n / 12.0) * q[n] for n in q)       # noqa: E731
                    / sum(math.exp(n / 12.0) for n in q))
    beta = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in rows) / W
    print(__doc__)
    print("=" * 84)
    print("beta as actually graded on the 2026-08-23 medians: {:.5f}  (rank {})\n"
          .format(beta, rank_of(beta)))
    print("{:<28}{:>8}{:>9}{:>9}{:>9}{:>10}{:>6}"
          .format("configuration", "wall", "quality", "RF", "NET",
                  "graded", "rank"))
    for lbl, fn in arms:
        try:
            w, q = ld(fn)
        except FileNotFoundError:
            print("{:<28}  (not measured yet)".format(lbl))
            continue
        gain = 100 * (wq(qm) - wq(q)) / wq(qm)
        for ra, tag in ((1.00, ""), (0.68, "   route A 0.68x")):
            num = rfnum = wall = 0.0
            for r in rows:
                n = r["n"]
                t = r["t"] * (w[n] / wm[n]) * ra if wm.get(n) else r["t"]
                wall += t
                f = max(0.7, (t / r["med"]) ** 0.3)
                rfnum += r["w"] * r["q"] * f
                num += r["w"] * r["q"] * (1 - gain / 100.0) * f
            rf = 100 * (beta - rfnum / W) / beta
            tot = num / W
            if ra == 1.00:
                print("{:<28}{:>7.1f}s{:>+8.3f}%{:>+8.3f}%{:>+8.3f}%{:>10.5f}{:>6}"
                      .format(lbl, wall, gain, rf, gain + rf, tot, rank_of(tot)))
            else:
                print("{:<28}{:>7.1f}s{:>8}{:>+8.3f}%{:>+8.3f}%{:>10.5f}{:>6}"
                      .format(tag, wall, "", rf, gain + rf, tot, rank_of(tot)))
    print("\nquality is measured on the same runs (in set, 100 cases).")
    print("RF is vs beta, per case, on the 2026-08-23 medians.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
