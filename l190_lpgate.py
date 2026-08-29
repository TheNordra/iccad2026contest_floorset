"""L190 - should the shape LP run AT ALL on this case? A per-case k in {0,1} gate.

THE PROBLEM. The LP at k=1 costs 20.8 grader-seconds against a free budget of
14.72 s on the 2026-08-23 medians, so it is unaffordable ON AVERAGE. But the
budget is not a pool: per case the slack runs from 0.84x to 2.96x. L157 already
built exactly this logic to choose passes 2 and 3; it never asked the prior
question, because until L188 measured it nobody had counted the FIRST pass's
seconds -- L172 flagged them missing and then everything was built on top anyway.

THE RULE, and there is no oracle in it:

    run the LP on case n  iff   t_pool(n) + dt_lp(n)  <=  0.3046 * M(n) * s

  t_pool(n)  our pool wall, transported to the grader by the calibration-free
             ratio  t_beta(n) * w_lpoff(n)/w_m73(n)   -- `f` never appears
  dt_lp(n)   the LP's own seconds, likewise transported: w_k1 - w_lpoff
  M(n)       the PUBLISHED per-case median, 2026-08-23
  s          a safety scale on the medians, since the final round's table is
             not this one. Everything is shown across s.

Route A multiplies the POOL only -- the LP is single-threaded scipy on the main
thread, so a frame queue does not touch it. Applying route A to the whole wall
(as l189_ladder did) slightly flatters the LP; this does not.

QUALITY is mixed per case from two measured runs, which is exact: _l181_cur
(LP off) and _l189_k1 (LP at k=1) are the same tree, same box, same day, so
picking one or the other per case is precisely what the gate would produce.
"""
import json
import math
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
THR = 0.7 ** (1 / 0.3)
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333),
         (6, 0.9552271810705998), (7, 0.9638548902636931),
         (8, 0.9795094024339005), (9, 1.0006127694878413),
         (10, 1.0598302507627029)]


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t - 1e-9) + 1


def ld(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


def main():
    wm, qm73 = ld("_l181_m73.json")
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "thin":
        # the THIN pool (M80/twins/L137 off). Its own wall is ~M73's, so the
        # whole 14.72s budget is still available for a selective LP -- on the
        # full pool the pool work had already eaten it.
        wo, qo = ld("_l181_nohint.json")      # thin, LP off
        wk, qk = ld("_l191_thinpool_lp.json") # thin, LP at k=1
        print(">>> THIN POOL variant <<<")
    else:
        wo, qo = ld("_l181_cur.json")        # LP off
        wk, qk = ld("_l189_k1.json")         # LP at k=1
    rows = M.rows_new()
    W = sum(r["w"] for r in rows)
    beta = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in rows) / W
    print(__doc__)
    print("=" * 80)
    print("beta as graded: {:.5f} (rank {})".format(beta, rank_of(beta)))

    # per-case, in grader seconds, calibration-free
    pool, dtlp = {}, {}
    for r in rows:
        n = r["n"]
        if not wm.get(n):
            continue
        k = r["t"] / wm[n]
        pool[n] = wo[n] * k
        dtlp[n] = max(0.0, (wk[n] - wo[n]) * k)
    print("\nLP cost per case, grader seconds:  total {:.2f}s   "
          "p10 {:.3f}  p50 {:.3f}  p90 {:.3f}"
          .format(sum(dtlp.values()),
                  sorted(dtlp.values())[10], sorted(dtlp.values())[50],
                  sorted(dtlp.values())[90]))

    def score(gate, ra=1.0):
        """Quality is transported PER CASE as the ratio to M73 on the same
        block count, then applied to the grader's own beta cost for that case.
        The earlier version multiplied our in-set costs by a global rescale,
        which mixes two corpora and does not reconcile with l189_ladder."""
        num = wall = den = 0.0
        for r in rows:
            n = r["n"]
            if n not in pool or not qm73.get(n):
                continue
            on = gate.get(n, 0)
            t = pool[n] * ra + (dtlp[n] if on else 0.0)
            wall += t
            qratio = (qk if on else qo)[n] / qm73[n]
            num += r["w"] * r["q"] * qratio * max(0.7, (t / r["med"]) ** 0.3)
            den += r["w"]
        return num / den, wall

    def report(lbl, gate, ra=1.0):
        g, wall = score(gate, ra)
        n_on = sum(gate.values())
        print("{:<30}{:>4}{:>9.1f}s{:>11.5f}{:>+9.3f}%{:>6}"
              .format(lbl, n_on, wall, g, 100 * (beta - g) / beta, rank_of(g)))

    ALL_OFF = {n: 0 for n in pool}
    ALL_ON = {n: 1 for n in pool}
    print("\n{:<30}{:>4}{:>10}{:>11}{:>10}{:>6}"
          .format("configuration", "LP", "wall", "graded", "vs beta", "rank"))
    for ra, tag in ((1.0, ""), (0.68, "  [route A 0.68x]")):
        if tag:
            print("  --- route A 0.68x ---")
        report("LP off everywhere" + tag, ALL_OFF, ra)
        report("LP on everywhere (k=1)" + tag, ALL_ON, ra)
        for s in (1.00, 0.90, 0.80):
            gate = {n: 1 if pool[n] + dtlp[n] <= THR * r_med * s else 0
                    for n, r_med in ((r["n"], r["med"]) for r in rows)
                    if n in pool}
            report("GATED, medians x{:.2f}{}".format(s, tag), gate, ra)
    print("\nLP column = how many of the 100 cases actually run the LP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
