"""L188 - what does the merged package actually score, and what rank is that?

CALIBRATION-FREE. `f` never appears. Both walls are measured on ONE box under
ONE configuration and only their RATIO is used, then applied to the grader's
own measurement of M73:

    t_ours_grader(n)  =  t_beta(n)  *  w_ours(n) / w_m73(n)

  t_beta(n)  the GRADER's per-case runtime for M73 (beta_evaluation_results)
  w_ours(n)  _l188_ours_full.json  -- the full shipped config, 100 cases
  w_m73(n)   _l181_m73.json        -- git 7f38893's wrapper, same box, same day

Both arms run ICCAD_ROUTE_A=0. Route A is verified result-neutral (single cases
bit-identical on/off; L177 det1 vs det2 matched 100/100 on cost AND positions
with it live), so switching it off changes wall only -- and it MUST be off in
both, because on this box's 16 physical cores it costs 2.9x while on the
grader's 48 it was projected to SAVE 32.2%. Leaving it on would measure this
box's oversubscription, not the package.

QUALITY comes from the same two runs, so it is measured, not assumed.

⚠️ WHAT THIS CANNOT KNOW. Route A has never run on the grader -- beta was M73,
which does not have it. If route A delivers anything like its projected -32.2%
at 48 real cores, our wall is LOWER than this and the score BETTER. If it
misbehaves, worse. That is the one term with no measurement behind it, and it
is why the answer below is a range with a named centre rather than a point.
"""
import json
import math
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M

DIR = Path(__file__).parent
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333),
         (6, 0.9552271810705998), (7, 0.9638548902636931),
         (8, 0.9795094024339005), (9, 1.0006127694878413),
         (10, 1.0598302507627029)]


def load(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t) + 1


def main():
    w_o, q_o = load("_l188_ours_full.json")
    w_m, q_m = load("_l181_m73.json")
    rows = M.rows_new()
    W = sum(r["w"] for r in rows)
    print(__doc__)
    print("=" * 78)

    wq = lambda q: (sum(math.exp(n / 12.0) * q[n] for n in q)      # noqa: E731
                    / sum(math.exp(n / 12.0) for n in q))
    qm, qo = wq(q_m), wq(q_o)
    gain = 100 * (qm - qo) / qm
    print("in-set quality, same box, same day:")
    print("   M73 (what beta shipped)  {:.6f}".format(qm))
    print("   merged package           {:.6f}     +{:.4f}%".format(qo, gain))
    print("wall, route A off, 100 cases:")
    print("   M73 {:.2f}s     ours {:.2f}s     {:.3f}x"
          .format(sum(w_m.values()), sum(w_o.values()),
                  sum(w_o.values()) / sum(w_m.values())))

    beta = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in rows) / W
    print("\nbeta as actually graded on the 2026-08-23 medians: {:.5f} (rank {})"
          .format(beta, rank_of(beta)))

    print("\n{:>26}{:>10}{:>11}{:>10}{:>7}"
          .format("route A on the grader", "our wall", "graded", "vs beta", "rank"))
    for lbl, ra in (("as measured here (1.00x)", 1.00),
                    ("half its claim (0.84x)", 0.84),
                    ("its full claim (0.68x)", 0.68),
                    ("mildly harmful (1.15x)", 1.15)):
        num = 0.0
        wall = 0.0
        for r in rows:
            n = r["n"]
            if n not in w_m or w_m[n] <= 0:
                t = r["t"]
            else:
                t = r["t"] * (w_o[n] / w_m[n]) * ra
            wall += t
            num += r["w"] * r["q"] * (1 - gain / 100.0) \
                * max(0.7, (t / r["med"]) ** 0.3)
        tot = num / W
        print("{:>26}{:>9.1f}s{:>11.5f}{:>+9.4f}%{:>7}"
              .format(lbl, wall, tot, 100 * (beta - tot) / beta, rank_of(tot)))

    print("\nthresholds 2026-08-23:")
    print("   " + "   ".join("r{} {:.5f}".format(a, b) for a, b in RANKS[:5]))
    print("\nThe quality gain is applied uniformly, which is what every")
    print("projection in this ledger does; per-case it is not uniform, so treat")
    print("the rank as the claim and the fifth decimal as decoration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
