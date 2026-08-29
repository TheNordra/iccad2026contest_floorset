"""L196 - is the affordability gate the RIGHT gate? Sweep it, cross-validated.

The shipped-candidate gate runs the LP iff it stays under the RF floor. On the
full pool that fires on 30 block counts of which only TWO are n>100, while n>100
carries 71% of the weight -- so it captures 12% of the LP's OOS quality (L194).
The rule is right for RF and wrong for quality.

This sweeps a one-parameter family and picks by CROSS-VALIDATION, never by the
sample it is scored on:

    gate_s(n) = 1  iff  t_pool(n) + dt_lp(n) <= 0.3046 * M(n) * s

  s < 1  stricter than the floor    s > 1  deliberately overspends
  s -> inf is "LP everywhere", s -> 0 is "LP off"

FIT ON s1 -> TEST ON s2, and FIT ON s2 -> TEST ON s1, reported separately. A
value of s that only works in the direction it was fitted is noise -- the rule
this ledger keeps re-learning (L127 tally fitting; today the twins, L171 and
the thin pool all moved against their in-set reading).

Everything is arm-mixed from arms already measured, so no new runs and no
oracle: the gate never sees a case's own cost, only its block count.
"""
import json
import math
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
THR = 0.7 ** (1 / 0.3)
BETA = 0.9265861161320369
Q_POOL_FULL = 0.3976 + 2.6588
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333),
         (6, 0.9552271810705998)]


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t - 1e-9) + 1


def ins(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


def oos(fn):
    return {r["test_id"]: r for r in json.load(open(DIR / fn))["test_results"]}


def main():
    wm, _ = ins("_l181_m73.json")
    wo, _ = ins("_l181_cur.json")
    wk, _ = ins("_l189_k1.json")
    rows = M.rows_new()
    W = sum(r["w"] for r in rows)
    beta = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in rows) / W
    pool, dt, med = {}, {}, {}
    for r in rows:
        n = r["n"]
        if not wm.get(n):
            continue
        k = r["t"] / wm[n]
        pool[n] = wo[n] * k
        dt[n] = max(0.0, (wk[n] - wo[n]) * k)
        med[n] = r["med"]

    A = {s: (oos("l194_{}_fulloff.json".format(s)),
             oos("l192_{}_full.json".format(s))) for s in ("s1", "s2")}

    def gate(s):
        return {n: 1 if pool[n] + dt[n] <= THR * med[n] * s else 0 for n in pool}

    def qual(g, s):
        off, on = A[s]
        ids = sorted(set(off) & set(on))
        w = lambda i: math.exp(off[i]["n"] / 12.0)                 # noqa: E731
        sw = sum(w(i) for i in ids)
        qo = sum(w(i) * off[i]["cost"] for i in ids) / sw
        qg = sum(w(i) * (on if g.get(off[i]["n"], 0) else off)[i]["cost"]
                 for i in ids) / sw
        return 100 * (qo - qg) / qo

    def rf(g, ra=1.0):
        num = 0.0
        for r in rows:
            n = r["n"]
            if n not in pool:
                continue
            t = pool[n] * ra + (dt[n] if g.get(n, 0) else 0.0)
            num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        return 100 * (beta - num / W) / beta

    print(__doc__)
    print("=" * 80)
    SC = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9]
    print("{:>7}{:>7}{:>10}{:>10}{:>10}{:>10}{:>8}"
          .format("s", "gate", "qual s1", "qual s2", "RF", "NET(mean)", "rank"))
    best = None
    for s in SC:
        g = gate(s)
        q1, q2 = qual(g, "s1"), qual(g, "s2")
        r_ = rf(g)
        net = (q1 + q2) / 2 + Q_POOL_FULL + r_
        gr = BETA * (1 - net / 100.0)
        lbl = "inf" if s > 1e8 else "{:.1f}".format(s)
        print("{:>7}{:>6}%{:>+9.3f}%{:>+9.3f}%{:>+9.3f}%{:>+9.3f}%{:>8}"
              .format(lbl, int(100 * sum(g.values()) / len(g)), q1, q2, r_,
                      net, rank_of(gr)))
        if best is None or net > best[0]:
            best = (net, s, gr)

    print("\n--- cross-validated: pick s on one sample, score on the other ---")
    for fit, test in (("s1", "s2"), ("s2", "s1")):
        pick = max(SC, key=lambda s: qual(gate(s), fit) + rf(gate(s)))
        g = gate(pick)
        net = qual(g, test) + Q_POOL_FULL + rf(g)
        gr = BETA * (1 - net / 100.0)
        lbl = "inf" if pick > 1e8 else "{:.1f}".format(pick)
        print("   fit {} -> s = {:<5} : on {} it scores NET {:+.3f}%  "
              "graded {:.5f}  rank {}"
              .format(fit, lbl, test, net, gr, rank_of(gr)))
    print("\n   for reference, the shipped-candidate gate is s = 1.0")
    print("   and 'LP everywhere' is s = inf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
