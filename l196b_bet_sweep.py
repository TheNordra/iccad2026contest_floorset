"""L196b - choose the gate IN THE REGIME WE ARE BETTING ON.

The decision taken is to bet on route A. L196 swept the gate at route-A
neutral and found s = 1.2 (cross-validated both directions). But route A frees
wall, which makes a larger s affordable -- choosing s at neutral and deploying
it under the bet optimises the wrong objective.

Same family, same cross-validation discipline, evaluated at ra = 0.68:

    gate_s(n) = 1  iff  t_pool(n) + dt_lp(n) <= 0.3046 * M(n) * s

All arms already measured; the gate never sees a case's own cost, only its
block count.
"""
import json
import math
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
THR = 0.7 ** (1 / 0.3)
BETA = 0.9265861161320369
Q_POOL_FULL = 0.3976 + 2.6588
SC = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 1e9]
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
A = {s: (oos("l194_{}_fulloff.json".format(s)), oos("l192_{}_full.json".format(s)))
     for s in ("s1", "s2")}


def gate(s):
    return {n: 1 if pool[n] + dt[n] <= THR * med[n] * s else 0 for n in pool}


def qual(g, s):
    off, on = A[s]
    ids = sorted(set(off) & set(on))
    w = lambda i: math.exp(off[i]["n"] / 12.0)                     # noqa: E731
    sw = sum(w(i) for i in ids)
    qo = sum(w(i) * off[i]["cost"] for i in ids) / sw
    qg = sum(w(i) * (on if g.get(off[i]["n"], 0) else off)[i]["cost"]
             for i in ids) / sw
    return 100 * (qo - qg) / qo


def rf(g, ra):
    num = 0.0
    for r in rows:
        n = r["n"]
        if n not in pool:
            continue
        t = pool[n] * ra + (dt[n] if g.get(n, 0) else 0.0)
        num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
    return 100 * (beta - num / W) / beta


def table(ra, title):
    print("\n=== {} ===".format(title))
    print("{:>7}{:>7}{:>12}{:>10}{:>10}{:>7}{:>11}"
          .format("s", "gate", "qual(mean)", "RF", "NET", "rank", "graded"))
    for sc in SC:
        g = gate(sc)
        q = (qual(g, "s1") + qual(g, "s2")) / 2
        r_ = rf(g, ra)
        net = q + Q_POOL_FULL + r_
        gr = BETA * (1 - net / 100.0)
        lbl = "inf" if sc > 1e8 else "{:.1f}".format(sc)
        print("{:>7}{:>6}%{:>+11.3f}%{:>+9.3f}%{:>+9.3f}%{:>7}{:>11.5f}"
              .format(lbl, int(100 * sum(g.values()) / len(g)), q, r_, net,
                      rank_of(gr), gr))
    for fit, test in (("s1", "s2"), ("s2", "s1")):
        pick = max(SC, key=lambda x: qual(gate(x), fit) + rf(gate(x), ra))
        g = gate(pick)
        net = qual(g, test) + Q_POOL_FULL + rf(g, ra)
        gr = BETA * (1 - net / 100.0)
        lbl = "inf" if pick > 1e8 else "{:.1f}".format(pick)
        print("   CV: fit {} -> s={:<5} scores NET {:+.3f}% on {}   "
              "graded {:.5f}  rank {}"
              .format(fit, lbl, net, test, gr, rank_of(gr)))


print(__doc__)
print("=" * 80)
print("beta as graded: {:.5f} (rank {})   r3 0.89933   r2 0.88819"
      .format(beta, rank_of(beta)))
table(1.00, "route A NEUTRAL (the downside)")
table(0.68, "route A 0.68x (the bet)")
