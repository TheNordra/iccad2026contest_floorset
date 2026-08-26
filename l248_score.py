"""L248 score - the lens-D size curve, priced.

L167 and L232 both measured the tier only at K=32, which sits past the pool's
max-bound -> sum-bound crossover (~65 profiles at n=120, from sum/48 = 2.501 s
vs d_max = 3.204 s on the shipped tree). Below the crossover an added profile
costs almost nothing in C++ wall; above it, every one lands on the critical
path. The quality curve turns out to be strongly concave -- K=6 already gives
65% of what K=32 gives -- so the interesting question is whether a short prefix
is net positive.

RUNTIME IS TAKEN FROM THE WHOLE-CASE RATIO, not from the profile-phase wall.
For a pool-SIZE change the profile model cannot see L167's serial proxy tax
(~71 ms per added profile on the main thread), and that tax is exactly what
killed the tier the first time. min-of-3; rep 1 ran alongside L249 and is
expected to be the slowest, which is what min-of-N is for -- contention only
ever adds time.

  <python> l248_score.py
"""
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent
KS = (0, 6, 12, 20, 32)
REPS = (1, 2, 3)


def main():
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l203_marginal_gate as G
    import l230_gate as L
    import l231_score as S
    L._load_tables()

    d = json.load(open(DIR / "l230_pool_new.json"))
    P0 = {int(k): v for k, v in d["POOL"].items()}
    DT0 = {int(k): v for k, v in d["DT"].items()}
    best = {}
    for arm in ("m6", "m2"):
        for i in (1, 2, 3):
            per = S.load_prof("l231_prof_{}_{}.txt".format(arm, i))
            if not per:
                continue
            for n in per:
                w = S.wall(per[n])
                if (arm, n) not in best or w < best[(arm, n)]:
                    best[(arm, n)] = w
    cm = st.median(1 - best[("m2", n)] / best[("m6", n)]
                   for n in G.NS if 60 < n <= 100)
    phi = []
    per = S.load_prof("l231_prof_m6_1.txt")
    rr = {r["block_count"]: r for r in
          json.load(open(DIR / "results_L231_m6_1.json"))["test_results"]}
    for n in per:
        if 60 < n <= 100 and rr[n]["runtime_seconds"] > 0:
            phi.append(min(1.0, S.wall(per[n]) / rr[n]["runtime_seconds"]))
    phim = st.median(phi)
    POOL = {n: (P0[n] * (1 - cm * phim) if 60 < n <= 100 else P0[n])
            for n in G.NS}
    FB = {(20, 60): 1.145, (60, 100): 1.175, (100, 121): 1.172}

    def bf(n):
        for (lo, hi), v in FB.items():
            if lo < n <= hi:
                return v
        return 1.0

    DT = {n: DT0[n] / bf(n) for n in G.NS}
    T0 = {n: POOL[n] + (DT[n] if L.SHIPPED.get(n, 1) else 0.0) for n in G.NS}
    TOTW = sum(r["w"] for r in G.ROWS)

    # ---- min-of-3 whole-case runtime, and rep-1 cost -----------------------
    tmin, cost = {}, {}
    for K in KS:
        for i in REPS:
            f = DIR / "results_L248_k{}_{}.json".format(K, i)
            if not f.exists():
                continue
            for r in json.load(open(f))["test_results"]:
                n = r["block_count"]
                k = (K, n)
                if k not in tmin or r["runtime_seconds"] < tmin[k]:
                    tmin[k] = r["runtime_seconds"]
                if i == 1:
                    cost[k] = r["cost"]
    have = [K for K in KS if (K, 21) in cost]
    reps = {K: sum(1 for i in REPS
                   if (DIR / "results_L248_k{}_{}.json".format(K, i)).exists())
            for K in KS}
    print("reps per K: " + "  ".join("K{}={}".format(K, reps[K]) for K in KS))
    if 0 not in have:
        print("no baseline")
        return 1

    W = lambda n: math.exp(n / 12.0)                              # noqa: E731
    ns_all = [n for n in G.NS if (0, n) in cost]
    SW = sum(W(n) for n in ns_all)
    q0 = sum(W(n) * cost[(0, n)] for n in ns_all) / SW
    qbase = sum(G.qual_pern(L.SHIPPED, s) for s in ("s1", "s2")) / 2
    OP, OD = dict(G.POOL), dict(G.DT)

    print()
    print("{:<5}{:>11}{:>11}{:>11}{:>10}{:>10}{:>6}"
          .format("K", "quality", "wall n>100", "wall all", "NET", "graded",
                  "rank"))
    print("-" * 65)
    for K in have:
        q = 100 * (q0 - sum(W(n) * cost[(K, n)] for n in ns_all) / SW) / q0
        hv = [n for n in ns_all if n > 100 and (K, n) in tmin]
        rr_hi = st.median(tmin[(K, n)] / tmin[(0, n)] for n in hv)
        rr_all = st.median(tmin[(K, n)] / tmin[(0, n)]
                           for n in ns_all if (K, n) in tmin)
        num = 0.0
        for n in G.NS:
            r_ = tmin.get((K, n), tmin.get((0, n), 1.0)) / tmin[(0, n)] \
                if (0, n) in tmin else 1.0
            row = G.ROW[n]
            num += row["w"] * row["q"] * max(0.7, (T0[n] * r_ / row["med"]) ** 0.3)
        num += sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
                   for r in G.ROWS if r["n"] not in POOL)
        rf = 100 * (G.BETA_NUM - num / TOTW) / G.BETA_NUM
        net = qbase + q + G.Q_POOL_FULL + rf
        gr = G.BETA * (1 - net / 100)
        print("{:<5}{:>+10.4f}%{:>10.3f}x{:>10.3f}x{:>+9.3f}%{:>10.5f}{:>6}"
              .format(K, q, rr_hi, rr_all, net, gr, G.rank_of(gr)))
    G.POOL, G.DT = OP, OD
    print("-" * 65)
    print("K=0 is the shipped pool (51). quality is IN SET and, for an ADD-only")
    print("tier, a floor rather than an optimistic estimate: selection is over a")
    print("superset and the proxy is oracle-perfect on heterogeneous candidates")
    print("(M76/M77). L167 measured this tier's OOS transfer at 75-122%.")
    print("r2 = 0.888187   r3 = 0.89933   shipped NET +5.224% / 0.87819")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
