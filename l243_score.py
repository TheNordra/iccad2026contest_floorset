"""L243 score - devex out of sample, and the NET it would have to justify.

The RF side is settled and small: the LP is 1.072x faster over 100 cases, which
stacks on L235's 1.170x for 1.254x, worth roughly +0.15pp once the gate is held
fixed. The question is entirely the quality side, because devex moves 66 of 100
layouts and the LP objective on 4 of them by up to 4e-2 -- in BOTH directions.

Bar: both samples non-negative, or the RF gain has to cover the loss with room.

  <python> l243_score.py
"""
import json
import os
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent


def load(fn):
    f = DIR / fn
    if not f.exists():
        return None
    return {r["test_id"]: r for r in json.load(open(f))["test_results"]}


def main():
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l203_marginal_gate as G
    import l230_gate as L
    import l231_score as S
    L._load_tables()

    q = {}
    ok = True
    for s in ("s1", "s2"):
        a = load("l243_{}_base.json".format(s))
        b = load("l243_{}_devex.json".format(s))
        if not a or not b:
            print("missing arm for {}: base={} devex={}".format(s, bool(a), bool(b)))
            return 1
        ids = sorted(set(a) & set(b))
        ia = sum(1 for i in ids if not a[i].get("feasible", True))
        ib = sum(1 for i in ids if not b[i].get("feasible", True))
        so, sn, cnt = {}, {}, {}
        for i in ids:
            n = a[i]["n"]
            so[n] = so.get(n, 0.0) + a[i]["cost"]
            sn[n] = sn.get(n, 0.0) + b[i]["cost"]
            cnt[n] = cnt.get(n, 0) + 1
        ns = [n for n in cnt if n in G.ROW]
        sw = sum(G.ROW[n]["w"] for n in ns)
        qo = sum(G.ROW[n]["w"] * so[n] / cnt[n] for n in ns) / sw
        qn = sum(G.ROW[n]["w"] * sn[n] / cnt[n] for n in ns) / sw
        d = 100 * (qo - qn) / qo
        q[s] = d
        mv = sum(1 for i in ids if a[i]["cost"] != b[i]["cost"])
        ws = sum(1 for i in ids if b[i]["cost"] > a[i]["cost"] + 1e-12)
        print("{}: {} cases  infeasible {}/{}  quality {:+.4f}%  moved {}  "
              "worse {}".format(s, len(ids), ia, ib, d, mv, ws))
        ok &= (ia == 0 and ib == 0)

    m = sum(q.values()) / len(q)
    print()
    print("mean OOS quality {:+.4f}%   both samples same sign: {}"
          .format(m, (q["s1"] > 0) == (q["s2"] > 0)))

    # ---- RF: the gate held fixed, dt_lp divided by the extra 1.072x ---------
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
    F0 = {(20, 60): 1.145, (60, 100): 1.175, (100, 121): 1.172}
    FD = {(20, 60): 1.079, (60, 100): 1.089, (100, 121): 1.057}

    def bf(t, n):
        for (lo, hi), v in t.items():
            if lo < n <= hi:
                return v
        return 1.0

    OP, OD = dict(G.POOL), dict(G.DT)

    def sc(P, T, extra=0.0):
        G.POOL, G.DT = P, T
        v = [G.qual_pern(L.SHIPPED, t) + G.Q_POOL_FULL + G.rf_at(L.SHIPPED, 1.0)
             for f, t in (("s1", "s2"), ("s2", "s1"))]
        G.POOL, G.DT = OP, OD
        return sum(v) / 2 + extra

    print()
    print("{:<38}{:>11}{:>10}{:>6}".format("configuration", "NET", "graded",
                                           "rank"))
    print("-" * 66)
    for rb in (0.72, 0.7682, 0.82):
        Pb = {n: (P0[n] * rb / 0.7682 if n > 100 else P0[n]) for n in G.NS}
        Pb = {n: (Pb[n] * (1 - cm * phim) if 60 < n <= 100 else Pb[n])
              for n in G.NS}
        Ta = {n: DT0[n] / bf(F0, n) for n in G.NS}
        Tb = {n: DT0[n] / (bf(F0, n) * bf(FD, n)) for n in G.NS}
        a = sc(Pb, Ta)
        b = sc(Pb, Tb, m)
        gr = G.BETA * (1 - b / 100)
        print("rb={:.4f}  shipped                    {:>+10.3f}%{:>10.5f}{:>6}"
              .format(rb, a, G.BETA * (1 - a / 100), G.rank_of(a and
                                                              G.BETA * (1 - a / 100))))
        print("rb={:.4f}  + devex (RF and quality)   {:>+10.3f}%{:>10.5f}{:>6}"
              "   {:+.3f} pp".format(rb, b, gr, G.rank_of(gr), b - a))
    print("-" * 66)
    go = ok and min(q.values()) >= 0.0
    print("L243_VERDICT={}".format("GO" if go else "NO-GO"))
    if not go:
        print("  the bar is BOTH samples non-negative; devex is a"
              " variance-increasing change and does not get the benefit of a"
              " mean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
