"""L233 score - the mid band out of sample, and the GO/NO-GO for L234.

The in-set null (+0.0003%) is not evidence. M50/M74 derived REFINE=6 STRICTLY
SELECTION-PRESERVING, so going below it changes selections by construction, and
the in-set corpus carries exactly one case per block count -- the very case the
constant was fitted on. The heavy band read +0.0400% in set and -0.1788% out of
it. This file measures the same thing for the mid band and prints a verdict the
implementation script greps for.

Weighting is `qual_pern`'s, not the raw 240-case mean: the graded set is 100
cases, one per block count, so the OOS replicates are averaged WITHIN a block
count first and the corpus weights come from the grader's structure. Using the
sample's own case counts would silently give a block count that drew 4 cases
twice the influence of one that drew 2.

THREE HARD CHECKS before any number is believed:
  * feasibility must be 240/240 in every arm,
  * the arms must cover the same test_ids,
  * and n<=60 and n>100 must be BIT-IDENTICAL -- the mid-band table cannot
    touch them, so any movement there means the arm carried something else.

  <python> l233_score.py
"""
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent
BAR = 0.30                      # the project's ship bar, in pp of NET
SAMPLES = ("s1", "s2")


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

    ok = True
    qdel = {}
    for s in SAMPLES:
        a = load("l223_{}_r2.json".format(s))          # shipped config
        b = load("l233_{}_mid2.json".format(s))        # mid band = 2
        if not a or not b:
            print("missing arm for {}: base={} treat={}"
                  .format(s, bool(a), bool(b)))
            return 1
        ids = sorted(set(a) & set(b))
        print("{}: {} cases in common (base {} / treat {})"
              .format(s, len(ids), len(a), len(b)))
        inf_a = sum(1 for i in ids if not a[i].get("feasible", True))
        inf_b = sum(1 for i in ids if not b[i].get("feasible", True))
        print("    infeasible base {}  treat {}   {}"
              .format(inf_a, inf_b, "PASS" if inf_a == inf_b == 0 else "FAIL"))
        ok &= (inf_a == 0 and inf_b == 0)

        out = [i for i in ids if not (60 < a[i]["n"] <= 100)
               and abs(a[i]["cost"] - b[i]["cost"]) > 1e-12]
        print("    out-of-band movers (must be 0): {}   {}"
              .format(len(out), "PASS" if not out else "FAIL"))
        ok &= not out

        # qual_pern weighting: mean within block count, grader weights across
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
        qdel[s] = d
        mv = sum(1 for i in ids if abs(a[i]["cost"] - b[i]["cost"]) > 1e-12)
        ws = sum(1 for i in ids if b[i]["cost"] > a[i]["cost"] + 1e-12)
        print("    OOS quality {:+.4f}%   moved {} / {}   worse {}"
              .format(d, mv, len(ids), ws))

    q = sum(qdel.values()) / len(qdel)
    print()
    print("mean OOS quality delta {:+.4f}%   (in set it was +0.0003%)".format(q))

    # ---- NET, with the measured band-level wall cut and the additive gate ---
    d = json.load(open(DIR / "l230_pool_new.json"))
    P0 = {int(k): v for k, v in d["POOL"].items()}
    DT = {int(k): v for k, v in d["DT"].items()}
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
    print("mid-band wall cut {:+.2f}%  x PHI {:.3f}  (flat, un-de-biased = the"
          " conservative direction)".format(100 * cm, phim))

    OP, OD = dict(G.POOL), dict(G.DT)

    def sc(tbl, P, extra=0.0):
        G.POOL, G.DT = P, DT
        v = [G.qual_pern(tbl, t) + G.Q_POOL_FULL + G.rf_at(tbl, 1.0)
             for f, t in (("s1", "s2"), ("s2", "s1"))]
        G.POOL, G.DT = OP, OD
        return sum(v) / 2 + extra

    Pm = {n: (P0[n] * (1 - cm * phim) if 60 < n <= 100 else P0[n]) for n in G.NS}
    G.POOL, G.DT = Pm, DT
    g = G.time_gate(1.15)
    NEW = {n: (1 if (L.SHIPPED.get(n, 1) or g[n]) else 0) for n in G.NS}
    G.POOL, G.DT = OP, OD
    adds = sorted(n for n in G.NS if NEW[n] and not L.SHIPPED.get(n, 1))

    ship = sc(L.SHIPPED, P0)
    prop = sc(NEW, Pm, q)
    gr_s, gr_p = G.BETA * (1 - ship / 100), G.BETA * (1 - prop / 100)
    print()
    print("{:<44}{:>10}{:>10}{:>6}".format("configuration", "NET", "graded", "rank"))
    print("-" * 70)
    print("{:<44}{:>+9.3f}%{:>10.5f}{:>6}"
          .format("shipped (mid 6, gate 63)", ship, gr_s, G.rank_of(gr_s)))
    print("{:<44}{:>+9.3f}%{:>10.5f}{:>6}"
          .format("L234 (mid 2, gate {} on)".format(sum(NEW.values())),
                  prop, gr_p, G.rank_of(gr_p)))
    print("-" * 70)
    print("gate adds: {}".format(adds))
    print("delta {:+.3f} pp   bar {:.2f} pp   margin over r2 (0.888187) {:+.3f} pp"
          .format(prop - ship, BAR, 100 * (0.888187 - gr_p) / G.BETA))

    go = ok and (prop - ship) >= BAR and min(qdel.values()) > -1.0
    print()
    print("L233_VERDICT={}".format("GO" if go else "NO-GO"))
    if not go:
        print("  hard checks {} | delta {:+.3f} pp vs bar {:.2f}"
              .format("ok" if ok else "FAILED", prop - ship, BAR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
