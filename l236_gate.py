"""L236 - what the L235 LP speedup buys, on the L234 package.

The speedup itself scores nothing: it returns wall on cases that already sit on
the RF floor, where max(0.7, R^0.3) has derivative 0. That is L155's argument
and it is still correct as far as it goes. What it misses is that the package
now carries `_L196_LPGATE`: the LP is worth +4.80% of in-set quality fully on,
the gate can only afford part of it, and a cheaper `dt_lp` moves the affordable
line. So the speedup is scored the only way it can be -- by RE-OPTIMISING THE
GATE at the new dt and taking the difference.

Inputs, all measured:
  l230_pool_new.json    POOL / DT on the post-REFINE tree, de-biased, band level
  l231_prof_m{6,2}_*    the mid-band wall cut L234 shipped (min-of-3)
  l235_ab_all.out       the per-case LP speedup, A/B, identity-gated

  <python> l236_gate.py [f]      f overrides the measured speedup
"""
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent


def measured_f():
    """Per-band LP speedup from the A/B run, or None."""
    f = DIR / "l235_ab_all.out"
    if not f.exists():
        return None
    rows = []
    for line in f.read_text().splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", line)
        if m:
            rows.append((int(m.group(2)), float(m.group(3)), float(m.group(4))))
    if not rows:
        return None
    out = {}
    for lo, hi in ((20, 60), (60, 100), (100, 121)):
        v = [(a, b) for n, a, b in rows if lo < n <= hi]
        if v:
            out[(lo, hi)] = sum(x[0] for x in v) / sum(x[1] for x in v)
    return out


def main():
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l203_marginal_gate as G
    import l230_gate as L
    import l231_score as S
    L._load_tables()          # SHIPPED is now the L234 table (71 on)

    d = json.load(open(DIR / "l230_pool_new.json"))
    P0 = {int(k): v for k, v in d["POOL"].items()}
    DT0 = {int(k): v for k, v in d["DT"].items()}

    # the mid-band cut L234 shipped
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
    print("L234 pool: mid band cut {:.2%} x PHI {:.3f}".format(cm, phim))

    fb = measured_f()
    if len(sys.argv) > 1 and sys.argv[1] not in ("x",):
        try:
            flat = float(sys.argv[1])
            fb = {(20, 60): flat, (60, 100): flat, (100, 121): flat}
            print("using the f given on the command line: {:.3f}x".format(flat))
        except ValueError:
            pass
    if fb:
        print("measured LP speedup by band: " + "  ".join(
            "{}-{} {:.3f}x".format(lo + 1, hi, v) for (lo, hi), v in
            sorted(fb.items())))
    else:
        print("!! no l235_ab_all.out yet -- run l235_lpbench.py ab --minn 0")
        fb = {(20, 60): 1.0, (60, 100): 1.0, (100, 121): 1.0}

    def band_f(n):
        for (lo, hi), v in fb.items():
            if lo < n <= hi:
                return v
        return 1.0

    OP, OD = dict(G.POOL), dict(G.DT)

    def sc(tbl, P, T):
        G.POOL, G.DT = P, T
        v = [G.qual_pern(tbl, t) + G.Q_POOL_FULL + G.rf_at(tbl, 1.0)
             for f, t in (("s1", "s2"), ("s2", "s1"))]
        G.POOL, G.DT = OP, OD
        return sum(v) / 2

    print()
    print("{:<34}{:>5}{:>11}{:>10}{:>6}{:>9}"
          .format("configuration", "on", "NET", "graded", "rank", "vs L234"))
    print("-" * 76)
    base = sc(L.SHIPPED, POOL, DT0)
    gr = G.BETA * (1 - base / 100)
    print("{:<34}{:>5}{:>+10.3f}%{:>10.5f}{:>6}{:>9}"
          .format("L234 as shipped", sum(L.SHIPPED.values()), base, gr,
                  G.rank_of(gr), "-"))
    best_row = None
    for tag, ff in (("measured", None), ("1.10x", 1.10), ("1.25x", 1.25),
                    ("1.50x", 1.50), ("2.00x", 2.00)):
        DT = {n: DT0[n] / (band_f(n) if ff is None else ff) for n in G.NS}
        G.POOL, G.DT = POOL, DT
        cands = {"gate unchanged": L.SHIPPED}
        for s in (1.15, 1.2, 1.25, 1.3, 1.35, 1.4):
            g = G.time_gate(s)
            cands["s={}".format(s)] = {
                n: (1 if (L.SHIPPED.get(n, 1) or g[n]) else 0) for n in G.NS}
        G.POOL, G.DT = OP, OD
        vals = {k: sc(v, POOL, DT) for k, v in cands.items()}
        b = max(vals, key=vals.get)
        gr = G.BETA * (1 - vals[b] / 100)
        adds = sorted(n for n in G.NS if cands[b].get(n)
                      and not L.SHIPPED.get(n, 1))
        print("{:<34}{:>5}{:>+10.3f}%{:>10.5f}{:>6}{:>+8.3f}pp"
              .format("LP {} -> best {}".format(tag, b),
                      sum(cands[b].values()), vals[b], gr, G.rank_of(gr),
                      vals[b] - base))
        if tag == "measured":
            best_row = (b, adds, vals[b], gr)
    print("-" * 76)
    if best_row:
        print("at the MEASURED speedup: {}  adds {}".format(best_row[0],
                                                            best_row[1]))
        print("margin over r2 (0.888187): {:+.3f} pp"
              .format(100 * (0.888187 - best_row[3]) / G.BETA))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
