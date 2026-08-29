"""L194 verdict - the LP gate's capture efficiency on the FULL pool, measured.

L193 measured it on the THIN pool: gate fires on 45% of block counts, collects
40% of the LP's OOS quality => efficiency 0.889. The full-pool candidate was
scored by APPLYING that 0.889 to a gate that fires on only 30%. This replaces
that extrapolation with a measurement.

The failure direction, stated before the data (L194 script header): the full
pool's own wall has already eaten most of the budget, so the 30% that survive
the gate are the cases with the MOST slack -- not necessarily the cases the LP
helps most. If the LP's value sits on heavy, low-slack cases, capture collapses.

    >= 25%  recommendation stands: FULL + gated LP, rank 4 floor / rank 2 upside
    15-25%  upside thins, still likely ahead of beta at neutral
    < 15%   the extrapolation was wrong; thin + gated LP is the only candidate
            whose capture rate is measured rather than assumed
"""
import json
import math
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
THR = 0.7 ** (1 / 0.3)
BETA = 0.9265861161320369
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333),
         (6, 0.9552271810705998)]
Q_POOL_FULL = 0.3976 + 2.6588      # full pool vs M73, OOS (L189 + L192)
Q_POOL_THIN = 0.3976               # thin pool vs M73


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t - 1e-9) + 1


def ins(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


def oos(fn):
    return {r["test_id"]: r
            for r in json.load(open(DIR / fn))["test_results"]}


def main():
    wm, _ = ins("_l181_m73.json")
    rows = M.rows_new()
    W = sum(r["w"] for r in rows)
    beta = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in rows) / W
    print(__doc__)
    print("=" * 78)

    def gate_for(off_fn, on_fn):
        wo, _ = ins(off_fn)
        wk, _ = ins(on_fn)
        pool, dt, g = {}, {}, {}
        for r in rows:
            n = r["n"]
            if not wm.get(n):
                continue
            k = r["t"] / wm[n]
            pool[n] = wo[n] * k
            dt[n] = max(0.0, (wk[n] - wo[n]) * k)
            g[n] = 1 if pool[n] + dt[n] <= THR * r["med"] else 0
        return pool, dt, g

    def capture(g, off_tag, on_tag):
        out = []
        for s in ("s1", "s2"):
            off, on = oos(off_tag.format(s)), oos(on_tag.format(s))
            ids = sorted(set(off) & set(on))
            w = lambda i: math.exp(off[i]["n"] / 12.0)             # noqa: E731
            sw = sum(w(i) for i in ids)
            qo = sum(w(i) * off[i]["cost"] for i in ids) / sw
            qn = sum(w(i) * on[i]["cost"] for i in ids) / sw
            qg = sum(w(i) * (on if g.get(off[i]["n"], 0) else off)[i]["cost"]
                     for i in ids) / sw
            out.append((100 * (qo - qn) / qo, 100 * (qo - qg) / qo))
        return out

    def rf(pool, dt, sel, ra):
        num = wall = 0.0
        for r in rows:
            n = r["n"]
            if n not in pool:
                continue
            t = pool[n] * ra + (dt[n] if sel.get(n, 0) else 0.0)
            wall += t
            num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        return 100 * (beta - num / W) / beta, wall

    print("{:<22}{:>7}{:>10}{:>10}{:>9}"
          .format("pool", "gate", "LP full", "LP gated", "capture"))
    results = {}
    for lbl, off_i, on_i, off_o, on_o, qpool in (
            ("thin (measured L193)", "_l181_nohint.json",
             "_l191_thinpool_lp.json", "l193_{}_thinoff.json",
             "l192_{}_thin.json", Q_POOL_THIN),
            ("FULL", "_l181_cur.json", "_l189_k1.json",
             "l194_{}_fulloff.json", "l192_{}_full.json", Q_POOL_FULL)):
        pool, dt, g = gate_for(off_i, on_i)
        try:
            c = capture(g, off_o, on_o)
        except FileNotFoundError:
            print("{:<22}{:>6}%   not finished".format(
                lbl, int(100 * sum(g.values()) / len(g))))
            continue
        lpf = sum(a for a, _ in c) / 2
        lpg = sum(b for _, b in c) / 2
        results[lbl] = (pool, dt, g, qpool, lpg)
        print("{:<22}{:>6}%{:>+9.3f}%{:>+9.3f}%{:>8.0f}%"
              .format(lbl, int(100 * sum(g.values()) / len(g)), lpf, lpg,
                      100 * lpg / lpf if lpf else 0))

    if "FULL" not in results:
        return 1
    print("\n{:<28}{:>8}{:>10}{:>10}{:>11}{:>6}"
          .format("configuration", "wall", "quality", "RF", "graded", "rank"))
    print("{:<28}{:>7.1f}s{:>10}{:>10}{:>11.5f}{:>6}"
          .format("beta (M73)", sum(r["t"] for r in rows), "-", "-",
                  beta, rank_of(beta)))
    for lbl, (pool, dt, g, qpool, lpg) in results.items():
        for ra, tag in ((1.0, ""), (0.68, "  [rA .68]")):
            r_, wall = rf(pool, dt, g, ra)
            q = qpool + lpg
            gr = BETA * (1 - (q + r_) / 100.0)
            print("{:<28}{:>7.1f}s{:>+9.3f}%{:>+9.3f}%{:>11.5f}{:>6}"
                  .format(lbl + " + gated LP" + tag, wall, q, r_, gr,
                          rank_of(gr)))
    print("\nr3 threshold 0.89933   r2 threshold 0.88819")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
