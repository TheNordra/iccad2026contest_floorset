"""L253 -- how far, in topology edit distance, is our layout from the label's?

HANDOFF_2026-08-28_RESEARCH.md sec.5 item 3: L128 says a topology cannot be
TRANSPLANTED (blend the label's aspect ratios 2% toward ours and the LP is already
worse than shipped; at 30% it is infeasible on 98/100). But L128 varied SHAPES on
the label's own arrangement. It never measured the distance between OUR
arrangement and the label's, which is the question that decides whether imitation
is even a coherent target.

Topology here is the project's OWN definition -- the max-gap pair relation that
build_and_solve derives at optimizer_constructive.py:3346-3357:

    g0 = x_j - (x_i + w_i)   i LEFT  of j        pick = argmax(g)
    g1 = x_i - (x_j + w_j)   j LEFT  of i
    g2 = y_j - (y_i + h_i)   i BELOW of j
    g3 = y_i - (y_j + h_j)   j BELOW of i

Two distances per pair of layouts, over all n(n-1)/2 pairs:

  d_pick  fraction of pairs whose argmax relation differs. This is the project's
          own notion, but it is NOISY: a pair separated on both axes flips its
          argmax under an arbitrarily small move without any reordering.

  d_hard  fraction of pairs whose FEASIBLE RELATION SETS are disjoint -- there is
          no relation that both layouts satisfy, so this pair genuinely has to be
          reordered. This is a lower bound on the number of edits, and it is the
          honest "edit distance".

Reads labels for OFFLINE DIAGNOSIS only (the 2026-08-05 ruling bans
label-supervised ML, not oracle probes -- same standing as L250/L251/L252).

  <python> l253_editdist.py --limit 40
"""
import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent
CACHE = DIR / "l252_cache.pkl"
TOL = 1e-9


def masks(P, I, J):
    """(mask, pick) over the upper-triangle pairs (I, J) of layout P."""
    P = np.asarray(P, dtype=np.float64)
    x, y, w, h = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    rx, ty = x + w, y + h
    g = np.empty((4, I.size), dtype=np.float64)
    g[0] = x[J] - rx[I]
    g[1] = x[I] - rx[J]
    g[2] = y[J] - ty[I]
    g[3] = y[I] - ty[J]
    m = np.zeros(I.size, dtype=np.uint8)
    for b in range(4):
        m |= ((g[b] >= -TOL).astype(np.uint8) << b)
    return m, g.argmax(axis=0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    sys.argv = ["x"]
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    import optimizer_constructive as oc

    if len(list(oc._pool_indices(120))) != 51:
        print("!! not the shipped pool")
        return 1
    RH = oc._RH

    C = pickle.load(open(CACHE, "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs(a.sample)}
    keys = [k for k in C if k[0] == a.sample]
    keys.sort(key=lambda k: -C[k]["n"])
    keys = keys[:a.limit]
    print("[l253] {} cases, sample {}".format(len(keys), a.sample))

    rows = []
    loaded = {}
    for kn, key in enumerate(keys):
        ck = key[1]
        e = C[key]
        fk, L, n = spec_of[ck]
        if fk not in loaded:
            loaded.clear()
            loaded[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(loaded[fk], L)
        lay["base"], _ = m67._baseline_official(lay)   # _cost needs it (l250 does the same)
        lab = [tuple(map(float, q)) for q in lay["tp"]]
        I, J = np.triu_indices(n, 1)
        lm, lp = masks(lab, I, J)

        idxs = sorted(e["recs"])
        met = [e["recs"][i] for i in idxs]
        A_hat = 1.035 * max(e["sumA"], 1e-9)
        hmin = min(m["hpwl"] for m in met) or 1.0
        prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
                for m in met]
        kpick = min(range(len(idxs)), key=lambda t: prox[t])

        M, PK, cost = [], [], []
        for m in met:
            mm, pp = masks(m["pos"], I, J)
            M.append(mm)
            PK.append(pp)
            try:
                cost.append(float(m67._cost(m["pos"], lay).cost))
            except Exception:
                cost.append(float("inf"))
        npair = float(I.size)
        d_hard = [float(np.count_nonzero((mm & lm) == 0)) / npair for mm in M]
        d_pick = [float(np.count_nonzero(pp != lp)) / npair for pp in PK]

        # the pool's own spread, same metric, so the two are comparable
        tot, cnt = 0.0, 0
        nn = [1e9] * len(M)
        for u in range(len(M)):
            for v in range(u + 1, len(M)):
                duv = float(np.count_nonzero((M[u] & M[v]) == 0)) / npair
                tot += duv
                cnt += 1
                if duv < nn[u]:
                    nn[u] = duv
                if duv < nn[v]:
                    nn[v] = duv
        d_int = tot / max(cnt, 1)
        # apples-to-apples: d_min is a MIN over the pool, so compare it to each
        # pool member's OWN nearest neighbour, not to the pool's mean.
        d_nn_med = float(np.median(np.asarray(nn)))
        d_nn_max = float(np.max(np.asarray(nn)))

        kclose = min(range(len(M)), key=lambda t: d_hard[t])
        kbest = min(range(len(M)), key=lambda t: cost[t])
        order = sorted(range(len(M)), key=lambda t: cost[t])
        rank_close = order.index(kclose)
        lab_cost = float(m67._cost(lay["tp"], lay).cost)

        # THE gradient test: across the 51 candidates of THIS case, does being
        # closer to the label's topology predict a lower true cost? Spearman
        # (Pearson on ranks) so a single outlier candidate cannot carry it.
        fin = [t for t in range(len(M)) if cost[t] < float("inf")]
        if len(fin) >= 4:
            rd = np.empty(len(fin))
            rc = np.empty(len(fin))
            for rank, t in enumerate(sorted(fin, key=lambda t: d_hard[t])):
                rd[fin.index(t)] = rank
            for rank, t in enumerate(sorted(fin, key=lambda t: cost[t])):
                rc[fin.index(t)] = rank
            sd, sc = rd.std(), rc.std()
            rho = float(((rd - rd.mean()) * (rc - rc.mean())).mean() / (sd * sc)) \
                if sd > 0 and sc > 0 else 0.0
        else:
            rho = float("nan")

        rows.append(dict(
            n=n, npool=len(M), d_pick=d_pick[kpick], d_hard=d_hard[kpick],
            d_min=min(d_hard), d_max=max(d_hard), d_int=d_int,
            d_best=d_hard[kbest], rank_close=rank_close,
            d_nn_med=d_nn_med, d_nn_max=d_nn_max,
            cost_pick=cost[kpick], cost_best=cost[kbest],
            cost_close=cost[kclose], lab=lab_cost, rho=rho))
        if (kn + 1) % 10 == 0:
            print("   {}/{}".format(kn + 1, len(keys)))

    SW = sum(math.exp(r["n"] / 12.0) for r in rows)

    def wm(f):
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rows) / SW

    print()
    print("=" * 78)
    print("L253 -- topology edit distance to the label, {} cases".format(len(rows)))
    print("=" * 78)
    print("  {:>5s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>7s}".format(
        "n", "d_pick", "d_hard", "d_min", "d_int", "d_best", "rk_cl"))
    for r in sorted(rows, key=lambda r: -r["n"])[:14]:
        print("  {:5d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:5d}/{:d}".format(
            r["n"], r["d_pick"], r["d_hard"], r["d_min"], r["d_int"],
            r["d_best"], r["rank_close"], r["npool"]))
    if len(rows) > 14:
        print("  ... {} more".format(len(rows) - 14))

    print()
    print("  weighted by exp(n/12):")
    print("    d_pick   shipped layout vs label, argmax relation  {:.4f}".format(
        wm(lambda r: r["d_pick"])))
    print("    d_hard   shipped layout vs label, MUST-reorder     {:.4f}".format(
        wm(lambda r: r["d_hard"])))
    print("    d_min    closest of the 51 to the label            {:.4f}".format(
        wm(lambda r: r["d_min"])))
    print("    d_max    furthest of the 51                        {:.4f}".format(
        wm(lambda r: r["d_max"])))
    print("    d_int    the pool's OWN mean pairwise distance     {:.4f}".format(
        wm(lambda r: r["d_int"])))
    print("    d_nn     median pool member's NEAREST neighbour     {:.4f}".format(
        wm(lambda r: r["d_nn_med"])))
    print("    d_nn_max the most ISOLATED pool member              {:.4f}".format(
        wm(lambda r: r["d_nn_max"])))
    print()
    print("  is there a gradient? (the 'monotone path' question)")
    print("    rank of the label-closest profile in true cost   {:.1f} / {:.0f}".format(
        wm(lambda r: r["rank_close"]), wm(lambda r: r["npool"])))
    print("    cost of the proxy pick                           {:.6f}".format(
        wm(lambda r: r["cost_pick"])))
    print("    cost of the label-CLOSEST profile                {:.6f}".format(
        wm(lambda r: r["cost_close"])))
    print("    cost of the true-best profile                    {:.6f}".format(
        wm(lambda r: r["cost_best"])))
    print("    the label itself                                 {:.6f}".format(
        wm(lambda r: r["lab"])))
    print("    d_hard of the true-best profile                  {:.4f}".format(
        wm(lambda r: r["d_best"])))

    good = [r for r in rows if r["rho"] == r["rho"]]
    mr = sum(r["rho"] for r in good) / max(len(good), 1)
    pos = sum(1 for r in good if r["rho"] > 0)
    print("    Spearman(d_hard, true cost) across the 51, per case:")
    print("      mean {:+.3f}   positive in {}/{} cases".format(mr, pos, len(good)))
    print("      (positive => closer to the label IS cheaper => a gradient exists)")
    print()
    print("  is the label an OUTLIER relative to our own pool?")
    print("    d_min(label)  {:.4f}   vs median member's nearest nb {:.4f}"
          "   ratio {:.2f}x".format(
              wm(lambda r: r["d_min"]), wm(lambda r: r["d_nn_med"]),
              wm(lambda r: r["d_min"]) / max(wm(lambda r: r["d_nn_med"]), 1e-9)))
    print("    d_min(label)  {:.4f}   vs MOST ISOLATED member      {:.4f}".format(
        wm(lambda r: r["d_min"]), wm(lambda r: r["d_nn_max"])))
    nin = sum(1 for r in rows if r["d_min"] <= r["d_nn_max"])
    print("    cases where the label is NOT more isolated than our own"
          " loneliest candidate: {}/{}".format(nin, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
