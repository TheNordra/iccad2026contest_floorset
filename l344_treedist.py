"""L344 -- how far are L340's GOOD trees from the LABEL's tree?

THE DECISION THIS FEEDS. Tier-1 path (3) is "predict the generator's B*-tree from
the netlist, supervised on tree_sol". L340 established that the B*-tree manifold
contains layouts that beat our packer (n=80, 5/5 seeds, median 1.1500 vs 1.2178)
and that the line dies on SEARCH cost, not on reachability. A supervised model
would replace the search -- but only if the label tree is where the good trees are.
If the good trees sit somewhere else, a model trained on label trees aims at the
wrong point, and path (3) has to be re-scoped to "learn the representation, not the
target" (HANDOFF_RESEARCH_AFTER_L340 sec.3).

WHY LAYOUT-RELATION SPACE AND NOT RAW TREE EDIT DISTANCE. A B*-tree is not a
canonical encoding: many trees decode to the same layout, so raw tree edit distance
counts differences that do not exist on the floor. The project already has the right
metric -- L253's d_hard, the max-gap pair relation that build_and_solve derives at
optimizer_constructive.py:3346-3357, imported here verbatim so the numbers land on
the SAME RULER as L253's (shipped vs label 0.0679, our pool's own internal spread
0.0756). A secondary, genuinely tree-flavoured fingerprint is also reported: the
B*-tree left-child rule IS horizontal abutment (x_c == x_p + w_p with y-overlap), so
the abutment-edge Jaccard measures the tree relation directly, from positions alone.

THE CONTROLS ARE THE POINT. A distance means nothing without them:
  * d(SA_i, SA_j) -- how far apart are two INDEPENDENTLY FOUND good trees? This is
    the scale. If d(SA, label) is no larger than this, the label sits inside the good
    basin and is a legitimate target.
  * d(random, label) -- chance level, from the SA's own random initial tree (iters=0).
  * d(ours, label) -- our shipped packer on the same instance, the L253 calibration.
  * the GRADIENT: d(SA, label) as iterations go 0 -> 10k -> 100k -> 2M. As the SA
    gets better, does it move TOWARD the label or not? This is the non-circular
    version of the question -- it needs no training and no model.

Offline oracle probe: reads labels for DIAGNOSIS only, trains nothing, ships nothing,
touches no file in the shipping path (2026-08-05 ruling, same standing as L250-L253).

Usage:  cd ship_final
        <python> l344_treedist.py [--ns 40,80,120] [--seeds 5] [--hw 2]
                                  [--iters 0,10000,100000,2000000]
"""
import argparse
import itertools
import json
import statistics
import sys

import numpy as np
import torch

from l253_editdist import masks
from l340_run import load, run

TOL = 1e-6
# Our shipped RF-SAFE candidate, Windows 48c, in-set 100 (SHIP_DECISION_2026-08-28).
OURS_JSON = "l313_win48_rfsafe.json"
# The full-ungate arm. L340's OURS[80]=1.2178 came from THIS, not from RF-SAFE.
OURS_JSON_ALT = "l294_gate0.json"


def label_rects(n):
    """fp_sol is a closed 5-vertex rectangle polygon per block -> (x, y, w, h)."""
    fp = torch.load("LiteTensorDataTest/config_%d/litelabel_1.pth" % n,
                    weights_only=False)[0][1]
    out = []
    for k in range(fp.shape[0]):
        p = fp[k].numpy()
        x0, y0 = float(p[:, 0].min()), float(p[:, 1].min())
        x1, y1 = float(p[:, 0].max()), float(p[:, 1].max())
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def ours_rects(n, path):
    for t in json.load(open(path))["test_results"]:
        if t.get("block_count") == n and t.get("positions"):
            return ([tuple(map(float, r)) for r in t["positions"]],
                    1 + 0.5 * (t["hpwl_gap"] + t["area_gap"]))
    return None, None


def d_hard(A, B, I, J):
    ma, _ = masks(A, I, J)
    mb, _ = masks(B, I, J)
    return float(np.count_nonzero((ma & mb) == 0)) / float(I.size)


def relset(A, I, J):
    """Mean number of the 4 relations a pair satisfies. LOOSE layouts score higher,
    and a higher value mechanically LOWERS d_hard against everything -- this is the
    confound the permutation null below removes."""
    m, _ = masks(A, I, J)
    return float(np.mean([bin(int(v)).count("1") for v in m]))


def d_perm(A, B, I, J, reps=20, seed=0):
    """Chance level AT THIS LAYOUT'S OWN DENSITY: permute which block is which in A,
    keeping A's geometry untouched, and measure again. d_hard/d_perm < 1 means the
    two layouts genuinely agree more than their densities alone would produce."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=np.float64)
    out = []
    for _ in range(reps):
        out.append(d_hard(A[rng.permutation(len(A))], B, I, J))
    return float(np.mean(out))


def abut_edges(R):
    """B*-tree left-child relation read off the floor: j starts exactly where i ends,
    and they overlap in y. Directed, so (i,j) != (j,i)."""
    R = np.asarray(R, dtype=np.float64)
    x, y, w, h = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    E = set()
    for i in range(len(R)):
        ri = x[i] + w[i]
        for j in range(len(R)):
            if i == j:
                continue
            if abs(x[j] - ri) <= TOL and y[j] < y[i] + h[i] - TOL \
                    and y[i] < y[j] + h[j] - TOL:
                E.add((i, j))
    return E


def jac(A, B):
    u = len(A | B)
    return len(A & B) / u if u else 1.0


def bottom_supported(R):
    R = np.asarray(R, dtype=np.float64)
    x, y, w, h = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    ok = 0
    for k in range(len(R)):
        if y[k] <= TOL:
            ok += 1
            continue
        if any(abs(y[j] + h[j] - y[k]) <= TOL and x[j] < x[k] + w[k] - TOL
               and x[k] < x[j] + w[j] - TOL for j in range(len(R)) if j != k):
            ok += 1
    return ok / len(R)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="40,80,120")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hw", type=float, default=2.0)
    ap.add_argument("--iters", default="0,10000,100000,2000000")
    a = ap.parse_args()
    NS = [int(v) for v in a.ns.split(",")]
    ITS = [int(v) for v in a.iters.split(",")]

    print("== L344: distance from L340's good trees to the LABEL's tree ==")
    print("   metric  = L253 d_hard (fraction of block pairs that MUST be reordered)")
    print("   ruler   = L253 measured shipped-vs-label 0.0679, pool-internal 0.0756")
    print("   HW      = %.2f * HW*,  %d seeds per point" % (a.hw, a.seeds))
    print()

    summary = []
    for n in NS:
        *_, arL, hpL, nb = load(n)
        meta, *_ = load(n)
        pre = [k for k in range(nb) if float(meta[k, 2]) != 0]
        fix = [k for k in range(nb) if float(meta[k, 1]) != 0]
        hw = (arL / hpL) * a.hw

        lab = label_rects(n)[:nb]
        our, ourq = ours_rects(n, OURS_JSON)
        alt, altq = ours_rects(n, OURS_JSON_ALT)
        I, J = np.triu_indices(nb, 1)
        lm, _ = masks(lab, I, J)

        print("-" * 78)
        print("n = %d   preplaced %d  fixed %d   label bottom-supported %.1f%%"
              % (nb, len(pre), len(fix), 100 * bottom_supported(lab)))
        print("   ours(RF-SAFE) quality %.4f   ours(gate0) quality %.4f"
              "   label quality 1.0000 by definition" % (ourq, altq))
        print()
        print("   %9s %8s %8s | %8s %8s %8s | %8s %8s %6s"
              % ("iters", "quality", "sd", "d->lab", "d_perm", "d/perm",
                 "d(i,j)", "abutJac", "bsup%"))

        rows = {}
        for it in ITS:
            layouts, qs = [], []
            for s in range(1, a.seeds + 1):
                r = run(n, hw, it, seed=s)
                layouts.append(r["pos"])
                qs.append(1 + 0.5 * (r["hg"] + r["ag"]))
            dl = [d_hard(p, lab, I, J) for p in layouts]
            dij = [d_hard(layouts[u], layouts[v], I, J)
                   for u, v in itertools.combinations(range(len(layouts)), 2)]
            eL = abut_edges(lab)
            aj = [jac(abut_edges(p), eL) for p in layouts]
            bs = [bottom_supported(p) for p in layouts]
            dp = [d_perm(p, lab, I, J) for p in layouts]
            rs = [relset(p, I, J) for p in layouts]
            rows[it] = dict(q=qs, dl=dl, dij=dij, aj=aj, bs=bs, lay=layouts,
                            dp=dp, rs=rs)
            print("   %9d %8.4f %8.4f | %8.4f %8.4f %8.4f | %8.4f %8.4f %6.1f%%"
                  % (it, statistics.median(qs),
                     statistics.stdev(qs) if len(qs) > 1 else 0.0,
                     mean(dl), mean(dp), mean(dl) / max(mean(dp), 1e-12),
                     mean(dij) if dij else float("nan"), mean(aj), 100 * mean(bs)))

        best = max(ITS)
        dl_our = d_hard(our, lab, I, J)
        dl_alt = d_hard(alt, lab, I, J)
        d_sa_our = [d_hard(p, our, I, J) for p in rows[best]["lay"]]
        print()
        pm_our = d_perm(our, lab, I, J)
        print("   CALIBRATION on this same instance")
        print("     d(ours RF-SAFE, label)                      %.4f"
              "   (chance at its own density %.4f -> %.2fx closer)"
              % (dl_our, pm_our, pm_our / max(dl_our, 1e-12)))
        print("     d(ours gate0,   label)                      %.4f" % dl_alt)
        print("     d(SA@%d, ours RF-SAFE)                 %.4f"
              % (best, mean(d_sa_our)))
        print("     ours bottom-supported                       %.1f%%"
              % (100 * bottom_supported(our)))
        print("     abutment Jaccard ours vs label              %.4f"
              % jac(abut_edges(our), abut_edges(lab)))
        summary.append(dict(n=nb, rows=rows, dl_our=dl_our, dl_alt=dl_alt,
                            best=best, d_sa_our=mean(d_sa_our), ourq=ourq,
                            pm_our=pm_our))
        print()

    print("=" * 78)
    print("VERDICT TABLE   (best-iteration cell, the trees that beat our packer)")
    print("=" * 78)
    print("  %5s %9s %9s %8s %9s %8s %9s %9s"
          % ("n", "d(SA,lab)", "d(SA,SA)", "SA/SASA", "chance", "SA/chnc",
             "d(our,lab)", "our/chnc"))
    for s in summary:
        b = s["rows"][s["best"]]
        dsa, dij, pm = mean(b["dl"]), mean(b["dij"]), mean(b["dp"])
        print("  %5d %9.4f %9.4f %8.2f %9.4f %8.2f %9.4f %9.2f"
              % (s["n"], dsa, dij, dsa / dij if dij else float("nan"), pm,
                 dsa / max(pm, 1e-12), s["dl_our"],
                 s["dl_our"] / max(s["pm_our"], 1e-12)))
    print()
    print("  READ IT LIKE THIS")
    print("    ratio ~ 1  -> the label sits INSIDE the good basin; independently")
    print("                  found good trees are as far from each other as from it")
    print("                  => training on label trees aims at the right place.")
    print("    ratio >> 1 -> good trees cluster somewhere the label is NOT")
    print("                  => the label is the wrong target; re-scope path (3).")
    print("    compare d(SA,lab) against d(rnd,lab) for the chance level, and")
    print("    against d(our,lab) for what our own packer already achieves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
