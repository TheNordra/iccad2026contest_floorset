"""L330: is the public 1M the same distribution as the validation 100?

Jimmy's report: the public 1M and the set used to GENERATE the test data differ --
"T2B relatedness" 97 % vs 30 % -- so the 1M is already over-fitted and ML trained on
it transfers badly (in-set good, beta bad).

What can and cannot be settled here:
  * 1M vs validation-100  -- BOTH have netlist and label, so every statistic of the
    generative law is directly comparable. Measurable.
  * the hidden beta set    -- we hold only per-case OUTCOMES (cost, gaps, runtime),
    no netlists and no layouts. Its generative law is NOT measurable from here.

So this probe answers the half that is answerable, and measures several candidate
readings of "T2B relatedness" so that whichever one Jimmy means can be compared.
"""
import glob
import math
import os
import sys
from collections import defaultdict

import torch


def stats_from(meta, b2b, p2b, pins, W, H, X, Y, n):
    """every candidate reading of 'how related are the pins to the blocks'"""
    c = torch.stack([X + W / 2, Y + H / 2], 1)
    out = {}
    P = pins[pins[:, 0] >= 0].float()
    pe = p2b[(p2b[:, 0] >= 0) & (p2b[:, 1] >= 0) & (p2b[:, 1] < n) & (p2b[:, 0] < len(P))]
    if len(P) and len(pe):
        Dp = torch.cdist(P, c, p=1)
        mxp = Dp.max()
        u = Dp / mxp
        m = torch.zeros(len(P), n)
        m.index_put_((pe[:, 0].long(), pe[:, 1].long()), torch.ones(len(pe)), accumulate=True)
        con = u[m > 0]
        allu = u.flatten()
        out["p2b_u_connected"] = float(con.mean())
        out["p2b_u_all"] = float(allu.mean())
        out["p2b_concentration"] = float(allu.mean() / con.mean()) if con.numel() else float("nan")
        out["p2b_blocks_covered"] = float((m.sum(0) > 0).float().mean())
        out["p2b_edges_per_block"] = float(len(pe)) / n
    e = b2b[(b2b[:, 0] >= 0) & (b2b[:, 1] >= 0) & (b2b[:, 0] < n) & (b2b[:, 1] < n)]
    e = e[e[:, 0] != e[:, 1]]
    if len(e) > 10:
        D = torch.cdist(c, c, p=1)
        mx = D.max()
        d = D[e[:, 0].long(), e[:, 1].long()] / mx
        w = e[:, 2]
        iu = torch.triu_indices(n, n, 1)
        allp = (D[iu[0], iu[1]] / mx)
        out["b2b_u_connected"] = float(d.mean())
        out["b2b_u_all"] = float(allp.mean())
        out["b2b_concentration"] = float(allp.mean() / d.mean())
        dm, wm = d.mean(), w.mean()
        num = ((d - dm) * (w - wm)).sum()
        den = ((d - dm) ** 2).sum().sqrt() * ((w - wm) ** 2).sum().sqrt()
        out["b2b_weight_r"] = float(num / (den + 1e-30))
        out["b2b_edges_per_block"] = float(len(e)) / n
    out["n"] = n
    out["util"] = float((W * H).sum() / ((X + W).max() - X.min()) / ((Y + H).max() - Y.min()))
    return out


def agg(rows):
    keys = sorted({k for r in rows for k in r})
    return {k: (sum(r[k] for r in rows if k in r) / max(1, sum(1 for r in rows if k in r)))
            for k in keys}


# ---- validation 100 -------------------------------------------------------
val = []
for f in sorted(glob.glob("LiteTensorDataTest/config_*/litedata_1.pth")):
    nn = int(os.path.basename(os.path.dirname(f)).split("_")[1])
    d = torch.load(f, weights_only=False)[0]
    _m, poly = torch.load(f.replace("litedata", "litelabel"), weights_only=False)[0]
    meta, b2b, p2b, pins = d[0], d[1], d[2], d[3]
    n = int((meta[:, 0] > 0).sum())
    X = torch.zeros(n); Y = torch.zeros(n); W = torch.zeros(n); H = torch.zeros(n)
    for k in range(n):
        p = poly[k]; v = p[p[:, 0] != -1]
        x0, y0 = v.min(dim=0).values; x1, y1 = v.max(dim=0).values
        X[k], Y[k], W[k], H[k] = x0, y0, x1 - x0, y1 - y0
    val.append(stats_from(meta, b2b, p2b, pins, W, H, X, Y, n))

# ---- the public 1M --------------------------------------------------------
tr = []
NSH = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for sh in sorted(glob.glob("C:/ICCAD_ml/floorset_lite/worker_*/layouts_*.th"))[:NSH]:
    meta, b2b, p2b, pins, tree, fp, metrics = torch.load(sh, weights_only=False)
    for b in range(meta.shape[0]):
        n = int((meta[b, :, 0] > 0).sum())
        if n < 5:
            continue
        tr.append(stats_from(meta[b], b2b[b], p2b[b], pins[b],
                             fp[b, :n, 0], fp[b, :n, 1], fp[b, :n, 2], fp[b, :n, 3], n))

A, B = agg(val), agg(tr)
print("== L330 public 1M vs validation 100 (%d vs %d layouts) ==" % (len(tr), len(val)))
print("   %-24s %12s %12s %10s" % ("statistic", "validation", "1M train", "ratio"))
for k in sorted(set(A) | set(B)):
    a, b = A.get(k, float("nan")), B.get(k, float("nan"))
    print("   %-24s %12.4f %12.4f %10s"
          % (k, a, b, "%.3f" % (a / b) if b else "-"))
