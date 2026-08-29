"""L327 step A: LEARN the generative law from the training set (vectorised).

FloorSet Algorithm 4 samples the netlist FROM the finished layout:
    PSim <- 1 - Normalize(pairwiseB2BDistance(F));  edges ~ Sample(p=PSim), with
    replacement; weights likewise.  So for a pair, two observables carry distance:
    the MULTIPLICITY m of the pair among the sampled edges, and the sampled weight w.
Learn E[u | m] and E[u | w] empirically -- u = per-layout normalised centre distance
-- rather than assuming the form of Normalize() or of the weight law.
"""
import glob
import json
import sys
from collections import defaultdict

import torch

SHARDS = sorted(glob.glob("C:/ICCAD_ml/floorset_lite/worker_*/layouts_*.th"))
NSH = int(sys.argv[1]) if len(sys.argv) > 1 else 4
acc = {k: defaultdict(lambda: [0.0, 0]) for k in ("m", "w", "pm", "pw")}
nlay = 0
wset = set()


def add(key, k, vals):
    if vals.numel():
        a = acc[key][k]
        a[0] += float(vals.sum()); a[1] += vals.numel()


for sh in SHARDS[:NSH]:
    meta, b2b, p2b, pins, tree, fp, metrics = torch.load(sh, weights_only=False)
    for b in range(meta.shape[0]):
        n = int((meta[b, :, 0] > 0).sum())
        if n < 5:
            continue
        c = torch.stack([fp[b, :n, 2] + fp[b, :n, 0] / 2,
                         fp[b, :n, 3] + fp[b, :n, 1] / 2], 1)
        D = torch.cdist(c, c, p=1)
        mx = D.max()
        if mx <= 0:
            continue
        U = D / mx
        M = torch.zeros(n, n)
        Ws = torch.zeros(n, n)
        e = b2b[b]
        e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0) & (e[:, 0] < n) & (e[:, 1] < n)]
        e = e[e[:, 0] != e[:, 1]]
        if e.numel():
            i = e[:, 0].long(); j = e[:, 1].long()
            lo = torch.minimum(i, j); hi = torch.maximum(i, j)
            M.index_put_((lo, hi), torch.ones(len(lo)), accumulate=True)
            Ws.index_put_((lo, hi), e[:, 2], accumulate=True)
            wset.update(e[:, 2].unique().tolist())
        iu = torch.triu_indices(n, n, 1)
        m = M[iu[0], iu[1]]; ws = Ws[iu[0], iu[1]]; u = U[iu[0], iu[1]]
        for k in range(5):
            add("m", k, u[(m == k) if k < 4 else (m >= 4)])
        wm = torch.where(m > 0, ws / m.clamp(min=1), torch.zeros_like(ws))
        pos = wm[m > 0]
        if pos.numel():
            base = pos.min()
            kk = torch.where(m > 0, (wm / base).round(), torch.zeros_like(wm))
            for kv in kk[m > 0].unique():
                if kv <= 12:
                    add("w", int(kv), u[(m > 0) & (kk == kv)])
        # pins
        P = pins[b]
        keep = (P[:, 0] >= 0)
        P = P[keep]
        if len(P) and n:
            Dp = torch.cdist(P.float(), c, p=1)
            mxp = Dp.max()
            if mxp > 0:
                Up = Dp / mxp
                Mp = torch.zeros(len(P), n); Wp = torch.zeros(len(P), n)
                pe = p2b[b]
                pe = pe[(pe[:, 0] >= 0) & (pe[:, 1] >= 0) & (pe[:, 1] < n)
                        & (pe[:, 0] < len(P))]
                if pe.numel():
                    Mp.index_put_((pe[:, 0].long(), pe[:, 1].long()),
                                  torch.ones(len(pe)), accumulate=True)
                    Wp.index_put_((pe[:, 0].long(), pe[:, 1].long()),
                                  pe[:, 2], accumulate=True)
                mf = Mp.flatten(); uf = Up.flatten(); wf = Wp.flatten()
                for k in range(4):
                    add("pm", k, uf[(mf == k) if k < 3 else (mf >= 3)])
                wmp = torch.where(mf > 0, wf / mf.clamp(min=1), torch.zeros_like(wf))
                posp = wmp[mf > 0]
                if posp.numel():
                    bp = posp.min()
                    kp = torch.where(mf > 0, (wmp / bp).round(), torch.zeros_like(wmp))
                    for kv in kp[mf > 0].unique():
                        if kv <= 12:
                            add("pw", int(kv), uf[(mf > 0) & (kp == kv)])
        nlay += 1

law = {}
print("== L327 the generative law, learned from %d training layouts ==" % nlay)
for key, title in (("m", "b2b MULTIPLICITY -> u"), ("w", "b2b WEIGHT -> u"),
                   ("pm", "p2b MULTIPLICITY -> u_pin"), ("pw", "p2b WEIGHT -> u_pin")):
    print("\n   %s" % title)
    law[key] = {}
    for k in sorted(acc[key]):
        s, c = acc[key][k]
        if c < 500:
            continue
        law[key][k] = s / c
        print("      %-6s n=%-10d E[u] = %.4f" % (k, c, s / c))
json.dump({"law": law, "nlay": nlay, "weights": sorted(wset)[:20]},
          open("l327_law.json", "w"), indent=1)
print("\n   saved -> l327_law.json     (weights seen: %s)" % sorted(wset)[:10])
