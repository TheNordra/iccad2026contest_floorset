"""L328 step B: invert the generative model into a POSITION TARGET, and score it.

The law (L327, 336 training layouts) turns each observable into an expected
normalised distance:

    b2b weight k=1..12  ->  u = 0.379 ... 0.092      (monotone at the top end)
    b2b multiplicity 0/1/2 -> 0.391 / 0.231 / 0.191
    p2b edge present    ->  u_pin = 0.0514  against a 0.4341 baseline  (8.4x)

u is normalised per layout by the max pairwise distance, so it needs a scale. Both
scales are predicted from the INPUT only: the layout bbox area is Sum(A)/0.971
(L320: label utilisation p50 0.9693, and w*h = A exactly), and the pin ring bounds
the layout in 100/100 cases.

Then solve for the centres by minimising L1 stress
    sum_ij  s_ij * ( |xi-xj| + |yi-yj| - d_ij )^2
with the pin terms supplying absolute position -- the pins are at known absolute
coordinates, so this is trilateration, not a free embedding.

Scored against the label's true centres, next to the same error for OUR OWN placer.
"""
import glob
import json
import math
import os
import sys

import torch

LAW = json.load(open("l327_law.json"))["law"]
UB_W = {int(k): v for k, v in LAW["w"].items()}
UB_M = {int(k): v for k, v in LAW["m"].items()}
UP = {int(k): v for k, v in LAW["pm"].items()}
UTIL = 0.971

LAB = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth")}
DAT = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litedata_1.pth")}
OURS = {t["block_count"]: t for t in json.load(open("l302_mix_1.json"))["test_results"]}


def load(n):
    _m, poly = torch.load(LAB[n], weights_only=False)[0]
    d = torch.load(DAT[n], weights_only=False)[0]
    meta, b2b, p2b, pins = d[0], d[1], d[2], d[3]
    C = []
    for k in range(n):
        p = poly[k]
        v = p[p[:, 0] != -1]
        x0, y0 = v.min(dim=0).values.tolist()
        x1, y1 = v.max(dim=0).values.tolist()
        C.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return meta[:n], b2b, p2b, pins, torch.tensor(C)


def solve(n, meta, b2b, p2b, pins, iters=600, lr=0.05):
    A = meta[:, 0]
    bbox_area = float(A.sum()) / UTIL
    P = pins[pins[:, 0] >= 0].float()
    if len(P) < 2:
        return None
    pw = float(P[:, 0].max() - P[:, 0].min()) or 1.0
    ph = float(P[:, 1].max() - P[:, 1].min()) or 1.0
    # outline: the pin ring's aspect, rescaled to the predicted area
    s = math.sqrt(bbox_area / (pw * ph))
    W, H = pw * s, ph * s
    mx = 0.85 * (W + H)                      # max pairwise block-centre distance
    corners = torch.tensor([[0.0, 0.0], [W, 0.0], [0.0, H], [W, H]])
    mxp = float(torch.cdist(P, corners, p=1).max())

    tgt, wt = [], []
    e = b2b[(b2b[:, 0] >= 0) & (b2b[:, 1] >= 0) & (b2b[:, 0] < n) & (b2b[:, 1] < n)]
    e = e[e[:, 0] != e[:, 1]]
    if e.numel():
        base = e[:, 2][e[:, 2] > 0].min()
        agg = {}
        for r in e:
            i, j, w = int(r[0]), int(r[1]), float(r[2])
            k = (min(i, j), max(i, j))
            a = agg.get(k, [0, 0.0]); a[0] += 1; a[1] += w; agg[k] = a
        for (i, j), (m, wsum) in agg.items():
            kk = max(1, min(12, int(round((wsum / m) / float(base)))))
            u = 0.5 * (UB_W.get(kk, UB_M.get(1, .23)) + UB_M.get(min(m, 2), .23))
            tgt.append((i, j, u * mx)); wt.append(1.0)
    pe = p2b[(p2b[:, 0] >= 0) & (p2b[:, 1] >= 0) & (p2b[:, 1] < n) & (p2b[:, 0] < len(P))]
    pin_terms = [(int(r[0]), int(r[1]), UP.get(1, .05) * mxp) for r in pe]

    c = torch.rand(n, 2) * torch.tensor([W, H])
    c.requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr * max(W, H) / 20)
    I = torch.tensor([t[0] for t in tgt]) if tgt else None
    Jv = torch.tensor([t[1] for t in tgt]) if tgt else None
    Dv = torch.tensor([t[2] for t in tgt]) if tgt else None
    PI = torch.tensor([t[0] for t in pin_terms]) if pin_terms else None
    PB = torch.tensor([t[1] for t in pin_terms]) if pin_terms else None
    PD = torch.tensor([t[2] for t in pin_terms]) if pin_terms else None
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.zeros(())
        if I is not None:
            d = (c[I] - c[Jv]).abs().sum(1)
            loss = loss + ((d - Dv) ** 2).mean()
        if PI is not None:
            d = (P[PI] - c[PB]).abs().sum(1)
            loss = loss + 3.0 * ((d - PD) ** 2).mean()      # pins are 8.4x sharper
        loss.backward()
        opt.step()
    return c.detach()


tot_i = tot_o = 0.0
cnt = wins = 0
for n in sorted(LAB):
    meta, b2b, p2b, pins, C = load(n)
    est = solve(n, meta, b2b, p2b, pins)
    if est is None:
        continue
    W = float(C[:, 0].max() - C[:, 0].min()); H = float(C[:, 1].max() - C[:, 1].min())
    scale = (W + H) or 1.0
    r = OURS.get(n)
    if not r or len(r.get("positions", [])) < n:
        continue
    ours = torch.tensor([[p[0] + p[2] / 2, p[1] + p[3] / 2] for p in r["positions"][:n]])
    ours = ours + (C.mean(0) - ours.mean(0))          # best translation for our layout
    e_i = float((est - C).abs().sum(1).mean()) / scale
    e_o = float((ours - C).abs().sum(1).mean()) / scale
    tot_i += e_i; tot_o += e_o; cnt += 1; wins += e_i < e_o

print("== L328 netlist inversion as a position target (%d cases) ==" % cnt)
print("   inverted generative model : mean error %.4f of bbox half-perimeter" % (tot_i / cnt))
print("   our shipped placer        : mean error %.4f" % (tot_o / cnt))
print("   L326 crude harmonic       : mean error 0.1267  (no distance info at all)")
print("   inversion closer on %d/%d cases" % (wins, cnt))
