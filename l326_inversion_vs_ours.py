"""L326: does inverting the generative model locate the label's blocks BETTER than
our own placer does?

The FloorSet paper's Algorithm 4 samples the netlist FROM the finished layout:
p(edge) and its weight are both increasing in 1 - normalised centre distance. L324
verified all three consequences on the 100 validation labels (weight-vs-distance
r = -0.558 in 100/100; connected pairs 1.51x closer; pin centroid at 0.32x naive
error). So the netlist is a NOISY OBSERVATION of the answer, and the question is how
much of the answer it carries.

Estimator, deliberately crude and completely label-free:
  1. anchor: each block with p2b nets gets the weight-weighted centroid of its pins
  2. propagate: iteratively relax every block toward the b2b-weight-weighted mean of
     its neighbours, keeping anchors pinned (a weighted-Laplacian harmonic extension)
  3. score: mean Manhattan error against the label's true centres, normalised by the
     label's bbox half-perimeter

Compared against the SAME error for our own shipped layout's centres. If the crude
estimator is competitive with a mature placer, the signal is real and unexploited.
"""
import glob
import json

import torch

# the config directory name IS the block count; results are keyed by test_id, so
# join the two on block_count rather than on position
import os
LAB = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth")}
DAT = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litedata_1.pth")}
OURS = {t["block_count"]: t for t in json.load(open("l302_mix_1.json"))["test_results"]}
NS = sorted(LAB)


def case(i):
    _m, poly = torch.load(LAB[i], weights_only=False)[0]
    d = torch.load(DAT[i], weights_only=False)[0]
    meta, b2b, p2b, pins = d[0], d[1], d[2], d[3]
    n = int((meta[:, 0] > 0).sum())
    C = []
    for k in range(n):
        p = poly[k]
        v = p[p[:, 0] != -1]
        x0, y0 = v.min(dim=0).values.tolist()
        x1, y1 = v.max(dim=0).values.tolist()
        C.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return n, C, b2b, p2b, pins


def estimate(n, b2b, p2b, pins, iters=200):
    anch = {}
    for e in p2b:
        pi, bi, w = int(e[0]), int(e[1]), float(e[2])
        if pi < 0 or bi < 0 or bi >= n or pi >= len(pins):
            continue
        sx, sy, sw = anch.get(bi, (0.0, 0.0, 0.0))
        anch[bi] = (sx + w * float(pins[pi][0]), sy + w * float(pins[pi][1]), sw + w)
    P = {}
    for bi, (sx, sy, sw) in anch.items():
        if sw > 0:
            P[bi] = (sx / sw, sy / sw)
    if not P:
        return None
    mx = sum(p[0] for p in P.values()) / len(P)
    my = sum(p[1] for p in P.values()) / len(P)
    pos = [P.get(k, (mx, my)) for k in range(n)]
    adj = [[] for _ in range(n)]
    for e in b2b:
        a, b, w = int(e[0]), int(e[1]), float(e[2])
        if a < 0 or b < 0 or a >= n or b >= n or a == b:
            continue
        adj[a].append((b, w)); adj[b].append((a, w))
    for _ in range(iters):                       # harmonic extension, anchors pinned
        new = list(pos)
        for k in range(n):
            if k in P or not adj[k]:
                continue
            sw = sum(w for _j, w in adj[k])
            new[k] = (sum(w * pos[j][0] for j, w in adj[k]) / sw,
                      sum(w * pos[j][1] for j, w in adj[k]) / sw)
        pos = new
    return pos


tot_e, tot_o, tot_n = 0.0, 0.0, 0
wins = 0
for i in NS:
    n, C, b2b, p2b, pins = case(i)
    est = estimate(n, b2b, p2b, pins)
    if est is None:
        continue
    W = max(c[0] for c in C) - min(c[0] for c in C)
    H = max(c[1] for c in C) - min(c[1] for c in C)
    scale = (W + H) or 1.0
    r = OURS.get(n)
    if not r or not r.get("positions") or len(r["positions"]) < n:
        continue
    ours = [(p[0] + p[2] / 2, p[1] + p[3] / 2) for p in r["positions"][:n]]
    # our layout may sit anywhere; give it the best translation (Procrustes on mean)
    dx = sum(c[0] for c in C) / n - sum(o[0] for o in ours) / n
    dy = sum(c[1] for c in C) / n - sum(o[1] for o in ours) / n
    e_est = sum(abs(est[k][0] - C[k][0]) + abs(est[k][1] - C[k][1]) for k in range(n)) / n
    e_our = sum(abs(ours[k][0] + dx - C[k][0]) + abs(ours[k][1] + dy - C[k][1])
                for k in range(n)) / n
    tot_e += e_est / scale; tot_o += e_our / scale; tot_n += 1
    wins += e_est < e_our

print("== L326 who locates the label's blocks better? (%d cases) ==" % tot_n)
print("   netlist inversion (pins + b2b harmonic) : mean error %.4f of bbox half-perimeter"
      % (tot_e / tot_n))
print("   OUR shipped placer (best translation)   : mean error %.4f"
      % (tot_o / tot_n))
print("   the crude estimator is closer on %d/%d cases" % (wins, tot_n))
print()
print("   note: the estimator uses NO label information and no placement reasoning -")
print("   no overlap handling, no shapes, no constraints. It is pure netlist inversion.")
