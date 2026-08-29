"""L324: verify that the netlist is an OBSERVATION of the label's geometry.

The FloorSet paper (arXiv 2405.05480, Algorithm 4) says the netlist is sampled FROM
the finished layout: PSim = 1 - Normalize(pairwiseB2BDistance(F)), then edges and
weights are drawn with probability proportional to PSim. If true, the netlist is not
a specification to be satisfied -- it is a noisy readout of the answer key.

Three independent checks, all against the 100 validation labels.
"""
import glob
import math

import torch

LAB = sorted(glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth"))
DAT = sorted(glob.glob("LiteTensorDataTest/config_*/litedata_1.pth"))


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


def pear(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) ** .5
    vb = sum((x - mb) ** 2 for x in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb + 1e-30)


rs_edge, rs_pin = [], []
conc, base = [], []
pin_err, naive_err = [], []
for i in range(100):
    n, C, b2b, p2b, pins = case(i)
    D = lambda a, b: abs(C[a][0] - C[b][0]) + abs(C[a][1] - C[b][1])
    # 1. does b2b WEIGHT anticorrelate with label centre distance?
    d, w = [], []
    for e in b2b:
        a, b, ww = int(e[0]), int(e[1]), float(e[2])
        if a < 0 or b < 0 or a >= n or b >= n or a == b:
            continue
        d.append(D(a, b)); w.append(ww)
    if len(d) > 20:
        rs_edge.append(pear(d, w))
    # 2. are CONNECTED pairs closer than random pairs?
    alld = [abs(C[a][0] - C[b][0]) + abs(C[a][1] - C[b][1])
            for a in range(n) for b in range(a + 1, n)]
    mx = max(alld) or 1
    conc.append(sum(d) / len(d) / mx)
    base.append(sum(alld) / len(alld) / mx)
    # 3. does a p2b-weighted pin centroid predict the label block centre?
    acc = {}
    for e in p2b:
        pi, bi, ww = int(e[0]), int(e[1]), float(e[2])
        if pi < 0 or bi < 0 or bi >= n or pi >= len(pins):
            continue
        px, py = float(pins[pi][0]), float(pins[pi][1])
        sx, sy, sw = acc.get(bi, (0.0, 0.0, 0.0))
        acc[bi] = (sx + ww * px, sy + ww * py, sw + ww)
    cx = sum(c[0] for c in C) / n
    cy = sum(c[1] for c in C) / n
    for bi, (sx, sy, sw) in acc.items():
        if sw <= 0:
            continue
        ex, ey = sx / sw, sy / sw
        pin_err.append(abs(ex - C[bi][0]) + abs(ey - C[bi][1]))
        naive_err.append(abs(cx - C[bi][0]) + abs(cy - C[bi][1]))

q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print("== L324 is the netlist a readout of the label's geometry? ==")
print("   1. b2b WEIGHT vs label centre distance, Pearson r per case:")
print("      p10 %+.3f   p50 %+.3f   p90 %+.3f   cases with r<0: %d/100"
      % (q(rs_edge, .1), q(rs_edge, .5), q(rs_edge, .9), sum(1 for r in rs_edge if r < 0)))
print("   2. mean distance, CONNECTED pairs vs ALL pairs (normalised):")
print("      connected p50 %.4f   all-pairs p50 %.4f   concentration %.2fx   better in %d/100"
      % (q(conc, .5), q(base, .5), q(base, .5) / q(conc, .5),
         sum(1 for a, b in zip(conc, base) if a < b)))
print("   3. p2b-weighted PIN CENTROID as a predictor of the label block centre:")
print("      %d blocks with p2b nets;  centroid error p50 %.2f   vs 'guess the middle' p50 %.2f"
      % (len(pin_err), q(pin_err, .5), q(naive_err, .5)))
print("      ratio of means: %.3f   (1.0 = no information)"
      % (sum(pin_err) / len(pin_err) / (sum(naive_err) / len(naive_err))))
