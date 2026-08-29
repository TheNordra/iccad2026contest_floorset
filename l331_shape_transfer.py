"""L331: can the SHAPE choice be learned on the 1M and used on the graded shape?

L330 confirms Jimmy: the public 1M and the validation 100 are NOT the same
distribution -- but the difference is entirely in the PIN (T2B) channel:

    p2b edges per block      8.50 (val)  vs  1.38 (1M)   6.2x
    p2b blocks covered       0.622       vs  0.262       2.4x
    p2b u | connected        0.0875      vs  0.0368      2.4x   <- 1M pins are
                                                                 far more informative
    b2b weight-vs-distance r -0.5564     vs  -0.5629     0.988  <- the LAW is stable
    utilisation              0.9708      vs  0.9737      0.997

So the netlist-to-geometry MECHANISM is the same; the SAMPLING DENSITY is not.
A model trained on the 1M will over-trust the pin channel -- which is exactly the
"in-set good, beta bad" failure mode.

But SHAPE selection does not read the netlist at all. It maps (area, constraints)
-> which integer divisor pair. This probe trains on the 1M and tests on the
validation 100, i.e. straight across the gap L330 just measured.
"""
import glob
import math
import os
import sys
from collections import defaultdict

import torch


def divpairs(A, lo=1 / 3, hi=3.0):
    A = int(round(A))
    out = []
    w = 1
    while w * w <= A:
        if A % w == 0:
            for a, b in ((w, A // w), (A // w, w)):
                if lo - 1e-12 <= a / b <= hi + 1e-12:
                    out.append((a, b))
        w += 1
    return sorted(set(out), key=lambda p: (max(p[0] / p[1], p[1] / p[0]), p[0]))


def harvest_train(nsh):
    rows = []
    for sh in sorted(glob.glob("C:/ICCAD_ml/floorset_lite/worker_*/layouts_*.th"))[:nsh]:
        meta, b2b, p2b, pins, tree, fp, metrics = torch.load(sh, weights_only=False)
        for b in range(meta.shape[0]):
            n = int((meta[b, :, 0] > 0).sum())
            for k in range(n):
                A = float(meta[b, k, 0])
                w, h = float(fp[b, k, 0]), float(fp[b, k, 1])
                rows.append((A, meta[b, k, 1:].tolist(), w, h))
    return rows


def harvest_val():
    rows = []
    for f in sorted(glob.glob("LiteTensorDataTest/config_*/litedata_1.pth")):
        meta = torch.load(f, weights_only=False)[0][0]
        _m, poly = torch.load(f.replace("litedata", "litelabel"), weights_only=False)[0]
        n = int((meta[:, 0] > 0).sum())
        for k in range(n):
            p = poly[k]; v = p[p[:, 0] != -1]
            x0, y0 = v.min(dim=0).values.tolist(); x1, y1 = v.max(dim=0).values.tolist()
            rows.append((float(meta[k, 0]), meta[k, 1:].tolist(), x1 - x0, y1 - y0))
    return rows


def rank_of(A, w, h):
    ps = divpairs(A)
    for i, (a, b) in enumerate(ps):
        if abs(a - w) < 1e-6 and abs(b - h) < 1e-6:
            return i, len(ps)
    return None, len(ps)


TR = harvest_train(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
VA = harvest_val()
print("== L331 shape-choice transfer: train on the 1M, test on the graded shape ==")
print("   train blocks %d   test blocks %d" % (len(TR), len(VA)))

# what does the label choose, as a rank among divisor pairs sorted by squareness?
def profile(rows, tag):
    rk = defaultdict(int)
    land = 0; tot = 0; multi = 0
    for A, c, w, h in rows:
        r, m = rank_of(A, w, h)
        if r is None:
            continue
        rk[min(r, 4)] += 1
        tot += 1
        if m > 1:
            multi += 1
        land += (w >= h)
    print("   %-12s rank0(squarest) %.3f  rank1 %.3f  rank2+ %.3f | landscape %.3f | "
          "blocks with a real choice %.3f"
          % (tag, rk[0] / tot, rk[1] / tot, sum(v for k, v in rk.items() if k >= 2) / tot,
             land / tot, multi / tot))
    return rk, tot


profile(TR, "1M train")
profile(VA, "validation")

# ---- the predictor: squarest, vs a rule learned on the 1M -----------------
# learn P(rank | #options, constraint flags, area magnitude) on the 1M
def key(A, c, m):
    return (min(m, 4), int(c[0] > 0), int(c[1] > 0), int(c[2] > 0), int(c[3] > 0),
            int(c[4] > 0), min(int(math.log10(max(A, 1))), 4))


tab = defaultdict(lambda: defaultdict(int))
for A, c, w, h in TR:
    r, m = rank_of(A, w, h)
    if r is None:
        continue
    tab[key(A, c, m)][r] += 1
pred = {k: max(v, key=v.get) for k, v in tab.items()}

def orient_key(A, c, m):
    return (min(m, 4), int(c[4]) if len(c) > 4 else 0)

ori = defaultdict(lambda: [0, 0])
for A, c, w, h in TR:
    _r, m = rank_of(A, w, h)
    o = ori[orient_key(A, c, m)]
    o[0] += (w >= h); o[1] += 1
oripred = {k: (v[0] / v[1] >= .5) for k, v in ori.items()}

for name, rows in (("1M (in-sample)", TR), ("validation (HELD OUT)", VA)):
    sq = ex = tot = choice = sq_c = ex_c = 0
    for A, c, w, h in rows:
        ps = divpairs(A)
        if not ps:
            continue
        r, m = rank_of(A, w, h)
        if r is None:
            continue
        tot += 1
        s = (r == 0)
        p = pred.get(key(A, c, m), 0)
        e = (r == p)
        sq += s; ex += e
        if m > 1:
            choice += 1; sq_c += s; ex_c += e
    print("   %-22s squarest %.4f | 1M-learned rule %.4f | (on blocks WITH a choice: "
          "%.4f vs %.4f, n=%d)"
          % (name, sq / tot, ex / tot, sq_c / max(choice, 1), ex_c / max(choice, 1), choice))
