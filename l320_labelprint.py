"""L320: fingerprint the LABEL generator from the labels themselves.

The score is a GAP against fp_sol, so the generator's inductive bias IS the target.
M40 asked whether a layout can be reconstructed from CONNECTIVITY (no). This asks a
different question: what regularities do the labels themselves carry, and do they
restrict the search space enough to be exploitable?
"""
import glob
import math
import sys
from collections import Counter

import torch


def load(i):
    f = sorted(glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth"))[i]
    metrics, poly = torch.load(f, weights_only=False)[0]
    g = sorted(glob.glob("LiteTensorDataTest/config_*/litedata_1.pth"))[i]
    data = torch.load(g, weights_only=False)[0]
    return metrics, poly, data[0]        # [100, 6] = area | 5 constraint cols


def rects(poly, n):
    out = []
    for k in range(n):
        p = poly[k]
        v = p[p[:, 0] != -1]
        x0, y0 = v.min(dim=0).values.tolist()
        x1, y1 = v.max(dim=0).values.tolist()
        out.append((x0, y0, x1, y1))
    return out


def is_slicing(rs, eps=1e-9):
    """Can this rectangle set be recursively cut by a full horizontal/vertical line?"""
    def rec(items, depth=0):
        if len(items) <= 1:
            return True
        if depth > 400:
            return False
        for axis in (0, 1):
            lo = min(r[axis] for r in items)
            hi = max(r[axis + 2] for r in items)
            cuts = sorted({r[axis + 2] for r in items if lo + eps < r[axis + 2] < hi - eps})
            for c in cuts:
                a = [r for r in items if r[axis + 2] <= c + eps]
                b = [r for r in items if r[axis] >= c - eps]
                if len(a) + len(b) == len(items) and a and b:
                    if rec(a, depth + 1) and rec(b, depth + 1):
                        return True
        return False
    return rec(list(rs))


N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
allint = slic = 0
utils, ars, arem, gaps, touch4 = [], [], [], [], 0
nb = []
for i in range(N):
    metrics, poly, data = load(i)
    area_t = data[:, 0]
    n = int((area_t > 0).sum())
    rs = rects(poly, n)
    nb.append(n)
    ints = all(float(v) == int(v) for r in rs for v in r)
    allint += ints
    W = max(r[2] for r in rs) - min(r[0] for r in rs)
    H = max(r[3] for r in rs) - min(r[1] for r in rs)
    used = sum((r[2] - r[0]) * (r[3] - r[1]) for r in rs)
    utils.append(used / (W * H))
    ars.append(max(W, H) / min(W, H))
    for r in rs:
        w, h = r[2] - r[0], r[3] - r[1]
        if w > 0 and h > 0:
            arem.append(max(w, h) / min(w, h))
    at = area_t.tolist()
    for k, r in enumerate(rs):
        if k < len(at) and at[k] > 0:
            gaps.append(((r[2] - r[0]) * (r[3] - r[1]) - at[k]) / at[k])
    if n <= 60:                       # slicing test is exponential; sample the small ones
        slic += is_slicing(rs)
    x0 = min(r[0] for r in rs); y0 = min(r[1] for r in rs)
    touch4 += (any(r[0] == x0 for r in rs) and any(r[1] == y0 for r in rs))

q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print("== L320 label fingerprint, %d cases (n = %d..%d) ==" % (N, min(nb), max(nb)))
print("   ALL COORDINATES INTEGER      : %d/%d cases" % (allint, N))
print("   utilisation  p10 %.4f  p50 %.4f  p90 %.4f  max %.4f"
      % (q(utils, .1), q(utils, .5), q(utils, .9), max(utils)))
print("   bbox aspect  p10 %.3f  p50 %.3f  p90 %.3f" % (q(ars, .1), q(ars, .5), q(ars, .9)))
print("   block aspect p10 %.3f  p50 %.3f  p90 %.3f  max %.2f"
      % (q(arem, .1), q(arem, .5), q(arem, .9), max(arem)))
print("   area vs target: p10 %+.5f  p50 %+.5f  p90 %+.5f  |max| %.5f"
      % (q(gaps, .1), q(gaps, .5), q(gaps, .9), max(abs(g) for g in gaps)))
sm = sum(1 for x in nb if x <= 60)
print("   SLICING (guillotine)         : %d/%d of the n<=60 cases" % (slic, sm))
print("   bbox has a block on x=min and y=min: %d/%d" % (touch4, N))
