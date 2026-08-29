"""L322: is the label layout the fixed point of a constructive rule?

If every label block is jammed against something on its left AND below, the layout is
BOTTOM-LEFT STABLE -- consistent with bottom-left-fill (BLF) packing. That matters
because it collapses the reconstruction problem: instead of searching free real
coordinates, one searches (shape choice, insertion order) for a DETERMINISTIC packer,
and L321 showed the shape choice is a median of 2 options.

NOTE this is NOT what M26 tested. M26 injected the label's ORDER into THIS project's
packer -- a fixed-outline greedy with candidate scoring over trial frames, on floats.
It never tested whether the label is the output of a bottom-left rule on an integer
grid with exact-area divisor-pair shapes.
"""
import glob
import sys

import torch

LAB = sorted(glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth"))
DAT = sorted(glob.glob("LiteTensorDataTest/config_*/litedata_1.pth"))


def case(i):
    _m, poly = torch.load(LAB[i], weights_only=False)[0]
    d = torch.load(DAT[i], weights_only=False)[0][0]
    n = int((d[:, 0] > 0).sum())
    rs = []
    for k in range(n):
        p = poly[k]
        v = p[p[:, 0] != -1]
        x0, y0 = v.min(dim=0).values.tolist()
        x1, y1 = v.max(dim=0).values.tolist()
        rs.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return rs, d[:n]


def ov(a, b):
    return (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
            and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])


def stable(rs, axis):
    """how many blocks CANNOT move one unit toward the origin along `axis`"""
    stuck = 0
    for k, r in enumerate(rs):
        if r[axis] == 0:
            stuck += 1
            continue
        m = list(r)
        m[axis] -= 1
        if any(ov(tuple(m), o) for j, o in enumerate(rs) if j != k):
            stuck += 1
    return stuck


def contacts(rs):
    """fraction of blocks whose left edge abuts something, and same for bottom"""
    L = B = 0
    for k, r in enumerate(rs):
        x, y, w, h = r
        if x == 0 or any(o[0] + o[2] == x and o[1] < y + h and y < o[1] + o[3]
                         for j, o in enumerate(rs) if j != k):
            L += 1
        if y == 0 or any(o[1] + o[3] == y and o[0] < x + w and x < o[0] + o[2]
                         for j, o in enumerate(rs) if j != k):
            B += 1
    return L, B


N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
tx = ty = tL = tB = tot = 0
both = 0
per = []
for i in range(N):
    rs, d = case(i)
    n = len(rs)
    sx, sy = stable(rs, 0), stable(rs, 1)
    L, B = contacts(rs)
    tx += sx; ty += sy; tL += L; tB += B; tot += n
    bothk = sum(1 for k, r in enumerate(rs)
                if (r[0] == 0 or any(o[0] + o[2] == r[0] and o[1] < r[1] + r[3] and r[1] < o[1] + o[3]
                                     for j, o in enumerate(rs) if j != k))
                and (r[1] == 0 or any(o[1] + o[3] == r[1] and o[0] < r[0] + r[2] and r[0] < o[0] + o[2]
                                      for j, o in enumerate(rs) if j != k)))
    both += bothk
    per.append(bothk / n)

print("== L322 is the label a constructive fixed point?  %d cases, %d blocks ==" % (N, tot))
print("   cannot move LEFT by 1  : %5d/%d  (%.1f %%)" % (tx, tot, 100 * tx / tot))
print("   cannot move DOWN by 1  : %5d/%d  (%.1f %%)" % (ty, tot, 100 * ty / tot))
print("   left edge ABUTS something or x=0 : %5d/%d  (%.1f %%)" % (tL, tot, 100 * tL / tot))
print("   bottom edge ABUTS or y=0         : %5d/%d  (%.1f %%)" % (tB, tot, 100 * tB / tot))
print("   BOTH (bottom-left supported)     : %5d/%d  (%.1f %%)" % (both, tot, 100 * both / tot))
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print("   per-case bottom-left-supported fraction: p10 %.3f  p50 %.3f  p90 %.3f  min %.3f"
      % (q(per, .1), q(per, .5), q(per, .9), min(per)))
