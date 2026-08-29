"""L332: what do 1M-LEARNED shapes buy in the L329 pipeline?

L331: the shape channel does NOT suffer Jimmy's distribution shift --
    rank0/rank1/rank2+   1M 0.368/0.469/0.163   validation 0.372/0.458/0.170
    landscape            0.619 vs 0.606
    a 1M-learned rule scores 0.5240 in-sample and 0.5082 HELD OUT (-3 % relative)
whereas the pin channel differs 6x in density and 2.4x in informativeness.

So the question is not whether shapes transfer -- they do -- but whether 51 %
accuracy is enough, given L329b measured label shapes at hpwl 0.047 and squarest
at 0.216.
"""
import glob
import math
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, "iccad2026contest")
from iccad2026_evaluate import (calculate_hpwl_b2b, calculate_hpwl_p2b,  # noqa: E402
                                calculate_bbox_area)
import l329_realise as R                                              # noqa: E402
import l328_invert as INV                                             # noqa: E402
import l331_shape_transfer as S                                       # noqa: E402

TR = S.TR
tab = defaultdict(lambda: defaultdict(int))
for A, c, w, h in TR:
    r, m = S.rank_of(A, w, h)
    if r is None:
        continue
    tab[S.key(A, c, m)][r] += 1
PRED = {k: max(v, key=v.get) for k, v in tab.items()}


def learned(A, c):
    ps = S.divpairs(A)
    if not ps:
        s = math.sqrt(A)
        return s, s
    m = len(ps)
    r = PRED.get(S.key(A, c, m), 0)
    return float(ps[min(r, m - 1)][0]), float(ps[min(r, m - 1)][1])


tot = defaultdict(lambda: [0.0, 0.0])
cnt = 0
for n in sorted(R.LAB):
    meta, b2b, p2b, pins, C = INV.load(n)
    lab = torch.load(R.LAB[n], weights_only=False)[0]
    m8, poly = lab[0], lab[1]
    hpb = float(m8[-2]) + float(m8[-1]); arb = float(m8[0])
    LW, LH, SW, SH, MW, MH = [], [], [], [], [], []
    for k in range(n):
        p = poly[k]; v = p[p[:, 0] != -1]
        x0, y0 = v.min(dim=0).values.tolist(); x1, y1 = v.max(dim=0).values.tolist()
        LW.append(x1 - x0); LH.append(y1 - y0)
        A = float(meta[k, 0]); c = meta[k, 1:].tolist()
        a, b = R.squarest(A); SW.append(a); SH.append(b)
        a, b = learned(A, c); MW.append(a); MH.append(b)
    labc = [(float(C[k, 0]), float(C[k, 1])) for k in range(n)]
    est = INV.solve(n, meta, b2b, p2b, pins)
    invc = [(float(est[k, 0]), float(est[k, 1])) for k in range(n)]
    for tag, (pts, W, H) in (("lab_c + lab_s", (labc, LW, LH)),
                             ("lab_c + squarest", (labc, SW, SH)),
                             ("lab_c + LEARNED", (labc, MW, MH)),
                             ("inv_c + squarest", (invc, SW, SH)),
                             ("inv_c + LEARNED", (invc, MW, MH))):
        x, y = R.decode(pts, W, H)
        pos = [(x[k], y[k], W[k], H[k]) for k in range(n)]
        hp = calculate_hpwl_b2b(pos, b2b) + calculate_hpwl_p2b(pos, p2b, pins)
        tot[tag][0] += max(0.0, (hp - hpb) / hpb)
        tot[tag][1] += max(0.0, (calculate_bbox_area(pos) - arb) / arb)
    cnt += 1

print("\n== L332 shapes learned on the 1M, in the L329 pipeline (%d cases) ==" % cnt)
print("   %-20s %9s %9s" % ("", "hpwl_gap", "area_gap"))
for k in ("lab_c + lab_s", "lab_c + LEARNED", "lab_c + squarest",
          "inv_c + LEARNED", "inv_c + squarest"):
    print("   %-20s %9.4f %9.4f" % (k, tot[k][0] / cnt, tot[k][1] / cnt))
print("   %-20s %9.4f %9.4f" % ("our shipped placer", 0.2402, 0.1176))
