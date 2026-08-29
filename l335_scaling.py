"""L335: how far does the B*-tree manifold go with compute?

L334: utilisation climbs monotonically with SA iterations and has NOT plateaued --
n=40 goes 0.790 -> 0.850 -> 0.9145 at 2k/10k/40k, n=80 goes 0.733 -> 0.769 -> 0.868.
At 40k, n=40 is ALREADY above our own placer's 0.877 and above the 85.4 % ceiling
L284 declared for the density axis.

That matters more than the number: L284's ceiling is a property of OUR packer's
reachable set, and this is a different reachable set. So the question is where the
curve goes, not whether it beats us.

(L334's fixed-outline arm was NOT an iteration-fair comparison -- it split its budget
across four aspect ratios, so each got a quarter. It is re-run fairly here.)
"""
import math, sys, time
import torch
from l333_btree_sa import DAT, LAB, divpairs
from l334_fixed_outline import anneal

NS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "40,80,120").split(",")]
ITS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "40000,160000").split(",")]
print("== L335 how far does the manifold go with compute? ==")
print("   %-5s %9s %11s %11s %11s %9s" % ("n","iters","label util","SA util","vs ours .877","time"))
for n in NS:
    meta = torch.load(DAT[n], weights_only=False)[0][0]
    _m, poly = torch.load(LAB[n], weights_only=False)[0]
    nb = int((meta[:,0] > 0).sum())
    shapes = [divpairs(float(meta[k,0])) for k in range(nb)]
    sumA = sum(int(round(float(meta[k,0]))) for k in range(nb))
    lw = lh = 0
    for k in range(nb):
        p = poly[k]; v = p[p[:,0] != -1]
        x1, y1 = v.max(dim=0).values.tolist()
        lw = max(lw,x1); lh = max(lh,y1)
    lu = sumA/(lw*lh)
    for it in ITS:
        t0 = time.time()
        W,H = anneal(nb, shapes, it, 7+n)
        u = sumA/(W*H)
        print("   %-5d %9d %11.4f %11.4f %11s %8.1fs"
              % (n, it, lu, u, "%+.4f" % (u-0.877), time.time()-t0))
