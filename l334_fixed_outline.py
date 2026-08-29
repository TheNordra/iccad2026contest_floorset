"""L334: is the 79.6 % of L333 a compute limit or the wrong objective?

L333 replayed the generator as "B*-tree SA minimising bounding-box area" and reached
only 0.7957 utilisation -- below our own placer's 0.877 and far below the label's
0.967. Two candidate causes, and they are separable:

  (a) too few iterations (1500 is nothing for SA at n=60);
  (b) the wrong objective. Parquet is a FIXED-OUTLINE floorplanner -- that is the
      whole Adya-Markov contribution -- and the FloorSet generator samples the
      outline FIRST (Algorithm 6 line 5: `W,H <- sampleOutline(...)`, then
      `runSA(W,H,...)`). Minimising area is a different and much worse-conditioned
      search than fitting a given box.

We can predict the outline without any label: L320 measured label utilisation at
p50 0.9693, and since w*h = A exactly, the target area is Sum(A)/0.971 to about
+-1 %. Only the aspect is unknown, so it is swept.

Objective (b) is Adya-Markov's: minimise the overflow outside the target box,
tie-broken by area, so the search has a gradient toward "fits" rather than
toward "smaller".
"""
import math
import random
import sys
import time

import torch

from l333_btree_sa import (DAT, LAB, attach, detach, divpairs, pack_ref,
                           rand_tree)


def anneal(n, shapes, iters, seed, target=None):
    """target=None -> minimise area (L333). target=(W,H) -> minimise overflow."""
    rng = random.Random(seed)
    L, R, par, root = rand_tree(n, rng)
    si = [0] * n

    def cost():
        W, H, _x, _y, _p = pack_ref(L, R, si, shapes, root, n)
        if target is None:
            return W * H, W, H
        tw, th = target
        over = max(0, W - tw) * th + max(0, H - th) * tw
        return over * 4 + W * H, W, H

    cur, W, H = cost()
    best = cur
    bestWH = (W, H)
    bestT = (L[:], R[:], par[:], si[:], root)
    T0 = max(cur * .05, 1.0)
    for it in range(iters):
        T = T0 * (1 - it / iters) ** 2 + 1e-9
        sL, sR, sP, sS, sRoot = L[:], R[:], par[:], si[:], root
        m = rng.random()
        if m < .40:
            k = rng.randrange(n)
            if len(shapes[k]) > 1:
                si[k] = rng.randrange(len(shapes[k]))
        elif m < .70:
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                for arr in (L, R):
                    for i in range(n):
                        if arr[i] == a:
                            arr[i] = b
                        elif arr[i] == b:
                            arr[i] = a
                par[a], par[b] = par[b], par[a]
                L[a], L[b] = L[b], L[a]
                R[a], R[b] = R[b], R[a]
                for c in (L[a], R[a]):
                    if c != -1:
                        par[c] = a
                for c in (L[b], R[b]):
                    if c != -1:
                        par[c] = b
                if root == a:
                    root = b
                elif root == b:
                    root = a
        else:
            k = rng.randrange(n)
            nr = detach(L, R, par, root, k)
            if nr is None:
                L, R, par, si, root = sL, sR, sP, sS, sRoot
                continue
            root = nr
            p = rng.randrange(n)
            while p == k:
                p = rng.randrange(n)
            attach(L, R, par, k, p, rng.randrange(2), rng)
        new, W, H = cost()
        if new <= cur or rng.random() < math.exp(min(0.0, (cur - new) / T)):
            cur = new
            if new < best:
                best = new
                bestWH = (W, H)
                bestT = (L[:], R[:], par[:], si[:], root)
        else:
            L, R, par, si, root = sL, sR, sP, sS, sRoot
    return bestWH


if __name__ == "__main__":
    NS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "40,80").split(",")]
    ITS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2000,10000").split(",")]
    ASPECTS = [1.0, 1.25, 1.5, 1.8]
    print("== L334 area objective vs FIXED-OUTLINE objective (the generator's own) ==")
    print("   outline target area = sum(A)/0.971, label-free; aspect swept %s\n" % ASPECTS)
    print("   %-5s %8s %11s %12s %12s %8s"
          % ("n", "iters", "label util", "area-obj", "outline-obj", "time"))
    for n in NS:
        meta = torch.load(DAT[n], weights_only=False)[0][0]
        _m, poly = torch.load(LAB[n], weights_only=False)[0]
        nb = int((meta[:, 0] > 0).sum())
        shapes = [divpairs(float(meta[k, 0])) for k in range(nb)]
        sumA = sum(int(round(float(meta[k, 0]))) for k in range(nb))
        lw = lh = 0
        for k in range(nb):
            p = poly[k]
            v = p[p[:, 0] != -1]
            x1, y1 = v.max(dim=0).values.tolist()
            lw = max(lw, x1)
            lh = max(lh, y1)
        lu = sumA / (lw * lh)
        for it in ITS:
            t0 = time.time()
            W, H = anneal(nb, shapes, it, 7 + n)
            ua = sumA / (W * H)
            ub = 0.0
            box = sumA / 0.971
            for asp in ASPECTS:
                tw = int(round(math.sqrt(box * asp)))
                th = int(round(box / max(tw, 1)))
                W2, H2 = anneal(nb, shapes, it // len(ASPECTS), 7 + n, (tw, th))
                ub = max(ub, sumA / (W2 * H2))
            print("   %-5d %8d %11.4f %12.4f %12.4f %7.1fs"
                  % (n, it, lu, ua, ub, time.time() - t0))
