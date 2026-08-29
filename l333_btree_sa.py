"""L333: replay the generator. B*-tree SA on the label's own space, AREA objective.

The FloorSet paper's Algorithm 6 is `runSA(W, H, n_parts, A_parts)` -- Parquet,
B*-tree simulated annealing, AREA only, with no connectivity in the argument list.
L325 verified the representation on the training shards: left child x = x_p + w_p
(4419/4419), right child x = x_p (3869/3869), y = contour (8400/8400).

This rebuilds that program on the space L320/L321 measured:
  * integer lattice
  * shapes = integer divisor pairs of the EXACT area, aspect <= 3
  * objective = bounding-box area, nothing else

The question is self-contained and needs no netlist, so it is immune to the
1M-vs-graded distribution shift L330 found: our packer's utilisation ceiling is
85.4 % (L284, and 87.7 % p50 for the mix arm at L323); the label's is 96.9 %.
Can this manifold reach it?

SCOPE: preplaced blocks cannot be expressed in a B*-tree, so they are packed like
any other block. This measures the REACHABLE DENSITY of the manifold, not a
submittable layout.
"""
import bisect
import glob
import math
import os
import random
import sys
import time

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
    return sorted(set(out)) or [(A, 1)]


def pack(L, R, si, shapes, root, n):
    """contour B*-tree decode. xs is sorted; hs[i] is the height on [xs[i], xs[i+1])."""
    xs = [0, 1 << 30]
    hs = [0, 0]
    W = H = 0
    st = [(root, 0)]
    while st:
        k, x = st.pop()
        w, h = shapes[k][si[k]]
        xe = x + w
        i = bisect.bisect_right(xs, x) - 1
        j = bisect.bisect_left(xs, xe)
        y = 0
        for q in range(i, j):
            if hs[q] > y:
                y = hs[q]
        top = y + h
        tail_h = hs[j - 1]
        seg_x = [x]
        seg_h = [top]
        if xe < xs[j] if j < len(xs) else True:
            seg_x.append(xe)
            seg_h.append(tail_h)
        xs[i + (xs[i] == x) if False else i:j] = seg_x
        hs[i:j] = seg_h
        if xs[i] != x:                       # left remnant kept
            xs.insert(i, xs[i])
        if xe > W:
            W = xe
        if top > H:
            H = top
        if R[k] != -1:
            st.append((R[k], x))
        if L[k] != -1:
            st.append((L[k], xe))
    return W, H


def pack_ref(L, R, si, shapes, root, n):
    """reference decode: simple, obviously correct, used to gate the fast one."""
    X = [0] * n
    Y = [0] * n
    placed = []
    W = H = 0
    st = [(root, 0)]
    while st:
        k, x = st.pop()
        w, h = shapes[k][si[k]]
        y = 0
        for (px, py, pw, ph) in placed:
            if px < x + w and x < px + pw:
                y = max(y, py + ph)
        X[k], Y[k] = x, y
        placed.append((x, y, w, h))
        W = max(W, x + w)
        H = max(H, y + h)
        if R[k] != -1:
            st.append((R[k], x))
        if L[k] != -1:
            st.append((L[k], x + w))
    return W, H, X, Y, placed


def rand_tree(n, rng):
    order = list(range(n))
    rng.shuffle(order)
    L = [-1] * n
    R = [-1] * n
    par = [-1] * n
    root = order[0]
    for idx in range(1, n):
        k = order[idx]
        p = order[rng.randrange(idx)]
        while True:
            if rng.random() < .5:
                if L[p] == -1:
                    L[p] = k; par[k] = p; break
                p = L[p]
            else:
                if R[p] == -1:
                    R[p] = k; par[k] = p; break
                p = R[p]
    return L, R, par, root


def detach(L, R, par, root, k):
    if L[k] != -1 and R[k] != -1:
        return None
    c = L[k] if L[k] != -1 else R[k]
    p = par[k]
    if p == -1:
        if c == -1:
            return None
        root = c
        par[c] = -1
    else:
        if L[p] == k:
            L[p] = c
        else:
            R[p] = c
        if c != -1:
            par[c] = p
    L[k] = R[k] = -1
    par[k] = -1
    return root


def attach(L, R, par, k, p, side, rng):
    if side == 0:
        c = L[p]; L[p] = k
    else:
        c = R[p]; R[p] = k
    par[k] = p
    if c != -1:
        if rng.random() < .5:
            L[k] = c
        else:
            R[k] = c
        par[c] = k


def anneal(n, shapes, iters, seed):
    rng = random.Random(seed)
    L, R, par, root = rand_tree(n, rng)
    si = [0] * n
    W, H, _X, _Y, _p = pack_ref(L, R, si, shapes, root, n)
    cur = W * H
    best = cur
    bestT = (L[:], R[:], par[:], si[:], root)
    T0 = cur * .05
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
        W, H, _X, _Y, _p = pack_ref(L, R, si, shapes, root, n)
        new = W * H
        if new <= cur or rng.random() < math.exp(min(0.0, (cur - new) / T)):
            cur = new
            if new < best:
                best = new
                bestT = (L[:], R[:], par[:], si[:], root)
        else:
            L, R, par, si, root = sL, sR, sP, sS, sRoot
    return bestT, best


LAB = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth")}
DAT = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litedata_1.pth")}

if __name__ == "__main__":
    ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    NS = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
          else [21, 40, 60, 80, 100, 120])
    print("== L333 B*-tree SA, AREA objective, integer lattice, divisor-pair shapes ==")
    print("   %d SA iterations per case" % ITERS)
    print()
    print("   %-5s %10s %11s %10s %11s %8s"
          % ("n", "sumA", "label util", "SA util", "gap closed", "time"))
    tu = tl = 0.0
    cnt = 0
    for n in NS:
        if n not in LAB:
            continue
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
        t0 = time.time()
        _bt, area = anneal(nb, shapes, ITERS, 1234 + n)
        dt = time.time() - t0
        su = sumA / area
        lu = sumA / (lw * lh)
        OURS = 0.877
        closed = (su - OURS) / (lu - OURS) if lu > OURS else float("nan")
        tu += su
        tl += lu
        cnt += 1
        print("   %-5d %10d %11.4f %10.4f %10.0f%% %7.1fs"
              % (n, sumA, lu, su, 100 * closed, dt))
    if cnt:
        print()
        print("   mean over %d cases: label %.4f   B*-tree SA %.4f   our mix arm 0.877"
              % (cnt, tl / cnt, tu / cnt))
