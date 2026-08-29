"""L281: how RIGID is the saturated chain?  Positions only, no dataset import.

If the critical chain runs through most of the blocks, the layout is a rigid
train and 'shorten the chain' is not a local edit.  If it is short, only a few
blocks set the width and the axis has a target.  Also reports how many blocks
lie on SOME maximum-length chain (slack-0 blocks), which is the set whose width
the bounding box is actually paying for.
"""
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent


def graphs(pos):
    n = len(pos)
    EH, EV = [], []
    for i in range(n):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = pos[j]
            g = (xj - (xi + wi), xi - (xj + wj),
                 yj - (yi + hi), yi - (yj + hj))
            k = 0
            for t in (1, 2, 3):
                if g[t] > g[k]:
                    k = t
            (EH if k < 2 else EV).append((i, j) if k in (0, 2) else (j, i))
    return EH, EV


def chain(n, edges, wt):
    """longest node-weighted path, plus the set of nodes on some longest path"""
    adj, radj, indeg = [[] for _ in range(n)], [[] for _ in range(n)], [0] * n
    for a, b in edges:
        adj[a].append(b)
        radj[b].append(a)
        indeg[b] += 1
    order, q, d = [], [i for i in range(n) if indeg[i] == 0], list(range(n))
    down = [0.0] * n
    up = [wt[i] for i in range(n)]
    head = 0
    while head < len(q):
        a = q[head]
        head += 1
        order.append(a)
        for b in adj[a]:
            if up[a] + wt[b] > up[b]:
                up[b] = up[a] + wt[b]
            indeg[b] -= 1
            if indeg[b] == 0:
                q.append(b)
    if len(order) != n:
        return None
    for a in reversed(order):
        best = 0.0
        for b in adj[a]:
            if down[b] + wt[b] > best:
                best = down[b] + wt[b]
        down[a] = best
    L = max(up[i] + down[i] for i in range(n))
    on = [i for i in range(n) if up[i] + down[i] > L - 1e-9]
    # longest single path length in nodes
    return L, on


aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
rows = []
for t in sorted(aj["test_results"], key=lambda t: t["test_id"]):
    pos = [tuple(p) for p in t["positions"]]
    n = len(pos)
    EH, EV = graphs(pos)
    rH = chain(n, EH, [p[2] for p in pos])
    rV = chain(n, EV, [p[3] for p in pos])
    if not rH or not rV:
        continue
    W0 = max(p[0] + p[2] for p in pos) - min(p[0] for p in pos)
    H0 = max(p[1] + p[3] for p in pos) - min(p[1] for p in pos)
    sH, sV = 1.0 - rH[0] / W0, 1.0 - rV[0] / H0
    tight = (sH if sH <= sV else sV)
    on = rH[1] if sH <= sV else rV[1]
    rows.append((t["test_id"], n, tight, len(on)))

sat = [r for r in rows if r[2] <= 1e-9]
print(f"in-set 100: {len(sat)} cases saturated on at least one axis")
fr = sorted(100.0 * r[3] / r[1] for r in sat)


def q(a, f):
    return a[min(int(f * len(a)), len(a) - 1)]


print(f"  blocks on a critical (zero-slack) chain, as % of the case's blocks:")
print(f"    min {fr[0]:.1f} %   p25 {q(fr, .25):.1f} %   p50 {q(fr, .5):.1f} %"
      f"   p75 {q(fr, .75):.1f} %   max {fr[-1]:.1f} %")
cnt = sorted(r[3] for r in sat)
print(f"  absolute count: min {cnt[0]}  p50 {q(cnt, .5)}  max {cnt[-1]}"
      f"   (n ranges {min(r[1] for r in sat)}..{max(r[1] for r in sat)})")
