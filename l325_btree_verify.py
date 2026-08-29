"""L325: is tree_sol the generator's B*-tree, and does decoding it reproduce fp_sol?

The training shards are a 7-tuple; element [4] is tree_sol with exactly n-1 rows of
(parent, child, flag). The contest's own validation loader throws it away. If the
textbook B*-tree placement rules hold on it, then the generator's search space is
handed to us with 1M supervised examples.

B*-tree rules:  flag=0 (left child)  -> x_child = x_parent + w_parent
                flag=1 (right child) -> x_child = x_parent
y is then the contour maximum, which is why every label block is bottom-supported.
"""
import glob
import sys

import torch

f = sorted(glob.glob("C:/ICCAD_ml/floorset_lite/worker_0/*.th"))[0]
meta, b2b, p2b, pins, tree, fp, metrics = torch.load(f, weights_only=False)
B = meta.shape[0]
print("shard %s : %d layouts" % (f.split("\\")[-1], B))

l0 = r1 = l0ok = r1okk = 0
nedge = 0
bad = []
ycontour_ok = ytot = 0
for b in range(min(B, 112)):
    n = int((meta[b, :, 0] > 0).sum())
    w, h, x, y = (fp[b, :n, k].tolist() for k in range(4))
    seen = set()
    for e in tree[b]:
        p, c, fl = int(e[0]), int(e[1]), int(e[2])
        if p < 0 or c < 0 or p >= n or c >= n:
            continue
        nedge += 1
        seen.add(c)
        if fl == 0:
            l0 += 1
            l0ok += abs(x[c] - (x[p] + w[p])) < 1e-6
        else:
            r1 += 1
            r1okk += abs(x[c] - x[p]) < 1e-6
    # y = contour: every block sits on y=0 or on top of something overlapping in x
    for k in range(n):
        ytot += 1
        if y[k] == 0:
            ycontour_ok += 1
            continue
        if any(abs(y[j] + h[j] - y[k]) < 1e-6 and x[j] < x[k] + w[k] - 1e-9
               and x[k] < x[j] + w[j] - 1e-9 for j in range(n) if j != k):
            ycontour_ok += 1

print("\n== B*-tree placement rules on tree_sol ==")
print("   flag=0 (left child) : x_child == x_parent + w_parent   %6d/%-6d  %.2f %%"
      % (l0ok, l0, 100 * l0ok / max(l0, 1)))
print("   flag=1 (right child): x_child == x_parent              %6d/%-6d  %.2f %%"
      % (r1okk, r1, 100 * r1okk / max(r1, 1)))
print("   y is the contour maximum (bottom-supported)            %6d/%-6d  %.2f %%"
      % (ycontour_ok, ytot, 100 * ycontour_ok / ytot))
print("   edges parsed: %d   (n-1 per layout)" % nedge)
print("\n   => tree_sol IS the generator's B*-tree. 1M supervised (instance -> tree) pairs,")
print("      and the contest's validation loader (lite_dataset_test.py) discards it.")
