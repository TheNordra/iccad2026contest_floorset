"""L174 - a hoisted `_proxy_metrics`, for benchmarking. OFFLINE, not shipped yet.

Everything in `_proxy_metrics` that depends only on (constraints, n) is computed
ONCE per case instead of once per profile. There are 51 profiles, so it is
computed 51 times today.

Hoisted (identical for all 51 calls):
    constraints[:n, 4].tolist()  /  [:n, 3]  /  [:n, 2]      3 torch->list
    nsoft            -- the sum of all three contributions is position-free
    ngrp, nmib
    the per-group index lists

And one algorithmic change that is NOT just caching: the shipped code finds each
group's members with

    for g in range(1, ngrp + 1):
        idx = [i for i in range(n) if int(clust_l[i]) == g]

which is O(n * ngrp). One bucketing pass is O(n). Same lists, same order --
`range(n)` ascending in both -- so the shapely union and the MIB shape set see
identical inputs in identical order.

EQUIVALENCE. The returned dict must be bit-identical, not merely close: the
proxy is the live selector's argmin, so a 1-ULP difference can pick a different
candidate and change the score. `l174_proxy_bench.py` asserts exact equality of
all three fields on real inputs before reporting any timing.
"""


def build_case_cache(constraints, n):
    """The position-independent half of _proxy_metrics, once per case."""
    ncols = constraints.shape[1] if constraints.dim() > 1 else 0
    if ncols <= 4:
        return (None, (), (), 0)
    bound_l = constraints[:n, 4].tolist()
    clust_l = constraints[:n, 3].tolist()
    mib_l = constraints[:n, 2].tolist()
    nsoft = sum(1 for b in bound_l if b != 0)

    # only the blocks that carry a boundary code matter to the loop
    bcodes = [(i, int(bound_l[i])) for i in range(n) if int(bound_l[i]) != 0]

    ngrp = int(max(clust_l)) if clust_l else 0
    buckets = [[] for _ in range(ngrp + 1)]
    for i in range(n):
        g = int(clust_l[i])
        if 1 <= g <= ngrp:
            buckets[g].append(i)
    gidx = []
    for g in range(1, ngrp + 1):
        idx = buckets[g]
        nsoft += max(0, len(idx) - 1)
        if len(idx) > 1:
            gidx.append(idx)

    nmib = int(max(mib_l)) if mib_l else 0
    mbuckets = [[] for _ in range(nmib + 1)]
    for i in range(n):
        g = int(mib_l[i])
        if 1 <= g <= nmib:
            mbuckets[g].append(i)
    midx = []
    for g in range(1, nmib + 1):
        idx = mbuckets[g]
        nsoft += max(0, len(idx) - 1)
        midx.append(idx)
    return (bcodes, gidx, midx, nsoft)


def proxy_metrics_hoisted(positions, area_targets, b2b, p2b, pins,
                          constraints, n, cache, mod):
    """_proxy_metrics with the cached half supplied. `mod` is the live module,
    so the hpwl helpers and the shapely names are exactly the shipped ones."""
    xmin = min(p[0] for p in positions)
    ymin = min(p[1] for p in positions)
    xmax = max(p[0] + p[2] for p in positions)
    ymax = max(p[1] + p[3] for p in positions)
    area = (xmax - xmin) * (ymax - ymin)
    hpwl = mod._hpwl_b2b_fast(positions, b2b) + mod._hpwl_p2b_fast(positions, p2b, pins)

    bcodes, gidx, midx, nsoft = cache
    vb = vg = vm = 0
    if bcodes is not None:
        eps = 1e-6
        for i, code in bcodes:
            bx, by, bw, bh = positions[i]
            ok = True
            if code & 1:
                ok = ok and abs(bx - xmin) < eps
            if code & 2:
                ok = ok and abs(bx + bw - xmax) < eps
            if code & 4:
                ok = ok and abs(by + bh - ymax) < eps
            if code & 8:
                ok = ok and abs(by - ymin) < eps
            if not ok:
                vb += 1
        if mod._SHAPELY:
            for idx in gidx:
                u = mod._unary_union([
                    mod._box(positions[i][0], positions[i][1],
                             positions[i][0] + positions[i][2],
                             positions[i][1] + positions[i][3]) for i in idx])
                if u.geom_type == "MultiPolygon":
                    vg += len(u.geoms) - 1
        for idx in midx:
            shapes = {(round(positions[i][2], 4), round(positions[i][3], 4))
                      for i in idx}
            vm += len(shapes) - 1
    vrel = (vb + vg + vm) / max(nsoft, 1)
    return {"area": area, "hpwl": hpwl, "vrel": vrel}
