"""L135 — where is the remaining soft-violation headroom in the SHIPPED result?

L131 closed the ULP family, and it turns out to have had exactly one member:

    grouping  `unary_union`, EXACT geometry, no tolerance      -> was exposed
    boundary  `abs(bx - x_min_bb) < eps`, **eps = 1e-6**       -> robust
    MIB       compares `round(dim, 4)`                         -> robust

So anything left is a REAL violation, and the question is whether any of them are
cheap to remove. This audits the shipped 48c result and reports, per violation:

  * grouping -- the true gap between the components of a split group, in units
    and as a fraction of the smaller block, plus whether the gap corridor is
    clear of other blocks (i.e. whether closing it is even geometrically
    available);
  * boundary -- how far the offending block is from the edge it must touch, and
    whether anything is in the way of sliding it there.

READ-ONLY. Reports; changes nothing.

  <python> -u l135_soft_audit.py results_L114_48c_lp_anchor.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

import torch                                                        # noqa: E402
from shapely.geometry import box                                    # noqa: E402
from shapely.ops import unary_union                                 # noqa: E402
from iccad2026_evaluate import ContestEvaluator                     # noqa: E402

EPS_BND = 1e-6


def overlaps_1d(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0) > 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()

    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    res = {r["test_id"]: r for r in json.load(open(a.results))["test_results"]}

    grp_rows, bnd_rows = [], []
    tot_g = tot_b = 0
    for idx in sorted(res):
        r = res[idx]
        P = r.get("positions")
        if not P:
            continue
        s = ev.dataset[idx]
        at, _b2b, _p2b, _pins, cons = s["input"]
        n = int((at != -1).sum().item())
        P = [list(map(float, p)) for p in P[:n]]
        polys = [box(p[0], p[1], p[0] + p[2], p[1] + p[3]) for p in P]

        # ---- grouping: real fragments and how wide the split is --------------
        clust = cons[:n, 3]
        ngrp = int(clust.max().item()) if clust.numel() else 0
        for g in range(1, ngrp + 1):
            mem = torch.where(clust == g)[0].tolist()
            if len(mem) < 2:
                continue
            u = unary_union([polys[i] for i in mem])
            if u.geom_type != "MultiPolygon":
                continue
            k = len(u.geoms) - 1
            tot_g += k
            # nearest approach between the two nearest components
            best = None
            for i in mem:
                for j in mem:
                    if i >= j:
                        continue
                    d = polys[i].distance(polys[j])
                    if d > 0 and (best is None or d < best[0]):
                        best = (d, i, j)
            if best is None:
                grp_rows.append((idx, g, k, float("nan"), float("nan"), "?"))
                continue
            d, i, j = best
            small = min(P[i][2] * P[i][3], P[j][2] * P[j][3]) ** 0.5
            # is the corridor between i and j clear?
            gap_poly = polys[i].union(polys[j]).envelope.difference(
                polys[i]).difference(polys[j])
            blocked = any(polys[t].intersects(gap_poly)
                          and polys[t].intersection(gap_poly).area > 1e-9
                          for t in range(n) if t not in (i, j))
            grp_rows.append((idx, g, k, d, d / max(small, 1e-9),
                             "blocked" if blocked else "CLEAR"))

        # ---- boundary: how far from the required edge ------------------------
        bnd = cons[:n, 4]
        if int((bnd != 0).sum().item()) == 0:
            continue
        x0 = min(p[0] for p in P)
        y0 = min(p[1] for p in P)
        x1 = max(p[0] + p[2] for p in P)
        y1 = max(p[1] + p[3] for p in P)
        for i in range(n):
            code = int(bnd[i].item())
            if not code:
                continue
            bx, by, bw, bh = P[i]
            miss = {}
            if code & 1 and abs(bx - x0) >= EPS_BND:
                miss["L"] = bx - x0
            if code & 2 and abs(bx + bw - x1) >= EPS_BND:
                miss["R"] = x1 - (bx + bw)
            if code & 4 and abs(by + bh - y1) >= EPS_BND:
                miss["T"] = y1 - (by + bh)
            if code & 8 and abs(by - y0) >= EPS_BND:
                miss["B"] = by - y0
            if not miss:
                continue
            tot_b += 1
            side, dist = min(miss.items(), key=lambda kv: abs(kv[1]))
            dim = (bw * bh) ** 0.5
            # is the slide path clear?
            if side in ("L", "R"):
                lo, hi = (x0, bx) if side == "L" else (bx + bw, x1)
                blocked = any(overlaps_1d(P[t][1], P[t][1] + P[t][3], by, by + bh)
                              and overlaps_1d(P[t][0], P[t][0] + P[t][2], lo, hi)
                              for t in range(n) if t != i)
            else:
                lo, hi = (y0, by) if side == "B" else (by + bh, y1)
                blocked = any(overlaps_1d(P[t][0], P[t][0] + P[t][2], bx, bx + bw)
                              and overlaps_1d(P[t][1], P[t][1] + P[t][3], lo, hi)
                              for t in range(n) if t != i)
            bnd_rows.append((idx, i, side, abs(dist), abs(dist) / max(dim, 1e-9),
                             "blocked" if blocked else "CLEAR"))

    print(f"\n=== L135 soft-violation audit: {a.results} ===\n")
    print(f"grouping violations (real fragments): {tot_g}")
    print(f"{'case':>5} {'grp':>4} {'k':>3} {'gap':>12} {'gap/blk':>9} {'corridor':>9}")
    for row in sorted(grp_rows, key=lambda r: r[3])[:a.show]:
        print(f"{row[0]:>5} {row[1]:>4} {row[2]:>3} {row[3]:>12.4g} "
              f"{row[4]:>9.4f} {row[5]:>9}")
    clear_g = sum(1 for r in grp_rows if r[5] == "CLEAR")
    print(f"  corridor CLEAR on {clear_g}/{len(grp_rows)} split groups")

    print(f"\nboundary violations: {tot_b}")
    print(f"{'case':>5} {'blk':>4} {'side':>5} {'dist':>12} {'dist/blk':>9} {'path':>9}")
    for row in sorted(bnd_rows, key=lambda r: r[4])[:a.show]:
        print(f"{row[0]:>5} {row[1]:>4} {row[2]:>5} {row[3]:>12.4g} "
              f"{row[4]:>9.4f} {row[5]:>9}")
    clear_b = sum(1 for r in bnd_rows if r[5] == "CLEAR")
    print(f"  slide path CLEAR on {clear_b}/{len(bnd_rows)} boundary violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
