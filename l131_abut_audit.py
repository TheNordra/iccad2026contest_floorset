"""L131 — does the SHIPPED path lose grouping violations to the same ULP?

L130 found that `origin + offset` emission does not abut in doubles: the
evaluator builds a block's far edge as `x + w`, and `(lx+ox)+w != lx+(ox+w)`, so
a packing that abuts exactly in exact arithmetic lands +-2.8e-14 off and
`unary_union` splits the group. In L129 that was worth vgrp 0.89 -> 0.42.

Nothing about that is specific to L129. This audits the SHIPPED result: for every
cluster group it counts the official components, then recounts them with the
polygons buffered by a hair. A component split that disappears under a 1e-9
buffer was never a real gap.

READ-ONLY. It reads a results json and touches nothing. The submission is
uploaded and frozen; this reports what a fix WOULD be worth, it does not make one.

  <python> -u l131_abut_audit.py results_L114_48c_lp_anchor.json
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

TOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()

    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    res = {r["test_id"]: r for r in json.load(open(a.results))["test_results"]}

    tot_g = tot_spur = 0
    rows = []
    for idx in sorted(res):
        r = res[idx]
        P = r.get("positions")
        if not P:
            continue
        s = ev.dataset[idx]
        at, _b2b, _p2b, _pins, cons = s["input"]
        n = int((at != -1).sum().item())
        clust = cons[:n, 3]
        ngrp = int(clust.max().item()) if clust.numel() else 0
        if ngrp <= 0:
            continue
        polys = [box(float(p[0]), float(p[1]),
                     float(p[0]) + float(p[2]), float(p[1]) + float(p[3]))
                 for p in P[:n]]
        g_off = g_on = 0
        worst = 0.0
        for g in range(1, ngrp + 1):
            mem = torch.where(clust == g)[0].tolist()
            gp = [polys[i] for i in mem]
            u = unary_union(gp)
            c_off = len(u.geoms) if u.geom_type == "MultiPolygon" else 1
            ub = unary_union([p.buffer(a.tol) for p in gp])
            c_on = len(ub.geoms) if ub.geom_type == "MultiPolygon" else 1
            g_off += c_off - 1
            g_on += c_on - 1
            if c_off > c_on:
                # how wide is the gap that the buffer closed?
                for i in mem:
                    for j in mem:
                        if i >= j:
                            continue
                        d = polys[i].distance(polys[j])
                        if 0 < d <= a.tol and d > worst:
                            worst = d
        tot_g += g_off
        tot_spur += g_off - g_on
        if g_off != g_on:
            rows.append((idx, n, g_off, g_on, g_off - g_on, worst))

    print(f"\n=== L131: spurious grouping components in {a.results} ===")
    print(f"tolerance {a.tol:g}\n")
    print(f"{'case':>5} {'n':>4} {'vgrp':>6} {'vgrp@tol':>9} {'spurious':>9} {'gap':>12}")
    for idx, n, off, on, sp, worst in rows:
        print(f"{idx:>5} {n:>4} {off:>6} {on:>9} {sp:>9} {worst:>12.3e}")
    print(f"\ntotal grouping violations      {tot_g}")
    print(f"of which are FP artefacts      {tot_spur}"
          f"   ({100 * tot_spur / max(tot_g, 1):.1f}%)")
    print(f"cases affected                 {len(rows)}")

    if tot_spur:
        print("\nThis tool COUNTS. It deliberately does not price.")
        print("  <python> l131_snap_verify.py <results.json>   <- prices it, by")
        print("  snapping the positions and re-running the OFFICIAL evaluator.")
        print("\n!! An earlier version of this file DID estimate the prize, as")
        print("  cost * exp(BETA * dV_rel), and reported +0.2196% against the")
        print("  official +0.0758% -- 3x too high. It read N_soft from the")
        print("  results json, an official results json HAS NO `nsoft` FIELD")
        print("  (that key is an L129 addition), the fallback was 1.0, and so")
        print("  `vr - spurious/N_soft` removed EVERY violation on those cases")
        print("  instead of one. A per-case normaliser that silently defaults to")
        print("  1.0 is a 3x error that still looks like a plausible number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
