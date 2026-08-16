"""L135 — snap near-miss BOUNDARY blocks, and re-score with the official evaluator.

L131 closed the ULP family (grouping was its only member: `unary_union` is exact,
while boundary uses eps=1e-6 and MIB rounds to 4dp). This is a different bug with
the same shape, one scale up.

`constructive_l124.cpp:51` sets `MARGIN = 1e-4`, and the frame is sized
`max(w, pre_w + MARGIN)` (`:683`, `:1972`). So the packing frame is deliberately
1e-4 LARGER than the extent it has to contain. Blocks packed against the frame
edge land on `X.0001` and define the bbox; a block sitting at the true `X.0`
then misses the bbox edge by exactly MARGIN -- and the evaluator's threshold is
**1e-6**, a hundred times tighter, so it scores a boundary violation.

Audited on the shipped 48c result: **11 blocks miss a required edge by exactly
1e-4 with a clear slide path**, e.g. case 54 blk 3 right edge 141.0 against bbox
xmax 141.0001, case 7 blk 4 top 100.0 against ymax 100.0001.

The snap slides such a block onto the edge. Safety, in the same terms as L131:
  * the move is <= SNAP_MAX (default 1e-3) and only ever TOWARD the bbox edge,
    so the bbox cannot grow and area_gap cannot get worse;
  * the destination is checked against every other block with the evaluator's
    own 1e-6 overlap rule, so it cannot create an overlap violation;
  * preplaced blocks are never moved (position is HARD);
  * widths and heights are never touched (dimensions and area are HARD).

READ-ONLY with respect to the submission.

  <python> -u l135_bnd_verify.py results_L114_48c_lp_anchor.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import ContestEvaluator, evaluate_solution   # noqa: E402
import optimizer_constructive as oc                                  # noqa: E402

SNAP_MAX = 1e-3
OVL = 1e-6
# 🚨 The evaluator counts a block as touching when it is within **1e-6**
# (iccad2026_evaluate.py:527). A block already inside that band is NOT violating,
# and nudging it anyway is not free: the nudge is ULP-scale, and an ULP is
# exactly enough to break a cluster abutment and CREATE a grouping violation.
#
# MEASURED, first version of this file which snapped anything within snap_max:
# 423 blocks moved, **0** boundary violations removed, case 82
# grouping_violations 0 -> 2, weighted total -1.1401%. Handoff §3.1 again --
# a stage that optimises one term must not quietly break another.
#
# So the lower bound is the evaluator's own threshold: only touch blocks that
# are genuinely scoring a violation.
SNAP_MIN = 1e-6


def snap_boundary(P, cons, n, snap_max=SNAP_MAX):
    """Slide genuinely-violating boundary blocks onto the edge they must touch."""
    moved = 0
    x0 = min(p[0] for p in P)
    y0 = min(p[1] for p in P)
    x1 = max(p[0] + p[2] for p in P)
    y1 = max(p[1] + p[3] for p in P)

    def free(i, nx, ny):
        w, h = P[i][2], P[i][3]
        for t in range(n):
            if t == i:
                continue
            ox = min(nx + w, P[t][0] + P[t][2]) - max(nx, P[t][0])
            oy = min(ny + h, P[t][1] + P[t][3]) - max(ny, P[t][1])
            if ox > OVL and oy > OVL:
                return False
        return True

    for i in range(n):
        code = int(cons[i][4])
        if not code or int(cons[i][1]) != 0:      # preplaced position is HARD
            continue
        bx, by, bw, bh = P[i]
        nx, ny = bx, by
        if code & 1 and SNAP_MIN <= bx - x0 <= snap_max:
            nx = x0
        if code & 2 and SNAP_MIN <= x1 - (bx + bw) <= snap_max:
            nx = x1 - bw
        if code & 4 and SNAP_MIN <= y1 - (by + bh) <= snap_max:
            ny = y1 - bh
        if code & 8 and SNAP_MIN <= by - y0 <= snap_max:
            ny = y0
        if (nx, ny) != (bx, by) and free(i, nx, ny):
            P[i][0], P[i][1] = nx, ny
            moved += 1
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--snap-max", type=float, default=SNAP_MAX)
    a = ap.parse_args()

    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    res = {r["test_id"]: r for r in json.load(open(a.results))["test_results"]}

    wsum = c_old = c_new = 0.0
    rows = []
    inf_b = inf_a = 0
    for idx in sorted(res):
        r = res[idx]
        if not r.get("positions"):
            continue
        s = ev.dataset[idx]
        at, b2b, p2b, pins, cons = s["input"]
        n = int((at != -1).sum().item())
        base, tp = ev._extract_baseline(idx, s["label"], b2b, p2b, pins, n)
        rt = float(r.get("runtime_seconds") or 1.0)

        def score(Q):
            return evaluate_solution({"positions": [list(q) for q in Q],
                                      "runtime": rt},
                                     base, cons[:n], b2b, p2b, pins, at[:n],
                                     target_positions=tp[:n], median_runtime=rt)

        P0 = [list(map(float, p)) for p in r["positions"][:n]]
        m0 = score(P0)
        P1 = [list(p) for p in P0]
        mv = snap_boundary(P1, cons, n, a.snap_max)
        # a boundary slide can separate the block from its cluster mates, so the
        # L131 abutment repair runs AFTER it, not instead of it
        if mv:
            P1 = [list(q) for q in oc._snap_group_abutment(
                [tuple(q) for q in P1], cons, n)]
        m1 = score(P1)

        w = math.exp(n / 12.0)
        wsum += w
        c_old += w * float(m0.cost)
        c_new += w * float(m1.cost)
        inf_b += 0 if m0.is_feasible else 1
        inf_a += 0 if m1.is_feasible else 1
        if abs(float(m1.cost) - float(m0.cost)) > 1e-12 or mv:
            rows.append((idx, n, mv, float(m0.cost), float(m1.cost),
                         int(m0.boundary_violations), int(m1.boundary_violations),
                         int(m0.overlap_violations), int(m1.overlap_violations)))

    print(f"\n=== L135 boundary snap: {a.results} (snap_max {a.snap_max:g}) ===\n")
    print(f"{'case':>5} {'n':>4} {'moved':>6} {'cost before':>12} {'cost after':>12} "
          f"{'vbnd':>9} {'overlap':>9}")
    for idx, n, mv, c0, c1, b0, b1, o0, o1 in rows:
        print(f"{idx:>5} {n:>4} {mv:>6} {c0:>12.6f} {c1:>12.6f} "
              f"{str(b0) + '->' + str(b1):>9} {str(o0) + '->' + str(o1):>9}")
    print(f"\ncases touched                  {len(rows)}")
    print(f"blocks snapped                 {sum(r[2] for r in rows)}")
    print(f"boundary violations removed    {sum(r[5] - r[6] for r in rows)}")
    print(f"infeasible before / after      {inf_b} / {inf_a}")
    print(f"\nweighted total BEFORE          {c_old / wsum:.12f}")
    print(f"weighted total AFTER           {c_new / wsum:.12f}")
    print(f"worth                          "
          f"{100 * (c_old - c_new) / max(c_old, 1e-9):+.4f}%")
    if inf_a > inf_b:
        print("\n!! SNAP CREATED AN INFEASIBILITY -- do not use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
