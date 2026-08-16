"""L131 verify — apply the abutment snap to the SHIPPED positions and re-score
with the OFFICIAL evaluator.

l131_abut_audit.py estimated the prize by scaling each case cost by
exp(BETA*dV_rel). This does not estimate anything: it snaps the positions,
re-runs `evaluate_solution`, and reports the official weighted total.

WHY A SNAP IS SAFE HERE, and it is worth stating because it is the whole reason
this can be a post-process instead of a C++ rebuild:

  * the gaps are ~1e-14 and the snap moves a block by at most TOL = 1e-9;
  * `check_overlap` (iccad2026_evaluate.py:223) ignores anything below **1e-6**
    on both axes -- "touching edges OK" -- so a 1e-9 move cannot create an
    overlap violation, with five orders of magnitude to spare;
  * preplaced blocks are never moved (their position is a HARD constraint);
  * widths and heights are never touched (dimensions and area are HARD).

READ-ONLY with respect to the submission: it reads a results json, snaps in
memory, and prints. It writes nothing into build_submission/ and does not
rebuild the binary.

  <python> -u l131_snap_verify.py results_L114_48c_lp_anchor.json
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

TOL = 1e-9


def snap_group(P, mem, is_pre, tol):
    """Close sub-tolerance gaps between members of one cluster group.

    A pair that should abut has its far/near edges equal in exact arithmetic and
    an ULP apart in doubles. Assigning `x_j = x_i + w_i` makes the two edges the
    identical float -- the same expression the evaluator uses to build block i's
    far edge -- so `unary_union` sees them touch.

    Repeated a few times so a chain a-b-c settles; each pass moves nothing by
    more than `tol`, so it converges immediately in practice.
    """
    moved = 0
    for _ in range(4):
        for ax in (0, 1):
            oth = 1 - ax
            order = sorted(mem, key=lambda i: P[i][ax])
            for a in order:
                for b in order:
                    if a == b or is_pre[b]:
                        continue
                    # must overlap on the OTHER axis to be a shared edge at all
                    lo = max(P[a][oth], P[b][oth])
                    hi = min(P[a][oth] + P[a][oth + 2], P[b][oth] + P[b][oth + 2])
                    if hi - lo <= 1e-9:
                        continue
                    far = P[a][ax] + P[a][ax + 2]
                    gap = P[b][ax] - far
                    if 0.0 < gap <= tol:
                        P[b][ax] = far
                        moved += 1
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()

    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    res = {r["test_id"]: r for r in json.load(open(a.results))["test_results"]}

    wsum = c_old = c_new = 0.0
    n_changed = n_moved = 0
    infeas_before = infeas_after = 0
    rows = []
    for idx in sorted(res):
        r = res[idx]
        if not r.get("positions"):
            continue
        s = ev.dataset[idx]
        at, b2b, p2b, pins, cons = s["input"]
        n = int((at != -1).sum().item())
        base, tp = ev._extract_baseline(idx, s["label"], b2b, p2b, pins, n)
        rt = float(r.get("runtime_seconds") or 1.0)

        def score(P):
            return evaluate_solution({"positions": [list(p) for p in P],
                                      "runtime": rt},
                                     base, cons[:n], b2b, p2b, pins, at[:n],
                                     target_positions=tp[:n], median_runtime=rt)

        P0 = [list(map(float, p)) for p in r["positions"][:n]]
        m0 = score(P0)

        is_pre = [cons[i][1] != 0 for i in range(n)]
        clust = cons[:n, 3]
        ngrp = int(clust.max().item()) if clust.numel() else 0
        P1 = [list(p) for p in P0]
        mv = 0
        for g in range(1, ngrp + 1):
            mem = [i for i in range(n) if int(clust[i]) == g]
            if len(mem) > 1:
                mv += snap_group(P1, mem, is_pre, a.tol)
        m1 = score(P1)

        w = math.exp(n / 12.0)
        wsum += w
        c_old += w * float(m0.cost)
        c_new += w * float(m1.cost)
        infeas_before += 0 if m0.is_feasible else 1
        infeas_after += 0 if m1.is_feasible else 1
        n_moved += mv
        if abs(float(m1.cost) - float(m0.cost)) > 1e-12:
            n_changed += 1
            rows.append((idx, n, float(m0.cost), float(m1.cost),
                         int(m0.grouping_violations), int(m1.grouping_violations),
                         int(m0.overlap_violations), int(m1.overlap_violations),
                         mv))

    print(f"\n=== L131 verify: official re-score after abutment snap ===")
    print(f"file {a.results}   tol {a.tol:g}\n")
    print(f"{'case':>5} {'n':>4} {'cost before':>12} {'cost after':>12} "
          f"{'vgrp':>9} {'overlap':>9} {'moves':>6}")
    for idx, n, c0, c1, g0, g1, o0, o1, mv in rows:
        print(f"{idx:>5} {n:>4} {c0:>12.6f} {c1:>12.6f} "
              f"{str(g0) + '->' + str(g1):>9} {str(o0) + '->' + str(o1):>9} {mv:>6}")
    print(f"\ncases changed                  {n_changed}")
    print(f"edges snapped (all cases)      {n_moved}")
    print(f"infeasible before / after      {infeas_before} / {infeas_after}")
    print(f"\nweighted total BEFORE          {c_old / wsum:.12f}")
    print(f"weighted total AFTER           {c_new / wsum:.12f}")
    print(f"worth                          "
          f"{100 * (c_old - c_new) / max(c_old, 1e-9):+.4f}%")
    if infeas_after > infeas_before:
        print("\n!! SNAP CREATED AN INFEASIBILITY -- do not use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
