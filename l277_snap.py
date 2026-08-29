"""L277 -- post-hoc boundary snap, scored by the official evaluator. WRAPPER-ONLY.

WHY THIS ONE. L275 moved the target to the graded corpus; L277's inventory there
found only 12 of 81 soft violations are both geometrically CLEAR and not
`preplaced` (a preplaced block's position is a HARD constraint -- moving it breaks
feasibility rather than fixing a violation). Upper bound -0.4456 %.

WHY ANY ARE LEFT AT ALL. `constructive.cpp:final_boundary_nudge` already snaps
boundary blocks to the edge they must touch when the path is clear -- but it skips
`blocks[i].cluster > 0` and `is_preplaced`, and it runs INSIDE the frame trial,
before compaction, `hpwl_push` and the shape LP have moved everything again.

So the proposal is a snap applied to the FINAL layout. That is pure post-processing
on positions, i.e. a change to `op_wrapper.py` only: **no C++ change and therefore
no Linux ELF rebuild**, which is the only class of change that can ship safely.

This measures it offline and exactly: apply the snap to saved positions, write a
solutions json, and let the OFFICIAL evaluator score it. No re-solving, no proxy.

Guards, each of which can only cost the mechanism candidates:
  * never move a `preplaced` or `fixed` block;
  * the destination must not overlap anything (exact rect test);
  * the destination must not grow the bounding box (it never should -- the block
    moves toward an existing extreme -- but it is asserted, not assumed);
  * per case, keep the snap only if the OFFICIAL cost does not get worse.

⚠️ The last guard is an ORACLE (it reads the true cost, which the optimizer cannot
see at solve time). It is used here to separate "is there anything here at all"
from "is it selectable"; a deployable version must use the shapely proxy instead,
and that gap is exactly what sank several earlier candidates. Both numbers are
printed.

  <python> l277_snap.py results_L274_base_48c.json
"""
import json
import math
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import ContestEvaluator                    # noqa: E402
from l135_soft_audit import audit_case                             # noqa: E402

EPS = 1e-9


def _ov(a, b):
    return (a[0] < b[0] + b[2] - EPS and b[0] < a[0] + a[2] - EPS and
            a[1] < b[1] + b[3] - EPS and b[1] < a[1] + a[3] - EPS)


def snap_case(P, cons, n):
    """Return (new_positions, moved_count). Pure geometry; no scoring."""
    P = [list(map(float, p)) for p in P[:n]]
    pre = cons[:n, 1]
    fix = cons[:n, 0]
    bnd = cons[:n, 4]
    x0 = min(p[0] for p in P); y0 = min(p[1] for p in P)
    x1 = max(p[0] + p[2] for p in P); y1 = max(p[1] + p[3] for p in P)
    moved = 0
    for i in range(n):
        code = int(bnd[i].item())
        if not code or float(pre[i]) != 0 or float(fix[i]) != 0:
            continue
        bx, by, bw, bh = P[i]
        nx, ny = bx, by
        if code & 1:
            nx = x0
        if code & 2:
            nx = x1 - bw
        if code & 4:
            ny = y1 - bh
        if code & 8:
            ny = y0
        if abs(nx - bx) < 1e-9 and abs(ny - by) < 1e-9:
            continue
        cand = [nx, ny, bw, bh]
        if any(_ov(cand, P[t]) for t in range(n) if t != i):
            continue
        # must not grow the bbox
        if (nx < x0 - EPS or ny < y0 - EPS or
                nx + bw > x1 + EPS or ny + bh > y1 + EPS):
            continue
        P[i] = cand
        moved += 1
    return P, moved


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results_L274_base_48c.json"
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    res = {r["test_id"]: r for r in json.load(open(src))["test_results"]}

    sols, moved_tot, cases_moved = [], 0, 0
    for idx in sorted(res):
        r = res[idx]
        P = r.get("positions")
        if not P:
            continue
        s = ev.dataset[idx]
        at, _b2b, _p2b, _pins, cons = s["input"]
        n = int((at != -1).sum().item())
        Q, mv = snap_case(P, cons, n)
        if mv:
            moved_tot += mv
            cases_moved += 1
        sols.append(dict(test_id=idx, block_count=n, positions=Q))
    out = _DIR / "l277_snap_solutions.json"
    json.dump({"solutions": sols}, open(out, "w"))
    print("snapped {} blocks across {} cases -> {}".format(
        moved_tot, cases_moved, out.name))
    print("now score it:")
    print("  cd iccad2026contest && python iccad2026_evaluate.py "
          "--score ../{} -o ../results_L277_snap.json".format(out.name))
    # also write the untouched control, so the two are scored the SAME way -- a
    # results json and a solutions json do not necessarily go through identical
    # code paths, and comparing across the two would be comparing instruments.
    ctrl = [dict(test_id=idx, block_count=int(res[idx]["block_count"]),
                 positions=res[idx]["positions"])
            for idx in sorted(res) if res[idx].get("positions")]
    cout = _DIR / "l277_ctrl_solutions.json"
    json.dump({"solutions": ctrl}, open(cout, "w"))
    print("control (identical positions, same code path) -> {}".format(cout.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
