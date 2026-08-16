"""L135b — pull the bbox in to the preplaced extent, and re-score officially.

THE BUG. `constructive_l124.cpp:51` sets `MARGIN = 1e-4` and the frame is sized
`max(w, pre_w + MARGIN)` (`:683`, `:1972`), so the outline is deliberately 1e-4
larger than the preplaced extent it must contain. A PREPLACED block that also
carries a boundary requirement then sits at the true edge (e.g. right edge
141.0) while the bbox runs to 141.0001 -- and the evaluator's threshold is 1e-6,
so it scores a boundary violation **that the layout can never satisfy**. Its
position is a HARD constraint, so it cannot be moved to meet the bbox.

Audited on the shipped 48c result: 11 such blocks, all preplaced.

THE ONLY AVAILABLE FIX is the other direction: pull the bbox IN to the preplaced
block, by sliding every block that overhangs past it back by the excess. Safe
only when nothing is in the way, so each side is attempted and kept only if it
verifies.

  * moves are <= 1e-3 and always INWARD, so the bbox shrinks (area_gap can only
    improve) and no block leaves the outline;
  * every destination is checked with the evaluator's own 1e-6 overlap rule;
  * preplaced blocks are never moved;
  * the L131 abutment repair runs afterwards, because an inward slide can
    separate a block from its cluster mates (measured: that is exactly how the
    first version of l135_bnd_verify.py lost 1.14%);
  * the whole case is REVERTED unless the official cost actually improves.

READ-ONLY with respect to the submission.

  <python> -u l135_shrink_verify.py results_L114_48c_lp_anchor.json
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

OVL = 1e-6
EPS = 1e-6
MAXPULL = 1e-3


def overlap_free(P, n, moved_idx):
    for i in moved_idx:
        for t in range(n):
            if t == i:
                continue
            ox = min(P[i][0] + P[i][2], P[t][0] + P[t][2]) - max(P[i][0], P[t][0])
            oy = min(P[i][1] + P[i][3], P[t][1] + P[t][3]) - max(P[i][1], P[t][1])
            if ox > OVL and oy > OVL:
                return False
    return True


def shrink_to_preplaced(P, cons, n):
    """For each side, if a preplaced boundary block misses by a hair, pull the
    overhanging blocks in by exactly that hair. Returns the list of sides done."""
    done = []
    for side in ("L", "R", "B", "T"):
        x0 = min(p[0] for p in P)
        y0 = min(p[1] for p in P)
        x1 = max(p[0] + p[2] for p in P)
        y1 = max(p[1] + p[3] for p in P)
        bit = {"L": 1, "R": 2, "T": 4, "B": 8}[side]

        # the smallest miss among preplaced blocks that require this side
        excess = None
        for i in range(n):
            if not (int(cons[i][4]) & bit) or int(cons[i][1]) == 0:
                continue
            bx, by, bw, bh = P[i]
            d = {"L": bx - x0, "R": x1 - (bx + bw),
                 "B": by - y0, "T": y1 - (by + bh)}[side]
            if EPS <= d <= MAXPULL and (excess is None or d < excess):
                excess = d
        if excess is None:
            continue

        # every block overhanging past the preplaced extent moves in by `excess`
        idx = []
        for i in range(n):
            if int(cons[i][1]) != 0:              # preplaced cannot move
                continue
            bx, by, bw, bh = P[i]
            if side == "R" and x1 - (bx + bw) < excess - 1e-12:
                idx.append(i)
            elif side == "L" and bx - x0 < excess - 1e-12:
                idx.append(i)
            elif side == "T" and y1 - (by + bh) < excess - 1e-12:
                idx.append(i)
            elif side == "B" and by - y0 < excess - 1e-12:
                idx.append(i)
        if not idx:
            continue
        # a preplaced block already sitting outside the shrunk outline would be
        # pushed out of the bbox -- then the pull cannot work at all
        blocked = False
        for i in range(n):
            if int(cons[i][1]) == 0:
                continue
            bx, by, bw, bh = P[i]
            if side == "R" and bx + bw > x1 - excess + 1e-12:
                blocked = blocked or (x1 - (bx + bw) < excess - 1e-12)
            if side == "L" and bx < x0 + excess - 1e-12:
                blocked = blocked or (bx - x0 < excess - 1e-12)
            if side == "T" and by + bh > y1 - excess + 1e-12:
                blocked = blocked or (y1 - (by + bh) < excess - 1e-12)
            if side == "B" and by < y0 + excess - 1e-12:
                blocked = blocked or (by - y0 < excess - 1e-12)
        if blocked:
            continue

        save = {i: (P[i][0], P[i][1]) for i in idx}
        for i in idx:
            if side == "R":
                P[i][0] -= excess
            elif side == "L":
                P[i][0] += excess
            elif side == "T":
                P[i][1] -= excess
            else:
                P[i][1] += excess
        if overlap_free(P, n, idx):
            done.append((side, excess, len(idx)))
        else:
            for i, (ox, oy) in save.items():
                P[i][0], P[i][1] = ox, oy
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
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
        done = shrink_to_preplaced(P1, cons, n)
        if done:
            P1 = [list(q) for q in oc._snap_group_abutment(
                [tuple(q) for q in P1], cons, n)]
        m1 = score(P1)
        # keep only what actually helps, judged by the official cost
        if not done or not m1.is_feasible or float(m1.cost) >= float(m0.cost):
            m1 = m0
            kept = []
        else:
            kept = done

        w = math.exp(n / 12.0)
        wsum += w
        c_old += w * float(m0.cost)
        c_new += w * float(m1.cost)
        inf_b += 0 if m0.is_feasible else 1
        inf_a += 0 if m1.is_feasible else 1
        if kept:
            rows.append((idx, n, kept, float(m0.cost), float(m1.cost),
                         int(m0.boundary_violations), int(m1.boundary_violations)))

    print(f"\n=== L135b bbox pull-in: {a.results} ===\n")
    print(f"{'case':>5} {'n':>4} {'cost before':>12} {'cost after':>12} {'vbnd':>9}  sides")
    for idx, n, kept, c0, c1, b0, b1 in rows:
        sides = ",".join(f"{s}{e:.0e}x{k}" for s, e, k in kept)
        print(f"{idx:>5} {n:>4} {c0:>12.6f} {c1:>12.6f} "
              f"{str(b0) + '->' + str(b1):>9}  {sides}")
    print(f"\ncases improved                 {len(rows)}")
    print(f"boundary violations removed    {sum(r[5] - r[6] for r in rows)}")
    print(f"infeasible before / after      {inf_b} / {inf_a}")
    print(f"\nweighted total BEFORE          {c_old / wsum:.12f}")
    print(f"weighted total AFTER           {c_new / wsum:.12f}")
    print(f"worth                          "
          f"{100 * (c_old - c_new) / max(c_old, 1e-9):+.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
