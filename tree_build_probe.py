"""OFFLINE ONLY — M29 Phase B: connectivity-driven builder prototype (never shipped).

Phase A (tree_decode_probe.py) established the renderer: exact X comes from the
B*-tree X-rule and, given a good Y-ORDER, gravity reproduces fp_sol. At TEST time
there is NO tree, so this prototype BUILDS a placement from connectivity only:
greedy insertion (most-connected-first), B*-tree-style candidate positions
(right of / above placed blocks), FREE ASPECT within +/-1% area, gravity render.

It scores the result on the validation set with the EXACT evaluator and compares
per-case to our shipped portfolio (optimizer_constructive_results.json, 1.3843).
First real headroom signal for the reconstruction route.

Scope: skips cases with preplaced blocks (a pure greedy packing cannot pin exact
positions); fixed-shape blocks use exact target dims; MIB groups share dims.

Run:  C:/Users/Nordra/.conda/envs/iccadv/python.exe tree_build_probe.py
"""
import sys, math, json
from pathlib import Path

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import (ContestEvaluator, evaluate_solution,
                                calculate_hpwl_b2b, calculate_hpwl_p2b,
                                calculate_bbox_area, compute_total_score)

ASPECTS = [1.0, 1.5, 2.0, 0.667, 0.5]   # w/h ratios tried for movable blocks
WIRE_W = 2.0                            # marginal-cost weight: area + WIRE_W*wire


def build_case(n, at, b2b, p2b, pins, cons, tp):
    """Greedy connectivity-driven packer with free aspect. Preplaced blocks are
    pinned at their exact target positions (obstacles); fixed-shape blocks use
    exact target dims; MIB groups share dims; everything else is free aspect."""
    fixed = [int(cons[i][0]) != 0 for i in range(n)]
    preplaced = [int(cons[i][1]) != 0 for i in range(n)]
    mib = [int(cons[i][2]) for i in range(n)]
    area = [float(at[i]) for i in range(n)]

    # adjacency
    adj = [[] for _ in range(n)]
    deg = [0.0] * n
    for e in b2b.tolist():
        i, j, w = int(e[0]), int(e[1]), float(e[2])
        if 0 <= i < n and 0 <= j < n:
            adj[i].append((j, w)); adj[j].append((i, w))
            deg[i] += w; deg[j] += w
    pin_adj = [[] for _ in range(n)]
    for e in p2b.tolist():
        pi, j, w = int(e[0]), int(e[1]), float(e[2])
        if 0 <= j < n and 0 <= pi < len(pins):
            pin_adj[j].append((float(pins[pi][0]), float(pins[pi][1]), w))

    # MIB group -> shared (w,h): square of mean area in group
    mib_dim = {}
    for g in set(mib):
        if g <= 0:
            continue
        idx = [i for i in range(n) if mib[i] == g]
        s = math.sqrt(sum(area[i] for i in idx) / len(idx))
        mib_dim[g] = (s, s)

    def dims_for(i):
        if fixed[i]:
            return [(float(tp[i][2]), float(tp[i][3]))]
        if mib[i] > 0:
            return [mib_dim[mib[i]]]
        out = []
        for r in ASPECTS:
            w = math.sqrt(area[i] * r); h = area[i] / w
            out.append((w, h))
        return out

    px = [0.0] * n; py = [0.0] * n; pw = [0.0] * n; ph = [0.0] * n
    placed = []; placed_set = set()

    def skyline(x0, x1):
        top = 0.0
        for c in placed:
            if px[c] < x1 - 1e-9 and px[c] + pw[c] > x0 + 1e-9:
                top = max(top, py[c] + ph[c])
        return top

    def wire(i, cx, cy):
        s = 0.0
        for (j, w) in adj[i]:
            if pw[j] > 0 or j in placed_set:
                s += w * (abs(cx - (px[j] + pw[j] / 2)) + abs(cy - (py[j] + ph[j] / 2)))
        for (xx, yy, w) in pin_adj[i]:
            s += w * (abs(cx - xx) + abs(cy - yy))
        return s

    def pack(Wt):
        """Greedy fill into a frame of target width Wt (prevents tower degeneracy)."""
        for i in range(n):
            px[i] = py[i] = pw[i] = ph[i] = 0.0
        placed.clear(); placed_set.clear()
        for i in range(n):
            if preplaced[i]:
                px[i] = float(tp[i][0]); py[i] = float(tp[i][1])
                pw[i] = float(tp[i][2]); ph[i] = float(tp[i][3])
                placed.append(i); placed_set.add(i)
        if not placed:
            sd = max(range(n), key=lambda i: deg[i])
            pw[sd], ph[sd] = dims_for(sd)[0]
            placed.append(sd); placed_set.add(sd)
        remaining = set(range(n)) - placed_set
        while remaining:
            nb = max(remaining,
                     key=lambda i: (sum(w for (j, w) in adj[i] if j in placed_set), deg[i]))
            bx1 = max(px[c] + pw[c] for c in placed)
            by1 = max(py[c] + ph[c] for c in placed)
            best = None
            xc = {0.0}
            for c in placed:
                xc.add(px[c] + pw[c]); xc.add(px[c])
            for (w, h) in dims_for(nb):
                cands = [x for x in xc if x + w <= Wt + 1e-6] or [0.0]
                for x in cands:
                    y = skyline(x, x + w)
                    cx, cy = x + w / 2, y + h / 2
                    darea = max(bx1, x + w) * max(by1, y + h) - bx1 * by1
                    sc = darea + WIRE_W * wire(nb, cx, cy)
                    if best is None or sc < best[0]:
                        best = (sc, x, y, w, h)
            _, x, y, w, h = best
            px[nb] = x; py[nb] = y; pw[nb] = w; ph[nb] = h
            placed.append(nb); placed_set.add(nb); remaining.discard(nb)
        return [(px[i], py[i], pw[i], ph[i]) for i in range(n)]

    total_area = sum(area)
    best_pos = None
    for asp in (0.6, 1.0, 1.6):
        Wt = math.sqrt(total_area * asp)
        pos = pack(Wt)
        score = calculate_bbox_area(pos) + WIRE_W * (
            calculate_hpwl_b2b(pos, b2b) + calculate_hpwl_p2b(pos, p2b, pins))
        if best_pos is None or score < best_pos[0]:
            best_pos = (score, pos)
    return best_pos[1]


def main():
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    PORT = json.load(open(_DIR / "iccad2026contest" / "optimizer_constructive_results.json"))
    ours = {t["test_id"]: t for t in PORT["test_results"]}

    rows = []
    for idx in range(100):
        s = ev.dataset[idx]
        at, b2b, p2b, pins, cons = s["input"]
        n = int((at != -1).sum().item())
        base, tp = ev._extract_baseline(idx, s["label"], b2b, p2b, pins, n)
        pos = build_case(n, at[:n], b2b, p2b, pins, cons[:n], tp[:n])
        if pos is None:
            continue
        m = evaluate_solution({"positions": pos, "runtime": 1.0}, base, cons[:n],
                              b2b, p2b, pins, at[:n], target_positions=tp[:n],
                              median_runtime=1.0)
        rows.append(dict(idx=idx, n=n, w=math.exp(n / 12.0),
                         cost=float(m.cost), feas=bool(m.is_feasible),
                         agap=float(m.area_gap), hgap=float(m.hpwl_gap),
                         our=float(ours[idx]["cost"])))

    nfe = sum(1 for r in rows if not r["feas"])
    wins = [r for r in rows if r["feas"] and r["cost"] < r["our"]]
    totW = sum(r["w"] for r in rows)
    print("=" * 72)
    print(f"builder prototype on {len(rows)}/100 no-preplaced cases  "
          f"(skipped {100 - len(rows)} preplaced)")
    print(f"infeasible: {nfe}/{len(rows)}")
    if rows:
        bt = sum(r["w"] * r["cost"] for r in rows) / totW
        ot = sum(r["w"] * r["our"] for r in rows) / totW
        print(f"weighted mean cost  builder={bt:.4f}  ours={ot:.4f}  "
              f"(portfolio Total=1.3843)")
        print(f"per-case WINS vs ours: {len(wins)}/{len(rows)}")
        for r in sorted(wins, key=lambda r: r["our"] - r["cost"], reverse=True)[:10]:
            print(f"  idx {r['idx']:>2} n={r['n']:>3}  builder {r['cost']:.4f} "
                  f"< ours {r['our']:.4f}  (agap {r['agap']:.3f} hgap {r['hgap']:.3f})")
        worst = sorted(rows, key=lambda r: r["cost"] - r["our"], reverse=True)[:5]
        print("worst vs ours:")
        for r in worst:
            print(f"  idx {r['idx']:>2} n={r['n']:>3}  builder {r['cost']:.4f} "
                  f"vs ours {r['our']:.4f}  feas={r['feas']}")


if __name__ == "__main__":
    main()
