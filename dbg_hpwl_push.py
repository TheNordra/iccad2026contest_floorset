"""Prototype: post-placement HPWL slack push on the PORTFOLIO's output positions.

hgap is the dominant cost lever (weighted 0.412 vs agap 0.228 vs vrel 0.040).
Compaction only packs toward frame faces (area) and can spread connected blocks
apart, lifting HPWL. This prototype slides every FREE SINGLE block (no boundary
code, no cluster, not preplaced) toward its connectivity-weighted L1-median within
available void, keeping it inside the current bbox.

Such moves are downside-free by construction: free singles don't define boundary
or grouping violations and stay inside the bbox, so area / bv / gf / mib are all
unchanged -- only HPWL drops. We additionally accept a move only if it lowers the
true HPWL (guard against the rare non-monotone case).

Measures the TRUE-cost headroom of this lever on the deployed layouts (portfolio
JSON positions) via the harness evaluate_solution. If meaningful, port to C++.

Run: python dbg_hpwl_push.py
"""
import json, math, sys
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import (ContestEvaluator, evaluate_solution,
                                calculate_hpwl_b2b, calculate_hpwl_p2b)

TOL = 1e-6
PASSES = 8


def _bbox(pos):
    xmin = min(p[0] for p in pos); ymin = min(p[1] for p in pos)
    xmax = max(p[0] + p[2] for p in pos); ymax = max(p[1] + p[3] for p in pos)
    return xmin, ymin, xmax, ymax


def _adj(b2b, p2b, n):
    """block -> [(neighbor, w)], block -> [(pin, w)]"""
    ba = [[] for _ in range(n)]; pa = [[] for _ in range(n)]
    for e in b2b:
        i, j, w = int(e[0]), int(e[1]), float(e[2])
        if w <= 0 or i < 0 or j < 0 or i >= n or j >= n:
            continue
        ba[i].append((j, w)); ba[j].append((i, w))
    for e in p2b:
        pi, bj, w = int(e[0]), int(e[1]), float(e[2])
        if w <= 0 or bj < 0 or bj >= n:
            continue
        pa[bj].append((pi, w))
    return ba, pa


def _wmedian(targets):
    """Weighted L1 median (minimiser of sum w*|x-t|) over [(t, w)]."""
    if not targets:
        return None
    targets = sorted(targets)
    total = sum(w for _, w in targets)
    acc = 0.0
    for t, w in targets:
        acc += w
        if acc >= total / 2.0:
            return t
    return targets[-1][0]


def push(pos, codes, clus, pre, b2b, p2b, pins, n):
    pos = [list(p) for p in pos]
    ba, pa = _adj(b2b, p2b, n)
    free = [i for i in range(n)
            if codes[i] == 0 and clus[i] == 0 and i not in pre]
    if not free:
        return [tuple(p) for p in pos]
    xmin, ymin, xmax, ymax = _bbox(pos)

    def hpwl_of(i, x, y):
        cx, cy = x + pos[i][2] / 2.0, y + pos[i][3] / 2.0
        h = 0.0
        for j, w in ba[i]:
            ncx = pos[j][0] + pos[j][2] / 2.0; ncy = pos[j][1] + pos[j][3] / 2.0
            h += w * (abs(cx - ncx) + abs(cy - ncy))
        for pidx, w in pa[i]:
            h += w * (abs(cx - pins[pidx][0]) + abs(cy - pins[pidx][1]))
        return h

    for _ in range(PASSES):
        moved = False
        for i in free:
            w_, h_ = pos[i][2], pos[i][3]
            # x-axis: free void interval [lo, hi] for the lower-left x of block i
            lo, hi = xmin, xmax - w_
            for j in range(n):
                if j == i:
                    continue
                if pos[j][1] < pos[i][1] + h_ - TOL and pos[i][1] < pos[j][1] + pos[j][3] - TOL:
                    if pos[j][0] + pos[j][2] <= pos[i][0] + TOL:
                        lo = max(lo, pos[j][0] + pos[j][2])
                    elif pos[j][0] >= pos[i][0] + w_ - TOL:
                        hi = min(hi, pos[j][0] - w_)
            tx = _wmedian([(pos[j][0] + pos[j][2] / 2.0, w) for j, w in ba[i]]
                          + [(pins[pidx][0], w) for pidx, w in pa[i]])
            if tx is not None and hi >= lo - TOL:
                nx = min(max(tx - w_ / 2.0, lo), hi)
                if abs(nx - pos[i][0]) > TOL and hpwl_of(i, nx, pos[i][1]) < hpwl_of(i, pos[i][0], pos[i][1]) - TOL:
                    pos[i][0] = nx; moved = True
            # y-axis
            lo, hi = ymin, ymax - h_
            for j in range(n):
                if j == i:
                    continue
                if pos[j][0] < pos[i][0] + w_ - TOL and pos[i][0] < pos[j][0] + pos[j][2] - TOL:
                    if pos[j][1] + pos[j][3] <= pos[i][1] + TOL:
                        lo = max(lo, pos[j][1] + pos[j][3])
                    elif pos[j][1] >= pos[i][1] + h_ - TOL:
                        hi = min(hi, pos[j][1] - h_)
            ty = _wmedian([(pos[j][1] + pos[j][3] / 2.0, w) for j, w in ba[i]]
                          + [(pins[pidx][1], w) for pidx, w in pa[i]])
            if ty is not None and hi >= lo - TOL:
                ny = min(max(ty - h_ / 2.0, lo), hi)
                if abs(ny - pos[i][1]) > TOL and hpwl_of(i, pos[i][0], ny) < hpwl_of(i, pos[i][0], pos[i][1]) - TOL:
                    pos[i][1] = ny; moved = True
        if not moved:
            break
    return [tuple(p) for p in pos]


def main():
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
    port = json.load(open(_DIR / "iccad2026contest" / "optimizer_constructive_results.json"))
    pjson = {t["test_id"]: t for t in port["test_results"]}

    totW = totC0 = totC1 = 0.0
    rows = []
    for idx in range(len(ev.dataset)):
        s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
        at, b2b, p2b, pins, cons = inp
        n = int((at != -1).sum().item())
        base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
        ps = [tuple(p) for p in pjson[idx]["positions"]]
        codes = [int(cons[i, 4]) if cons.dim() > 1 and cons.shape[1] > 4 else 0 for i in range(n)]
        clus = [int(cons[i, 3]) if cons.dim() > 1 and cons.shape[1] > 3 else 0 for i in range(n)]
        pre = {i for i in range(n) if cons.dim() > 1 and cons.shape[1] > 1 and int(cons[i, 1]) != 0}
        pinsl = [(float(pins[k][0]), float(pins[k][1])) for k in range(pins.shape[0])]
        ps2 = push(ps, codes, clus, pre, b2b.tolist(), p2b.tolist(), pinsl, n)

        m0 = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                               b2b, p2b, pins, at[:n], target_positions=tp[:n], median_runtime=1.0)
        m1 = evaluate_solution({'positions': ps2, 'runtime': 1.0}, base, cons[:n],
                               b2b, p2b, pins, at[:n], target_positions=tp[:n], median_runtime=1.0)
        w = math.exp(n / 12.0)
        totW += w; totC0 += w * m0.cost; totC1 += w * m1.cost
        rows.append((idx, n, w, m0.cost, m1.cost, m0.hpwl_gap, m1.hpwl_gap,
                     m0.area_gap, m1.area_gap, m1.is_feasible,
                     m0.boundary_violations, m1.boundary_violations,
                     m0.grouping_violations, m1.grouping_violations))

    nreg = sum(1 for r in rows if r[4] > r[3] + 1e-6)
    print(f"Total Score: orig={totC0/totW:.4f}  pushed={totC1/totW:.4f}  "
          f"delta={100*(totC1-totC0)/totC0:+.2f}%")
    print(f"regressions: {nreg}/100 cases")
    print(f"{'case':>4} {'n':>4} {'wt%':>5} {'cost0':>6} {'cost1':>6} "
          f"{'hg0':>6} {'hg1':>6} {'ag0':>6} {'ag1':>6} {'bv':>7} {'gf':>7} {'feas':>5}")
    rows.sort(key=lambda r: -(r[2] * (r[3] - r[4])))
    for (idx, n, w, c0, c1, h0, h1, a0, a1, fe, b0, b1, g0, g1) in rows[:30]:
        print(f"{idx:>4} {n:>4} {100*w/totW:5.2f} {c0:6.3f} {c1:6.3f} "
              f"{h0:6.3f} {h1:6.3f} {a0:6.3f} {a1:6.3f} {b0:>3}->{b1:<3} "
              f"{g0:>3}->{g1:<3} {'' if fe else 'INFEAS'}")


if __name__ == "__main__":
    main()
