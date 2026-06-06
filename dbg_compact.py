"""Prototype: boundary-contact-preserving compaction on constructive.exe output.

Measures the area-gap ceiling achievable by squeezing void out of the packed
layout WITHOUT moving preplaced blocks, creating overlaps, breaking boundary
edge-contact, or fragmenting clusters. Selection mirrors the C++ layout_score
(downside-protected), and true cost comes from the harness evaluate_solution.

Run: python dbg_compact.py            (base profile, all 100 cases)
"""
import os, sys, math, subprocess
from pathlib import Path
import torch
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import (ContestEvaluator, evaluate_solution,
                                calculate_hpwl_b2b, calculate_hpwl_p2b)
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

EXE = str(_DIR / "constructive.exe")
TOL = 1e-6
B_LEFT, B_RIGHT, B_TOP, B_BOTTOM = 1, 2, 4, 8


def _bbox(xs, ys, ws, hs):
    xmin = min(xs); ymin = min(ys)
    xmax = max(xs[i] + ws[i] for i in range(len(xs)))
    ymax = max(ys[i] + hs[i] for i in range(len(xs)))
    return xmin, ymin, xmax, ymax


def _yov(i, j, ys, hs):
    return ys[i] < ys[j] + hs[j] - TOL and ys[j] < ys[i] + hs[i] - TOL


def _xov(i, j, xs, ws):
    return xs[i] < xs[j] + ws[j] - TOL and xs[j] < xs[i] + ws[i] - TOL


def pack_left(xs, ys, ws, hs, pre, anchor):
    n = len(xs); new = list(xs); placed = [False] * n
    for i in sorted(range(n), key=lambda i: xs[i]):
        if i in pre:
            new[i] = xs[i]; placed[i] = True; continue
        lb = anchor
        for j in range(n):
            if placed[j] and _yov(i, j, ys, hs):
                lb = max(lb, new[j] + ws[j])
        new[i] = min(xs[i], lb); placed[i] = True
    return new


def pack_right(xs, ys, ws, hs, pre, anchor):
    n = len(xs); new = list(xs); placed = [False] * n
    for i in sorted(range(n), key=lambda i: -(xs[i] + ws[i])):
        if i in pre:
            new[i] = xs[i]; placed[i] = True; continue
        ub = anchor
        for j in range(n):
            if placed[j] and _yov(i, j, ys, hs):
                ub = min(ub, new[j])
        new[i] = max(xs[i], ub - ws[i]); placed[i] = True
    return new


def pack_down(xs, ys, ws, hs, pre, anchor):
    n = len(ys); new = list(ys); placed = [False] * n
    for i in sorted(range(n), key=lambda i: ys[i]):
        if i in pre:
            new[i] = ys[i]; placed[i] = True; continue
        lb = anchor
        for j in range(n):
            if placed[j] and _xov(i, j, xs, ws):
                lb = max(lb, new[j] + hs[j])
        new[i] = min(ys[i], lb); placed[i] = True
    return new


def pack_up(xs, ys, ws, hs, pre, anchor):
    n = len(ys); new = list(ys); placed = [False] * n
    for i in sorted(range(n), key=lambda i: -(ys[i] + hs[i])):
        if i in pre:
            new[i] = ys[i]; placed[i] = True; continue
        ub = anchor
        for j in range(n):
            if placed[j] and _xov(i, j, xs, ws):
                ub = min(ub, new[j])
        new[i] = max(ys[i], ub - hs[i]); placed[i] = True
    return new


def _overlaps(i, nx, ny, xs, ys, ws, hs):
    n = len(xs)
    for j in range(n):
        if j == i:
            continue
        if (nx < xs[j] + ws[j] - TOL and xs[j] < nx + ws[i] - TOL and
                ny < ys[j] + hs[j] - TOL and ys[j] < ny + hs[i] - TOL):
            return True
    return False


def boundary_nudge(xs, ys, ws, hs, codes, pre, clus):
    """Snap non-cluster single boundary blocks to the (possibly new) bbox edge."""
    n = len(xs)
    xmin, ymin, xmax, ymax = _bbox(xs, ys, ws, hs)
    for i in range(n):
        if codes[i] == 0 or i in pre or clus[i] > 0:
            continue
        nx, ny = xs[i], ys[i]
        if codes[i] & B_LEFT:   nx = xmin
        if codes[i] & B_RIGHT:  nx = xmax - ws[i]
        if codes[i] & B_BOTTOM: ny = ymin
        if codes[i] & B_TOP:    ny = ymax - hs[i]
        if not _overlaps(i, nx, ny, xs, ys, ws, hs):
            xs[i], ys[i] = nx, ny


def count_bv(xs, ys, ws, hs, codes):
    n = len(xs)
    xmin, ymin, xmax, ymax = _bbox(xs, ys, ws, hs)
    bad = 0
    for i in range(n):
        c = codes[i]
        if c == 0:
            continue
        ok = True
        if c & B_LEFT:   ok = ok and abs(xs[i] - xmin) <= TOL
        if c & B_RIGHT:  ok = ok and abs(xs[i] + ws[i] - xmax) <= TOL
        if c & B_TOP:    ok = ok and abs(ys[i] + hs[i] - ymax) <= TOL
        if c & B_BOTTOM: ok = ok and abs(ys[i] - ymin) <= TOL
        if not ok:
            bad += 1
    return bad


def count_gf(xs, ys, ws, hs, clus):
    """Group fragments via union-find on edge-touching (C++ 1e-3 tol)."""
    n = len(xs)
    groups = {}
    for i in range(n):
        if clus[i] > 0:
            groups.setdefault(clus[i], []).append(i)
    frag = 0
    for m in groups.values():
        k = len(m)
        if k <= 1:
            continue
        par = list(range(k))
        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for a in range(k):
            for b in range(a + 1, k):
                ia, ib = m[a], m[b]
                tx = ((abs(xs[ia] + ws[ia] - xs[ib]) <= 1e-3 or
                       abs(xs[ib] + ws[ib] - xs[ia]) <= 1e-3) and
                      (ys[ia] < ys[ib] + hs[ib] - TOL and ys[ib] < ys[ia] + hs[ia] - TOL))
                ty = ((abs(ys[ia] + hs[ia] - ys[ib]) <= 1e-3 or
                       abs(ys[ib] + hs[ib] - ys[ia]) <= 1e-3) and
                      (xs[ia] < xs[ib] + ws[ib] - TOL and xs[ib] < xs[ia] + ws[ia] - TOL))
                if tx or ty:
                    par[find(a)] = find(b)
        roots = {find(a) for a in range(k)}
        frag += len(roots) - 1
    return frag


def layout_score(xs, ys, ws, hs, codes, clus, b2b, p2b, pins, n):
    xmin, ymin, xmax, ymax = _bbox(xs, ys, ws, hs)
    area = (xmax - xmin) * (ymax - ymin)
    pos = [(xs[i], ys[i], ws[i], hs[i]) for i in range(n)]
    hpwl = calculate_hpwl_b2b(pos, b2b) + calculate_hpwl_p2b(pos, p2b, pins)
    bv = count_bv(xs, ys, ws, hs, codes)
    gf = count_gf(xs, ys, ws, hs, clus)
    hw = 0.12 if n >= 116 else 0.06
    return area + hw * hpwl + 150000.0 * bv + 6500.0 * gf


def _nsoft(codes, clus, ws, hs, n):
    ns = sum(1 for c in codes if c)
    cl = {}; mb = {}
    for i in range(n):
        if clus[i] > 0:
            cl[clus[i]] = cl.get(clus[i], 0) + 1
    for g, c in cl.items():
        ns += max(0, c - 1)
    return max(ns, 1)


def csc(xs, ys, ws, hs, codes, clus, b2b, p2b, pins, n, nsoft):
    """Proxy of the true cost for compaction selection: weights bv and gf equally
    and normalises by nsoft (mirrors cost's exp(2*vrel))."""
    xmin, ymin, xmax, ymax = _bbox(xs, ys, ws, hs)
    area = (xmax - xmin) * (ymax - ymin)
    pos = [(xs[i], ys[i], ws[i], hs[i]) for i in range(n)]
    hpwl = calculate_hpwl_b2b(pos, b2b) + calculate_hpwl_p2b(pos, p2b, pins)
    bv = count_bv(xs, ys, ws, hs, codes)
    gf = count_gf(xs, ys, ws, hs, clus)
    hw = 0.12 if n >= 116 else 0.06
    return (area + hw * hpwl) * math.exp(2.0 * (bv + gf) / nsoft)


def compact(pos, codes, pre, clus, b2b, p2b, pins, n):
    """Return best layout among original + directional-pack candidates."""
    xs0 = [p[0] for p in pos]; ys0 = [p[1] for p in pos]
    ws = [p[2] for p in pos]; hs = [p[3] for p in pos]
    nsoft = _nsoft(codes, clus, ws, hs, n)

    def mk(xs, ys):
        xs, ys = list(xs), list(ys)
        boundary_nudge(xs, ys, ws, hs, codes, pre, clus)
        return xs, ys

    xmin, ymin, xmax, ymax = _bbox(xs0, ys0, ws, hs)
    cands = [(list(xs0), list(ys0))]  # original (no extra nudge; exe already did)

    xL = pack_left(xs0, ys0, ws, hs, pre, xmin)
    xR = pack_right(xs0, ys0, ws, hs, pre, xmax)
    yD = pack_down(xs0, ys0, ws, hs, pre, ymin)
    yU = pack_up(xs0, ys0, ws, hs, pre, ymax)
    cands.append(mk(xL, ys0))
    cands.append(mk(xR, ys0))
    cands.append(mk(xs0, yD))
    cands.append(mk(xs0, yU))
    # combos: pack one axis, then the other on the packed result
    for xx in (xL, xR):
        yd = pack_down(xx, ys0, ws, hs, pre, ymin)
        yu = pack_up(xx, ys0, ws, hs, pre, ymax)
        cands.append(mk(xx, yd))
        cands.append(mk(xx, yu))
    for yy in (yD, yU):
        xl = pack_left(xs0, yy, ws, hs, pre, xmin)
        xr = pack_right(xs0, yy, ws, hs, pre, xmax)
        cands.append(mk(xl, yy))
        cands.append(mk(xr, yy))

    best = None; best_sc = None
    for xs, ys in cands:
        sc = csc(xs, ys, ws, hs, codes, clus, b2b, p2b, pins, n, nsoft)
        if best_sc is None or sc < best_sc:
            best_sc = sc; best = (xs, ys)
    xs, ys = best
    return [(xs[i], ys[i], ws[i], hs[i]) for i in range(n)]


def main():
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
    totW = 0.0; totC0 = 0.0; totC1 = 0.0
    rows = []
    for idx in range(len(ev.dataset)):
        s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
        at, b2b, p2b, pins, cons = inp
        n = int((at != -1).sum().item())
        base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
        otp = build_opt_target_pos(tp, cons, n)
        txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
        ps = _parse_output(subprocess.run([EXE], input=txt, capture_output=True,
                                          text=True).stdout, n)
        codes = [int(cons[i, 4]) if cons.dim() > 1 and cons.shape[1] > 4 else 0
                 for i in range(n)]
        clus = [int(cons[i, 3]) if cons.dim() > 1 and cons.shape[1] > 3 else 0
                for i in range(n)]
        pre = {i for i in range(n)
               if cons.dim() > 1 and cons.shape[1] > 1 and int(cons[i, 1]) != 0}
        ps2 = compact(ps, codes, pre, clus, b2b, p2b, pins, n)

        m0 = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                               b2b, p2b, pins, at[:n], target_positions=tp[:n],
                               median_runtime=1.0)
        m1 = evaluate_solution({'positions': ps2, 'runtime': 1.0}, base, cons[:n],
                               b2b, p2b, pins, at[:n], target_positions=tp[:n],
                               median_runtime=1.0)
        w = math.exp(n / 12.0)
        totW += w; totC0 += w * m0.cost; totC1 += w * m1.cost
        rows.append((idx, n, w, m0.cost, m1.cost, m0.area_gap, m1.area_gap,
                     m0.hpwl_gap, m1.hpwl_gap, m1.is_feasible,
                     m0.boundary_violations, m1.boundary_violations,
                     m0.grouping_violations, m1.grouping_violations))

    nreg = sum(1 for r in rows if r[4] > r[3] + 1e-6)
    wreg = sum(r[2] * (r[4] - r[3]) for r in rows if r[4] > r[3] + 1e-6) / totW
    print(f"Total Score: orig={totC0/totW:.4f}  compacted={totC1/totW:.4f}  "
          f"delta={100*(totC1-totC0)/totC0:+.2f}%")
    print(f"regressions: {nreg}/100 cases, weighted cost added={wreg:+.4f}")
    print(f"{'case':>4} {'n':>4} {'wt%':>5} {'cost0':>6} {'cost1':>6} "
          f"{'ag0':>6} {'ag1':>6} {'hg0':>6} {'hg1':>6} {'bv':>7} {'gf':>7} {'feas':>5}")
    rows.sort(key=lambda r: -(r[2] * (r[3] - r[4])))  # biggest cost win first
    for (idx, n, w, c0, c1, a0, a1, h0, h1, fe, b0, b1, g0, g1) in rows[:30]:
        print(f"{idx:>4} {n:>4} {100*w/totW:5.2f} {c0:6.3f} {c1:6.3f} "
              f"{a0:6.3f} {a1:6.3f} {h0:6.3f} {h1:6.3f} {b0:>3}->{b1:<3} "
              f"{g0:>3}->{g1:<3} {'' if fe else 'INFEAS'}")


if __name__ == "__main__":
    main()
