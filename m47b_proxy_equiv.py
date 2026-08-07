#!/usr/bin/env python3
"""M47b equivalence gate: fast tolist-based proxy vs shipped _proxy_metrics.

The fast path must be BIT-IDENTICAL: same fp32->fp64 exact widening
(tolist == float(tensor scalar)), same edge order, same accumulation order,
same formulas on python floats. Any mismatch on any (case, profile) is FAIL.
Runs 3 representative profiles x 100 cases; also times both versions.
"""
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet")
sys.path.insert(0, str(REPO / "iccad2026contest"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import optimizer_constructive as oc  # noqa: E402
import iccad2026_evaluate as ev_mod  # noqa: E402
from optimizer_claude import _serialize_input  # noqa: E402


# ---- candidate fast implementations (to be merged into the wrapper if PASS) --
def _hpwl_b2b_fast(positions, b2b):
    if b2b is None or len(b2b) == 0:
        return 0.0
    total_wl = 0.0
    np_ = len(positions)
    for r in b2b.tolist():
        if r[0] == -1:
            continue
        i, j, weight = int(r[0]), int(r[1]), r[2]
        if i < np_ and j < np_:
            x1 = positions[i][0] + positions[i][2] / 2
            y1 = positions[i][1] + positions[i][3] / 2
            x2 = positions[j][0] + positions[j][2] / 2
            y2 = positions[j][1] + positions[j][3] / 2
            total_wl += weight * (abs(x2 - x1) + abs(y2 - y1))
    return total_wl


def _hpwl_p2b_fast(positions, p2b, pins):
    if p2b is None or len(p2b) == 0:
        return 0.0
    total_wl = 0.0
    np_ = len(positions)
    pins_l = pins.tolist() if pins is not None else []
    for r in p2b.tolist():
        if r[0] == -1:
            continue
        pin_idx, block_idx, weight = int(r[0]), int(r[1]), r[2]
        if block_idx < np_ and pin_idx < len(pins_l):
            px, py = pins_l[pin_idx][0], pins_l[pin_idx][1]
            bx = positions[block_idx][0] + positions[block_idx][2] / 2
            by = positions[block_idx][1] + positions[block_idx][3] / 2
            total_wl += weight * (abs(px - bx) + abs(py - by))
    return total_wl


def _proxy_metrics_fast(positions, area_targets, b2b, p2b, pins, constraints, n):
    xmin = min(p[0] for p in positions); ymin = min(p[1] for p in positions)
    xmax = max(p[0] + p[2] for p in positions); ymax = max(p[1] + p[3] for p in positions)
    area = (xmax - xmin) * (ymax - ymin)
    hpwl = _hpwl_b2b_fast(positions, b2b) + _hpwl_p2b_fast(positions, p2b, pins)

    ncols = constraints.shape[1] if constraints.dim() > 1 else 0
    vb = vg = vm = 0
    nsoft = 0
    if ncols > 4:
        bound_l = constraints[:n, 4].tolist()
        clust_l = constraints[:n, 3].tolist()
        mib_l = constraints[:n, 2].tolist()
        nsoft = sum(1 for b in bound_l if b != 0)
        eps = 1e-6
        for i in range(n):
            code = int(bound_l[i])
            if code == 0:
                continue
            bx, by, bw, bh = positions[i]
            ok = True
            if code & 1: ok = ok and abs(bx - xmin) < eps
            if code & 2: ok = ok and abs(bx + bw - xmax) < eps
            if code & 4: ok = ok and abs(by + bh - ymax) < eps
            if code & 8: ok = ok and abs(by - ymin) < eps
            if not ok:
                vb += 1
        ngrp = int(max(clust_l)) if clust_l else 0
        for g in range(1, ngrp + 1):
            idx = [i for i in range(n) if int(clust_l[i]) == g]
            nsoft += max(0, len(idx) - 1)
            if len(idx) > 1 and oc._SHAPELY:
                u = oc._unary_union([oc._box(positions[i][0], positions[i][1],
                                             positions[i][0] + positions[i][2],
                                             positions[i][1] + positions[i][3])
                                     for i in idx])
                if u.geom_type == "MultiPolygon":
                    vg += len(u.geoms) - 1
        nmib = int(max(mib_l)) if mib_l else 0
        for g in range(1, nmib + 1):
            idx = [i for i in range(n) if int(mib_l[i]) == g]
            nsoft += max(0, len(idx) - 1)
            shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in idx}
            vm += len(shapes) - 1
    vrel = (vb + vg + vm) / max(nsoft, 1)
    return {"area": area, "hpwl": hpwl, "vrel": vrel}
# ------------------------------------------------------------------------------

ev = ev_mod.ContestEvaluator(data_path=str(REPO), verbose=False)
ev._load_dataset()
opt = oc.MyOptimizer(verbose=False)
assert opt._ok

PROF_IDX = (0, 2, 18)   # base, FREE-stack max-setters #2/#18
bad = 0
t_old = t_new = 0.0
ncmp = 0
for idx in range(len(ev.dataset)):
    sample = ev.dataset[idx]
    area_target, b2b, p2b, pins, constraints = sample['input']
    n = int((area_target != -1).sum().item())
    _, target_pos = ev._extract_baseline(idx, sample['label'], b2b, p2b, pins, n)
    opt_target_pos = torch.full((n, 4), -1.0)
    if target_pos is not None and constraints is not None:
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(n):
            if nc > 1 and constraints[i, 1] != 0:
                tx, ty, tw, th = target_pos[i]
                opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
            elif nc > 0 and constraints[i, 0] != 0:
                _, _, tw, th = target_pos[i]
                opt_target_pos[i, 2] = tw
                opt_target_pos[i, 3] = th
    inp = _serialize_input(n, area_target, b2b, p2b, pins, constraints,
                           opt_target_pos, gnn_hint=None)
    for pi in PROF_IDX:
        pos = oc._run_profile(oc._PROFILES[pi], inp, n)
        if pos is None:
            continue
        t0 = time.perf_counter()
        m_old = oc._proxy_metrics(pos, area_target, b2b, p2b, pins, constraints, n)
        t1 = time.perf_counter()
        m_new = _proxy_metrics_fast(pos, area_target, b2b, p2b, pins, constraints, n)
        t2 = time.perf_counter()
        t_old += t1 - t0
        t_new += t2 - t1
        ncmp += 1
        if m_old != m_new:
            bad += 1
            print(f"MISMATCH case {idx} prof {pi}: old={m_old} new={m_new}")
    if idx % 20 == 19:
        print(f"...{idx+1} cases, {ncmp} comparisons, mismatches={bad}, "
              f"old={t_old:.2f}s new={t_new:.2f}s")

print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {ncmp} comparisons, mismatches={bad}")
print(f"proxy time old={t_old:.2f}s new={t_new:.2f}s speedup x{t_old/max(t_new,1e-9):.1f}")
