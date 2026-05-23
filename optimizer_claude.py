#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer (C++ backend)
=======================================================
Python wrapper that compiles optimizer_claude.cpp on first run,
then delegates all work to the C++ binary via subprocess.

The C++ binary implements:
  - Skyline BL packing (overlap-free by construction)
  - Connectivity-driven initial permutation (greedy TSP on b2b graph)
  - SA with: swap, relocate, connectivity move, resize, rotate, MIB unify
  - All constraints: fixed, preplaced, MIB, cluster, boundary
"""

import math
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent / "iccad2026contest"))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)

_DIR = Path(__file__).parent
_CPP  = _DIR / "optimizer_claude.cpp"
_BIN  = _DIR / "optimizer_claude.exe"
_GPP  = r"C:\msys64\ucrt64\bin\g++.exe"

# ── Fallback Python SA (used if C++ compile fails) ────────────────────────────
import random

COL_FIXED     = 0
COL_PREPLACED = 1
COL_MIB       = 2
COL_CLUSTER   = 3
COL_BOUNDARY  = 4

BOUNDARY_LEFT         = 1
BOUNDARY_RIGHT        = 2
BOUNDARY_TOP          = 4
BOUNDARY_BOTTOM       = 8
BOUNDARY_TOP_LEFT     = 5
BOUNDARY_TOP_RIGHT    = 6
BOUNDARY_BOTTOM_LEFT  = 9
BOUNDARY_BOTTOM_RIGHT = 10

TIME_LIMIT = 1.55
AREA_TOL   = 0.005


class Skyline:
    def __init__(self):
        self.pts: List[Tuple[float, float]] = [(0.0, 0.0)]

    def _height_at(self, x: float) -> float:
        for i, (sx, sh) in enumerate(self.pts):
            nx = self.pts[i + 1][0] if i + 1 < len(self.pts) else float('inf')
            if sx <= x < nx:
                return sh
        return 0.0

    def max_height(self, x0: float, x1: float) -> float:
        h = 0.0
        for i, (sx, sh) in enumerate(self.pts):
            nx = self.pts[i + 1][0] if i + 1 < len(self.pts) else x1 + 1.0
            if sx >= x1:
                break
            if nx <= x0:
                continue
            h = max(h, sh)
        return h

    def raise_region(self, x0: float, x1: float, new_h: float):
        all_xs = sorted(set([sx for sx, _ in self.pts] + [x0, x1]))
        new_pts: List[Tuple[float, float]] = []
        prev_h: float = -1.0
        for x in all_xs:
            h = new_h if x0 <= x < x1 else self._height_at(x)
            if h != prev_h:
                new_pts.append((x, h))
                prev_h = h
        self.pts = new_pts if new_pts else [(0.0, 0.0)]

    def find_bl(self, w: float, h: float) -> Tuple[float, float]:
        best_x, best_y = 0.0, float('inf')
        for sx, _ in self.pts:
            y = self.max_height(sx, sx + w)
            if y < best_y or (y == best_y and sx < best_x):
                best_y = y
                best_x = sx
        return best_x, best_y

    def place(self, x: float, y: float, w: float, h: float):
        self.raise_region(x, x + w, y + h)


def skyline_decode(perm, widths, heights, preplaced, n):
    sky = Skyline()
    pos = [(0.0, 0.0, 0.0, 0.0)] * n
    for bid, (px, py, pw, ph) in preplaced.items():
        pos[bid] = (px, py, pw, ph)
        sky.place(px, py, pw, ph)
    for bid in perm:
        w, h = widths[bid], heights[bid]
        x, y = sky.find_bl(w, h)
        pos[bid] = (x, y, w, h)
        sky.place(x, y, w, h)
    return pos


def violation_cost(positions, mib_groups, cluster_groups, boundary_blocks, n_soft):
    if n_soft == 0:
        return 0.0
    v = 0.0
    for members in mib_groups.values():
        shapes = set()
        for b in members:
            _, _, w, h = positions[b]
            shapes.add((round(w, 3), round(h, 3)))
        v += len(shapes) - 1
    for members in cluster_groups.values():
        if len(members) < 2:
            continue
        adj = {b: [] for b in members}
        for ii in range(len(members)):
            bi = members[ii]
            xi, yi, wi, hi = positions[bi]
            for jj in range(ii + 1, len(members)):
                bj = members[jj]
                xj, yj, wj, hj = positions[bj]
                oy = min(yi + hi, yj + hj) - max(yi, yj)
                ox = min(xi + wi, xj + wj) - max(xi, xj)
                touch = (
                    (oy > 1e-6 and (abs(xi + wi - xj) < 1e-4 or abs(xj + wj - xi) < 1e-4)) or
                    (ox > 1e-6 and (abs(yi + hi - yj) < 1e-4 or abs(yj + hj - yi) < 1e-4))
                )
                if touch:
                    adj[bi].append(bj); adj[bj].append(bi)
        visited, components = set(), 0
        for start in members:
            if start not in visited:
                components += 1
                stack = [start]
                while stack:
                    cur = stack.pop()
                    if cur in visited: continue
                    visited.add(cur)
                    stack.extend(adj[cur])
        v += components - 1
    if boundary_blocks:
        x_min = min(p[0]        for p in positions)
        y_min = min(p[1]        for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        for b, flag in boundary_blocks:
            x, y, w, h = positions[b]
            lt = abs(x     - x_min) < 1e-4
            rt = abs(x + w - x_max) < 1e-4
            bt = abs(y     - y_min) < 1e-4
            tt = abs(y + h - y_max) < 1e-4
            ok = {
                BOUNDARY_LEFT:         lt, BOUNDARY_RIGHT:        rt,
                BOUNDARY_TOP:          tt, BOUNDARY_BOTTOM:        bt,
                BOUNDARY_TOP_LEFT:     tt and lt, BOUNDARY_TOP_RIGHT:    tt and rt,
                BOUNDARY_BOTTOM_LEFT:  bt and lt, BOUNDARY_BOTTOM_RIGHT: bt and rt,
            }.get(flag, False)
            if not ok: v += 1
    return v / n_soft


def python_sa_solve(block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints, target_positions):
    """Pure-Python fallback SA (same as previous v4 approach)."""
    from collections import defaultdict
    t0 = time.time()
    n = block_count
    W_AREA_PY = 0.008
    W_VIOL_PY = 8.0

    is_fixed     = [False] * n
    is_preplaced = [False] * n
    mib_groups_d    : Dict[int, List[int]] = defaultdict(list)
    cluster_groups_d: Dict[int, List[int]] = defaultdict(list)
    boundary_blocks_d: List[Tuple[int, int]] = []

    for i in range(n):
        c = constraints[i]
        if int(c[COL_PREPLACED].item()) == 1:
            is_preplaced[i] = True
        elif int(c[COL_FIXED].item()) == 1:
            is_fixed[i] = True
        mib_id = int(c[COL_MIB].item())
        if mib_id > 0: mib_groups_d[mib_id].append(i)
        cl_id = int(c[COL_CLUSTER].item())
        if cl_id > 0: cluster_groups_d[cl_id].append(i)
        bflag = int(c[COL_BOUNDARY].item())
        if bflag > 0: boundary_blocks_d.append((i, bflag))

    n_soft = (len(boundary_blocks_d)
              + sum(max(0, len(g) - 1) for g in cluster_groups_d.values())
              + sum(max(0, len(g) - 1) for g in mib_groups_d.values()))

    widths: List[float] = []
    heights: List[float] = []
    for i in range(n):
        tp = target_positions
        if tp is not None and float(tp[i, 2]) > 0 and float(tp[i, 3]) > 0:
            widths.append(float(tp[i, 2]))
            heights.append(float(tp[i, 3]))
        else:
            area = float(area_targets[i]) if float(area_targets[i]) > 0 else 1.0
            side = math.sqrt(area)
            widths.append(side); heights.append(side)

    for members in mib_groups_d.values():
        soft_m = [b for b in members if not is_preplaced[b] and not is_fixed[b]]
        if soft_m:
            rw, rh = widths[soft_m[0]], heights[soft_m[0]]
            for b in soft_m: widths[b] = rw; heights[b] = rh

    preplaced = {}
    for i in range(n):
        if is_preplaced[i] and target_positions is not None:
            preplaced[i] = (float(target_positions[i, 0]), float(target_positions[i, 1]),
                            float(target_positions[i, 2]), float(target_positions[i, 3]))

    free_blocks = [i for i in range(n) if not is_preplaced[i]]
    resizable   = [i for i in free_blocks if not is_fixed[i]]

    perm = sorted(free_blocks, key=lambda i: widths[i] * heights[i], reverse=True)

    def proxy_cost(pos):
        hpwl = (calculate_hpwl_b2b(pos, b2b_connectivity) +
                calculate_hpwl_p2b(pos, p2b_connectivity, pins_pos))
        area = calculate_bbox_area(pos)
        viol = violation_cost(pos, mib_groups_d, cluster_groups_d,
                              boundary_blocks_d, n_soft)
        return hpwl + area * W_AREA_PY + viol * W_VIOL_PY

    cur_pos  = skyline_decode(perm, widths, heights, preplaced, n)
    cur_cost = proxy_cost(cur_pos)
    best_perm  = perm[:]; best_w = widths[:]; best_h = heights[:]
    best_pos   = cur_pos; best_cost = cur_cost

    mib_free = {}
    for gid, members in mib_groups_d.items():
        s = [b for b in members if not is_preplaced[b] and not is_fixed[b]]
        if len(s) >= 2: mib_free[gid] = s

    n_free = len(free_blocks)
    temp = 200.0; min_temp = 0.1; alpha = 0.93

    while temp > min_temp and (time.time() - t0) < TIME_LIMIT:
        new_perm   = perm[:]
        new_widths = widths[:]
        new_heights = heights[:]
        move = random.random()

        if move < 0.35 and n_free >= 2:
            i, j = random.sample(range(n_free), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        elif move < 0.50 and n_free >= 2:
            i = random.randint(0, n_free - 1)
            j = random.randint(0, n_free - 1)
            blk = new_perm.pop(i)
            new_perm.insert(j, blk)
        elif move < 0.65 and resizable:
            b = random.choice(resizable)
            area = float(area_targets[b])
            if area > 0:
                r = random.uniform(0.2, 5.0)
                nw = math.sqrt(area * r)
                new_widths[b] = nw; new_heights[b] = area / nw
        elif move < 0.75 and resizable:
            b = random.choice(resizable)
            new_widths[b], new_heights[b] = new_heights[b], new_widths[b]
        else:
            if mib_free:
                gid = random.choice(list(mib_free.keys()))
                members = mib_free[gid]
                leader  = random.choice(members)
                lw, lh  = new_widths[leader], new_heights[leader]
                l_area  = lw * lh
                for b in members:
                    if b == leader: continue
                    area_b = float(area_targets[b])
                    if area_b > 0:
                        if abs(l_area - area_b) / area_b <= AREA_TOL:
                            new_widths[b] = lw; new_heights[b] = lh
                        else:
                            r = lw / lh if lh > 1e-9 else 1.0
                            nw = math.sqrt(area_b * r)
                            new_widths[b] = nw; new_heights[b] = area_b / nw
                    else:
                        new_widths[b] = lw; new_heights[b] = lh
            elif n_free >= 2:
                i, j = random.sample(range(n_free), 2)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

        new_pos  = skyline_decode(new_perm, new_widths, new_heights, preplaced, n)
        new_cost = proxy_cost(new_pos)
        delta = new_cost - cur_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            perm = new_perm; widths = new_widths; heights = new_heights
            cur_cost = new_cost; cur_pos = new_pos
            if cur_cost < best_cost:
                best_cost = cur_cost; best_perm = perm[:]
                best_w = widths[:]; best_h = heights[:]
                best_pos = new_pos

        temp *= alpha

    return best_pos


# ── C++ compilation ──────────────────────────────────────────────────────────
_CPP_COMPILED = False

def _ensure_compiled():
    global _CPP_COMPILED
    if _CPP_COMPILED:
        return True
    if _BIN.exists():
        if _CPP.exists() and _BIN.stat().st_mtime >= _CPP.stat().st_mtime:
            _CPP_COMPILED = True
            return True
    try:
        result = subprocess.run(
            [_GPP, "-O3", "-std=c++17", "-o", str(_BIN), str(_CPP)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            _CPP_COMPILED = True
            return True
        else:
            print(f"[optimizer_claude] C++ compile failed:\n{result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[optimizer_claude] Compile error: {e}", file=sys.stderr)
        return False


def _serialize_input(block_count, area_targets, b2b_connectivity,
                     p2b_connectivity, pins_pos, constraints, target_positions):
    """Serialize problem data to the text format expected by optimizer_claude.cpp."""
    buf = StringIO()
    n = block_count
    buf.write(f"{n}\n")

    # area targets
    buf.write(" ".join(f"{float(area_targets[i]):.10f}" for i in range(n)) + "\n")

    # b2b edges
    b2b_valid = []
    if b2b_connectivity is not None and len(b2b_connectivity) > 0:
        for edge in b2b_connectivity:
            if int(edge[0]) == -1: continue
            b2b_valid.append((int(edge[0]), int(edge[1]), float(edge[2])))
    buf.write(f"{len(b2b_valid)}\n")
    for i, j, w in b2b_valid:
        buf.write(f"{i} {j} {w:.10f}\n")

    # p2b edges
    p2b_valid = []
    if p2b_connectivity is not None and len(p2b_connectivity) > 0:
        for edge in p2b_connectivity:
            if int(edge[0]) == -1: continue
            p2b_valid.append((int(edge[0]), int(edge[1]), float(edge[2])))
    buf.write(f"{len(p2b_valid)}\n")
    for pi, bi, w in p2b_valid:
        buf.write(f"{pi} {bi} {w:.10f}\n")

    # pins
    n_pins = 0
    pins_list = []
    if pins_pos is not None and len(pins_pos) > 0:
        for p in pins_pos:
            if float(p[0]) == -1 and float(p[1]) == -1: continue
            pins_list.append((float(p[0]), float(p[1])))
            n_pins += 1
    buf.write(f"{n_pins}\n")
    for px, py in pins_list:
        buf.write(f"{px:.10f} {py:.10f}\n")

    # constraints
    for i in range(n):
        c = constraints[i] if constraints is not None else [0]*5
        fx = int(c[0]) if hasattr(c[0], 'item') else int(c[0])
        pp = int(c[1]) if hasattr(c[1], 'item') else int(c[1])
        mib= int(c[2]) if hasattr(c[2], 'item') else int(c[2])
        cl = int(c[3]) if hasattr(c[3], 'item') else int(c[3])
        bnd= int(c[4]) if hasattr(c[4], 'item') else int(c[4])
        buf.write(f"{fx} {pp} {mib} {cl} {bnd}\n")

    # target positions
    for i in range(n):
        if target_positions is not None:
            tx = float(target_positions[i, 0])
            ty = float(target_positions[i, 1])
            tw = float(target_positions[i, 2])
            th = float(target_positions[i, 3])
        else:
            tx = ty = tw = th = -1.0
        buf.write(f"{tx:.10f} {ty:.10f} {tw:.10f} {th:.10f}\n")

    return buf.getvalue()


def _parse_output(text: str, n: int) -> List[Tuple[float, float, float, float]]:
    lines = text.strip().split("\n")
    assert int(lines[0]) == n, f"Expected {n} blocks, got {lines[0]}"
    positions = []
    for i in range(1, n + 1):
        vals = lines[i].split()
        positions.append((float(vals[0]), float(vals[1]),
                          float(vals[2]), float(vals[3])))
    return positions


# ── Optimizer class ───────────────────────────────────────────────────────────
class MyOptimizer(FloorplanOptimizer):
    """
    C++ SA floorplanner with Python fallback.

    C++ backend: skyline BL packer + connectivity-driven SA
      - Overlap-free by construction (skyline)
      - Area tolerance: resize preserves area exactly; fixed/preplaced untouched
      - Soft constraints: MIB, cluster, boundary handled in SA cost
    """

    W_AREA = 0.008
    W_VIOL = 8.0

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        _ensure_compiled()

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Tuple[float, float, float, float]]:

        if not _CPP_COMPILED:
            # Fallback to Python SA
            return python_sa_solve(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions)

        inp = _serialize_input(
            block_count, area_targets, b2b_connectivity,
            p2b_connectivity, pins_pos, constraints, target_positions)

        def _run_binary(flags=()):
            result = subprocess.run(
                [str(_BIN)] + list(flags),
                input=inp, capture_output=True, text=True, timeout=55.0
            )
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError(f"C++ failed: {result.stderr[:200]}")
            return _parse_output(result.stdout, block_count)

        def _has_overlap(positions):
            from iccad2026_evaluate import check_overlap
            return check_overlap(positions) > 0

        try:
            positions = _run_binary()
            # If max_width caused overlaps, retry without it
            if _has_overlap(positions):
                positions = _run_binary(("--no-width",))
            return positions
        except Exception as e:
            if self.verbose:
                print(f"[optimizer_claude] C++ error: {e}, falling back to Python", file=sys.stderr)
            return python_sa_solve(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions)
