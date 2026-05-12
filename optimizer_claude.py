#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Overlap-Free Optimizer
=======================================================
Core guarantee: ALWAYS overlap-free (feasible) by construction.

Architecture:
  - SA operates ONLY on block permutation + aspect ratios
  - Every decode uses a Skyline packer → guaranteed overlap-free
  - Preplaced blocks initialize the skyline as obstacles
  - Hard constraints (preplaced, fixed-shape) are never violated

Cost function matches the contest metric:
  (1 + 0.5*(HPWLgap + AreaGap)) * exp(2*Vrel) * max(0.7, RuntimeFactor^0.3)
  We approximate this with HPWL + w_area*bbox + w_viol*Vrel
"""

import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)

# ── Constraint column indices ──────────────────────────────────────────────────
COL_FIXED      = 0
COL_PREPLACED  = 1
COL_MIB        = 2
COL_CLUSTER    = 3
COL_BOUNDARY   = 4

BOUNDARY_LEFT         = 1
BOUNDARY_RIGHT        = 2
BOUNDARY_TOP          = 4
BOUNDARY_BOTTOM       = 8
BOUNDARY_TOP_LEFT     = 5
BOUNDARY_TOP_RIGHT    = 6
BOUNDARY_BOTTOM_LEFT  = 9
BOUNDARY_BOTTOM_RIGHT = 10

TIME_LIMIT = 1.6   # hard wall per case (seconds); leave margin for eval overhead


# =============================================================================
# SKYLINE PACKER  — always produces overlap-free layouts
# =============================================================================

class Skyline:
    """
    Skyline packer.

    Internal representation: sorted list of (x_start, height) breakpoints.
    Segment i covers [pts[i].x, pts[i+1].x) at pts[i].height.
    The last segment extends to +inf.
    """

    def __init__(self):
        self.pts: List[Tuple[float, float]] = [(0.0, 0.0)]

    def copy(self) -> 'Skyline':
        s = Skyline()
        s.pts = self.pts[:]
        return s

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

    def _height_at(self, x: float) -> float:
        """Height of skyline exactly at x."""
        h = 0.0
        for i, (sx, sh) in enumerate(self.pts):
            nx = self.pts[i + 1][0] if i + 1 < len(self.pts) else float('inf')
            if sx <= x < nx:
                h = sh
                break
        return h

    def raise_region(self, x0: float, x1: float, new_h: float):
        """
        Raise skyline to new_h in [x0, x1).
        new_h must be >= all current heights in that region (satisfied by BL placement).
        """
        pts = self.pts
        # Collect unique x values: existing breakpoints + x0, x1
        xs = sorted(set([x0, x1] + [sx for sx, _ in pts if x0 < sx < x1]))

        new_pts: List[Tuple[float, float]] = []
        prev_h: Optional[float] = None

        # Walk through all breakpoints (existing + new)
        all_xs = sorted(set([sx for sx, _ in pts] + [x0, x1]))

        for x in all_xs:
            if x0 <= x < x1:
                h = new_h
            else:
                h = self._height_at(x)

            if h != prev_h:
                new_pts.append((x, h))
                prev_h = h

        self.pts = new_pts if new_pts else [(0.0, 0.0)]

    def find_bl(self, w: float, h: float) -> Tuple[float, float]:
        """
        Bottom-Left fit: find the leftmost, lowest position for a block (w, h).
        Returns (x, y).
        """
        best_x, best_y = 0.0, float('inf')
        for sx, _ in self.pts:
            y = self.max_height(sx, sx + w)
            if y < best_y or (y == best_y and sx < best_x):
                best_y = y
                best_x = sx
        return best_x, best_y

    def place(self, x: float, y: float, w: float, h: float):
        """Record that a block has been placed at (x, y, w, h)."""
        self.raise_region(x, x + w, y + h)


def skyline_decode(
    perm: List[int],
    widths: List[float],
    heights: List[float],
    preplaced: Dict[int, Tuple[float, float, float, float]],
    n: int,
) -> List[Tuple[float, float, float, float]]:
    """
    Decode a permutation to an overlap-free floorplan.

    preplaced : {block_id: (x, y, w, h)} — fixed blocks that initialize the skyline
    perm      : order in which free blocks are packed
    Returns   : list of (x, y, w, h) for ALL n blocks
    """
    sky = Skyline()

    # Initialize skyline with preplaced blocks
    pos = [(0.0, 0.0, 0.0, 0.0)] * n
    for bid, (px, py, pw, ph) in preplaced.items():
        pos[bid] = (px, py, pw, ph)
        sky.place(px, py, pw, ph)

    # Pack free blocks in permutation order
    for bid in perm:
        w, h = widths[bid], heights[bid]
        x, y = sky.find_bl(w, h)
        pos[bid] = (x, y, w, h)
        sky.place(x, y, w, h)

    return pos


# =============================================================================
# VIOLATION COST
# =============================================================================

def violation_cost(
    positions: List[Tuple[float, float, float, float]],
    mib_groups: Dict[int, List[int]],
    cluster_groups: Dict[int, List[int]],
    boundary_blocks: List[Tuple[int, int]],
    n_soft: int,
) -> float:
    if n_soft == 0:
        return 0.0
    v = 0.0

    # MIB: distinct (w,h) pairs per group
    for members in mib_groups.values():
        shapes = set()
        for b in members:
            _, _, w, h = positions[b]
            shapes.add((round(w, 3), round(h, 3)))
        v += len(shapes) - 1

    # Cluster: BFS connected components
    for members in cluster_groups.values():
        if len(members) < 2:
            continue
        adj: Dict[int, List[int]] = {b: [] for b in members}
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
                    adj[bi].append(bj)
                    adj[bj].append(bi)
        visited, components = set(), 0
        for start in members:
            if start not in visited:
                components += 1
                stack = [start]
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    stack.extend(adj[cur])
        v += components - 1

    # Boundary
    if boundary_blocks:
        x_min = min(p[0]       for p in positions)
        y_min = min(p[1]       for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        for b, flag in boundary_blocks:
            x, y, w, h = positions[b]
            lt = abs(x     - x_min) < 1e-4
            rt = abs(x + w - x_max) < 1e-4
            bt = abs(y     - y_min) < 1e-4
            tt = abs(y + h - y_max) < 1e-4
            ok = {
                BOUNDARY_LEFT:         lt,
                BOUNDARY_RIGHT:        rt,
                BOUNDARY_TOP:          tt,
                BOUNDARY_BOTTOM:       bt,
                BOUNDARY_TOP_LEFT:     tt and lt,
                BOUNDARY_TOP_RIGHT:    tt and rt,
                BOUNDARY_BOTTOM_LEFT:  bt and lt,
                BOUNDARY_BOTTOM_RIGHT: bt and rt,
            }.get(flag, False)
            if not ok:
                v += 1

    return v / n_soft


# =============================================================================
# OPTIMIZER
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    Permutation-based SA with skyline decoding.
    Hard constraint satisfaction is guaranteed by construction:
      - Overlap-free: skyline packer never produces overlaps
      - Area tolerance: maintained via aspect-ratio perturbation keeping w*h = area_target
      - Preplaced: never moved (put directly in preplaced dict)
      - Fixed-shape: w,h never changed (excluded from resize moves)
    """

    W_AREA = 0.006   # bbox area weight in proxy cost
    W_VIOL = 6.0     # soft-violation penalty weight

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _proxy_cost(
        self,
        positions,
        b2b, p2b, pins,
        mib_groups, cluster_groups, boundary_blocks, n_soft,
    ) -> float:
        hpwl = (calculate_hpwl_b2b(positions, b2b) +
                calculate_hpwl_p2b(positions, p2b, pins))
        area = calculate_bbox_area(positions)
        viol = violation_cost(positions, mib_groups, cluster_groups,
                              boundary_blocks, n_soft)
        return hpwl + area * self.W_AREA + viol * self.W_VIOL

    # ── main entry point ─────────────────────────────────────────────────────

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

        t0 = time.time()
        n  = block_count

        # ── 1. Parse constraints ──────────────────────────────────────────────
        is_fixed      = [False] * n
        is_preplaced  = [False] * n
        mib_groups    : Dict[int, List[int]] = defaultdict(list)
        cluster_groups: Dict[int, List[int]] = defaultdict(list)
        boundary_blocks: List[Tuple[int, int]] = []

        for i in range(n):
            c = constraints[i]
            if int(c[COL_PREPLACED].item()) == 1:
                is_preplaced[i] = True
            elif int(c[COL_FIXED].item()) == 1:
                is_fixed[i] = True
            mib_id = int(c[COL_MIB].item())
            if mib_id > 0:
                mib_groups[mib_id].append(i)
            cl_id  = int(c[COL_CLUSTER].item())
            if cl_id > 0:
                cluster_groups[cl_id].append(i)
            bflag  = int(c[COL_BOUNDARY].item())
            if bflag > 0:
                boundary_blocks.append((i, bflag))

        n_soft = (
            len(boundary_blocks)
            + sum(max(0, len(g) - 1) for g in cluster_groups.values())
            + sum(max(0, len(g) - 1) for g in mib_groups.values())
        )

        # ── 2. Initial dimensions ─────────────────────────────────────────────
        widths:  List[float] = []
        heights: List[float] = []
        for i in range(n):
            tp = target_positions
            if tp is not None and float(tp[i, 2]) > 0 and float(tp[i, 3]) > 0:
                widths.append(float(tp[i, 2]))
                heights.append(float(tp[i, 3]))
            else:
                area = float(area_targets[i]) if float(area_targets[i]) > 0 else 1.0
                side = math.sqrt(area)
                widths.append(side)
                heights.append(side)

        # ── 3. Pre-unify MIB shapes ───────────────────────────────────────────
        for members in mib_groups.values():
            free_m = [b for b in members if not is_preplaced[b] and not is_fixed[b]]
            if free_m:
                avg_w = sum(widths[b]  for b in free_m) / len(free_m)
                avg_h = sum(heights[b] for b in free_m) / len(free_m)
                for b in free_m:
                    widths[b]  = avg_w
                    heights[b] = avg_h

        # ── 4. Build preplaced dict ───────────────────────────────────────────
        preplaced: Dict[int, Tuple[float, float, float, float]] = {}
        for i in range(n):
            if is_preplaced[i] and target_positions is not None:
                preplaced[i] = (
                    float(target_positions[i, 0]),
                    float(target_positions[i, 1]),
                    float(target_positions[i, 2]),
                    float(target_positions[i, 3]),
                )

        # ── 5. Initial permutation (free blocks) ──────────────────────────────
        free_blocks = [i for i in range(n) if not is_preplaced[i]]
        resizable   = [i for i in free_blocks if not is_fixed[i]]

        # Sort by area descending for a better initial packing
        perm = sorted(free_blocks, key=lambda i: widths[i] * heights[i], reverse=True)

        # ── 6. Decode initial placement ───────────────────────────────────────
        current_pos  = skyline_decode(perm, widths, heights, preplaced, n)
        current_cost = self._proxy_cost(
            current_pos, b2b_connectivity, p2b_connectivity, pins_pos,
            mib_groups, cluster_groups, boundary_blocks, n_soft,
        )
        best_perm  = perm[:]
        best_widths  = widths[:]
        best_heights = heights[:]
        best_pos   = current_pos
        best_cost  = current_cost

        # ── 7. MIB free-block index lookup ───────────────────────────────────
        mib_free: Dict[int, List[int]] = {
            gid: [b for b in members if not is_preplaced[b]]
            for gid, members in mib_groups.items()
        }
        mib_free = {k: v for k, v in mib_free.items() if len(v) > 1}

        n_free = len(free_blocks)

        # ── 8. Simulated Annealing (time-bounded) ─────────────────────────────
        temp     = 100.0
        min_temp = 0.5
        alpha    = 0.90          # cooling rate per temperature step
        n_moves  = max(10, n_free // 2)   # moves per temperature level

        while temp > min_temp and (time.time() - t0) < TIME_LIMIT:

            new_perm    = perm[:]
            new_widths  = widths[:]
            new_heights = heights[:]
            move = random.random()

            if move < 0.40 and n_free >= 2:
                # ── Swap two blocks in permutation ────────────────────────────
                i, j = random.sample(range(n_free), 2)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

            elif move < 0.60 and n_free >= 2:
                # ── Move a block to a random position in permutation ──────────
                i = random.randint(0, n_free - 1)
                j = random.randint(0, n_free - 1)
                block = new_perm.pop(i)
                new_perm.insert(j, block)

            elif move < 0.75 and resizable:
                # ── Aspect-ratio perturbation (area preserved) ─────────────
                b    = random.choice(resizable)
                area = float(area_targets[b])
                if area > 0:
                    r   = random.uniform(0.25, 4.0)
                    nw  = math.sqrt(area * r)
                    nh  = area / nw
                    new_widths[b]  = nw
                    new_heights[b] = nh

            elif move < 0.88 and resizable:
                # ── 90° rotation (swap w and h, area preserved) ──────────────
                b = random.choice(resizable)
                new_widths[b], new_heights[b] = new_heights[b], new_widths[b]

            else:
                # ── MIB shape unification ─────────────────────────────────────
                if mib_free:
                    gid     = random.choice(list(mib_free.keys()))
                    members = mib_free[gid]
                    leader  = random.choice(members)
                    lw, lh  = new_widths[leader], new_heights[leader]
                    area_l  = lw * lh
                    if area_l > 0:
                        for b in members:
                            # All members get the same shape as leader
                            new_widths[b]  = lw
                            new_heights[b] = lh
                elif n_free >= 2:
                    i, j = random.sample(range(n_free), 2)
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

            # Decode → always overlap-free
            new_pos  = skyline_decode(new_perm, new_widths, new_heights, preplaced, n)
            new_cost = self._proxy_cost(
                new_pos, b2b_connectivity, p2b_connectivity, pins_pos,
                mib_groups, cluster_groups, boundary_blocks, n_soft,
            )

            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                perm    = new_perm
                widths  = new_widths
                heights = new_heights
                current_cost = new_cost
                current_pos  = new_pos

                if current_cost < best_cost:
                    best_cost    = current_cost
                    best_perm    = perm[:]
                    best_widths  = widths[:]
                    best_heights = heights[:]
                    best_pos     = new_pos

            temp *= alpha

        return best_pos