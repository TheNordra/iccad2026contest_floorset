#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Portfolio Optimizer
====================================================
Runs optimizer_claude.exe with N different initial-perm hints IN PARALLEL,
picks whichever produced the lowest proxy_cost.

Each profile gets the full 8s SA budget; total wall time is bounded by the
slowest profile (parallel) + a small selection overhead.

Profiles vary only in the `gnn_hint` (cx, cy) per block fed to the C++ binary,
which is used purely as a tie-break for the initial permutation ordering inside
each priority bucket (cluster / LEFT-BOTTOM boundary / regular / RIGHT-TOP
boundary). Shapes and SA hyperparameters are unchanged.

Profiles:
  - "gnn":          real FloorplanNet centers (current single-run behavior)
  - "connectivity": no hint -> C++ uses pure connectivity-driven perm
  - "area_desc":    sort big blocks first (BL-friendly, pack large then fill)
  - "area_asc":     sort small blocks first (occasional win on tight layouts)

Disable individual profiles via ICCAD_PORTFOLIO_PROFILES env var
(comma-separated). Default: all four.
"""

import concurrent.futures
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)

# Reuse infrastructure from the single-run wrapper.
import optimizer_claude as _oc
from optimizer_claude import (
    _ensure_compiled, _serialize_input, _parse_output, _BIN,
    _gnn_centers, python_sa_solve, violation_cost,
    COL_FIXED, COL_PREPLACED, COL_MIB, COL_CLUSTER, COL_BOUNDARY,
)

# Contest-shape proxy (adapted from teammate codex v7 absarea selector):
#   proxy = (1 + 0.5 * (area_gap_est + hpwl_rel)) * exp(2.0 * v_rel)
# where:
#   area_gap_est = (area - 1.035 * sum_block_area) / (1.035 * sum_block_area),
#                  clamped >= 0 (no reward for being smaller than estimate)
#   hpwl_rel     = (hpwl - best_hpwl_in_pool) / best_hpwl_in_pool, clamped >= 0
#   v_rel        = soft_violation_count / max(1, n_soft)
# This matches contest cost shape (Cost = (1+α·gap)·exp(β·V_rel)) much better
# than a flat linear combination, and properly penalizes the exponential
# violation factor.
_PROXY_ALPHA = 0.5
_PROXY_BETA  = 2.0
_AREA_OVERHEAD = 1.035  # estimated bbox area / sum_block_area baseline

_ALL_PROFILES = ["gnn", "connectivity", "area_desc", "area_asc"]
_PROFILES = os.environ.get("ICCAD_PORTFOLIO_PROFILES", "")
if _PROFILES:
    _PROFILES = [p.strip() for p in _PROFILES.split(",") if p.strip()]
    for p in _PROFILES:
        assert p in _ALL_PROFILES, f"Unknown profile: {p}"
else:
    _PROFILES = _ALL_PROFILES

# Optional: turn off GNN profile if torch/.pth missing — `_gnn_centers` returns
# None in that case and the corresponding profile devolves into a no-hint run,
# duplicating "connectivity". We drop GNN from the list to save one subprocess.
def _gnn_available() -> bool:
    try:
        return _oc._load_gnn() is not None
    except Exception:
        return False


def _make_hint(
    profile: str,
    block_count: int,
    area_targets,
    b2b,
    p2b,
    pins,
) -> Optional[List[Tuple[float, float]]]:
    """Build a per-block (cx, cy) hint for a given profile.

    Returns None to signal "no hint" — the C++ binary then uses its default
    connectivity-driven perm.
    """
    n = block_count
    if profile == "gnn":
        return _gnn_centers(n, area_targets, b2b, p2b, pins)
    if profile == "connectivity":
        return None
    if profile == "area_desc":
        # C++ sorts ascending by (cx+cy) — make biggest area's sum the most
        # negative so it comes first.
        return [(-float(area_targets[i]), -float(area_targets[i])) for i in range(n)]
    if profile == "area_asc":
        return [(float(area_targets[i]), float(area_targets[i])) for i in range(n)]
    raise ValueError(f"Unknown profile: {profile}")


def _run_one_profile(
    profile: str,
    block_count, area_targets, b2b, p2b, pins, constraints, target_positions,
) -> Tuple[str, Optional[list], float]:
    """Run the C++ binary once with the given profile's hint.
    Returns (profile, positions or None, wall_time_sec).
    """
    t0 = time.time()
    hint = _make_hint(profile, block_count, area_targets, b2b, p2b, pins)
    inp = _serialize_input(
        block_count, area_targets, b2b, p2b, pins, constraints,
        target_positions, gnn_hint=hint,
    )
    try:
        result = subprocess.run(
            [str(_BIN)], input=inp,
            capture_output=True, text=True, timeout=55.0,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return (profile, None, time.time() - t0)
        positions = _parse_output(result.stdout, block_count)
        return (profile, positions, time.time() - t0)
    except Exception as e:
        print(f"[portfolio] profile={profile} failed: {e}", file=sys.stderr)
        return (profile, None, time.time() - t0)


def _build_constraint_structures(block_count, constraints):
    """Extract (mib_groups, cluster_groups, boundary_blocks, n_soft) used by
    `violation_cost`."""
    mib_groups = defaultdict(list)
    cluster_groups = defaultdict(list)
    boundary_blocks = []
    for i in range(block_count):
        c = constraints[i]
        mib_id = int(c[COL_MIB].item())
        if mib_id > 0:
            mib_groups[mib_id].append(i)
        cl_id = int(c[COL_CLUSTER].item())
        if cl_id > 0:
            cluster_groups[cl_id].append(i)
        bf = int(c[COL_BOUNDARY].item())
        if bf > 0:
            boundary_blocks.append((i, bf))
    n_soft = (
        len(boundary_blocks)
        + sum(max(0, len(g) - 1) for g in cluster_groups.values())
        + sum(max(0, len(g) - 1) for g in mib_groups.values())
    )
    return mib_groups, cluster_groups, boundary_blocks, n_soft


def _metrics_for_positions(positions, block_count, area_targets, b2b, p2b, pins,
                           constraints):
    """Returns (hpwl, area, v_rel)."""
    mib_g, cl_g, bdry, n_soft = _build_constraint_structures(block_count, constraints)
    hpwl = (
        calculate_hpwl_b2b(positions, b2b)
        + calculate_hpwl_p2b(positions, p2b, pins)
    )
    area = calculate_bbox_area(positions)
    # violation_cost already returns v_total / n_soft (i.e., v_rel)
    v_rel = violation_cost(positions, mib_g, cl_g, bdry, n_soft)
    return hpwl, area, v_rel


def _pick_best(profile_results, block_count, area_targets, b2b, p2b, pins, constraints):
    """Apply contest-shape proxy across all profile results, pick min.

    profile_results: dict {profile_name: (positions, wall_time)}
    Returns: (best_profile, best_positions, debug_info_dict)
    """
    # Pass 1: compute metrics
    metrics = {}
    for profile, (positions, _dt) in profile_results.items():
        h, a, v = _metrics_for_positions(
            positions, block_count, area_targets, b2b, p2b, pins, constraints,
        )
        metrics[profile] = (h, a, v)

    # Baselines for relative scoring
    best_hpwl = min(m[0] for m in metrics.values()) or 1.0
    sum_block_area = float(sum(float(area_targets[i]) for i in range(block_count)))
    est_area = _AREA_OVERHEAD * sum_block_area if sum_block_area > 0 else 1.0

    best_profile, best_proxy = None, float("inf")
    debug = {}
    for profile, (h, a, v) in metrics.items():
        area_gap = max(0.0, (a - est_area) / est_area) if est_area > 0 else 0.0
        hpwl_rel = max(0.0, (h - best_hpwl) / best_hpwl) if best_hpwl > 0 else 0.0
        proxy = (1.0 + _PROXY_ALPHA * (area_gap + hpwl_rel)) * math.exp(_PROXY_BETA * v)
        debug[profile] = (proxy, h, a, v)
        if proxy < best_proxy:
            best_proxy = proxy
            best_profile = profile

    best_positions = profile_results[best_profile][0]
    return best_profile, best_positions, debug


class MyOptimizer(FloorplanOptimizer):
    """Portfolio wrapper that runs multiple optimizer_claude profiles and
    returns whichever produced the lowest proxy_cost."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        _ensure_compiled()
        # Prune GNN profile if GNN isn't loadable — it would just duplicate
        # the "connectivity" profile and waste one parallel slot.
        if "gnn" in _PROFILES and not _gnn_available():
            print("[portfolio] GNN not available; dropping 'gnn' profile",
                  file=sys.stderr)
            self.profiles = [p for p in _PROFILES if p != "gnn"]
        else:
            self.profiles = list(_PROFILES)
        if not self.profiles:
            self.profiles = ["connectivity"]

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:

        if not _oc._CPP_COMPILED:
            return python_sa_solve(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions,
            )

        # Fan out: one thread per profile, each spawning its own C++ subprocess.
        results: dict = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.profiles)
        ) as ex:
            futures = {
                ex.submit(
                    _run_one_profile, p, block_count, area_targets,
                    b2b_connectivity, p2b_connectivity, pins_pos,
                    constraints, target_positions,
                ): p for p in self.profiles
            }
            for fut in concurrent.futures.as_completed(futures):
                profile, positions, dt = fut.result()
                if positions is not None:
                    results[profile] = (positions, dt)

        if not results:
            # All profiles failed — fall back to Python SA.
            return python_sa_solve(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions,
            )

        # Pick best by contest-shape proxy.
        best_profile, best_pos, debug = _pick_best(
            results, block_count, area_targets, b2b_connectivity,
            p2b_connectivity, pins_pos, constraints,
        )

        if self.verbose:
            parts = []
            for p, (proxy, h, a, v) in debug.items():
                dt = results[p][1]
                marker = "*" if p == best_profile else " "
                parts.append(
                    f"{marker}{p}: proxy={proxy:.3f} hpwl={h:.1f} "
                    f"area={a:.1f} vrel={v:.3f} t={dt:.1f}s"
                )
            print(f"[portfolio] " + " | ".join(parts), file=sys.stderr)

        return best_pos
