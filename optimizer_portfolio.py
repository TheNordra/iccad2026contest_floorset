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
  - "gnn":           real FloorplanNet centers (current single-run behavior)
  - "connectivity":  no hint -> C++ uses pure connectivity-driven perm
  - "area_desc":     sort big blocks first (BL-friendly, pack large then fill)
  - "area_asc":      sort small blocks first (occasional win on tight layouts)
  - "pin_centroid":  per-block hint = mean of connected pin positions
  - "degree_desc":   sort most-connected blocks first
  - "degree_asc":    sort least-connected blocks first
  - "high_boundary": connectivity perm + W_BOUNDARY=100 (10x default) — attacks
                     violation-heavy big-n cases by amplifying the soft boundary
                     gradient during SA
  - "low_viol":      connectivity perm + W_VIOL_MULT=0.5 — SA favors HPWL/area
                     minimization; wins low-violation cases
  - "high_viol":     connectivity perm + W_VIOL_MULT=2.0 — SA prioritizes hard-
                     violation elimination (second lever vs the boundary gradient)

Select profiles via ICCAD_PORTFOLIO_PROFILES env var (comma-separated). Default
is the 8 proven profiles; low_viol/high_viol are recognized but excluded from the
default (they regressed the total — see _DEFAULT_PROFILES).
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

# Near-tie tie-break (2026-06-02): when two profiles' proxies are within this
# relative margin, the proxy cannot reliably distinguish them — its HPWL/area
# baselines are *estimates* (best-in-pool / 1.035·ΣA), unreliable at sub-% gaps,
# whereas v_rel is computed exactly. So among near-ties, pick the lowest v_rel.
# Offline proxy_search.py over a 100-case true-cost dump: 3.1288 -> 3.0979
# (-1.0%); robust basin 0.02-0.03. See proxy_analysis.py / proxy_search.py.
# Oracle-selector ceiling for the 8-profile set is ~3.03 (3.0% below this proxy),
# so selection tuning helps but cannot alone break 3.00 — that needs the placer.
_PROXY_TIE_MARGIN = float(os.environ.get("ICCAD_PROXY_TIE_MARGIN", "0.02"))

# Per-case debug log: appended one JSON line per solve() with all profile metrics
# + chosen winner. Enabled when ICCAD_PORTFOLIO_DEBUG_LOG points to a path.
_DEBUG_LOG_PATH = os.environ.get("ICCAD_PORTFOLIO_DEBUG_LOG", "")
_DEBUG_LOG_COUNTER = 0

# All recognized profiles (validation set for the ICCAD_PORTFOLIO_PROFILES env var).
_ALL_PROFILES = [
    "gnn", "connectivity", "area_desc", "area_asc",
    "pin_centroid", "degree_desc", "degree_asc",
    "high_boundary", "low_viol", "high_viol",
]
# Default portfolio = the 8 proven profiles (Total Score 3.0554).
# low_viol/high_viol (W_VIOL ×0.5/×2.0) were tested 2026-06-02: they win 19/100
# cases by proxy but REGRESS the total (3.0554 -> 3.0859) — the contest-shape
# proxy mis-picks them on <1.5%-margin wins. Profile-count scaling has saturated;
# the bottleneck is now selector accuracy, not candidate diversity. Kept available
# via ICCAD_PORTFOLIO_PROFILES for re-test under an improved selector.
_DEFAULT_PROFILES = [
    "gnn", "connectivity", "area_desc", "area_asc",
    "pin_centroid", "degree_desc", "degree_asc",
    "high_boundary",
]

# Per-profile env-var overrides passed to the C++ subprocess. Empty dict = use
# the binary's calibrated defaults.
#   high_boundary: amplify soft boundary gradient (wins violation-heavy big-n).
#   low_viol:  halve the calibrated hard-violation weight -> SA spends its budget
#              on HPWL/area minimization; wins low-violation cases.
#   high_viol: double it -> SA prioritizes eliminating hard violations; a second
#              lever (vs boundary gradient) for violation-heavy cases.
_PROFILE_ENV: dict = {
    "high_boundary": {"ICCAD_W_BOUNDARY": "100"},
    "low_viol": {"ICCAD_W_VIOL_MULT": "0.5"},
    "high_viol": {"ICCAD_W_VIOL_MULT": "2.0"},
}
_PROFILES = os.environ.get("ICCAD_PORTFOLIO_PROFILES", "")
if _PROFILES:
    _PROFILES = [p.strip() for p in _PROFILES.split(",") if p.strip()]
    for p in _PROFILES:
        assert p in _ALL_PROFILES, f"Unknown profile: {p}"
else:
    _PROFILES = _DEFAULT_PROFILES

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
    if profile == "pin_centroid":
        return _pin_centroid_hint(n, p2b, pins)
    if profile == "degree_desc":
        return _degree_hint(n, b2b, p2b, descending=True)
    if profile == "degree_asc":
        return _degree_hint(n, b2b, p2b, descending=False)
    if profile in ("high_boundary", "low_viol", "high_viol"):
        # No hint — C++ uses connectivity perm. Divergence comes from the env-var
        # weight override (set in _PROFILE_ENV), not the perm.
        return None
    raise ValueError(f"Unknown profile: {profile}")


def _pin_centroid_hint(n, p2b, pins):
    """Per-block hint = mean (px, py) over connected pins; fallback (0,0)."""
    sx = [0.0] * n
    sy = [0.0] * n
    cnt = [0] * n
    if p2b is not None and pins is not None and len(p2b) > 0 and len(pins) > 0:
        for edge in p2b:
            pi = int(edge[0]); bi = int(edge[1])
            if pi == -1 or bi == -1 or bi >= n or pi >= len(pins):
                continue
            px = float(pins[pi][0]); py = float(pins[pi][1])
            if px == -1 and py == -1:
                continue
            sx[bi] += px; sy[bi] += py; cnt[bi] += 1
    out = []
    for i in range(n):
        if cnt[i] > 0:
            out.append((sx[i] / cnt[i], sy[i] / cnt[i]))
        else:
            out.append((0.0, 0.0))
    return out


def _degree_hint(n, b2b, p2b, descending=True):
    """Sort blocks by total (b2b + p2b) edge-weight degree."""
    deg = [0.0] * n
    if b2b is not None and len(b2b) > 0:
        for edge in b2b:
            i = int(edge[0]); j = int(edge[1])
            if i == -1 or j == -1: continue
            w = float(edge[2])
            if 0 <= i < n: deg[i] += w
            if 0 <= j < n: deg[j] += w
    if p2b is not None and len(p2b) > 0:
        for edge in p2b:
            pi = int(edge[0]); bi = int(edge[1])
            if pi == -1 or bi == -1: continue
            w = float(edge[2])
            if 0 <= bi < n: deg[bi] += w
    sign = -1.0 if descending else 1.0
    return [(sign * deg[i], sign * deg[i]) for i in range(n)]


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
    # Build env for this subprocess (inherit + per-profile overrides)
    env_extra = _PROFILE_ENV.get(profile)
    proc_env = None
    if env_extra:
        proc_env = os.environ.copy()
        proc_env.update(env_extra)
    try:
        result = subprocess.run(
            [str(_BIN)], input=inp,
            capture_output=True, text=True, timeout=55.0,
            env=proc_env,
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

    proxies = {}
    debug = {}
    for profile, (h, a, v) in metrics.items():
        area_gap = max(0.0, (a - est_area) / est_area) if est_area > 0 else 0.0
        hpwl_rel = max(0.0, (h - best_hpwl) / best_hpwl) if best_hpwl > 0 else 0.0
        proxy = (1.0 + _PROXY_ALPHA * (area_gap + hpwl_rel)) * math.exp(_PROXY_BETA * v)
        proxies[profile] = proxy
        debug[profile] = (proxy, h, a, v)

    # Pick the min-proxy profile, but on near-ties (within _PROXY_TIE_MARGIN)
    # defer to the exactly-computed violation signal: the proxy's HPWL/area
    # baselines are estimated and unreliable at sub-% margins, while v_rel is
    # exact. metrics[p] = (hpwl, area, v_rel).
    lo = min(proxies.values())
    if _PROXY_TIE_MARGIN > 0:
        near = [p for p, pr in proxies.items() if pr <= lo * (1.0 + _PROXY_TIE_MARGIN)]
    else:
        near = [p for p, pr in proxies.items() if pr <= lo]
    best_profile = min(near, key=lambda p: (metrics[p][2], proxies[p]))

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

        if _DEBUG_LOG_PATH:
            global _DEBUG_LOG_COUNTER
            import json as _json
            entry = {
                "seq": _DEBUG_LOG_COUNTER,
                "n": block_count,
                "winner": best_profile,
                "profiles": {
                    p: {"proxy": float(prx), "hpwl": float(h),
                        "area": float(a), "vrel": float(v),
                        "t": float(results[p][1])}
                    for p, (prx, h, a, v) in debug.items()
                },
            }
            _DEBUG_LOG_COUNTER += 1
            with open(_DEBUG_LOG_PATH, "a") as f:
                f.write(_json.dumps(entry) + "\n")

        return best_pos
