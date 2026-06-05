#!/usr/bin/env python3
"""
Constructive-placer PORTFOLIO wrapper.

Drives constructive.exe (C++ port of the teammate's constraint-aware constructive
floorplanner). Runs several deterministic profiles in parallel and selects the
best with a BASELINE-FREE proxy of the contest cost:

    cost  = 0.5*(area/A + hpwl/H) * exp(2*vrel)
    proxy = (area/Â + hpwl/hmin) * exp(2*vrel)     (Â = 1.035*ΣblockArea, hmin =
                                                    min hpwl over profiles)

vrel is exact from (positions, constraints); area/hpwl are emitted by the C++ on
stderr ("METRICS area hpwl vbd vcl vmb nsoft"). Offline the proxy matched the
oracle ceiling almost exactly (1.6060 vs 1.6057) because constructive is
deterministic — no SA timing noise. Profiles vary boundary aspect (the highest-
leverage diversity axis) plus wire/anchor weights via env knobs.

Single best profile ~1.695; portfolio ~1.606. Set ICCAD_CONSTRUCTIVE_SINGLE=1 to
run only the base profile. ICCAD_CONSTRUCTIVE_BIN overrides the binary path.
"""
import concurrent.futures
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import (
    FloorplanOptimizer, calculate_hpwl_b2b, calculate_hpwl_p2b,
)
from optimizer_claude import _serialize_input, _parse_output, python_sa_solve

try:
    from shapely.geometry import box as _box
    from shapely.ops import unary_union as _unary_union
    _SHAPELY = True
except Exception:
    _SHAPELY = False

_BIN = Path(os.environ.get("ICCAD_CONSTRUCTIVE_BIN", str(_DIR / "constructive.exe")))

# Profiles validated by portfolio_ceiling.py. Aspect is the key diversity axis
# (high LR aspect -> low vBd on violation-heavy cases). Adding profiles is
# downside-protected: the proxy picks per-case, so a never-best profile costs
# only runtime. Oracle 1.5839 / proxy 1.5840 with this 11-profile set
# (vs 7-profile 1.6057/1.6060). wire_xhi dropped (0.9% win, ~0 oracle gain).
_PROFILES: List[Dict[str, str]] = [
    {},                                                                       # base
    {"ICCAD_WIRE_MULT": "2.0"},                                               # wire_hi
    {"ICCAD_ANCHOR_W": "0.04"},                                               # anc_lo
    {"ICCAD_WIRE_MULT": "0.5", "ICCAD_ANCHOR_W": "0.20"},                     # area_lean
    {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286"},                   # aspect_hi
    {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20"},                    # aspect_xhi
    {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286", "ICCAD_WIRE_MULT": "2.0"},  # asp_wire
    {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143"},                   # aspect_v7
    {"ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10"},                   # aspect_v10
    {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "2.0"},  # asp7_wire
    {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_ANCHOR_W": "0.04"},   # asp5_anclo
]
_RH = 1.0  # relative weight of the hpwl term in the proxy (swept optimum)


def _ensure_compiled() -> bool:
    src = _DIR / "constructive.cpp"
    if _BIN.exists() and _BIN.stat().st_mtime >= src.stat().st_mtime:
        return True
    for gpp in (r"C:\msys64\ucrt64\bin\g++.exe", "g++"):
        try:
            r = subprocess.run(
                [gpp, "-O3", "-std=c++17", "-o", str(_BIN), str(src)],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                return True
            print(f"[constructive] compile failed:\n{r.stderr}", file=sys.stderr)
        except Exception as e:
            print(f"[constructive] compile error with {gpp}: {e}", file=sys.stderr)
    return _BIN.exists()


def _proxy_metrics(positions, area_targets, b2b, p2b, pins, constraints, n):
    """Baseline-free (area, hpwl, vrel), computed EXACTLY like the harness so the
    live selector matches the offline-validated proxy. The C++ emits its own vrel
    too, but its union-find grouping (1e-3 tol) disagrees with shapely on ~34% of
    cases; replicating the harness here recovers the oracle-level selection."""
    xmin = min(p[0] for p in positions); ymin = min(p[1] for p in positions)
    xmax = max(p[0] + p[2] for p in positions); ymax = max(p[1] + p[3] for p in positions)
    area = (xmax - xmin) * (ymax - ymin)
    hpwl = calculate_hpwl_b2b(positions, b2b) + calculate_hpwl_p2b(positions, p2b, pins)

    ncols = constraints.shape[1] if constraints.dim() > 1 else 0
    vb = vg = vm = 0
    nsoft = 0
    if ncols > 4:
        bound = constraints[:n, 4]; clust = constraints[:n, 3]; mib = constraints[:n, 2]
        nsoft = int((bound != 0).sum().item())
        eps = 1e-6
        for i in range(n):
            code = int(bound[i].item())
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
        ngrp = int(clust.max().item()) if clust.numel() else 0
        for g in range(1, ngrp + 1):
            idx = [i for i in range(n) if int(clust[i].item()) == g]
            nsoft += max(0, len(idx) - 1)
            if len(idx) > 1 and _SHAPELY:
                u = _unary_union([_box(positions[i][0], positions[i][1],
                                       positions[i][0] + positions[i][2],
                                       positions[i][1] + positions[i][3]) for i in idx])
                if u.geom_type == "MultiPolygon":
                    vg += len(u.geoms) - 1
        nmib = int(mib.max().item()) if mib.numel() else 0
        for g in range(1, nmib + 1):
            idx = [i for i in range(n) if int(mib[i].item()) == g]
            nsoft += max(0, len(idx) - 1)
            shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in idx}
            vm += len(shapes) - 1
    vrel = (vb + vg + vm) / max(nsoft, 1)
    return {"area": area, "hpwl": hpwl, "vrel": vrel}


def _run_profile(env_over: Dict[str, str], inp: str, n: int):
    """Run one profile; return positions or None."""
    env = dict(os.environ)
    env.update(env_over)
    try:
        r = subprocess.run([str(_BIN)], input=inp, capture_output=True,
                           text=True, timeout=55.0, env=env)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return _parse_output(r.stdout, n)
    except Exception:
        return None


class MyOptimizer(FloorplanOptimizer):
    """Constructive fixed-outline placer, portfolio + proxy selection."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._ok = _ensure_compiled()
        self._single = os.environ.get("ICCAD_CONSTRUCTIVE_SINGLE") == "1"
        if not self._ok:
            print("[constructive] binary unavailable; falling back to python SA",
                  file=sys.stderr)

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
        if not self._ok:
            return python_sa_solve(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints,
                                   target_positions)
        inp = _serialize_input(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints, target_positions, gnn_hint=None,
        )
        profiles = _PROFILES[:1] if self._single else _PROFILES

        if len(profiles) == 1:
            positions_list = [_run_profile(profiles[0], inp, block_count)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(profiles)) as ex:
                futs = [ex.submit(_run_profile, p, inp, block_count) for p in profiles]
                positions_list = [f.result() for f in futs]

        cands = [pos for pos in positions_list if pos is not None]
        if not cands:
            print("[constructive] all profiles failed; python SA fallback",
                  file=sys.stderr)
            return python_sa_solve(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints,
                                   target_positions)
        if len(cands) == 1:
            return cands[0]

        # Baseline-free proxy selection: cost ~ (area/A + hpwl/H)*exp(2*vrel).
        metrics = [_proxy_metrics(pos, area_targets, b2b_connectivity,
                                  p2b_connectivity, pins_pos, constraints, block_count)
                   for pos in cands]
        sumA = sum(max(0.0, float(area_targets[i])) for i in range(block_count))
        A_hat = 1.035 * max(sumA, 1e-9)
        hmin = min(m["hpwl"] for m in metrics) or 1.0
        best_pos, best_proxy = cands[0], float("inf")
        for pos, m in zip(cands, metrics):
            proxy = (m["area"] / A_hat + _RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
            if proxy < best_proxy:
                best_proxy, best_pos = proxy, pos
        return best_pos
