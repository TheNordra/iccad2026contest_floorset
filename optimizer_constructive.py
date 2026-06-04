#!/usr/bin/env python3
"""
Constructive-placer wrapper. Drives constructive.exe (C++ port of the teammate's
constraint-aware constructive floorplanner) using the SAME stdin wire format as
optimizer_claude (reused _serialize_input / _parse_output).

Rationale: our SA + skyline-BL placer is architecturally capped at Total Score
~3.27 (proven by the oracle-perm experiment). The teammate's constructive
boundary-aware placer reaches ~1.74 on the same validation set. This is a clean
C++ reimplementation of that architecture in our codebase (Milestone 1).

Set ICCAD_CONSTRUCTIVE_BIN to override the binary path.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import FloorplanOptimizer
from optimizer_claude import _serialize_input, _parse_output, python_sa_solve

_BIN = Path(os.environ.get("ICCAD_CONSTRUCTIVE_BIN", str(_DIR / "constructive.exe")))


def _ensure_compiled():
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


class MyOptimizer(FloorplanOptimizer):
    """Constructive fixed-outline placer."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._ok = _ensure_compiled()
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
        try:
            r = subprocess.run([str(_BIN)], input=inp, capture_output=True,
                               text=True, timeout=55.0)
            if r.returncode != 0 or not r.stdout.strip():
                raise RuntimeError(f"rc={r.returncode} stderr={r.stderr[:200]}")
            return _parse_output(r.stdout, block_count)
        except Exception as e:
            print(f"[constructive] run failed ({e}); python SA fallback",
                  file=sys.stderr)
            return python_sa_solve(block_count, area_targets, b2b_connectivity,
                                   p2b_connectivity, pins_pos, constraints,
                                   target_positions)
