#!/usr/bin/env python3
"""
Oracle BL upper bound experiment.

Pre-loads fp_sol for every validation case via the dataset, then on solve():
  1. Look up the input fingerprint (block_count + area_targets + constraints)
  2. Retrieve fp_sol = (x, y, w, h) per block
  3. Use fp_sol (w, h) as the shape for every block
  4. Sort non-preplaced blocks by fp_sol (x + y) as the BL perm
  5. Decode via skyline_decode (same packer as optimizer_claude)

Three modes (set ICCAD_ORACLE_MODE):
  - "bl"   (default): oracle perm + oracle shape -> Python skyline BL packer (no SA)
  - "raw" : return fp_sol verbatim (sanity: should match teammate's 1.1079)
  - "exe" : oracle perm fed to optimizer_claude.exe as GNN hint, default
            sqrt(area) shapes, full SA refinement.  Closest to "v3 ranking
            predicts perfect perm, rest of pipeline unchanged" question.

This answers "what's the BL packer ceiling given perfect ranking + shapes?"
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import FloorplanOptimizer, ContestEvaluator
from litetestLoader import FloorplanDatasetLiteTest

# Reuse the same skyline packer the C++ binary mirrors.
from optimizer_claude import (
    Skyline, skyline_decode,
    COL_FIXED, COL_PREPLACED,
)

_MODE = os.environ.get("ICCAD_ORACLE_MODE", "bl").lower()
assert _MODE in ("bl", "raw", "exe"), \
    f"ICCAD_ORACLE_MODE must be 'bl', 'raw', or 'exe', got {_MODE}"

if _MODE == "exe":
    # Make sure optimizer_claude's real GNN is OFF; we'll provide our own hint.
    os.environ["ICCAD_DISABLE_GNN"] = "1"
    import subprocess
    import optimizer_claude as _oc
    _oc._ensure_compiled()


def _val(x, i, j=None):
    try:
        v = x[i] if j is None else x[i, j]
    except Exception:
        v = x[i] if j is None else x[i][j]
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def _tensor_values(x, rows):
    try:
        shape = tuple(int(v) for v in x.shape)
    except Exception:
        shape = (len(x), len(x[0]) if len(x) else 0)
    if not shape:
        return ()
    if len(shape) == 1:
        return tuple(round(_val(x, i), 6) for i in range(min(rows, shape[0])))
    out = []
    for i in range(min(rows, shape[0])):
        out.append(tuple(round(_val(x, i, j), 6) for j in range(shape[1])))
    return tuple(out)


def _edge_digest(mat):
    try:
        shape = tuple(int(v) for v in mat.shape)
    except Exception:
        return ((), ())
    if len(shape) < 2:
        return (shape, ())
    rows, cols = shape[0], shape[1]
    vals = []
    for r in range(rows):
        if cols >= 3:
            vals.append((
                int(round(_val(mat, r, 0))),
                int(round(_val(mat, r, 1))),
                round(_val(mat, r, 2), 8),
            ))
        else:
            vals.append(tuple(round(_val(mat, r, c), 8) for c in range(cols)))
    return (shape, tuple(vals))


def _pin_digest(pins):
    try:
        shape = tuple(int(v) for v in pins.shape)
    except Exception:
        return ((), ())
    if len(shape) < 2:
        return (shape, ())
    return (
        shape,
        tuple(
            (round(_val(pins, i, 0), 6), round(_val(pins, i, 1), 6))
            for i in range(shape[0])
        ),
    )


def _fingerprint(block_count, area, b2b, p2b, pins, cons):
    try:
        a_shape = tuple(int(v) for v in area.shape)
    except Exception:
        a_shape = (len(area),)
    try:
        c_shape = tuple(int(v) for v in cons.shape)
    except Exception:
        c_shape = (len(cons), len(cons[0]) if len(cons) else 0)
    return (
        int(block_count),
        a_shape,
        _tensor_values(area, block_count),
        c_shape,
        _tensor_values(cons, block_count),
        _edge_digest(b2b),
        _edge_digest(p2b),
        _pin_digest(pins),
    )


class MyOptimizer(FloorplanOptimizer):
    """Oracle BL upper bound optimizer.

    On first solve(), pre-loads every validation sample, builds a fingerprint
    index → fp_sol map.  Then each solve() looks up its input and returns
    either fp_sol verbatim (mode=raw) or an oracle-BL reconstruction (mode=bl).
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._label_index: Optional[Dict[tuple, tuple]] = None
        self._misses = 0
        self._hits = 0

    def _ensure_index(self):
        if self._label_index is not None:
            return
        self._label_index = {}
        try:
            ds = FloorplanDatasetLiteTest("../")
            evaluator = ContestEvaluator("../", verbose=False)
            for idx in range(len(ds)):
                sample = ds[idx]
                area, b2b, p2b, pins, cons = sample["input"]
                n = int((area != -1).sum().item())
                _, positions = evaluator._extract_baseline(
                    idx, sample["label"], b2b, p2b, pins, n
                )
                key = _fingerprint(n, area, b2b, p2b, pins, cons)
                self._label_index[key] = tuple(positions)
        except Exception as e:
            print(f"[oracle_perm] index build failed: {e}", file=sys.stderr)
            self._label_index = {}

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
        self._ensure_index()
        key = _fingerprint(
            block_count, area_targets, b2b_connectivity,
            p2b_connectivity, pins_pos, constraints,
        )
        fp_sol = self._label_index.get(key) if self._label_index else None
        if fp_sol is None:
            self._misses += 1
            # No oracle available — return a degenerate stack (will score badly).
            n = block_count
            return [(0.0, 0.0, 1.0, 1.0)] * n
        self._hits += 1

        n = block_count
        if _MODE == "raw":
            return [tuple(fp_sol[i]) for i in range(n)]

        if _MODE == "exe":
            # Feed fp_sol (x, y) as the GNN hint; C++ sorts by (x+y) -> oracle
            # perm.  Default sqrt(area) shapes; SA does its full 8s refinement.
            hint = [(float(fp_sol[i][0]), float(fp_sol[i][1])) for i in range(n)]
            inp = _oc._serialize_input(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions,
                gnn_hint=hint,
            )
            result = subprocess.run(
                [str(_oc._BIN)], input=inp,
                capture_output=True, text=True, timeout=55.0,
            )
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError(f"C++ failed: {result.stderr[:200]}")
            return _oc._parse_output(result.stdout, block_count)

        # _MODE == "bl": oracle perm + oracle shapes via skyline BL packer
        widths = [float(fp_sol[i][2]) for i in range(n)]
        heights = [float(fp_sol[i][3]) for i in range(n)]

        preplaced = {}
        for i in range(n):
            if constraints is not None and int(_val(constraints, i, COL_PREPLACED)) == 1:
                preplaced[i] = (
                    float(fp_sol[i][0]), float(fp_sol[i][1]),
                    widths[i], heights[i],
                )

        free = [i for i in range(n) if i not in preplaced]
        # Sort by fp_sol (x + y) — the oracle BL ordering.
        free.sort(key=lambda i: float(fp_sol[i][0]) + float(fp_sol[i][1]))

        return skyline_decode(free, widths, heights, preplaced, n)
