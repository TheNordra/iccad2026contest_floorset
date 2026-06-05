#!/usr/bin/env python3
"""Throwaway optimizer: dumps the serialized C++ input for a case, then returns a
dummy placement. Used to capture faithful VM smoke-test inputs (the harness builds
the exact tensors the real solver would get). Safe to delete."""
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import FloorplanOptimizer
from optimizer_claude import _serialize_input

_OUT = _DIR / "vm_smoke"
_OUT.mkdir(exist_ok=True)


class MyOptimizer(FloorplanOptimizer):
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        s = _serialize_input(block_count, area_targets, b2b_connectivity,
                             p2b_connectivity, pins_pos, constraints,
                             target_positions, gnn_hint=None)
        (_OUT / f"input_n{block_count}.txt").write_text(s)
        print(f"[dump] wrote input_n{block_count}.txt ({len(s)} bytes)", file=sys.stderr)
        return [(0.0, 0.0, 1.0, 1.0)] * block_count
