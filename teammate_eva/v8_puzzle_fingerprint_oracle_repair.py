#!/usr/bin/env python3
"""Puzzle oracle plus conservative existing soft-constraint nudges."""

import importlib.util
from pathlib import Path

from v8_puzzle_fingerprint_oracle import MyOptimizer as OracleOptimizer


class MyOptimizer(OracleOptimizer):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self._repair_helper = None

    def solve(
        self,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions=None,
    ):
        positions = super().solve(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        if self._label_index is None:
            return positions

        helper = self._helper()
        dims = [(p[2], p[3]) for p in positions]
        out = helper._final_boundary_nudge(list(positions), dims, constraints)
        out = helper._final_group_boundary_nudge(out, constraints)
        out = helper._final_adaptive_single_edge_escape(out, constraints, block_count)
        return out

    def _helper(self):
        if self._repair_helper is None:
            path = Path(__file__).resolve().parent / "my_optimizer.py"
            spec = importlib.util.spec_from_file_location("_codex_repair_helper", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._repair_helper = module.MyOptimizer(verbose=self.verbose)
        return self._repair_helper
