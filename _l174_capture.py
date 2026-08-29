"""L174 capture shim -- OFFLINE ONLY, never shipped.

An optimizer whose solve() pickles its inputs for the heaviest case and returns
a trivial legal-ish layout so the harness finishes fast. Used to get REAL
(constraints, b2b, p2b, pins, area_targets) for the _proxy_metrics benchmark
without running the placer.

    python iccad2026_evaluate.py --evaluate ../_l174_capture.py --test-id 99 -o nul
"""
import pickle
from pathlib import Path

OUT = Path(__file__).resolve().parent / "_l174_case.pkl"


class MyOptimizer:
    def solve(self, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints, target_positions=None):
        import math
        import torch
        with open(OUT, "wb") as fh:
            pickle.dump({
                "block_count": int(block_count),
                "area_targets": area_targets,
                "b2b": b2b_connectivity,
                "p2b": p2b_connectivity,
                "pins": pins_pos,
                "constraints": constraints,
            }, fh)
        # a trivial grid so the harness has something to score
        n = int(block_count)
        side = int(math.ceil(math.sqrt(n)))
        out = []
        for i in range(n):
            a = float(area_targets[i]) if float(area_targets[i]) > 0 else 1.0
            w = math.sqrt(a)
            out.append([(i % side) * w, (i // side) * w, w, w])
        return torch.tensor(out, dtype=torch.float32)
