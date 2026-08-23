"""L141 - is the OOS boundary deficit a PACKING failure or a CAPACITY failure?

L140 says boundary is the only violation family on held-out data with real
headroom (MIB is 98.8% arithmetically forced, grouping corridors are blocked).
Before building any mechanism, this asks which of the two possible causes it is:

  CAPACITY  the blocks required to touch one side do not FIT along it. Blocks
            touching the left edge all sit at x = x_min, so their y-intervals
            are disjoint and sum(h_i) <= H is a hard requirement of the achieved
            frame. If demand > capacity, `demand - capacity` worth of blocks
            MUST violate -- no packing order or repair pass can help, and the
            lever is shape (LR/TB aspect, the boundary-aspect knob) or frame.

  PACKING   demand fits, and the placer simply did not put them there. Then the
            lever is ordering / candidate generation / a repair pass.

Reported per side (L, R capacity H; T, B capacity W) on the achieved layout, so
"capacity" means "capacity of the frame the placer actually chose".

READ-ONLY.

  <python> -u l141_edge_capacity.py l140_oos_s1_c48.json --sample s1
  <python> -u l141_edge_capacity.py results_L136_48c_anchor.json --inset
"""
import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

import torch                                                        # noqa: E402

EPS = 1e-6
BITS = {1: "L", 2: "R", 4: "T", 8: "B"}


def _cases(a):
    """Yield (id, positions, cons, n, weight-n) for either corpus."""
    blob = json.load(open(a.results))
    rows = blob["test_results"]
    if a.inset:
        from iccad2026_evaluate import ContestEvaluator
        ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
        ev._load_dataset()
        for r in rows:
            if not r.get("positions"):
                continue
            at, _b, _p, _pi, cons = ev.dataset[r["test_id"]]["input"]
            yield r["test_id"], r["positions"], cons, \
                int((at != -1).sum().item())
        return
    import m77_oos_probe as M
    import m67_oos_probe as m67
    by_key = {r["key"]: r for r in rows}
    by_file = defaultdict(list)
    for ck, fk, lay_id, n in M._specs(a.sample or blob.get("sample", "s1")):
        if ck in by_key:
            by_file[fk].append((ck, lay_id, n))
    for fk, items in by_file.items():
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in items:
            lay = m67._load_case(d, lay_id)
            r = by_key[ck]
            yield r["test_id"], r["positions"], lay["cons"], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--sample", default="")
    ap.add_argument("--inset", action="store_true")
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()

    per_side, rows = [], []
    forced = packed = 0
    for idx, P, cons, n in _cases(a):
        P = [list(map(float, p)) for p in P[:n]]
        bnd = cons[:n, 4]
        if int((bnd != 0).sum().item()) == 0:
            continue
        x0 = min(p[0] for p in P)
        y0 = min(p[1] for p in P)
        x1 = max(p[0] + p[2] for p in P)
        y1 = max(p[1] + p[3] for p in P)
        W, H = x1 - x0, y1 - y0
        for bit, side in BITS.items():
            req = [i for i in range(n) if int(bnd[i].item()) & bit]
            if not req:
                continue
            cap = H if side in "LR" else W
            # extent each required block consumes ALONG the edge
            ext = {i: (P[i][3] if side in "LR" else P[i][2]) for i in req}
            demand = sum(ext.values())
            touch = {1: lambda p: abs(p[0] - x0) < EPS,
                     2: lambda p: abs(p[0] + p[2] - x1) < EPS,
                     4: lambda p: abs(p[1] + p[3] - y1) < EPS,
                     8: lambda p: abs(p[1] - y0) < EPS}[bit]
            miss = [i for i in req if not touch(P[i])]
            # how many MUST miss: greedily keep the smallest until the edge
            # is full (the achieved frame is taken as given)
            room, kept = cap, 0
            for i in sorted(req, key=lambda j: ext[j]):
                if ext[i] <= room + EPS:
                    room -= ext[i]
                    kept += 1
            floor = len(req) - kept
            forced += min(floor, len(miss))
            packed += max(0, len(miss) - floor)
            per_side.append((idx, side, len(req), len(miss), floor,
                             demand / max(cap, 1e-12)))
        rows.append(idx)

    tot_miss = sum(r[3] for r in per_side)
    print(f"\n=== L141 edge capacity: {Path(a.results).name} "
          f"({len(rows)} cases with boundary constraints) ===\n")
    print(f"side-misses {tot_miss}   forced by capacity {forced}   "
          f"packing-avoidable {packed}")
    print(f"  (a block with a corner code can miss on two sides, so side-misses "
          f"exceed the {tot_miss and ''}per-block boundary count)")
    r = [s[5] for s in per_side]
    print(f"\ndemand/capacity over all constrained sides: median {st.median(r):.3f}"
          f"   p90 {sorted(r)[int(0.9 * len(r))]:.3f}   max {max(r):.3f}")
    print(f"  sides over capacity (>1): {sum(1 for x in r if x > 1)}/{len(r)}")
    viol = [s for s in per_side if s[3]]
    if viol:
        rv = [s[5] for s in viol]
        print(f"  restricted to sides that MISSED: median {st.median(rv):.3f}"
              f"   over capacity {sum(1 for x in rv if x > 1)}/{len(rv)}")
    ok = [s for s in per_side if s[5] <= 1 and s[3]]
    print(f"\nsides that FIT but still missed: {len(ok)}  "
          f"({sum(s[3] for s in ok)} misses)")
    print(f"{'case':>5} {'side':>5} {'req':>4} {'miss':>5} {'floor':>6} "
          f"{'dem/cap':>8}")
    for s in sorted(viol, key=lambda s: -s[3])[:a.show]:
        print(f"{s[0]:>5} {s[1]:>5} {s[2]:>4} {s[3]:>5} {s[4]:>6} {s[5]:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
