#!/usr/bin/env python3
"""M47 A1 probe: decompose MyOptimizer.solve() wall time inside the scored window.

    total = [serialize + pool setup] + pool_span (C++ subprocess phase, parallel)
            + serial shapely proxy tail + selection loop

The tail (everything after the last profile finishes) is the latency the
proxy-into-threads lever can remove. Never shipped; measurement only.
"""
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(r"C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet")
sys.path.insert(0, str(REPO / "iccad2026contest"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import optimizer_constructive as oc  # noqa: E402
import iccad2026_evaluate as ev_mod  # noqa: E402

ev = ev_mod.ContestEvaluator(data_path=str(REPO), verbose=False)
ev._load_dataset()
opt = oc.MyOptimizer(verbose=False)
assert opt._ok, "constructive.exe unavailable"

lock = threading.Lock()
cur = None

_orig_serialize = oc._serialize_input
_orig_proxy = oc._proxy_metrics
_orig_run = oc._run_profile


def timed_serialize(*a, **k):
    t0 = time.perf_counter()
    r = _orig_serialize(*a, **k)
    cur["ser"] += time.perf_counter() - t0
    return r


def timed_proxy(*a, **k):
    t0 = time.perf_counter()
    r = _orig_proxy(*a, **k)
    cur["proxy"].append(time.perf_counter() - t0)
    return r


def timed_run(env_over, inp, n):
    t0 = time.perf_counter()
    r = _orig_run(env_over, inp, n)
    t1 = time.perf_counter()
    with lock:
        cur["prof"].append((t0, t1))
    return r


oc._serialize_input = timed_serialize
oc._proxy_metrics = timed_proxy
oc._run_profile = timed_run

# pick cases: every n>100 (the RF-scoring band) + a few mid/small references
picks = []
mid_left, small_left = 3, 2
for idx in range(len(ev.dataset)):
    at = ev.dataset[idx]['input'][0]
    n = int((at != -1).sum().item())
    if n > 100:
        picks.append((idx, n))
    elif 60 < n <= 100 and mid_left:
        picks.append((idx, n)); mid_left -= 1
    elif n <= 40 and small_left:
        picks.append((idx, n)); small_left -= 1

rows = []
print(f"{'case':>4} {'n':>4} {'k':>3} {'total':>7} {'ser':>6} {'pre':>6} "
      f"{'pool':>7} {'pmax':>7} {'proxyS':>7} {'tail':>7} {'tail%':>6}")
for idx, n in picks:
    sample = ev.dataset[idx]
    inputs, labels = sample['input'], sample['label']
    area_target, b2b, p2b, pins, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    baseline, target_pos = ev._extract_baseline(
        idx, labels, b2b, p2b, pins, block_count)

    # verbatim from iccad2026_evaluate.py:869-881
    opt_target_pos = torch.full((block_count, 4), -1.0)
    if target_pos is not None and constraints is not None:
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(block_count):
            is_fixed = nc > 0 and constraints[i, 0] != 0
            is_preplaced = nc > 1 and constraints[i, 1] != 0
            if is_preplaced:
                tx, ty, tw, th = target_pos[i]
                opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
            elif is_fixed:
                _, _, tw, th = target_pos[i]
                opt_target_pos[i, 2] = tw
                opt_target_pos[i, 3] = th

    cur = {"ser": 0.0, "proxy": [], "prof": []}
    t0 = time.perf_counter()
    positions = opt.solve(block_count, area_target, b2b, p2b, pins,
                          constraints, opt_target_pos)
    total = time.perf_counter() - t0
    assert positions is not None and len(positions) == block_count

    starts = [s for s, e in cur["prof"]]
    ends = [e for s, e in cur["prof"]]
    pool_span = max(ends) - min(starts)
    pre = min(starts) - t0                      # serialize + pool setup
    pool_end = max(ends) - t0
    proxy_sum = sum(cur["proxy"])
    prof_max = max(e - s for s, e in cur["prof"])
    tail = total - pool_end                     # serial proxy + selection
    row = dict(case=idx, n=n, k=len(cur["prof"]), total=total,
               ser=cur["ser"], pre=pre, pool_span=pool_span,
               prof_max=prof_max, proxy_sum=proxy_sum, tail=tail,
               tail_pct=100.0 * tail / total)
    rows.append(row)
    print(f"{idx:>4} {n:>4} {row['k']:>3} {total:>7.3f} {cur['ser']:>6.3f} "
          f"{pre:>6.3f} {pool_span:>7.3f} {prof_max:>7.3f} {proxy_sum:>7.3f} "
          f"{tail:>7.3f} {row['tail_pct']:>5.1f}%")

big = [r for r in rows if r["n"] > 100]
tot = sum(r["total"] for r in big)
tail = sum(r["tail"] for r in big)
pre = sum(r["pre"] for r in big)
print(f"\nn>100 aggregate: total={tot:.2f}s  tail={tail:.2f}s "
      f"({100*tail/tot:.1f}%)  pre={pre:.2f}s ({100*pre/tot:.1f}%)  "
      f"tail+pre={100*(tail+pre)/tot:.1f}%")

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(rows, indent=1))
print(f"saved -> {out}")
