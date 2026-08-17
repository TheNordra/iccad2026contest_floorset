"""L137 gate 0 -- PRICE the GORDIAN hint before wiring it into the shipped path.

Standing rule since L125, and this session has now been burned twice by skipping
the honest version of it (L133's `--dt 0` tautology, L135's estimate that missed
by 2x). So: measure, do not project.

WHAT IS BEING PRICED. The hint is computed ONCE PER CASE on the main thread,
before the profiles are launched, and its output is N per-block centres passed
through `_serialize_input`'s existing gnn_hint channel. Per M67-E's wall model

    W(pool, cores) = max( max_k dt_k , sum_k dt_k / cores , sum_k pt_k )

main-thread Python lands in the SERIAL term, so it adds to the per-case wall
whether or not the profiles have spare cores. That is the number that matters.

BUDGET. Beta results are not in, so the ceiling is alpha-anchored (m67e_rf48):
cost-weighted RF 0.708, i.e. already at the 0.70 floor, with ~1.7x margin on the
heavy band and the MID band NOT floored at all. Un-floored cases pay for added
runtime directly, so the mid band is the binding constraint, not the heavy one.

  <python> -u l137_hint_price.py
  <python> -u l137_hint_price.py --minn 80
"""
import argparse
import json
import math
import statistics as st
import time

import l129_global_placer as L


def hint_centres(c):
    """Everything the deployed hint would have to compute, in deployment order.

    Mirrors l129_global_placer.place() up to and including the alternation, and
    stops there: the hint is the CENTRES. lp_polish, legalise and refine_area --
    which L134 measured as 95.8% of an L129 case -- are not involved, because the
    C++ does the placement.
    """
    v = L.case_view(c)
    DIMS = {i: L.choose_dims(v, i) for i in range(v["n"])}
    if L.MIB_UNIFY:
        L.unify_mib(v, DIMS)
    units = L.build_units(v, DIMS)
    cx, cy, uof = L.gordian(v, units)
    out = [(0.0, 0.0)] * v["n"]
    for k, u in enumerate(units):
        for t, i in enumerate(u["mem"]):
            ox, oy = u["off"][t]
            w, h = u["dims"][i]
            out[i] = (float(cx[k] + ox + w / 2.0), float(cy[k] + oy + h / 2.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minn", type=int, default=0)
    ap.add_argument("--anchor", default="results_L136_48c_anchor.json")
    a = ap.parse_args()
    L.GORDIAN = True                      # the hint IS the alternation

    ship = {r["test_id"]: float(r["runtime_seconds"])
            for r in json.load(open(a.anchor))["test_results"]}

    cases = [c for c in L.CASES if c["n"] >= a.minn]
    rows = []
    for c in cases:
        hint_centres(c)                                   # warm caches once
        best = min(_time_once(c) for _ in range(3))       # min-of-3, as L128 did
        rows.append((c["idx"], c["n"], best, ship.get(c["idx"], float("nan"))))

    ts = [r[2] for r in rows]
    ws = sum(math.exp(r[1] / 12.0) for r in rows)
    wt = sum(math.exp(r[1] / 12.0) * r[2] for r in rows) / ws
    wship = sum(math.exp(r[1] / 12.0) * r[3] for r in rows) / ws

    print(f"\n=== L137 gate 0: price of the GORDIAN hint ({len(rows)} cases) ===\n")
    print(f"  hint  mean {st.mean(ts) * 1000:7.2f} ms   median "
          f"{st.median(ts) * 1000:7.2f} ms   max {max(ts) * 1000:8.2f} ms")
    print(f"  hint  weighted {wt * 1000:.2f} ms")
    print(f"  ship  weighted {wship * 1000:.2f} ms  (per-case wall, 48c anchor)")
    print(f"  => added to the SERIAL term: {100 * wt / wship:.3f}% of the wall\n")

    print(f"{'case':>5} {'n':>4} {'hint ms':>9} {'ship s':>8} {'% of wall':>10}")
    worst = sorted(rows, key=lambda r: -(r[2] / max(r[3], 1e-9)))[:8]
    for idx, n, t, s in worst:
        print(f"{idx:>5} {n:>4} {t * 1000:>9.2f} {s:>8.3f} {100 * t / max(s, 1e-9):>9.2f}%")
    print("\n  (worst 8 by share of that case's own wall -- the mid band is the")
    print("   binding one: m67e_rf48 says it is NOT floored, so it pays directly)")
    return 0


def _time_once(c):
    t = time.perf_counter()
    hint_centres(c)
    return time.perf_counter() - t


if __name__ == "__main__":
    raise SystemExit(main())
