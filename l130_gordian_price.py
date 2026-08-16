"""L130 gate 0 -- PRICE the GORDIAN alternation before building it.

Standing rule since L125: a mechanism is priced before it is built, because a
quality win that raises the 48-core wall is worth negative NET. L129 is already
unpriced at 2.3s/case against a ~1.0-1.5s wall, so the question here is narrow
and answerable without writing the solver:

    HOW MUCH of L129's per-case time is stage A (the quadratic solve + the one
    destructive spreading pass), and what would replacing it with L levels of
    constrained re-solves cost?

GORDIAN's alternation only touches stage A. Stage C (legalise + refine_area) and
stage D (lp_polish) are unchanged in COUNT -- they run once on whatever centres
stage A hands over. So the added price is bounded by

    (levels) x (one constrained solve at that level's size)

and the constrained solve is the unconstrained one plus R Lagrange rows, i.e. a
dense (U+R) system instead of a dense U system. Both are microseconds-to-ms at
U<=120. This script measures all of it rather than asserting it.

  <python> -u l130_gordian_price.py            # 30-case sample
  <python> -u l130_gordian_price.py --limit 100
"""
import argparse
import math
import time

import numpy as np

import l129_global_placer as L


class Clock:
    """Nesting-aware timer: refine_area CALLS legalise, so a naive wrapper
    double-counts the flip search into the legalise column and the stages stop
    summing to the total. Time inside refine_area is booked to refine_area."""

    def __init__(self):
        self.store = {}
        self.depth = 0

    def wrap(self, fn, key, opaque=False):
        def w(*a, **k):
            if self.depth:                       # already inside an opaque stage
                return fn(*a, **k)
            self.depth = 1 if opaque else 0
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.store[key] = self.store.get(key, 0.0) + (time.perf_counter() - t)
                self.depth = 0
        return w


def price_stages(limit):
    """Per-stage wall time of the CURRENT placer, weighted the way the score is."""
    keys = ("units", "global_place", "spread", "legalise", "refine_area", "lp_polish")
    tot = {k: 0.0 for k in keys}
    wtot = {k: 0.0 for k in keys}
    wsum = 0.0
    per_case = []

    orig = {k: getattr(L, k) for k in ("build_units", "global_place", "spread",
                                       "legalise", "refine_area", "lp_polish")}
    for c in L.CASES[:limit]:
        clk = Clock()
        store = clk.store
        L.build_units = clk.wrap(orig["build_units"], "units")
        L.global_place = clk.wrap(orig["global_place"], "global_place")
        L.spread = clk.wrap(orig["spread"], "spread")
        L.legalise = clk.wrap(orig["legalise"], "legalise")
        L.refine_area = clk.wrap(orig["refine_area"], "refine_area", opaque=True)
        t0 = time.perf_counter()
        P = L.place(c)
        t_place = time.perf_counter() - t0
        t_lp = 0.0
        if P is not None:
            t1 = time.perf_counter()
            L.lp_polish(c, P)
            t_lp = time.perf_counter() - t1
        store["lp_polish"] = t_lp
        w = math.exp(c["n"] / 12.0)
        wsum += w
        for k in keys:
            tot[k] += store.get(k, 0.0)
            wtot[k] += w * store.get(k, 0.0)
        per_case.append((c["idx"], c["n"], t_place + t_lp, dict(store),
                         P is not None))
    for k, v in orig.items():
        setattr(L, k, v)
    return tot, wtot, wsum, per_case


def price_solve(sizes, levels_of):
    """Cost of ONE dense unconstrained solve vs ONE KKT solve with R region rows.

    GORDIAN's constrained re-solve is

        min x'Lx - 2b'x  s.t.  Ax = u        (one row per region: the region's
                                              area-weighted centre of gravity)
      =>  [[2L, A'], [A, 0]] [x; lam] = [2b; u]

    which is a dense (U+R)x(U+R) solve. R doubles per level, so the last level is
    the expensive one and is what this measures.
    """
    rows = []
    rng = np.random.default_rng(0)
    for U in sizes:
        M = rng.normal(size=(U, U))
        Lap = M @ M.T + U * np.eye(U)
        b = rng.normal(size=U)
        r = min(9, int(np.ceil(np.log2(max(U, 2)))))

        t = time.perf_counter()
        for _ in range(20):
            np.linalg.solve(Lap, b)
        t_un = (time.perf_counter() - t) / 20

        tot_kkt = 0.0
        for lvl in range(1, r + 1):
            R = min(2 ** lvl, U)
            A = np.zeros((R, U))
            for k in range(U):
                A[k % R, k] = 1.0
            K = np.zeros((U + R, U + R))
            K[:U, :U] = 2 * Lap
            K[:U, U:] = A.T
            K[U:, :U] = A
            K[U:, U:] = 1e-9 * np.eye(R)
            rhs = np.concatenate([2 * b, rng.normal(size=R)])
            t = time.perf_counter()
            for _ in range(5):
                np.linalg.solve(K, rhs)
            tot_kkt += (time.perf_counter() - t) / 5
        rows.append((U, r, t_un, tot_kkt))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    a = ap.parse_args()

    print(f"\n=== L130 gate 0: pricing the GORDIAN alternation ===")
    print(f"sample: first {a.limit} cases\n")

    tot, wtot, wsum, per_case = price_stages(a.limit)
    grand = sum(tot.values())
    wgrand = sum(wtot.values())
    print(f"{'stage':<16} {'raw s':>10} {'raw %':>8} {'weighted s':>12} {'wt %':>8}")
    for k in ("units", "global_place", "spread", "legalise", "refine_area",
              "lp_polish"):
        print(f"{k:<16} {tot[k]:>10.3f} {100*tot[k]/max(grand,1e-9):>7.2f}% "
              f"{wtot[k]/wsum:>12.4f} {100*wtot[k]/max(wgrand,1e-9):>7.2f}%")
    print(f"{'TOTAL':<16} {grand:>10.3f} {100.0:>7.2f}% {wgrand/wsum:>12.4f} "
          f"{100.0:>7.2f}%")

    stage_a = tot["global_place"] + tot["spread"]
    wa = wtot["global_place"] + wtot["spread"]
    print(f"\nstage A (global_place + spread) = {100*stage_a/max(grand,1e-9):.2f}% raw, "
          f"{100*wa/max(wgrand,1e-9):.2f}% weighted")

    sizes = sorted({c[1] for c in per_case})
    sizes = [s for s in sizes if s >= 8][-6:]
    print(f"\n--- one solve, by unit count (dense) ---")
    print(f"{'U':>5} {'levels':>7} {'unconstrained s':>17} {'sum KKT s':>12} {'ratio':>8}")
    for U, r, t_un, t_kkt in price_solve(sizes, None):
        print(f"{U:>5} {r:>7} {t_un:>17.6f} {t_kkt:>12.6f} "
              f"{t_kkt/max(t_un,1e-12):>7.1f}x")

    print(f"\n--- projected price of the alternation ---")
    # replacing one solve with `levels` KKT solves; spreading cost is unchanged
    # in kind (one bisection per level instead of one total)
    big = price_solve([max(sizes)], None)[0]
    add = big[3] - big[2]
    n_units_note = ("solves are per UNIT, and units <= blocks (clusters collapse), "
                    "so these are upper bounds")
    print(f"  worst case in sample: U~{big[0]}, {big[1]} levels")
    print(f"  added solve time     : {add*1000:.3f} ms/case")
    print(f"  current weighted case: {wgrand/wsum:.3f} s")
    print(f"  => added share       : {100*add/max(wgrand/wsum,1e-9):.4f}%")
    print(f"  ({n_units_note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
