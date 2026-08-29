"""L310 -- re-price the candidates with f used ONLY where f belongs.

Every RF bill so far has taken a locally measured `dt` -- a difference of two
whole-run walls -- and divided the lot by one machine factor.  Two things are
wrong with that and L307/L308 fix both:

  1. the pool phase and the LP phase have DIFFERENT machine factors.  The pool is
     43 subprocesses: 48 cores on the grader (one wave, wall = slowest profile)
     against 16 physical here (sum-bound).  The LP is one thread after the pool.
     Only the LP needs f; the pool's factor is measurable as a same-phase ratio.
  2. `dt` from wall differencing carries this box's 8 % session-to-session drift.
     `ICCAD_LP_TIMING=1` measures the LP directly.

Model, per case, with everything on the right measured:

    grader(arm) = grader_beta * (pool_local(arm) / pool_local(beta))   +  LP_local(arm) / f
                  \______________ same-phase ratio, f cancels ______/     \___ f here ___/

    pool_local(x) = wall_local(x) - LP_local(x)
"""
import json, math, pickle, statistics, sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402
import l299_project2 as J                                         # noqa: E402

LB = J.LB
TH = 0.7 ** (1 / 0.3)


def beta_rows():
    B = {r["block_count"]: r for r in json.load(
        open(DIR / "beta_2026-08-16" / "beta_evaluation_results.json"))["test_results"]}
    M = {x["n"]: x["med"] for x in P.load()}
    W = {x["n"]: x["w"] for x in P.load()}
    return B, M, W


def local(js, lplog=None):
    """per-case wall and LP wall (seconds) for one arm, keyed by block count"""
    d = {r["block_count"]: dict(wall=r.get("runtime_seconds", 0.0), lp=0.0)
         for r in json.load(open(DIR / js))["test_results"]}
    if lplog:
        import l309_lptime as L
        for n, (cpu, wall) in L.load(DIR / lplog).items():
            if n in d:
                d[n]["lp"] = wall
    return d


def price(arm_js, arm_log, base_js, base_log, label, f, rows):
    B, MED, W = rows
    A, Bl = local(arm_js, arm_log), local(base_js, base_log)
    gt = {}
    for n in B:
        pool_a = max(1e-9, A[n]["wall"] - A[n]["lp"])
        pool_b = max(1e-9, Bl[n]["wall"] - Bl[n]["lp"])
        gt[n] = B[n]["runtime_seconds"] * (pool_a / pool_b) + A[n]["lp"] / f
    num = den = 0.0
    fl = 0
    for n in B:
        rf = max(0.7, (gt[n] / MED[n]) ** 0.3)
        fl += rf <= 0.7 + 1e-12
        num += W[n] * rf * B[n]["cost"] / max(0.7, (B[n]["runtime_seconds"] / MED[n]) ** 0.3)
        den += W[n]
    return sum(gt.values()), fl, num / den


if __name__ == "__main__":
    print("use l311_final.py")
