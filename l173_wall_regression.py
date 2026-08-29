"""L173 - the banked stack's WALL relative to the beta package has never been
priced, and on the same box it is 6.2x with the shape LP switched OFF.

MEASUREMENT (all on WSL2, 32 logical cores, ICCAD_ADAPTIVE_CORES=48, the same
harness and the same flags):

    M73, the package the graders actually ran        141.07 s   1.41 s/case
    current tree, LP off  (l166 LANE 1)              877.67 s   8.78 s/case   6.22x
    current tree, L147 off(l166 LANE 2)             1073.68 s  10.74 s/case   7.61x
    current tree, SHIPPED (l166 LANE 3)             1022.14 s  10.22 s/case   7.25x

    per-n, LP OFF vs M73:
        n  21- 40    7.88s ->   56.00s    7.10x
        n  41- 60   21.15s ->   63.36s    3.00x
        n  61- 80   22.96s ->   96.04s    4.18x
        n  81-100   38.86s ->  202.30s    5.21x
        n 101-120   50.22s ->  459.97s    9.16x     <- 71.4% of the corpus weight

The LP is not the cause: LANE 1 has it off and is already 6.22x. It is the POOL.

WHY IT WAS NEVER SEEN. Every runtime verdict in this ledger prices a mechanism
as a DELTA against the tree it was measured on. Nothing ever compared the tree
against M73, because M73's 52.07s was treated as our runtime -- HANDOFF
2026-08-24 states "beta's 52.07s contains no LP" and then adds LP seconds to
it. That is only valid if the pool did not move. It moved.

WHAT THIS IS NOT. It is not proof that the grader will see 7x. The wall is
max-setter bound only while the pool fits the box, and CLAUDE.md records that
detected cores is an UPPER BOUND on effective cores (16 logical ~ 10 effective
here). WSL reports 32; M73's measured c* was p50 19.3 / max 22.5, i.e. it was
BARELY max-bound on that box. A larger, heavier pool is not, so part of this
6-9x is over-subscription that a real 48-core grader would not pay.

But part of it transfers regardless of core count: M79/M80's knob vectors are
recorded in CLAUDE.md as "5-12s per case, they become the 48-core max-setter on
their own", and a max-setter that got slower raises the wall on ANY core count.

This script prices the range instead of guessing the point.
"""
import json
import math
import statistics as st

import l146_rf_price as L
import l172_depthmap as M

THR = L.THR
# rank thresholds from the 2026-08-23 leaderboard
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333)]
# measured per-n wall ratio, current tree with LP OFF vs M73, WSL32 @48c
BAND = [(21, 40, 7.10), (41, 60, 3.00), (61, 80, 4.18),
        (81, 100, 5.21), (101, 120, 9.16)]


def ratio(n):
    for lo, hi, r in BAND:
        if lo <= n <= hi:
            return r
    return BAND[-1][2]


def rank_of(total):
    return sum(1 for _, t in RANKS if t < total) + 1


def project(rows, wall_mult_of, quality_pct):
    """graded total when each case's wall is scaled and quality improves."""
    num = den = 0.0
    for r in rows:
        t = r["t"] * wall_mult_of(r["n"])
        num += r["w"] * r["q"] * (1 - quality_pct / 100.0) \
            * max(0.7, (t / r["med"]) ** 0.3)
        den += r["w"]
    return num / den


def main():
    rows = M.rows_new()
    print(__doc__)
    print("=" * 78)
    print("2026-08-23 thresholds:  " + "   ".join(
        "r{} {:.5f}".format(k, v) for k, v in RANKS))

    print("\n=== if the measured per-n ratio transfers, damped by `share` ===")
    print("share = 0 means the grader pays none of it (pure over-subscription")
    print("artefact); share = 1 means it pays all of it.\n")
    print("{:>7}{:>10}{:>12}{:>12}{:>12}{:>12}"
          .format("share", "our wall", "q=+0%", "q=+2.0%", "q=+3.0%", "q=+4.0%"))
    for share in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        mult = lambda n, s=share: 1.0 + s * (ratio(n) - 1.0)   # noqa: E731
        wall = sum(r["t"] * mult(r["n"]) for r in rows)
        cells = ""
        for q in (0.0, 2.0, 3.0, 4.0):
            tot = project(rows, mult, q)
            cells += "{:.5f} r{}  ".format(tot, rank_of(tot))
        print("{:>7.2f}{:>9.1f}s   {}".format(share, wall, cells))

    print("\n=== the break-even: how much wall can the stack pay for +3.0%? ===")
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        mult = lambda n, s=mid: 1.0 + s * (ratio(n) - 1.0)     # noqa: E731
        if project(rows, mult, 3.0) < 0.9265861161320369:
            lo = mid
        else:
            hi = mid
    mult = lambda n, s=lo: 1.0 + s * (ratio(n) - 1.0)          # noqa: E731
    print("a +3.0% quality stack stops beating the BETA SCORE at share {:.3f}"
          .format(lo))
    print("   = our wall {:.1f}s (beta was 52.07s), i.e. {:.2f}x aggregate"
          .format(sum(r["t"] * mult(r["n"]) for r in rows),
                  sum(r["t"] * mult(r["n"]) for r in rows) / 52.07))
    lo3, hi3 = 0.0, 1.0
    for _ in range(40):
        mid = (lo3 + hi3) / 2
        mm = lambda n, s=mid: 1.0 + s * (ratio(n) - 1.0)       # noqa: E731
        if project(rows, mm, 3.0) < 0.8993286931994098:
            lo3 = mid
        else:
            hi3 = mid
    mm = lambda n, s=lo3: 1.0 + s * (ratio(n) - 1.0)           # noqa: E731
    print("it stops reaching RANK 3 at share {:.3f} = wall {:.1f}s = {:.2f}x"
          .format(lo3, sum(r["t"] * mm(r["n"]) for r in rows),
                  sum(r["t"] * mm(r["n"]) for r in rows) / 52.07))

    print("\n=== what the beta grader would have to look like for share ~ 1 ===")
    print("M73 on WSL32@48c was {:.2f}x its own graded wall (141.07 / 52.07),"
          .format(141.07 / 52.07))
    print("which L160/L161 read as the machine-speed factor f. If the current")
    print("pool is ALSO max-setter bound on the grader, the same f applies and")
    print("share = 1. If it has outgrown the box, share < 1 and the excess is")
    print("over-subscription. THE ONE MEASUREMENT THAT SETTLES IT is c* for the")
    print("current pool: c* <= effective cores  =>  max-bound  =>  share ~ 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
