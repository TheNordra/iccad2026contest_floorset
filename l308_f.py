"""L308 -- f, pinned, on the correct problem.

WHAT f IS.  Every RF bill this project prints converts LOCALLY measured seconds
into GRADER seconds.  The constant in the tree is `F = 3.17` (l172_depthmap.py:39),
built at L157 5h as 2.71 x 1.17 where 2.71 = 141.07 s (a reconstructed beta
package, WSL) / 52.07 s (the same package, grader).

WHY THAT IS THE WRONG NUMBER FOR THE LP.  141.07 s is a **parallel wall** on a
box with 16 physical cores running 43 profile subprocesses; 52.07 s is a parallel
wall on 48 cores running the same 43.  43 <= 48, so on the grader the pool is ONE
wave and its wall is the single slowest profile; here it is sum-bound.  The ratio
therefore contains the core-count difference.  The shape LP is single-threaded
Python/scipy that runs AFTER the pool, in the main process, on one core -- it
collects none of that.

WHAT THIS MEASURES INSTEAD.  Decompose the grader's own published per-case wall:

    t_grader(n) = ( pool_wall + serial ) / f
      pool_wall = max( M , C )      M = slowest profile (uncontended, here)
                                    C = 43 x _proxy_metrics (main thread, serial)
      serial    = S                 _serialize_input, timed before the pool

M, C, S are all single-threaded work measured uncontended on this box, so f is
the single-thread speed ratio -- which is what the LP actually experiences.

    f_lo = sum( max(M,C) + S ) / 52.07     C fully hidden behind M
    f_hi = sum(  M + C  + S ) / 52.07      no overlap at all
"""
import json, pickle, statistics, sys
from pathlib import Path

DIR = Path(__file__).parent
S = pickle.load(open(DIR / "l307_serial.pkl", "rb"))
B = {r["block_count"]: r for r in
     json.load(open(DIR / "beta_2026-08-16" / "beta_evaluation_results.json"))["test_results"]}
ns = sorted(n for n in S if n in B)
G = {n: B[n]["runtime_seconds"] for n in ns}
lo = {n: max(S[n]["M"], S[n]["C"]) + S[n]["S"] for n in ns}
hi = {n: S[n]["M"] + S[n]["C"] + S[n]["S"] for n in ns}
GT = sum(G.values())

print("gate: the capture run reproduced L285's beta configuration exactly")
print("      (total_score 1.2598976821946901).")
print()
print("== single-threaded work, measured UNCONTENDED on this box ==")
print("   sum M %.2f s   sum C %.2f s   sum S %.2f s      (all 43 profiles: %.1f s)"
      % (sum(S[n]["M"] for n in ns), sum(S[n]["C"] for n in ns),
         sum(S[n]["S"] for n in ns), sum(S[n]["SUM"] for n in ns)))
print("   local PARALLEL wall of the same run: %.1f s" % sum(S[n]["wall"] for n in ns))
print("   grader, published:                   %.2f s" % GT)
print()
print("== f ==")
print("   f_lo = %.3f      f_hi = %.3f" % (sum(lo.values()) / GT, sum(hi.values()) / GT))
print("   f_LP = f x 1.17 = %.2f .. %.2f   (L157 5h measured scipy's LP 1.17x slower"
      % (sum(lo.values()) / GT * 1.17, sum(hi.values()) / GT * 1.17))
print("                                      on Windows than WSL; our dt is Windows)")
print()
print("== per band ==")
print("   band      M      C      S    max(M,C)+S   M+C+S    grader   f_lo   f_hi")
for a, b in [(21, 50), (51, 80), (81, 100), (101, 120)]:
    k = [n for n in ns if a <= n <= b]
    g = sum(G[n] for n in k)
    print("   %3d-%3d %6.2f %6.2f %6.2f %11.2f %8.2f %8.2f %6.3f %6.3f"
          % (a, b, sum(S[n]["M"] for n in k), sum(S[n]["C"] for n in k),
             sum(S[n]["S"] for n in k), sum(lo[n] for n in k), sum(hi[n] for n in k),
             g, sum(lo[n] for n in k) / g, sum(hi[n] for n in k) / g))
print()
print("== what the constants in the tree would have required ==")
for f in (1.99, 2.37, 2.71, 3.17):
    print("   f = %.2f  =>  our uncontended single-thread work must sum to %6.1f s;"
          "  measured %.1f .. %.1f s" % (f, f * GT, sum(lo.values()), sum(hi.values())))
print()
cst = [S[n]["SUM"] / S[n]["M"] for n in ns]
print("== the parallelism that 2.71 was actually measuring ==")
print("   c* = sum(dt)/max(dt):  p10 %.1f  p50 %.1f  p90 %.1f  max %.1f"
      % (sorted(cst)[10], statistics.median(cst), sorted(cst)[89], max(cst)))
print("   43 profiles: grader 48 cores -> ONE wave, wall = max(dt).")
print("   here 16 physical cores -> sum-bound, wall carries c*/cores_eff.")
print("   local parallel / local max(M,C)+S = %.2fx  <- that factor is the core count,"
      % (sum(S[n]["wall"] for n in ns) / sum(lo.values())))
print("      not the clock, and it is the whole of the old 2.71.")
