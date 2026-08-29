"""L305 -- f, pinned.

    t_grader(case) = ( pool_wall + serial_python ) / r        [grader is ONE wave]
      pool_wall   = max( M , C )     M = slowest profile subprocess (uncontended)
                                     C = 43 x _proxy_metrics, main thread, serial
      serial      = S                _serialize_input + argmin

Everything on the left is published per case (beta_evaluation_results.json);
everything on the right is measured uncontended on this box (l302/l304).  `r` is
therefore the SINGLE-THREAD speed ratio between this box and the grader -- which
is what `f` has to be, because the shape LP is single-threaded work that runs
after the pool, on one core, and gets no benefit from the grader's core count.

Two brackets, differing only in how much of C hides behind M:
    r_lo  = sum( max(M,C) + S ) / 52.07      full overlap
    r_hi  = sum(  M + C   + S ) / 52.07      no overlap
"""
import json, math, pickle, statistics, sys
from pathlib import Path

DIR = Path(__file__).parent
S = pickle.load(open(DIR / "l302_serial.pkl", "rb"))
PY = pickle.load(open(DIR / "l304_pyparts.pkl", "rb"))
B = {r["block_count"]: r for r in
     json.load(open(DIR / "beta_2026-08-16" / "beta_evaluation_results.json"))["test_results"]}
ns = sorted(n for n in S if n in B and n in PY)

M = {n: max(S[n]["dt"]) for n in ns}
SUMDT = {n: sum(S[n]["dt"]) for n in ns}
C = {n: PY[n]["C"] for n in ns}
SB = {n: PY[n]["s_build"] for n in ns}
G = {n: B[n]["runtime_seconds"] for n in ns}

lo = {n: max(M[n], C[n]) + SB[n] for n in ns}
hi = {n: M[n] + C[n] + SB[n] for n in ns}
r_lo = sum(lo.values()) / sum(G.values())
r_hi = sum(hi.values()) / sum(G.values())

print("== cost centres, measured UNCONTENDED on this box (beta configuration) ==")
print("   sum over 100 cases:   M %.2f s   C %.2f s   S %.2f s   (all 43 profiles: %.2f s)"
      % (sum(M.values()), sum(C.values()), sum(SB.values()), sum(SUMDT.values())))
print("   grader, published:    %.2f s   (52.0712 on the leaderboard)" % sum(G.values()))
print()
print("== f, pinned ==")
print("   r_lo (C fully hidden behind M) = %.3f" % r_lo)
print("   r_hi (no overlap at all)       = %.3f" % r_hi)
print("   ->  f  = %.2f .. %.2f   for single-threaded work" % (r_lo, r_hi))
print("   ->  f_LP = f x 1.17 = %.2f .. %.2f   (L157 5h: scipy LP is 1.17x slower"
      " on Windows than on WSL, and our dt is Windows-measured)" % (r_lo * 1.17, r_hi * 1.17))
print()

print("== does the decomposition explain the grader's PER-CASE walls? ==")


def fit(pred):
    xs = [pred[n] for n in ns]; ys = [G[n] for n in ns]
    a = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)   # zero-intercept
    yh = [x * a for x in xs]
    ybar = sum(ys) / len(ys)
    r2 = 1 - sum((ys[i] - yh[i]) ** 2 for i in range(len(ys))) / sum((v - ybar) ** 2 for v in ys)
    return 1 / a, r2, yh


for lbl, pred in (("max(M,C)+S", lo), ("M+C+S", hi), ("M only", M),
                  ("sum of all 43 profiles", SUMDT)):
    inv, r2, yh = fit(pred)
    bands = []
    for a_, b_ in [(21, 50), (51, 80), (81, 100), (101, 120)]:
        idx = [i for i, n in enumerate(ns) if a_ <= n <= b_]
        bands.append(sum(G[ns[i]] - yh[i] for i in idx) / len(idx))
    print("   %-24s implied r %6.3f   R2 %6.3f   band residuals %s"
          % (lbl, inv, r2, " ".join("%+.3f" % v for v in bands)))

print()
print("== what the old constants would have required ==")
for f in (1.99, 2.37, 2.71, 3.17):
    need = f * sum(G.values())
    print("   f = %.2f  =>  our uncontended single-thread work would have to sum to"
          " %.1f s; measured %.1f .. %.1f s" % (f, need, sum(lo.values()), sum(hi.values())))
print()
print("== why L157's 2.71 is a PARALLELISM ratio, not a clock ratio ==")
cst = [SUMDT[n] / M[n] for n in ns]
print("   c* = sum(dt)/max(dt): p10 %.1f  p50 %.1f  p90 %.1f  max %.1f"
      % (sorted(cst)[10], statistics.median(cst), sorted(cst)[89], max(cst)))
print("   grader: 43 profiles on 48 cores -> ONE wave -> pool wall = max(dt).")
print("   this box: 43 profiles on 16 physical cores -> sum-bound; the local wall"
      " carries a factor c*/cores_eff that the grader does not.")
