"""L303 -- pin f by regressing the GRADER's per-case wall on the uncontended
local max-profile time.

The identification, and why it is clean for the beta package:

  * the beta pool is **43 profiles** and the grader has **48 cores** -> ONE wave,
    so the grader's pool phase is EXACTLY the slowest single profile.  No c*
    argument is needed (and c* is up to 36.3 here, not the 22.5 CLAUDE.md
    records, so the argument would have been shaky).
  * that slowest profile is a single-threaded C++ run.  Measured uncontended on
    this box by `l302_replay.py` (one subprocess at a time).
  * the grader measured `runtime_seconds` = the wall of `solve()`, published per
    case in beta_evaluation_results.json.

  t_grader(n) = max_dt_local(n) / r_cpp  +  overhead_grader(n)

`r_cpp` is the single-thread compute ratio between this box and the grader, and
it is the right proxy for `f` -- the shape LP is also single-threaded numeric
work run after the pool, in the same process, on the same core.

Cases are matched by block count: both corpora are one case per n = 21..120.
"""
import csv, json, math, pickle, statistics, sys
from pathlib import Path

DIR = Path(__file__).parent
S = pickle.load(open(DIR / "l302_serial.pkl", "rb"))
B = {r["block_count"]: r for r in
     json.load(open(DIR / "beta_2026-08-16" / "beta_evaluation_results.json"))["test_results"]}

ns = sorted(k for k in S if k in B)
x = [max(S[n]["dt"]) for n in ns]          # uncontended local max profile, s
y = [B[n]["runtime_seconds"] for n in ns]  # grader wall, s


def ols(X, y):
    """X: list of feature rows (with intercept already included)."""
    k = len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(k)] for a in range(k)]
    v = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(k)]
    # gaussian elimination
    M = [row[:] + [v[i]] for i, row in enumerate(A)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for cc in range(c, k + 1):
                M[r][cc] -= f * M[c][cc]
    b = [M[i][k] / M[i][i] for i in range(k)]
    yh = [sum(b[a] * X[i][a] for a in range(k)) for i in range(len(X))]
    ybar = sum(y) / len(y)
    ss = sum((y[i] - yh[i]) ** 2 for i in range(len(y)))
    st = sum((v - ybar) ** 2 for v in y)
    return b, 1 - ss / st, yh


print("== the data ==")
print("   uncontended local max profile : p10 %.3f  p50 %.3f  p90 %.3f  max %.3f s"
      % (sorted(x)[10], statistics.median(x), sorted(x)[89], max(x)))
print("   grader per-case wall          : p10 %.3f  p50 %.3f  p90 %.3f  max %.3f s"
      % (sorted(y)[10], statistics.median(y), sorted(y)[89], max(y)))
print("   grader total %.2f s  (leaderboard 52.07)" % sum(y))
print()

print("== model 1:  t_grader = a * max_dt_local + c ==")
b, r2, yh = ols([[xi, 1.0] for xi in x], y)
print("   a = %.4f  ->  r_cpp = 1/a = %.3f      c = %.4f s      R2 = %.4f"
      % (b[0], 1 / b[0] if b[0] > 0 else float("nan"), b[1], r2))

print("\n== model 2:  + a term linear in n (serial python work grows with n) ==")
b2, r22, yh2 = ols([[x[i], ns[i], 1.0] for i in range(len(ns))], y)
print("   a = %.4f  ->  r_cpp = %.3f      b_n = %.5f s/block      c = %.4f s      R2 = %.4f"
      % (b2[0], 1 / b2[0] if b2[0] > 0 else float("nan"), b2[1], b2[2], r22))

print("\n== model 3:  + n^2 (proxy metrics are O(n^2) shapely, GIL-serial) ==")
b3, r23, yh3 = ols([[x[i], ns[i] ** 2 / 1e4, 1.0] for i in range(len(ns))], y)
print("   a = %.4f  ->  r_cpp = %.3f      b_n2 = %.5f      c = %.4f s      R2 = %.4f"
      % (b3[0], 1 / b3[0] if b3[0] > 0 else float("nan"), b3[1], b3[2], r23))

print("\n== residual structure of model 1, by band ==")
for lo, hi in [(21, 50), (51, 80), (81, 100), (101, 120)]:
    idx = [i for i, n in enumerate(ns) if lo <= n <= hi]
    print("   n %3d-%3d  mean resid %+.4f s   mean x %.3f   mean y %.3f"
          % (lo, hi, sum(y[i] - yh[i] for i in idx) / len(idx),
             sum(x[i] for i in idx) / len(idx), sum(y[i] for i in idx) / len(idx)))

print("\n== sanity: what does r_cpp = 1 (same single-thread speed) require? ==")
c1 = [y[i] - x[i] for i in range(len(ns))]
print("   implied grader overhead t_grader - max_dt_local :")
print("      p10 %+.3f  p50 %+.3f  p90 %+.3f  min %+.3f s  (negatives are impossible)"
      % (sorted(c1)[10], statistics.median(c1), sorted(c1)[89], min(c1)))
neg = sum(1 for v in c1 if v < 0)
print("      cases where the grader wall is SHORTER than our uncontended max profile: %d/100"
      % neg)
print("      -> those cases prove r_cpp > 1 on their own")
lb = max(x[i] / y[i] for i in range(len(ns)))
print("   hard lower bound on r_cpp (overhead >= 0):  r_cpp >= %.3f" % lb)
