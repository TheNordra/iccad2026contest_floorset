"""L285: the shipped package's 48-core runtime, answered the only sound way.

WHY NOT MEASURE IT DIRECTLY.  This box has 32 logical cores, and forcing
ICCAD_ADAPTIVE_CORES=48 changes pool SELECTION only -- route A then fans the
frame-trial loop 48 ways onto 32 cores, so the local 127.4 s is an
oversubscription artefact, not a 48-core projection.  Local wall noise is also
>= 20 % (CLAUDE.md).  And `audit_cache_ship.pkl`, the usual offline route, is
STALE: its signature carries REFINE band (60,100]->6 / (100,inf)->4 while the
shipped wrapper is L223/L231's 2/2, and its pool-best reproduces the shipped
per-case cost on only 4/100 cases (weighted 1.292646 vs 1.226325, +5.4 %).

WHAT IS SOUND.  We hold our own per-case runtimes AS MEASURED BY THE GRADER
(beta_2026-08-16/beta_evaluation_results.json, 52.0712 s total) and the
per-case medians the grader prices against (the 2026-08-23 republication).  So
instead of guessing an absolute, invert the question:

    how much slower may the package get before it loses a rank?

    RF_i   = max(0.7, (t_i / M_i) ** 0.3)        iccad2026_evaluate.py:552
    total  = sum_i w_i * q_i * RF_i / sum_i w_i,  w_i = exp(n_i / 12)

Sweep a slowdown factor s on the measured runtime vector.  This is a multiplier,
which l276_price.py rightly warns is not how an added-time DISTRIBUTION behaves
-- but here the question genuinely is a scaling one ("the same placer, more
work per case"), and the per-case floor structure is preserved exactly because
every t_i is scaled and every M_i is real.

QUALITY TRANSFER, stated as the assumption it is.  The hidden set's per-case
quality for the CURRENT code is unknown -- it has never been submitted.  The
in-set 48c total went 1.295548 (M73, the code that produced the beta row) ->
1.226325 (now), i.e. -5.34 %.  That factor is applied to the beta hidden raw.
L275's warning applies: in-set and hidden are not the same corpus.  A
sensitivity band is printed rather than a single number.
"""
import csv
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).parent
BETA = json.load(open(DIR / "beta_2026-08-16" / "beta_evaluation_results.json"))
MED = {}
with open(DIR / "beta_2026-08-23" /
          "C_median_runtimes_beta_hidden_update.csv") as f:
    for r in csv.DictReader(f):
        MED[int(r["test_id"])] = float(r["median_runtime_s"])

tr = BETA["test_results"]
cases = [(t["test_id"], t["block_count"], t["runtime_seconds"], t["cost"])
         for t in tr]
Wt = {ci: math.exp(n / 12.0) for ci, n, _t, _c in cases}
WS = sum(Wt.values())
T0 = sum(t for _ci, _n, t, _c in cases)


def cwrf(s):
    """cost-weighted RF at slowdown factor s, and the graded total per unit raw"""
    num = den = 0.0
    nfloor = 0
    for ci, n, t, q in cases:
        rf = max(0.7, (s * t / MED[ci]) ** 0.3)
        if rf <= 0.7 + 1e-12:
            nfloor += 1
        num += Wt[ci] * q * rf
        den += Wt[ci] * q
    return num / den, nfloor


raw_beta = sum(Wt[ci] * q for ci, _n, _t, q in cases) / WS
r1, nf1 = cwrf(1.0)
print("== gate: reproduce the beta row from its own parts ==")
print(f"   raw   computed {raw_beta:.10f}   leaderboard 1.3206649447461247")
print(f"   cwRF  computed {r1:.6f}")
print(f"   total computed {raw_beta * r1:.9f}   leaderboard 0.9265861161320369")
print(f"   runtime {T0:.4f} s   leaderboard 52.07122778892517")
print(f"   cases sitting on the RF floor at s=1: {nf1}/100")

INSET_M73, INSET_NOW = 1.295548, 1.226325
f = INSET_NOW / INSET_M73
raw_now = raw_beta * f
print(f"\n== quality transfer (an assumption, not a measurement) ==")
print(f"   in-set 48c  M73 {INSET_M73}  ->  now {INSET_NOW}   "
      f"factor {f:.5f} ({100 * (f - 1):+.2f} %)")
print(f"   projected hidden-set raw for the current package: {raw_now:.6f}")

BOARD = [(1, 0.8586322662042342), (2, 0.888187391),
         (3, 0.8993286931994098), (4, 0.9265861161320369)]
print(f"\n== headroom: how much slower before each rank is lost ==")
print(f"   {'need to stay under':>22}{'req cwRF':>11}{'max s':>9}"
       f"{'max runtime':>14}")
for rank, thr in BOARD[1:]:
    need = thr / raw_now
    lo, hi = 1.0, 64.0
    if cwrf(lo)[0] > need:
        print(f"   rank {rank - 1:<2} (< {thr:.6f}) : ALREADY LOST at s=1")
        continue
    for _ in range(60):
        mid = (lo + hi) / 2
        if cwrf(mid)[0] <= need:
            lo = mid
        else:
            hi = mid
    print(f"   rank {rank - 1:<2} (< {thr:.6f}){need:>11.5f}{lo:>9.2f}x"
          f"{T0 * lo:>12.1f} s")

print(f"\n== where the current package would land, at several slowdowns ==")
print(f"   {'s':>6}{'runtime':>11}{'cwRF':>10}{'total':>11}{'rank':>7}"
      f"{'floor cases':>13}")
for s in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
    c, nf = cwrf(s)
    tot = raw_now * c
    rank = 1 + sum(1 for _r, th in BOARD if th < tot)
    rank = max(rank, 1)
    print(f"   {s:>5.2f}x{T0 * s:>10.1f}s{c:>10.5f}{tot:>11.5f}{rank:>7}"
          f"{nf:>13}")

print(f"\n== sensitivity of the quality-transfer assumption ==")
print(f"   {'in-set gain kept':>18}{'raw':>10}{'total @ s=1':>14}{'rank':>7}")
for keep in (1.0, 0.75, 0.5, 0.25, 0.0):
    rn = raw_beta * (1.0 - (1.0 - f) * keep)
    tot = rn * r1
    rank = 1 + sum(1 for _r, th in BOARD if th < tot)
    print(f"   {100 * keep:>16.0f} %{rn:>10.5f}{tot:>14.5f}{max(rank, 1):>7}")
