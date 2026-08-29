"""L311 -- price the candidates with f measured (L308) and applied ONLY to the LP.

    grader(arm)_i = grader_beta_i * ( pool_local(arm)_i / pool_local(beta)_i )
                    + LP_local(arm)_i / f_i

The first term is a SAME-PHASE ratio, so the machine factor cancels there; the
second is the only place f is needed, and f is now measured per band rather than
imported.  `pool_local = wall_local - LP_local`, and LP_local comes from
ICCAD_LP_TIMING rather than from differencing two whole-run walls.

Quality is L299's per-component projection onto the graded corpus (baselines A
and B).  Output is the projected graded total against the leaderboard.
"""
import json, math, pickle, statistics, sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402
import l299_project2 as J                                         # noqa: E402
import l309_lptime as L                                           # noqa: E402

RANK = J.LB
BETA = {r["block_count"]: r for r in json.load(
    open(DIR / "beta_2026-08-16" / "beta_evaluation_results.json"))["test_results"]}
MED = {x["n"]: x["med"] for x in P.load()}
W = {x["n"]: x["w"] for x in P.load()}
SER = pickle.load(open(DIR / "l307_serial.pkl", "rb"))


def band_f(mode="lo"):
    """per-band single-thread ratio from L308's decomposition"""
    out = {}
    for a, b in [(21, 50), (51, 80), (81, 100), (101, 120)]:
        k = [n for n in SER if a <= n <= b]
        num = sum((max(SER[n]["M"], SER[n]["C"]) if mode == "lo"
                   else SER[n]["M"] + SER[n]["C"]) + SER[n]["S"] for n in k)
        den = sum(BETA[n]["runtime_seconds"] for n in k)
        for n in k:
            out[n] = num / den
    return out


def arm(js, log):
    d = {r["block_count"]: dict(wall=r.get("runtime_seconds", 0.0), lp=0.0)
         for r in json.load(open(DIR / js))["test_results"]}
    if log and (DIR / log).exists():
        for n, (cpu, wall) in L.load(DIR / log).items():
            if n in d:
                d[n]["lp"] = wall
    return d


def price(a_js, a_log, b_js, b_log, F):
    A, B = arm(a_js, a_log), arm(b_js, b_log)
    gt, num, den, fl = {}, 0.0, 0.0, 0
    for n in BETA:
        pa = max(1e-9, A[n]["wall"] - A[n]["lp"])
        pb = max(1e-9, B[n]["wall"] - B[n]["lp"])
        gt[n] = BETA[n]["runtime_seconds"] * (pa / pb) + A[n]["lp"] / F[n]
        rf = max(0.7, (gt[n] / MED[n]) ** 0.3)
        rf0 = max(0.7, (BETA[n]["runtime_seconds"] / MED[n]) ** 0.3)
        fl += rf <= 0.7 + 1e-12
        num += W[n] * rf / rf0
        den += W[n] * 1.0
    return sum(gt.values()), fl, num / den
