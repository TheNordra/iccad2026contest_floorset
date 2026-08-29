"""L295 -- the rank-1 target, stated as an exact function of (quality, runtime).

The project's ship bar has been "+0.30 % NET" for weeks.  That bar came from
"do not lose rank 2".  The user's question is different: *what does it take to
pass rank 1?*  This module answers it from the real grader data, so no candidate
has to be judged against a proxy again.

    cost_i = (1 + 0.5*(hpwl_gap + area_gap)) * exp(2*vrel) * max(0.7,(t_i/M_i)^0.3)
    total  = sum_i w_i cost_i / sum_i w_i,      w_i = exp(n_i/12)
                                            -- iccad2026_evaluate.py, beta report

Inputs are the two real files: our per-case beta result (the only per-case
measurement of us BY the grader) and the 2026-08-23 republished medians.

A candidate is described by two numbers:
    dq  relative change of raw quality      (-0.01 = 1 % better)
    s   multiplier on our runtime vector    (1.0 = the beta package's speed)
"""
import csv, json, math, sys
from pathlib import Path

DIR = Path(__file__).parent
BETA = DIR / "beta_2026-08-16" / "beta_evaluation_results.json"
MEDS = DIR / "beta_2026-08-23" / "C_median_runtimes_beta_hidden_update.csv"

LB = {1: 0.8586322662042342, 2: 0.888187391, 3: 0.8993286931994098,
      4: 0.9265861161320369, 5: 0.9507093062865333}

# L285: shipped package measured at 0.8548-0.8679 x the beta package's wall.
SHIP_S = 0.8679


def rows():
    B = {r["test_id"]: r for r in json.load(open(BETA))["test_results"]}
    M = {}
    with open(MEDS) as f:
        for row in csv.DictReader(f):
            k = list(row)
            M[int(row[k[0]])] = float(row["median_runtime_s"])
    out = []
    for i in sorted(B):
        r = B[i]
        q = (1 + 0.5 * (r["hpwl_gap"] + r["area_gap"])) * math.exp(2 * r["violations_relative"])
        out.append(dict(i=i, n=r["block_count"], w=math.exp(r["block_count"] / 12.0),
                        h=r["hpwl_gap"], a=r["area_gap"], v=r["violations_relative"],
                        t=r["runtime_seconds"], med=M[i], q=q))
    return out


def score(R, dq=0.0, s=1.0, dt=None):
    """dq: relative quality change (per case, uniform).  s: runtime multiplier.
       dt: optional f(row)->added GRADER seconds, applied after s."""
    num = den = 0.0
    fl = 0
    for r in R:
        t = r["t"] * s + (dt(r) if dt else 0.0)
        rf = max(0.7, (t / r["med"]) ** 0.3)
        fl += rf <= 0.7 + 1e-12
        num += r["w"] * r["q"] * (1 + dq) * rf
        den += r["w"]
    tot = num / den
    raw = sum(r["w"] * r["q"] * (1 + dq) for r in R) / den
    return tot, raw, tot / raw, fl, sum(r["t"] * s + (dt(r) if dt else 0.0) for r in R)


def need(R, s=1.0, target=LB[1], dt=None):
    """relative quality improvement required to reach `target` at runtime scale s"""
    tot, raw, cw, fl, T = score(R, 0.0, s, dt)
    return target / tot - 1.0, cw, fl, T


if __name__ == "__main__":
    R = rows()
    tot, raw, cw, fl, T = score(R)
    print("gate: beta row recomputed  raw %.10f  cwRF %.6f  total %.10f  (lb %.10f)"
          % (raw, cw, tot, LB[4]))
    print()
    print("== what the shipped package projects to ==")
    for tag, dq in [("as beta (M73)", 0.0), ("in-set delta -5.34%%", -0.0534),
                    ("x93%% transfer  -4.97%%", -0.0497), ("x75%%  -4.01%%", -0.0401),
                    ("x50%%  -2.67%%", -0.0267)]:
        tt, rr, cc, ff, TT = score(R, dq, SHIP_S)
        rank = sum(1 for v in LB.values() if v < tt) + 1
        print("  %-24s raw %.5f  cwRF %.5f  total %.6f  floor %3d/100  wall %.1fs  -> rank %d"
              % (tag, rr, cc, tt, ff, TT, rank))
    print()
    print("== the target: quality still needed to BEAT each rank, at runtime scale s ==")
    print("   (baseline = shipped package = beta quality x 0.9503, s=0.8679)")
    base_dq = -0.0497
    print("    s     wall     cwRF   floor   need-vs-r1  need-vs-r2  need-vs-r3")
    for s in [0.5, 0.7, 0.8679, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]:
        tt, rr, cc, ff, TT = score(R, base_dq, s)
        n1 = LB[1] / tt - 1
        n2 = LB[2] / tt - 1
        n3 = LB[3] / tt - 1
        print("  %5.3f  %6.1fs  %.5f  %3d   %+8.3f%%  %+8.3f%%  %+8.3f%%"
              % (s, TT, cc, ff, 100 * n1, 100 * n2, 100 * n3))
    print()
    print("== component sensitivity on the GRADED corpus (weighted) ==")
    W = sum(r["w"] for r in R)
    for lab, f in [("hpwl_gap -> 0", lambda r: (1 + 0.5 * r["a"]) * math.exp(2 * r["v"])),
                   ("area_gap -> 0", lambda r: (1 + 0.5 * r["h"]) * math.exp(2 * r["v"])),
                   ("vrel     -> 0", lambda r: (1 + 0.5 * (r["h"] + r["a"]))),
                   ("vrel     x0.5", lambda r: (1 + 0.5 * (r["h"] + r["a"])) * math.exp(r["v"])),
                   ("vrel     x0.75", lambda r: (1 + 0.5 * (r["h"] + r["a"])) * math.exp(1.5 * r["v"]))]:
        q2 = sum(r["w"] * f(r) for r in R) / W
        print("   %-16s raw %.5f  (%+.2f %%)" % (lab, q2, 100 * (q2 / raw - 1)))
