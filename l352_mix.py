"""L352 -- re-derive the mix-vs-RF-SAFE ladder under L348's corrected baseline.

WHY. L348 found that `l296_project`'s DQ_SHIP encodes **D**'s in-set gain (-5.34 %), not
RF-SAFE's (-6.199 %), so every absolute graded projection this session quoted was the D
arm. SHIP_DECISION's NET table is arm-RELATIVE (differences), so that bias does not touch
it -- but its RANK statements ("mix needs f_eff >= 3.40 for rank 1") need an ABSOLUTE
baseline, which is exactly what was stale. This re-derives the ladder self-consistently.

MODEL, stated so the assumptions are visible:
  quality   per-arm DQ = 0.931 x (that arm's measured in-set gain over M73 @48c),
            applied to the graded corpus's own per-case (h, a, v). 0.931 is the transfer
            coefficient DQ_SHIP already encodes; L350 measured the aggregate transports
            between corpora to +/-0.5 %, which is the error bar carried at the end.
  runtime   grader_rt(arm, i) = beta_rt_i x SHIP_S x (t_local(arm,i) / t_local(D,i))
            rf_i(phi)         = max(0.7, (grader_rt_i / (M_i x phi)) ** 0.3)
            phi sweeps the single unobservable: machine speed x median drift. phi > 1
            means we are relatively faster (medians grew, or the grader is quicker).

All three arms use per-case LOCAL runtimes measured on the same platform (Windows 48c),
so their ratios are comparable even though their absolute values are not the grader's.

Offline analysis. Nothing shipped, nothing on the shipping path.
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
from l296_project import BETA, MEDS, RANK1, SHIP_S  # noqa: E402


def graded():
    """Same rows as l296_project.graded(), but keeping the RAW grader runtime and the
    published median so the RF term can be re-modelled per arm."""
    import csv
    from fractions import Fraction
    B = {r["test_id"]: r for r in json.load(open(BETA))["test_results"]}
    M = {}
    with open(MEDS) as f:
        for row in csv.DictReader(f):
            k = list(row)
            M[int(row[k[0]])] = float(row["median_runtime_s"])
    out = []
    for i in sorted(B):
        r = B[i]
        v = r["violations_relative"]
        fr = Fraction(v).limit_denominator(4000) if v > 0 else Fraction(0)
        out.append(dict(i=i, n=r["block_count"], w=math.exp(r["block_count"] / 12.0),
                        h=r["hpwl_gap"], a=r["area_gap"], v=v,
                        V=fr.numerator, NS=(fr.denominator if fr.numerator else 1),
                        rt_raw=r["runtime_seconds"], med=M[i]))
    return out

M73_48 = 1.295547821428148          # the arms' common in-set reference
COEF = 0.931                        # transfer coefficient DQ_SHIP encodes (L348)
ARMS = [("D", "l302_ship_1.json"),
        ("RF-SAFE", "l313_win48_rfsafe.json"),
        ("mix", "l303_mixpkg_c48.json")]


def local(path):
    out = {}
    for t in json.load(open(DIR / path))["test_results"]:
        out[t["block_count"]] = dict(
            rt=t.get("runtime_seconds") or 0.0,
            q=(1 + 0.5 * (max(0.0, t["hpwl_gap"]) + max(0.0, t["area_gap"])))
            * math.exp(2.0 * t["violations_relative"]),
            w=math.exp(t["block_count"] / 12.0))
    return out


def wtot(d):
    return sum(v["w"] * v["q"] for v in d.values()) / sum(v["w"] for v in d.values())


def main():
    R = graded()
    L = {nm: local(p) for nm, p in ARMS}
    print("== L352: mix vs RF-SAFE under L348's corrected baseline ==")
    print()
    print("   %-9s %14s %10s %10s %10s"
          % ("arm", "in-set @48c", "gain vs M73", "DQ", "local wall"))
    DQ = {}
    for nm, _ in ARMS:
        t = wtot(L[nm])
        g = t / M73_48 - 1
        DQ[nm] = COEF * g
        print("   %-9s %14.9f %+9.3f %% %+9.3f %% %9.1f s"
              % (nm, t, 100 * g, 100 * DQ[nm], sum(v["rt"] for v in L[nm].values())))
    print()

    # per-case local runtime ratio of each arm to D (same platform, so comparable)
    def proj(nm, phi):
        num = den = 0.0
        for r in R:
            n = r["n"]
            a, d = L[nm].get(n), L["D"].get(n)
            ratio = (a["rt"] / d["rt"]) if (a and d and d["rt"] > 0) else 1.0
            grt = r["rt_raw"] * SHIP_S * ratio
            rf = max(0.7, (grt / (r["med"] * phi)) ** 0.3)
            q = (1 + 0.5 * (r["h"] + r["a"])) * math.exp(2 * r["v"]) * (1 + DQ[nm])
            num += r["w"] * q * rf
            den += r["w"]
        return num / den

    print("   phi sweeps machine-speed x median-drift. phi=1 reproduces the beta run's")
    print("   own runtime/median ratio scaled by SHIP_S; larger phi = we are relatively")
    print("   faster (medians grew, or the grader is quicker).")
    print()
    print("   %6s | %-22s | %-22s | %s"
          % ("phi", "RF-SAFE", "mix", "mix beats rank1?"))
    print("   %6s | %10s %10s | %10s %10s |"
          % ("", "score", "vs rank1", "score", "vs rank1"))
    rows = []
    for phi in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        a, b = proj("RF-SAFE", phi), proj("mix", phi)
        rows.append((phi, a, b))
        print("   %6.2f | %10.6f %+9.3f%% | %10.6f %+9.3f%% | %s"
              % (phi, a, 100 * (a / RANK1 - 1), b, 100 * (b / RANK1 - 1),
                 "YES" if b < RANK1 else "no"))
    print()
    # where does mix cross rank 1, and where does it overtake RF-SAFE?
    def bisect(f, lo, hi):
        for _ in range(60):
            m = (lo + hi) / 2
            if f(m):
                hi = m
            else:
                lo = m
        return (lo + hi) / 2
    lo, hi = 0.3, 8.0
    if proj("mix", hi) < RANK1 < proj("mix", lo):
        print("   mix reaches rank 1 at phi >= %.3f" % bisect(lambda p: proj("mix", p) < RANK1, lo, hi))
    else:
        print("   mix does not cross rank 1 anywhere in phi in [%.1f, %.1f]" % (lo, hi))
    if proj("mix", hi) < proj("RF-SAFE", hi) and proj("mix", lo) > proj("RF-SAFE", lo):
        print("   mix overtakes RF-SAFE at phi >= %.3f"
              % bisect(lambda p: proj("mix", p) < proj("RF-SAFE", p), lo, hi))
    print()
    print("   SHIP_DECISION's ladder for comparison (its own absolute scale):")
    print("     f_eff 3.17 -> 0.85675 (rank 1) | 2.84 -> 0.86506 | 2.38 -> 0.88430")
    print("     and it put D+RF-SAFE at 0.86726-0.86994.")
    print("   L348's corrected RF-SAFE projection is 0.871174, i.e. this scale sits")
    print("   ~0.4-0.5 %% ABOVE SHIP_DECISION's. A uniform shift of that size moves every")
    print("   ladder entry the same way, mix included.")
    print()
    print("   L350 error bar on any of these: +/-0.5 %% (aggregate cross-corpus drift).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
