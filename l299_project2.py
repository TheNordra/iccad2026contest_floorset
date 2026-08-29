"""L299 -- the graded projection, done per component and with both baselines.

L296 used one geometry factor `g` and a flat -4.97 % for "the shipped package vs
the M73 one that actually ran on the graded corpus".  Both are too coarse, and in
opposite directions:

  * the corpora have different MIXES.  in-set hgap:agap = 0.2484 : 0.1355 (65/35),
    graded = 0.2174 : 0.2051 (51/49).  A mechanism that is mostly an AREA
    mechanism (the shape LP is) is therefore worth MORE on the graded corpus than
    a single blended `g` says.
  * a flat quality delta for the shipped package assumes every component improved
    in proportion.  It did not: area -26.6 %, hpwl -4.1 %, vrel -6.3 %.

So: project (hgap, agap, vrel) separately, and print BOTH baselines --
  A  flat: graded quality x (1 + dq),  dq = -4.97 %  (L287's 93 % transfer)
  B  component: apply the measured (g_h, g_a, phi) of ship-vs-M73 to the graded
     corpus's own per-case values
-- because the difference between them is a real uncertainty, not a rounding one.
"""
import csv, json, math, sys
from fractions import Fraction
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402

BETA = DIR / "beta_2026-08-16" / "beta_evaluation_results.json"
MEDS = DIR / "beta_2026-08-23" / "C_median_runtimes_beta_hidden_update.csv"
LB = {1: 0.8586322662042342, 2: 0.888187391, 3: 0.8993286931994098,
      4: 0.9265861161320369}
SHIP_S = 0.8679
F = 3.17
DQ_FLAT = -0.0497
M73 = "l285_betacfg.json"
# real M73 in-set 48c = 1.295548 (L285 s2); the M73-LIKE arm is 1.259898, so the
# part of the gain no kill switch can revert -- L131/L136 fixes, M74 constant
# regen -- is a factor 0.97248 that B must carry flat because it cannot be
# decomposed into components.
CODE = 1.259898 / 1.295548


def graded():
    B = {r["test_id"]: r for r in json.load(open(BETA))["test_results"]}
    M = {}
    with open(MEDS) as f:
        for row in csv.DictReader(f):
            k = list(row); M[int(row[k[0]])] = float(row["median_runtime_s"])
    return [dict(i=i, n=B[i]["block_count"], w=math.exp(B[i]["block_count"] / 12.0),
                 h=B[i]["hpwl_gap"], a=B[i]["area_gap"], v=B[i]["violations_relative"],
                 rf=max(0.7, (B[i]["runtime_seconds"] * SHIP_S / M[i]) ** 0.3))
            for i in sorted(B)]


def factors(base_json, arm_json):
    b = {r["test_id"]: r for r in json.load(open(DIR / base_json))["test_results"]}
    a = {r["test_id"]: r for r in json.load(open(DIR / arm_json))["test_results"]}
    s = lambda d, k: sum(math.exp(d[i]["block_count"] / 12.0) * max(0.0, d[i][k]) for i in b if i in a)
    return (s(a, "hpwl_gap") / s(b, "hpwl_gap"),
            s(a, "area_gap") / s(b, "area_gap"),
            s(a, "violations_relative") / max(1e-12, s(b, "violations_relative")))


def total(rows, gh, ga, phi, dq=0.0, dtg=None):
    num = den = 0.0
    for r in rows:
        rf = r["rf"] if dtg is None else None
        if dtg is not None:
            rf = dtg(r)
        q = (1 + 0.5 * (r["h"] * gh + r["a"] * ga)) * math.exp(2 * r["v"] * phi) * (1 + dq)
        num += r["w"] * q * rf; den += r["w"]
    return num / den


def rf_after(rows, dt, sel=None):
    """RF factor per case with dt (local seconds, by block count) added."""
    meds = {x["n"]: x["med"] for x in P.load()}
    def f(r):
        d = dt.get(r["n"], 0.0)
        if sel is not None and r["n"] not in sel:
            d = 0.0
        t = r["rf"]  # unused
        return None
    # simpler: rebuild from P.load()
    return None


if __name__ == "__main__":
    R = graded()
    SHIP = "l294_ship.json"
    if sys.argv[1:2] and sys.argv[1].startswith("--base="):
        SHIP = sys.argv.pop(1).split("=", 1)[1]
    gh0, ga0, ph0 = factors(M73, "l285_lp_on.json")
    print("ship vs M73-like, in-set component factors:  hgap x%.4f  agap x%.4f  vrel x%.4f"
          % (gh0, ga0, ph0))
    tA = total(R, 1, 1, 1, DQ_FLAT)
    tB = total(R, gh0, ga0, ph0) * CODE
    print()
    print("shipped package projected onto the graded corpus")
    print("   A  flat  -4.97 %%     %.6f   rank %d"
          % (tA, sum(1 for v in LB.values() if v < tA) + 1))
    print("   B  component-wise x code %.6f   rank %d"
          % (tB, sum(1 for v in LB.values() if v < tB) + 1))
    print("   need to beat rank 1 (%.6f):  A %+.3f %%   B %+.3f %%"
          % (LB[1], 100 * (LB[1] / tA - 1), 100 * (LB[1] / tB - 1)))
    print()
    arms = sys.argv[1:] or ["l294_gate0.json", "l290_inset_lp2.json", "l293_k4.json",
                            "l293_k8.json", "l296_A1.json"]
    Rows = [dict(x, t=x["t"] * SHIP_S) for x in P.load()]
    b0 = P.total(Rows)
    print("  %-22s %7s %7s %7s | %9s | %10s %10s | %9s %9s"
          % ("arm", "g_h", "g_a", "phi", "RF", "A total", "B total", "A vs r1", "B vs r1"))
    for f in arms:
        if not (DIR / f).exists():
            print("  %-22s (missing)" % f); continue
        gh, ga, ph = factors(SHIP, f)
        dt = P.dt_by_n(DIR / SHIP, DIR / f)
        import statistics
        fb = statistics.mean(dt.values()) if dt else 0.0
        rf = (P.total(Rows, lambda x: max(0.0, dt.get(x["n"], fb)) / F) - b0) / b0
        a1 = total(R, gh, ga, ph, DQ_FLAT) * (1 + rf)
        b1 = total(R, gh0 * gh, ga0 * ga, ph0 * ph) * CODE * (1 + rf)
        print("  %-22s %7.4f %7.4f %7.4f | %+8.4f%% | %10.6f %10.6f | %+8.3f%% %+8.3f%%%s"
              % (f.replace(".json", ""), gh, ga, ph, 100 * rf, a1, b1,
                 100 * (LB[1] / a1 - 1), 100 * (LB[1] / b1 - 1),
                 "  <== r1" if min(a1, b1) < LB[1] else ""))
