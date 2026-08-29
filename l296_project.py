"""L296 -- project a mechanism measured on ANY corpus onto the GRADED corpus.

WHY THIS EXISTS.  Every quality verdict in this project is read off the in-set
100.  On the two components that are geometry (hpwl_gap, area_gap) that is
defensible.  On the third it is not:

    weighted vrel      in-set 100   0.01407   (54/100 cases carry one)
                       beta hidden  0.04252   (88/100)   <- the corpus that is GRADED
                       OOS s1       0.08620   (240/240)

    soft violations    in-set 100   66        beta hidden 152

The graded numbers are not an estimate: `violations_relative` is the exact
rational V / N_soft, so `Fraction(v).limit_denominator()` recovers BOTH the
violation count and N_soft for every one of the 100 hidden cases (88/88 exact).

Consequence: the in-set UNDERSTATES any mechanism that buys violations with
geometry by roughly 3x, and OOS overstates it by 2x.  L278 stated the rule
("a corpus can only vote on a mechanism whose antecedent it contains"); this
turns it into a number.

The projection is first order and deliberately crude in one direction only:
    g   = relative change of the geometry term  0.5*(hgap+agap)
    phi = relative change of the violation count
and both are applied to the graded corpus's OWN per-case distribution.  It
assumes the mechanism's RELATIVE effect transfers, which is the same assumption
L287 measured at 93 % for the package as a whole.

  <python> l296_project.py base.json arm.json [label]
"""
import csv, json, math, sys
from fractions import Fraction
from pathlib import Path

DIR = Path(__file__).parent
BETA = DIR / "beta_2026-08-16" / "beta_evaluation_results.json"
MEDS = DIR / "beta_2026-08-23" / "C_median_runtimes_beta_hidden_update.csv"
RANK1 = 0.8586322662042342
SHIP_S = 0.8679          # L285: shipped wall / beta wall
DQ_SHIP = -0.0497        # 93 % transfer of the in-set -5.34 % since M73


def graded():
    B = {r["test_id"]: r for r in json.load(open(BETA))["test_results"]}
    M = {}
    with open(MEDS) as f:
        for row in csv.DictReader(f):
            k = list(row); M[int(row[k[0]])] = float(row["median_runtime_s"])
    out = []
    for i in sorted(B):
        r = B[i]
        v = r["violations_relative"]
        fr = Fraction(v).limit_denominator(4000) if v > 0 else Fraction(0)
        out.append(dict(i=i, n=r["block_count"], w=math.exp(r["block_count"] / 12.0),
                        h=r["hpwl_gap"], a=r["area_gap"], v=v,
                        V=fr.numerator, NS=(fr.denominator if fr.numerator else 1),
                        rf=max(0.7, (r["runtime_seconds"] * SHIP_S / M[i]) ** 0.3)))
    return out


def summarise(base_json, arm_json):
    b = {r["test_id"]: r for r in json.load(open(base_json))["test_results"]}
    a = {r["test_id"]: r for r in json.load(open(arm_json))["test_results"]}
    W = gb = ga = vb = va = 0.0
    for i in b:
        if i not in a:
            continue
        w = math.exp(b[i]["block_count"] / 12.0)
        W += w
        gb += w * 0.5 * (max(0, b[i]["hpwl_gap"]) + max(0, b[i]["area_gap"]))
        ga += w * 0.5 * (max(0, a[i]["hpwl_gap"]) + max(0, a[i]["area_gap"]))
        vb += w * b[i]["violations_relative"]
        va += w * a[i]["violations_relative"]
    return ga / gb, (va / vb if vb > 0 else 1.0), gb / W, vb / W


def project(g, phi, dq=DQ_SHIP, rows=None):
    R = rows or graded()
    num0 = num1 = den = 0.0
    for r in R:
        q0 = (1 + 0.5 * (r["h"] + r["a"])) * math.exp(2 * r["v"]) * (1 + dq)
        q1 = (1 + 0.5 * (r["h"] + r["a"]) * g) * math.exp(2 * r["v"] * phi) * (1 + dq)
        num0 += r["w"] * q0 * r["rf"]; num1 += r["w"] * q1 * r["rf"]; den += r["w"]
    return num0 / den, num1 / den


if __name__ == "__main__":
    R = graded()
    base = sys.argv[1] if len(sys.argv) > 1 else "l285_lp_on.json"
    arms = sys.argv[2:] or ["l296_A1.json", "l296_A2.json"]
    t0, _ = project(1.0, 1.0, rows=R)
    print("graded projection of the shipped package: %.6f   rank-1 = %.6f  (need %+.3f %%)"
          % (t0, RANK1, 100 * (RANK1 / t0 - 1)))
    print()
    print("  %-22s %9s %9s | %9s %9s %9s"
          % ("arm", "g", "phi", "in-set", "graded", "vs rank1"))
    for f in arms:
        p = DIR / f
        if not p.exists():
            print("  %-22s (missing)" % f); continue
        g, phi, gb, vb = summarise(DIR / base, p)
        jb = json.load(open(DIR / base)); ja = json.load(open(p))
        ins = 100 * (ja["total_score"] / jb["total_score"] - 1)
        _, t1 = project(g, phi, rows=R)
        print("  %-22s %9.5f %9.5f | %+8.4f%% %+8.4f%% %9.4f%%"
              % (f.replace(".json", ""), g, phi, ins, 100 * (t1 / t0 - 1),
                 100 * (RANK1 / t1 - 1)))
    print()
    print("  in-set geometry term %.5f  vrel %.5f" % (gb, vb))
    print("  graded geometry term %.5f  vrel %.5f"
          % (sum(r["w"] * 0.5 * (r["h"] + r["a"]) for r in R) / sum(r["w"] for r in R),
             sum(r["w"] * r["v"] for r in R) / sum(r["w"] for r in R)))
    print()
    print("  sensitivity on the GRADED corpus (what one unit of each is worth):")
    for lab, gg, pp in [("geometry -1 %", 0.99, 1.0), ("violations -10 %", 1.0, 0.90),
                        ("violations -25 %", 1.0, 0.75), ("violations -50 %", 1.0, 0.50)]:
        _, t1 = project(gg, pp, rows=R)
        print("     %-18s %+8.4f %%   -> vs rank1 %+8.4f %%"
              % (lab, 100 * (t1 / t0 - 1), 100 * (RANK1 / t1 - 1)))
