"""L343 -- L296 x L342: what is one violation actually worth, on the FINAL corpus?

WHY NOW. Q&A A21 confirms the final is scored on the SAME hidden testcases as beta.
That promotes L296's reverse-solve (`Fraction(vrel).limit_denominator()` recovers V and
N_soft exactly, 88/88) from "a characterisation of the beta corpus" to "the per-case data
of the corpus we will actually be graded on". L342 then established that the violation
term, not geometry, is where the money moves: L340's SA bought 7 % of geometry and paid a
3.0x violation multiplier for it.

So stack them and ask the question neither answered on its own:

  A. What does removing ONE violation from case i save, in % of total score?
  B. What is the BREAK-EVEN price in geometry for that removal -- how much worse is the
     layout allowed to get before the trade stops paying?
  C. The global exchange rate, recomputed per EQUAL relative change (L296 sec.3's table
     compares 10 % of violations against 1 % of geometry and annotates it "5x more per
     relative point"; that is not a per-relative-point comparison, so recompute it).
  D. The TARGETING oracle: is choosing WHICH violation to remove worth more than removing
     the same number at random? This is the part that is actionable label-free -- w_i is
     exp(n/12) from the block count and N_soft is countable from `constraints`, both
     available at solve() time with no fitting and no label.
  E. Is N_soft actually informative, or does it just track n?

Pure analysis on data already on disk (l296_project.graded() + the dataset's own
constraints). No SA runs, no training, no shipping path touched.
"""
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from l296_project import DQ_SHIP, RANK1, graded  # noqa: E402


def q_of(r, dV=0, dG=0.0):
    """Per-case quality*violation factor, with V shifted by dV and the geometry term
    0.5*(h+a) shifted by dG."""
    G = 0.5 * (r["h"] + r["a"]) + dG
    V = max(0, r["V"] + dV)
    return (1 + G) * math.exp(2.0 * V / r["NS"]) * (1 + DQ_SHIP)


def total(R, mod=None):
    num = den = 0.0
    for r in R:
        q = mod(r) if mod else q_of(r)
        num += r["w"] * q * r["rf"]
        den += r["w"]
    return num / den


def main():
    R = graded()
    T0 = total(R)
    print("== L343: L296 x L342 -- the price of one violation on the FINAL corpus ==")
    print("   (A21: final uses the SAME hidden testcases as beta, so this IS the")
    print("    grading corpus, not a sample of it)")
    print()
    print("   shipped projection %.6f   rank-1 %.6f   gap %+.3f %%"
          % (T0, RANK1, 100 * (T0 / RANK1 - 1)))
    print("   violations: %d cases carry one, %d total (reduced-count lower bound)"
          % (sum(1 for r in R if r["V"] > 0), sum(r["V"] for r in R)))
    print()

    # ---- A. marginal saving of removing ONE violation, per case ----------------
    for r in R:
        if r["V"] > 0:
            r["save"] = 100 * (1 - total(R, lambda x, t=r:
                                         q_of(x, dV=-1 if x is t else 0)) / T0)
        else:
            r["save"] = 0.0
    hit = sorted([r for r in R if r["V"] > 0], key=lambda r: -r["save"])
    print("A. REMOVING ONE VIOLATION -- what each case is worth (%% of total score)")
    print("   %5s %5s %5s %6s %8s %9s %9s"
          % ("rank", "case", "n", "N_soft", "V", "save %", "cumul %"))
    c = 0.0
    for k, r in enumerate(hit[:8]):
        c += r["save"]
        print("   %5d %5d %5d %6d %8d %9.4f %9.4f"
              % (k + 1, r["i"], r["n"], r["NS"], r["V"], r["save"], c))
    tot_all = sum(r["save"] for r in hit)
    print("   %5s %5s %5s %6s %8s %9s %9.4f  (all %d cases, first-order)"
          % ("...", "", "", "", "", "", tot_all, len(hit)))
    print()
    need = 100 * (T0 / RANK1 - 1)

    def joint(k):
        top = {id(x) for x in hit[:k]}
        return 100 * (1 - total(R, lambda x: q_of(x, dV=-1 if id(x) in top else 0)) / T0)
    kneed = next((k for k in range(1, len(hit) + 1) if joint(k) >= need), None)
    print("   gap to rank-1 is %+.3f %%  (this projection's DQ_SHIP models ~the D arm;"
          % need)
    print("    SHIP_DECISION puts D+RF-SAFE at 1.00-1.32 %% behind rank 1)")
    print("   EXACT joint removal, best-first:  k=1 %.4f  k=2 %.4f  k=3 %.4f  k=5 %.4f"
          % (joint(1), joint(2), joint(3), joint(5)))
    print("   -> **%s of the %d violated cases closes the %+.2f %% gap; 2 close 1.32 %%**"
          % (kneed, len(hit), need))
    print()

    # ---- B. break-even geometry price -----------------------------------------
    print("B. BREAK-EVEN GEOMETRY PRICE of one violation   delta* = (1+G)(exp(2/N_soft)-1)")
    print("   w_i and rf_i CANCEL: both terms carry them, so the trade WITHIN a case")
    print("   depends only on N_soft and that case's own geometry term G.")
    print()
    print("   %6s %8s %10s %12s %14s"
          % ("N_soft", "cases", "G(median)", "delta*", "as %% of G"))
    for lo, hi in ((1, 14), (15, 24), (25, 34), (35, 49), (50, 999)):
        sel = [r for r in R if lo <= r["NS"] <= hi and r["V"] > 0]
        if not sel:
            continue
        G = statistics.median(0.5 * (r["h"] + r["a"]) for r in sel)
        ds = [(1 + 0.5 * (r["h"] + r["a"])) * (math.exp(2.0 / r["NS"]) - 1)
              for r in sel]
        d = statistics.median(ds)
        print("   %6s %8d %10.4f %12.4f %13.1f %%"
              % ("%d-%d" % (lo, min(hi, max(r["NS"] for r in sel))), len(sel), G, d,
                 100 * d / G))
    # numerical verification that the analytic delta* is right
    r = hit[0]
    dstar = (1 + 0.5 * (r["h"] + r["a"])) * (math.exp(2.0 / r["NS"]) - 1)
    a = total(R, lambda x, t=r: q_of(x, dV=-1 if x is t else 0))
    b = total(R, lambda x, t=r: q_of(x, dV=-1 if x is t else 0,
                                     dG=dstar if x is t else 0.0))
    print()
    print("   verify on case %d (N_soft %d): remove 1 violation -> %.6f, then pay"
          " delta*=%.4f of geometry -> %.6f   (T0 %.6f, residual %+.2e)"
          % (r["i"], r["NS"], a, dstar, b, T0, b - T0))
    print()

    # ---- C. global exchange rate, per EQUAL relative change ---------------------
    print("C. GLOBAL EXCHANGE RATE, per equal relative change")
    print("   %-26s %10s" % ("change", "total %"))
    rows = []
    for lab, g, phi in (("geometry  -1 %", 0.99, 1.0), ("geometry  -10 %", 0.90, 1.0),
                        ("violations -1 %", 1.0, 0.99), ("violations -10 %", 1.0, 0.90),
                        ("violations -25 %", 1.0, 0.75)):
        t = total(R, lambda x, g=g, phi=phi: (1 + 0.5 * (x["h"] + x["a"]) * g)
                  * math.exp(2.0 * x["V"] * phi / x["NS"]) * (1 + DQ_SHIP))
        rows.append((lab, 100 * (t / T0 - 1)))
        print("   %-26s %+10.4f" % (lab, 100 * (t / T0 - 1)))
    g1 = -rows[0][1]
    v1 = -rows[2][1]
    print()
    print("   per 1 relative %%: geometry %.4f  violations %.4f  ->  **geometry is"
          " %.2fx more valuable per relative point**" % (g1, v1, g1 / v1))
    print("   L296 sec.3 annotated its table \"5x more per relative point\" for")
    print("   violations, but that row compares 10 %% of violations against 1 %% of")
    print("   geometry. Per EQUAL relative change the ordering is the other way.")
    print("   What IS true: violations are far more CONCENTRATED (see A and D), so a")
    print("   targeted violation fix beats a diffuse geometry gain of the same size.")
    print()

    # ---- D. the targeting oracle ------------------------------------------------
    print("D. TARGETING ORACLE -- is knowing WHICH violation to remove worth anything?")
    print("   %5s %12s %12s %10s"
          % ("k", "best-first %", "uniform %", "ratio"))
    tv = sum(r["V"] for r in R)
    for k in (1, 2, 3, 5, 10, 25):
        bf = sum(r["save"] for r in hit[:k])
        # uniform: remove the same COUNT spread proportionally over all violations
        frac = 1.0 - k / tv
        tu = total(R, lambda x, f=frac: (1 + 0.5 * (x["h"] + x["a"]))
                   * math.exp(2.0 * x["V"] * f / x["NS"]) * (1 + DQ_SHIP))
        un = 100 * (1 - tu / T0)
        print("   %5d %12.4f %12.4f %10.2fx" % (k, bf, un, bf / max(un, 1e-9)))
    print()
    print("   both columns remove the SAME NUMBER of violations. The ratio is the value")
    print("   of AIM alone -- and aim is available at solve() time with no label and no")
    print("   fitting: w_i = exp(n/12) from block_count, N_soft countable from")
    print("   `constraints` (boundary + grouping + MIB).")
    print()

    # ---- E. is N_soft informative beyond n? ------------------------------------
    print("E. IS N_soft INFORMATIVE, OR DOES IT JUST TRACK n?")
    ns = [r["NS"] for r in R if r["V"] > 0]
    nn = [r["n"] for r in R if r["V"] > 0]
    vv = [r["v"] for r in R if r["V"] > 0]

    def pear(a, b):
        ma, mb = statistics.mean(a), statistics.mean(b)
        sa = statistics.pstdev(a) or 1e-12
        sb = statistics.pstdev(b) or 1e-12
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a) / (sa * sb)
    print("   r(N_soft, n)        %+.3f   -> N_soft is %s"
          % (pear(ns, nn),
             "largely predicted by n" if abs(pear(ns, nn)) > 0.8 else
             "NOT just a proxy for n"))
    vc = [r["V"] for r in R if r["V"] > 0]
    print("   r(N_soft, vrel)     %+.3f   <- NOT evidence of anything: vrel = V/N_soft,"
          % pear(ns, vv))
    print("                                so this is negative by arithmetic alone.")
    print("   r(N_soft, V)        %+.3f   <- THIS is the real test. If ~0, our violation"
          % pear(ns, vc))
    print("                                COUNT does not scale with the number of soft")
    print("                                constraints, so small-N_soft cases are")
    print("                                punished purely by the division.")
    print("   V: mean %.2f  p50 %d  max %d   |  V on N_soft<=24: mean %.2f (%d cases)"
          % (statistics.mean(vc), statistics.median(vc), max(vc),
             statistics.mean([r["V"] for r in R if 0 < r["V"] and r["NS"] <= 24]),
             sum(1 for r in R if r["V"] > 0 and r["NS"] <= 24)))
    print("   N_soft: min %d  p50 %d  max %d   |  N_soft/n: p50 %.3f"
          % (min(ns), statistics.median(ns), max(ns),
             statistics.median(a / b for a, b in zip(ns, nn))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
