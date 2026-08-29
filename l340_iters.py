"""L340 iterations sweep -- the only axis that can still change the verdict.

WHY THIS AND NOT THE WEIGHT. The HW axis turned out to sit inside the seed noise
(l340_seed.py: HW=2 vs HW=4 is p~0.08 at N=5, and seed 1 happened to be the best of
5 in the HW=2 cell, which manufactured the whole apparent ordering). What is NOT
inside the noise is runtime: 26/78/154 s per case at n=40/80/120 and 2M iterations,
against a ~1.4 s per-case budget -- 19x/56x/110x over. So the live question is how
fast quality decays as iterations drop, and whether anything survives at 1.4 s.

PROTOCOL. Equal seed count at every point (min-of-N is biased downward with N, so
unequal N is not comparable -- L296/L298). The MEDIAN is the estimate; min is
reported only to show the spread, never as the headline.

HW is FIXED at one multiple across all n so the curves are comparable. Confound
worth stating: the best HW may itself move with the iteration budget (a coarser
search may want a different weight), and this sweep cannot see that.

Usage: python l340_iters.py <n-list> <HW-mult> <iters-list> <n-seeds>
  e.g. python l340_iters.py 40,80,120 2 10000,30000,100000,300000,1000000,2000000 3
"""
import statistics
import sys

from l340_run import load, run

OURS = {40: 1.1140, 80: 1.2178, 120: 1.2136}   # per-case, from L340_HANDOFF
BUDGET = 1.4                                    # seconds per case, current shipped

if __name__ == "__main__":
    NS = [int(x) for x in sys.argv[1].split(",")]
    MU = float(sys.argv[2])
    ITS = [int(x) for x in sys.argv[3].split(",")]
    S = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    print("== L340 iterations sweep: HW = %.2f*HW*, %d seeds/point (equal N) ==" % (MU, S))
    print("   estimate = MEDIAN of %d seeds. budget line = %.1f s/case." % (S, BUDGET))
    for n in NS:
        *_, arL, hpL, _nb = load(n)
        hw = (arL / hpL) * MU
        print("\n   n = %d   (ours = %.4f)" % (n, OURS.get(n, float("nan"))))
        print("   %9s %9s %9s %9s %9s %8s  %s"
              % ("iters", "median", "min", "max", "spread", "med.time", "vs ours"))
        for it in ITS:
            qs, ts = [], []
            for s in range(1, S + 1):
                r = run(n, hw, it, seed=s)
                qs.append(1 + 0.5 * (r["hg"] + r["ag"]))
                ts.append(r["dt"])
            med, mt = statistics.median(qs), statistics.median(ts)
            d = med - OURS.get(n, float("nan"))
            print("   %9d %9.4f %9.4f %9.4f %9.4f %7.1fs  %+.4f %s%s"
                  % (it, med, min(qs), max(qs), max(qs) - min(qs), mt, d,
                     "WIN " if d < 0 else "LOSE",
                     "  <= AFFORDABLE" if mt <= BUDGET else ""))
