"""L340 seed-noise probe -- the measurement this line has never had.

WHY. Every anneal in L333-L336b and L340 runs exactly ONE seed per cell, and the
seed is a deterministic function of n (l333: 1234+n, l334/l336/l336b: 7+n) or
hardcoded 1 (l340_run.run). So no within-cell spread has ever been measured, and
two anomalies in the HW sweep are exactly what unmeasured spread looks like:

  * n=80, HW 1->2 improves BOTH hpwl_gap and area_gap. A weight change should
    produce a trade, not dominance.
  * n=80 is non-monotone and jagged in HW: 2 -> 1.1404, 4 -> 1.1845, 8 -> 1.1634.
    HW=4 is worse than both its neighbours.

PROTOCOL. L296/L298 measured that min-of-N is biased downward with N, so cells with
different repeat counts cannot be compared on their minimum. Every cell here gets
the SAME seed count, and the full distribution is printed, not just the best.

Usage: python l340_seed.py <n> <HW-multiples> <iters> <n-seeds>
   e.g. python l340_seed.py 80 2,4 2000000 5
"""
import statistics
import sys

from l340_run import load, run

if __name__ == "__main__":
    n = int(sys.argv[1])
    MULTS = [float(x) for x in sys.argv[2].split(",")]
    it = int(sys.argv[3])
    S = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    *_, arL, hpL, _nb = load(n)
    star = arL / hpL
    print("== L340 seed spread: n=%d  %d iterations  %d seeds/cell (equal N) =="
          % (n, it, S))
    print()
    cells = {}
    for mu in MULTS:
        print("   HW = %.2f * HW*" % mu)
        print("   %-6s %9s %9s %9s %8s"
              % ("seed", "hpwl_gap", "area_gap", "quality", "time"))
        qs = []
        for s in range(1, S + 1):
            r = run(n, star * mu, it, seed=s)
            q = 1 + 0.5 * (r["hg"] + r["ag"])
            qs.append(q)
            print("   %-6d %9.4f %9.4f %9.4f %7.1fs"
                  % (s, r["hg"], r["ag"], q, r["dt"]))
        cells[mu] = qs
        print("   -> min %.4f  median %.4f  max %.4f  SPREAD %.4f  sd %.4f"
              % (min(qs), statistics.median(qs), max(qs), max(qs) - min(qs),
                 statistics.stdev(qs) if len(qs) > 1 else 0.0))
        print()
    if len(MULTS) > 1:
        print("   VERDICT -- is the between-HW gap readable against within-cell spread?")
        ms = sorted(cells)
        worst = max(max(cells[m]) - min(cells[m]) for m in ms)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                d = statistics.median(cells[a]) - statistics.median(cells[b])
                print("   HW %.2f vs %.2f: median gap %+.4f   worst within-cell "
                      "spread %.4f   %s"
                      % (a, b, d, worst,
                         "READABLE" if abs(d) > worst else "NOT READABLE (noise)"))
