"""L205 - the number that decides route A, and it is NOT a wall time.

The ledger closed route A as "a permanent bet, cannot be measured here"
(HANDOFF 2026-08-25 sec.5) after trying to measure its WALL on this box. That
is genuinely impossible: wall depends on the machine. But the thing route A's
payoff turns on is a property of the WORKLOAD, and workload ratios transport
off this box exactly the way `t_ours/t_m73` does.

THE MECHANISM, read out of the code rather than assumed:

  * solve() submits ALL profiles at once --
    `ThreadPoolExecutor(max_workers=len(profiles))`, 51 profiles in the
    48-core configuration. The grader has 48 cores. So during the profile
    phase the grader is SATURATED (51 runnable on 48), and route A's premise
    -- "route A only converts IDLE cores into wall" -- has no idle cores.
  * `_route_a_cores()` deliberately ignores ICCAD_ADAPTIVE_CORES and sizes the
    frame queue from the REAL core count, so route A cannot oversubscribe. It
    is not concurrency that costs; it is WORK. L110 measured route A doing
    1.44x the work of the plain path (2.03x before the two-phase fix).

On a saturated box the two makespans are

    plain    ~ max( D_max , W/cores )        one subprocess per profile,
                                             frames run SEQUENTIALLY inside
    route A  >= 1.44 * W/cores               same frames, one shared queue

with W = k*D_mean over the k profiles the case actually ran, so route A wins a
case iff

    D_max / D_mean  >  1.44 * k / cores

D_max/D_mean is dimensionless. Every profile on this box is slowed by the same
oversubscription factor, so the ratio survives the transport that the wall does
not. "Route A cannot be measured here" was true of its WALL and false of its
WIN CONDITION.

  <python> l205_imbalance.py <stats-file> [...]

Input is the file ICCAD_PROFILE_TIMING points at in `optimizer_l205probe.py`:
one `<block_count> <profile_idx> <seconds>` line per profile, written under a
lock. (The first version printed to stderr from 51 threads and lost ~10% of the
records to interleaving -- 5100 emitted, 4588 parseable. That is not a cosmetic
loss: it biases D_max DOWN and k DOWN, which move the verdict in OPPOSITE
directions, so a corrupted run cannot be repaired after the fact. Hence the
completeness assertion below, which is the whole reason this file re-reads a
count it could have assumed.)
"""
import sys
from collections import defaultdict
from pathlib import Path

import l172_depthmap as M

CORES = 48
WORK = 1.44                      # L110's measured route-A work multiplier
EXPECT = 51                      # profiles in the 48-core configuration


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    per = defaultdict(list)
    for f in argv:
        for line in Path(f).read_text(errors="replace").splitlines():
            p = line.split()
            if len(p) == 3:
                try:
                    per[int(p[0])].append(float(p[2]))
                except ValueError:
                    pass
    if not per:
        print("no records found -- was ICCAD_PROFILE_TIMING set to this path?")
        return 1

    short = {n: len(v) for n, v in per.items() if len(v) != EXPECT}
    print("=" * 78)
    print("COMPLETENESS: {} block counts, {} records, expected {} each"
          .format(len(per), sum(len(v) for v in per.values()), EXPECT))
    if short:
        print("  !! {} block counts do not have exactly {} records: {}"
              .format(len(short), EXPECT, sorted(short.items())[:8]))
        print("  !! D_max is biased DOWN and k DOWN by missing records, and")
        print("     those bias the verdict in OPPOSITE directions. Do not read")
        print("     the table below until this line says every count is full.")
    else:
        print("  every block count has all {} profiles -- no records lost."
              .format(EXPECT))

    rows = {r["n"]: r for r in M.rows_new()}
    print("=" * 78)
    print("PROFILE-DURATION IMBALANCE   (route A wins case n iff ratio > "
          "1.44*k/{})".format(CORES))
    print("=" * 78)
    print("{:>6}{:>5}{:>9}{:>9}{:>8}{:>8}{:>9}{:>9}"
          .format("n", "k", "D_mean", "D_max", "ratio", "thr", "verdict", "ra"))
    detail = []
    for n in sorted(per):
        d = per[n]
        if len(d) < 2:
            continue
        k = len(d)
        mean, mx = sum(d) / k, max(d)
        ratio = mx / mean if mean else 0.0
        thr = WORK * k / CORES
        Wt = mean * k
        plain = max(mx, Wt / CORES)
        ra = (WORK * Wt / CORES) / plain if plain else 1.0
        detail.append((n, k, mean, mx, ratio, thr, ra,
                       rows[n]["w"] if n in rows else 0.0))
    for n, k, mean, mx, ratio, thr, ra, w in detail:
        if n % 9 == 0 or ratio > thr:
            print("{:>6}{:>5}{:>9.3f}{:>9.3f}{:>8.3f}{:>8.3f}{:>9}{:>9.3f}"
                  .format(n, k, mean, mx, ratio, thr,
                          "routeA" if ratio > thr else "plain", ra))
    rs = sorted(r for _n, _k, _m, _x, r, _t, _r, _w in detail)
    q = lambda p: rs[min(len(rs) - 1, int(p * len(rs)))]      # noqa: E731
    wins = sum(w for *_x, ra, w in detail if ra < 1.0)
    tot = sum(w for *_x, _ra, w in detail) or 1.0
    print("-" * 78)
    print("cases {}   ratio  min {:.3f}  p25 {:.3f}  median {:.3f}  p75 {:.3f}"
          "  max {:.3f}".format(len(rs), rs[0], q(.25), q(.5), q(.75), rs[-1]))
    print("cases route A would speed up: {} of {}   WEIGHTED share {:.1f}%"
          .format(sum(1 for *_x, ra, _w in detail if ra < 1.0), len(detail),
                  100 * wins / tot))
    num = sum(w * ra for *_x, ra, w in detail)
    print("WEIGHTED MEAN ra = {:.4f}   (<1 route A saves wall, >1 it costs)"
          .format(num / tot))
    print()
    print("TWO BIASES, pointing in OPPOSITE directions -- a verdict is only")
    print("safe if it survives both:")
    print("  (a) AGAINST route A: this credits it with PERFECT packing of the")
    print("      frame queue, which _run_profile_route_a's submission rule")
    print("      (stop once the prefix holds max_trials successes) does not")
    print("      give. Real makespan is worse than 1.44*W/cores, so real ra is")
    print("      HIGHER than printed.")
    print("  (b) FOR route A: this box runs 51 profiles on 32 logical cores,")
    print("      1.6x oversubscribed, where the grader runs 51 on 48 (1.06x).")
    print("      Under heavy oversubscription short profiles finish early and")
    print("      hand their share to the long ones, so D_max/D_mean measured")
    print("      HERE is COMPRESSED. The grader's imbalance is >= this, which")
    print("      makes real ra LOWER than printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
