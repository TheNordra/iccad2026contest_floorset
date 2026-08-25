"""L231 - is a UNIFORM REFINE band the right object at all?

The 48-core wall is max-setter bound: wall(n) = max(max_p d_p, sum_p d_p / 48).
_M49_REFINE_BAND cuts REFINE on ALL 51 profiles at a block count, but only the
profiles at or near the top of the duration distribution can move that wall.
Every other profile pays quality for zero wall.

This reads the L219 sweep's per-profile durations (l219_prof_r{4,3,2,1}.txt,
100 cases x 51 profiles each) and asks:

  1. how far below the wall does the pool sit -- i.e. how many profiles could
     keep (or raise) their REFINE budget for free?
  2. what does the wall actually do if REFINE=2 is applied ONLY to profiles
     near the top, and the rest are left at 6 (or raised)?

Simulation only -- no shipping file is touched.
"""
import collections
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent
CORES = 48


def load(fn):
    per = collections.defaultdict(dict)
    p = DIR / fn
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        s = line.split()
        if len(s) == 3:
            per[int(s[0])][int(s[1])] = float(s[2])
    return per


def wall(d):
    v = list(d.values())
    return max(max(v), sum(v) / CORES)


def main():
    R = {r: load("l219_prof_r{}.txt".format(r)) for r in (4, 3, 2, 1)}
    R = {k: v for k, v in R.items() if v}
    print("arms loaded:", sorted(R))
    ns = sorted(n for n in R[4] if n > 100)
    print("heavy band block counts:", len(ns))

    print()
    print("=== 1. how far below the wall does the pool sit? (REFINE=4 arm) ===")
    print("{:>6}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}"
          .format("n", "pool", "wall", "d_max", "sum/48", "<70%w", "<50%w"))
    fr = []
    for n in ns[:6] + ns[-4:]:
        d = R[4][n]
        w = wall(d)
        v = sorted(d.values(), reverse=True)
        print("{:>6}{:>8}{:>10.3f}{:>10.3f}{:>10.3f}{:>10}{:>10}"
              .format(n, len(d), w, v[0], sum(v) / CORES,
                      sum(1 for x in v if x < 0.70 * w),
                      sum(1 for x in v if x < 0.50 * w)))
    for n in ns:
        d = R[4][n]
        w = wall(d)
        fr.append(sum(1 for x in d.values() if x < 0.70 * w) / len(d))
    print("  across the whole heavy band, median share of profiles below 70% of"
          " the wall: {:.0%}".format(st.median(fr)))

    print()
    print("=== 2. per-profile REFINE sensitivity, REFINE 4 -> 2 ===")
    rat = collections.defaultdict(list)
    for n in ns:
        for p in R[4][n]:
            if p in R[2][n] and R[4][n][p] > 1e-6:
                rat[p].append(R[2][n][p] / R[4][n][p])
    med = {p: st.median(v) for p, v in rat.items()}
    allv = sorted(med.values())
    print("  per-profile median ratio: p10 {:.3f}  p50 {:.3f}  p90 {:.3f}"
          .format(allv[len(allv) // 10], st.median(allv),
                  allv[-max(1, len(allv) // 10)]))
    print("  profiles that barely move (ratio > 0.95): {} of {}"
          .format(sum(1 for v in med.values() if v > 0.95), len(med)))

    print()
    print("=== 3. SELECTIVE cut: REFINE=2 only on the top-k, rest stay at 4 ===")
    print("  the wall is a max, so cutting the tail cannot help it; the")
    print("  question is how small k can be before the wall stops falling.")
    print("{:>6}{:>12}{:>12}{:>12}{:>12}{:>12}"
          .format("k", "wall ratio", "vs all-51", "profiles cut", "n>100",
                  "recovered"))
    base = {n: wall(R[4][n]) for n in ns}
    full = {n: wall(R[2][n]) for n in ns}
    fullcut = st.median(full[n] / base[n] for n in ns)
    for k in (1, 2, 3, 5, 8, 12, 20, 30, 51):
        rr = []
        for n in ns:
            d4, d2 = R[4][n], R[2][n]
            order = sorted(d4, key=lambda p: -d4[p])[:k]
            mix = {p: (d2[p] if p in order and p in d2 else d4[p]) for p in d4}
            rr.append(wall(mix) / base[n])
        m = st.median(rr)
        print("{:>6}{:>12.4f}{:>12.4f}{:>12}{:>12}{:>11.0%}"
              .format(k, m, fullcut, k, len(ns),
                      (1 - m) / (1 - fullcut) if fullcut < 1 else 0))
    print()
    print("  'recovered' = share of the FULL uniform cut's wall saving that a")
    print("  top-k-only cut already delivers. If k=8 recovers ~100%, then 43 of")
    print("  the 51 profiles are paying quality for nothing.")

    print()
    print("=== 4. the same question upward: raise the tail to REFINE=1..? ===")
    print("  (uses the r1 arm as the cheap end; the expensive end needs a new")
    print("   arm, so this only bounds the SHAPE of the trade.)")
    if 1 in R:
        rr = []
        for n in ns:
            d2, d1 = R[2][n], R[1][n]
            w = wall(d2)
            order = sorted(d2, key=lambda p: -d2[p])[:8]
            mix = {p: (d1[p] if p in order and p in d1 else d2[p]) for p in d2}
            rr.append(wall(mix) / w)
        print("  top-8 at REFINE=1, rest at 2: wall x{:.4f} (median)"
              .format(st.median(rr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
