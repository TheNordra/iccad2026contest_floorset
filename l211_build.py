"""L211 - build and price a per-block-count pool drop table.

WHY THIS IS NOT ALREADY DONE, AND WHY IT MIGHT STILL BE WORTH SOMETHING.

The 48-core wall is max-setter bound: uncontended, D_max = 1.492*D_mean while
W/48 = 1.0625*D_mean, so the case wall is set by the slowest single profile and
not by total work. Dropping the slowest profiles is therefore the only runtime
lever left after L155/L156 closed the LP ones.

M41 and M42 already took the free part of it -- they drop max-setters that the
proxy never selects, derived strictly selection-preserving. So the 51 profiles
that remain are, by construction, the ones that sometimes win. Cutting further
必然 costs quality, and the ledger has twice measured that fixed drop sets kill
winners out of sample (L138/L139: 12 of 22 held-out winners removed).

So the question this file answers is narrow and quantitative: **the wall lever
is worth up to +1.7pp of RF at k=12 -- is the quality it costs less than that?**

Two things it does NOT assume:

  * that the slowest profile is identifiable. It is not, from one run: the
    argmax agrees across three runs in only 6/100 block counts, because the top
    of the distribution is a PLATEAU (median 2 profiles within 2% of D_max, 7
    within 10%) and run-to-run wall noise swamps the gaps. The table is built
    from the MEAN of the runs, and `l205b_compare`-style cross-validation showed
    a fitted table captures only 35-49% of the oracle saving.
  * that the profile-phase saving is the case saving. It is not: M47 measured
    the serial proxy tail at 29% of the n>100 wall, so a profile-phase cut of x%
    is a case-wall cut of ~0.71x%.

  <python> l211_build.py <k> [out.json]
"""
import collections
import json
import os
import sys
from pathlib import Path

DIR = Path(__file__).parent
RUNS = ["l205_prof_r1.txt", "l205_prof_r2.txt", "l205b_prof_seq.txt"]
CORES = 48
PHI = 0.71                # M47: the share of case wall route-A/pool work owns


def load(fn):
    per = collections.defaultdict(dict)
    f = DIR / fn
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) == 3:
            per[int(p[0])][int(p[1])] = float(p[2])
    return per


def main(argv):
    k = int(argv[0]) if argv else 8
    out = argv[1] if len(argv) > 1 else "l211_drop_k{}.json".format(k)

    runs = [r for r in (load(f) for f in RUNS) if r]
    if not runs:
        print("no duration runs found")
        return 1
    ns = sorted(set.intersection(*[set(r) for r in runs]))

    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import optimizer_constructive as O

    table, sizes = {}, []
    for n in ns:
        pos = sorted(set.intersection(*[set(r[n]) for r in runs]))
        mean = {i: sum(r[n][i] for r in runs) / len(runs) for i in pos}
        # position in the selected list -> ORIGINAL _PROFILES index, which is
        # what _pool_indices returns and what a shipped table would have to be
        # keyed on. Positions are not stable against any future pool change.
        idx = O._pool_indices(n)
        if len(idx) != len(pos):
            print("  !! n={}: pool is {} but {} positions were measured -- "
                  "the runs do not match this pool".format(n, len(idx), len(pos)))
            return 1
        slow = sorted(mean, key=mean.get, reverse=True)[:k]
        table[str(n)] = sorted(idx[i] for i in slow)
        sizes.append(len(idx) - k)

    (DIR / out).write_text(json.dumps(table, indent=0, sort_keys=True))
    glob = collections.Counter(i for v in table.values() for i in v)
    print("wrote {}  ({} block counts, drop {} each, pool {} -> {})"
          .format(out, len(table), k, sizes[0] + k, sizes[0]))
    print("distinct profiles ever dropped: {} of {}"
          .format(len(glob), len(O._pool_indices(ns[0]))))
    print("most-dropped: {}".format(
        ", ".join("#{}x{}".format(i, c) for i, c in glob.most_common(8))))

    # projected wall saving, measured on the run NOT used to rank (the seq one
    # is in the mean, so this is optimistic -- the honest number is the 35-49%
    # capture rate from the two-parallel-fit cross-validation)
    base = tot = 0.0
    for n in ns:
        d = runs[-1][n]
        keep = [t for i, t in d.items() if O._pool_indices(n)[i] not in
                set(table[str(n)])]
        base += max(max(d.values()), sum(d.values()) / CORES)
        tot += max(max(keep), sum(keep) / CORES)
    w = 100 * (tot / base - 1)
    print("profile-phase wall {:+.2f}%  ->  case wall {:+.2f}% (x{:.2f}) "
          "->  RF {:+.3f}pp".format(w, w * PHI, PHI, -0.3 * w * PHI))
    print("   ^ unweighted and fitted-on-this-run; treat as the optimistic end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
