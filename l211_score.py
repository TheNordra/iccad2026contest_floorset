"""L211 score - does the wall the pool drop buys cost less quality than it earns?

The arithmetic that makes this worth asking: the 48-core wall is max-setter
bound, dropping the k slowest profiles per block count cuts the profile phase
4-9%, and M47's 29% serial tail turns that into 3-7% of case wall, i.e.
+0.9 to +2.0pp of RF (score enters as t^0.3). Rank 3 needs +1.69pp over the
current +1.260% NET, so the lever is arithmetically in range.

The counter-argument the ledger supplies, and which this file tests rather than
assumes: M41/M42 already dropped every max-setter that never wins, so what is
left necessarily costs quality; and L138/L139 measured fixed drop sets removing
12 of 22 held-out winners.

Two things reported separately, because they fail differently:

  K0     the probe with NO drop table must reproduce the shipped arm
         bit-for-bit. If it does not, the probe is what is being measured and
         every row below is meaningless.
  Kk     quality delta vs the shipped arm, against the RF the same table buys.

!! IN-SET ONLY. Every one of this project's offline advantages shrank or
reversed out of sample, and pool pruning is the specific mechanism that did so
twice. A positive row here is a licence to run the two OOS 240-case samples,
not a result.

  <python> l211_score.py
"""
import json
import math
from pathlib import Path

DIR = Path(__file__).parent
PHI = 0.71
CORES = 48
BASE = "results_L209_det1.json"


def load(fn):
    f = DIR / fn
    if not f.exists():
        return None
    return {r["test_id"]: r for r in json.load(open(f))["test_results"]}


def wq(d, ids):
    w = lambda i: math.exp(d[i]["block_count"] / 12.0)         # noqa: E731
    return sum(w(i) * d[i]["cost"] for i in ids) / sum(w(i) for i in ids)


def wall_saving(k):
    """profile-phase saving of table k, on the sequential (uncontended) run."""
    import collections
    per = collections.defaultdict(dict)
    f = DIR / "l205b_prof_seq.txt"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) == 3:
            per[int(p[0])][int(p[1])] = float(p[2])
    t = DIR / "l211_drop_k{}.json".format(k)
    if not t.exists():
        return None
    tab = {int(a): set(b) for a, b in json.loads(t.read_text()).items()}
    import os
    import sys
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import optimizer_constructive as O
    base = tot = 0.0
    for n, d in per.items():
        idx = O._pool_indices(n)
        keep = [v for i, v in d.items() if idx[i] not in tab.get(n, set())]
        if not keep:
            continue
        base += max(max(d.values()), sum(d.values()) / CORES)
        tot += max(max(keep), sum(keep) / CORES)
    return 100 * (tot / base - 1)


def main():
    base = load(BASE)
    if not base:
        print("no baseline {} -- run the L209 gates first".format(BASE))
        return 1
    k0 = load("results_L211_k0.json")
    print("=" * 78)
    if k0:
        ids = sorted(set(base) & set(k0))
        c = sum(1 for i in ids if base[i]["cost"] == k0[i]["cost"])
        p = sum(1 for i in ids if base[i].get("positions") == k0[i].get("positions"))
        ok = c == len(ids) and p == len(ids)
        print("K0 PROBE INERT: cost {}/{}  positions {}/{}   {}"
              .format(c, len(ids), p, len(ids), "PASS" if ok else "FAIL"))
        if not ok:
            print("  !! the probe itself changes the result -- nothing below is "
                  "attributable to the drop table.")
            return 1
    else:
        print("K0 PROBE INERT: NOT RUN -- rows below are unattributed")
    print("=" * 78)
    print("{:>4}{:>7}{:>12}{:>10}{:>12}{:>12}{:>10}"
          .format("k", "pool", "quality", "moved", "prof wall", "RF", "NET"))
    print("-" * 78)
    for k in (3, 8, 12):
        d = load("results_L211_k{}.json".format(k))
        if not d:
            continue
        ids = sorted(set(base) & set(d))
        q = 100 * (wq(base, ids) - wq(d, ids)) / wq(base, ids)
        mv = sum(1 for i in ids if d[i]["cost"] != base[i]["cost"])
        ws = sum(1 for i in ids if d[i]["cost"] > base[i]["cost"] + 1e-12)
        w = wall_saving(k)
        rf = -0.3 * w * PHI if w is not None else float("nan")
        print("{:>4}{:>7}{:>+11.4f}%{:>7}/{:<3}{:>+11.2f}%{:>+11.3f}{:>+9.3f}"
              .format(k, 51 - k, q, mv, ws, w, rf, q + rf))
    print("-" * 78)
    print("quality: + is BETTER. moved = a/b, b of them worse.")
    print("prof wall: profile-phase only. RF applies M47's x{:.2f} tail and the"
          .format(PHI))
    print("           t^0.3 exponent. NET = quality + RF, both in points of score.")
    print()
    print("Current NET vs beta is +1.260%; rank 3 needs +2.95% (graded 0.89933).")
    print("!! IN-SET ONLY -- pool pruning is the exact mechanism that reversed")
    print("   out of sample twice (L138/L139). A positive NET here licenses the")
    print("   two OOS 240-case samples, it does not conclude anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
