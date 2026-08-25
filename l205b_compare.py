"""L205b - contended vs uncontended imbalance, and what route A costs either way.

The decision hangs on one number and one threshold:

    route A wins case n  iff  D_max/D_mean  >  1.44 * k / cores  = 1.530

L205 measured the ratio with all 51 profiles running at once on 32 logical
cores (1.6x oversubscribed) and got a median of 1.377 -- below the threshold,
so route A loses. But that setup COMPRESSES the ratio: short profiles finish
early and hand their share to the long ones, pulling D_max down toward D_mean.
The grader is barely oversubscribed (51 on 48 = 1.06x), so its ratio is at
least as large as the measured one, and the verdict only needs the true median
to rise 11% (1.377 -> 1.530) to flip.

This file puts the two side by side. The uncontended column is measured with
ICCAD_PROF_SEQ=1 (one profile at a time), which has no scheduler in it at all,
so it is the ratio the workload actually has -- and the upper end of what the
grader can show.

  <python> l205b_compare.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
CORES = 48
WORK = 1.44
EXPECT = 51
BETA_CROSS = 1.052        # L204: the ra at which we stop beating beta


def load(fn):
    per = defaultdict(list)
    f = DIR / fn
    if not f.exists():
        return None
    for line in f.read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) == 3:
            try:
                per[int(p[0])].append(float(p[2]))
            except ValueError:
                pass
    return per


def stats(per, rows):
    out = {}
    for n, d in per.items():
        if len(d) < 2:
            continue
        k = len(d)
        mean, mx = sum(d) / k, max(d)
        Wt = mean * k
        plain = max(mx, Wt / CORES)
        out[n] = {"k": k, "mean": mean, "max": mx,
                  "ratio": mx / mean if mean else 0.0,
                  "thr": WORK * k / CORES,
                  "ra": (WORK * Wt / CORES) / plain if plain else 1.0,
                  "w": rows[n]["w"] if n in rows else 0.0}
    return out


def summarise(tag, S):
    rs = sorted(v["ratio"] for v in S.values())
    q = lambda p: rs[min(len(rs) - 1, int(p * len(rs)))]      # noqa: E731
    tot = sum(v["w"] for v in S.values()) or 1.0
    ra = sum(v["w"] * v["ra"] for v in S.values()) / tot
    wins = sum(v["w"] for v in S.values() if v["ra"] < 1.0)
    nwin = sum(1 for v in S.values() if v["ra"] < 1.0)
    print("{:<14}{:>7.3f}{:>8.3f}{:>8.3f}{:>8.3f}{:>9}{:>10.1f}%{:>10.4f}"
          .format(tag, rs[0], q(.25), q(.5), q(.75),
                  "{}/{}".format(nwin, len(S)), 100 * wins / tot, ra))
    return ra


def main():
    rows = {r["n"]: r for r in M.rows_new()}
    par = load("l205_prof_r1.txt")
    par2 = load("l205_prof_r2.txt")
    seq = load("l205b_prof_seq.txt")
    if par is None:
        print("no parallel run found (l205_prof_r1.txt)")
        return 1

    for nm, p in (("parallel r1", par), ("parallel r2", par2), ("sequential", seq)):
        if p is None:
            continue
        short = {n: len(v) for n, v in p.items() if len(v) != EXPECT}
        print("{:<14} {} block counts, {} records{}"
              .format(nm, len(p), sum(len(v) for v in p.values()),
                      "" if not short else
                      "   !! INCOMPLETE: {}".format(sorted(short.items())[:6])))
    print()
    print("{:<14}{:>7}{:>8}{:>8}{:>8}{:>9}{:>11}{:>10}"
          .format("run", "min", "p25", "median", "p75", "A wins", "wtd share",
                  "mean ra"))
    print("-" * 76)
    ras = {}
    for nm, p in (("parallel r1", par), ("parallel r2", par2),
                  ("sequential", seq)):
        if p is None:
            continue
        ras[nm] = summarise(nm, stats(p, rows))
    print("-" * 76)
    print("threshold on the ratio: 1.44 * 51/48 = {:.3f}"
          .format(WORK * EXPECT / CORES))
    print()

    if seq is None:
        print("The sequential run has not landed yet -- it is the one that")
        print("removes the compression bias, so the verdict waits for it.")
        return 0

    Sp, Ss = stats(par, rows), stats(seq, rows)
    common = sorted(set(Sp) & set(Ss))
    lift = [Ss[n]["ratio"] / Sp[n]["ratio"] for n in common if Sp[n]["ratio"]]
    lift.sort()
    print("COMPRESSION, measured directly: sequential ratio / parallel ratio")
    print("   median {:.3f}   p25 {:.3f}   p75 {:.3f}   over {} block counts"
          .format(lift[len(lift) // 2], lift[len(lift) // 4],
                  lift[3 * len(lift) // 4], len(lift)))
    print("   (>1 means the parallel run WAS compressing the imbalance, i.e.")
    print("    the parallel verdict understated route A)")
    print()
    ra_seq = ras.get("sequential", 1.0)
    print("VERDICT")
    print("   uncontended weighted mean ra = {:.4f}".format(ra_seq))
    print("   L204: we stop beating beta at ra = {:.3f}".format(BETA_CROSS))
    if ra_seq > BETA_CROSS:
        print("   => route A ON is WORSE THAN BETA on this model. Turn it off:")
        print("      _route_a_default() -> 0, which lands the package at the")
        print("      certain 0.91491 / rank 4 with the LP still on.")
    elif ra_seq > 1.0:
        print("   => route A COSTS wall but stays inside the beta margin.")
        print("      It is a pure loss against the certain route-A-OFF")
        print("      configuration (0.91491), which dominates it.")
    else:
        print("   => route A SAVES wall even uncontended. The bet is live and")
        print("      the ledger's projection direction is supported.")
    print()
    print("   Residual bias, one-sided and AGAINST route A: this credits it")
    print("   with perfect packing of the frame queue, which its own")
    print("   submission rule does not give. True ra is >= the number above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
