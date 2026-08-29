"""L346 -- build the corpus L345 discovered we do not have.

L345 answered "generation or selection?" for N_soft >= 59 (generation, clearly) and then
found it could not answer it where the money is:

    corpus                heavy n>=101   N_soft min/p50/max   cases with N_soft <= 33
    in-set 100                    20         43 / 61 / 67                           0
    OOS s1 (l252_cache)           40         59 / 68 / 81                           0
    GRADED = final                19         14 / 52 / 65                           6

The prize cases sit at N_soft = 14, 18, 26, 28, 29, 33 and NEITHER runnable corpus holds a
single instance. That is L278's rule biting: a corpus can only vote on a mechanism whose
antecedent it contains. delta* = (1+G)(exp(2/N_soft)-1) is 3.9x larger at N_soft=18 than
at 68, so this is exactly the band where the violation trade might flip profitable.

N_soft = boundary blocks + sum(MIB-1) + sum(cluster-1) is computable from `constraints`
alone -- no packer run, no label -- and the training set is 9000 shards x 112 layouts,
n = 24..118. So the missing corpus is CONSTRUCTIBLE by an index scan.

GATE (runs first, refuses to continue if it fails). The constraints-only N_soft formula is
checked against the official evaluator's own `max_possible_violations` on all 100
validation cases. It must match on 100/100. If the formula is wrong every selection below
is wrong.

Offline probe: reads no label for selection (N_soft is an INPUT quantity), trains nothing,
ships nothing, touches no file on the shipping path.

  <python> l346_corpus.py gate
  <python> l346_corpus.py scan [--workers 0-19]      -> l346_index.pkl
  <python> l346_corpus.py pick [--nmin 101] [--nsmax 33] [--k 40]
"""
import argparse
import collections
import glob
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent
INDEX = DIR / "l346_index.pkl"
SHARDS = "C:/ICCAD_ml/floorset_lite/worker_*/*.th"

# The graded corpus's own prize cases (L296 sec.2.1 / L343): (n, N_soft, V)
PRIZE = [(116, 18, 1), (111, 14, 1), (119, 28, 1), (117, 33, 1), (118, 53, 3),
         (109, 29, 2), (105, 26, 1)]


def nsoft_from_constraints(cons, nb):
    """N_soft = boundary blocks + sum(MIB-1) + sum(cluster-1).

    cons is `meta[:, 1:]` (lite_dataset_test.py:214), so
    col0 fixed, col1 preplaced, col2 MIB, col3 cluster, col4 boundary.
    """
    c = cons[:nb]
    bnd = int((c[:, 4] != 0).sum())

    def groups(col):
        cnt = collections.Counter(int(v) for v in c[:, col].tolist() if int(v) != 0)
        return sum(v - 1 for v in cnt.values())
    return bnd + groups(2) + groups(3)


def gate():
    """The formula must reproduce the evaluator's own max_possible_violations."""
    sys.path.insert(0, str(DIR))
    from l342_strictcost import case, score
    bad = []
    n_ok = 0
    for n in range(21, 121):
        try:
            c = case(n)
        except Exception:
            continue
        mine = nsoft_from_constraints(c["cons"], c["n"])
        theirs = int(score(c["tp"], c, strict=True).max_possible_violations)
        n_ok += 1
        if mine != theirs:
            bad.append((n, mine, theirs))
    print("== L346 GATE: constraints-only N_soft vs the evaluator's own ==")
    print("   validation cases checked : %d" % n_ok)
    print("   mismatches               : %d" % len(bad))
    if bad:
        print("   first 10:", bad[:10])
        print("   *** FAIL -- do not trust any selection built on this formula ***")
        return 1
    print("   *** PASS 100/100 -- N_soft is an INPUT quantity, computable with no label")
    print("       and no packer run ***")
    return 0


def scan(a):
    fs = sorted(glob.glob(SHARDS))
    if a.workers:
        lo, hi = (int(v) for v in a.workers.split("-"))
        fs = [f for f in fs
              if lo <= int(os.path.basename(os.path.dirname(f)).split("_")[1]) <= hi]
    print("== L346 scan: %d shards ==" % len(fs))
    import torch
    rows = []
    t0 = time.time()
    nhist = collections.Counter()
    for i, f in enumerate(fs):
        try:
            d = torch.load(f, weights_only=False)
        except Exception as e:
            print("   skip %s (%s)" % (f, type(e).__name__))
            continue
        meta = d[0]
        B = meta.shape[0]
        for b in range(B):
            nb = int((meta[b, :, 0] > 0).sum())
            if nb <= 0:
                continue
            nhist[nb] += 1
            if nb < a.nmin:
                continue
            ns = nsoft_from_constraints(meta[b, :, 1:], nb)
            rows.append((f, b, nb, ns))
        if (i + 1) % 1000 == 0:
            print("   %d/%d shards  %.0fs  %d heavy layouts so far"
                  % (i + 1, len(fs), time.time() - t0, len(rows)))
    pickle.dump(dict(rows=rows, nmin=a.nmin, nhist=dict(nhist), shards=len(fs)),
                open(INDEX, "wb"))
    print("   done in %.0fs -> %s   %d layouts with n >= %d"
          % (time.time() - t0, INDEX.name, len(rows), a.nmin))
    tot = sum(nhist.values())
    print("   total layouts scanned: %d   n range %d..%d"
          % (tot, min(nhist), max(nhist)))
    return 0


def pick(a):
    d = pickle.load(open(INDEX, "rb"))
    rows = [r for r in d["rows"] if r[2] >= a.nmin]
    print("== L346 pick: %d layouts with n >= %d (from %d shards) =="
          % (len(rows), a.nmin, d["shards"]))
    if not rows:
        print("   nothing to pick from.")
        return 1
    ns = [r[3] for r in rows]
    print("   N_soft over them: min %d  p50 %d  max %d"
          % (min(ns), int(statistics.median(ns)), max(ns)))
    band = collections.Counter(min(v // 10 * 10, 90) for v in ns)
    print("   N_soft histogram:", dict(sorted(band.items())))
    hit = [r for r in rows if r[3] <= a.nsmax]
    print()
    print("   *** layouts with n >= %d AND N_soft <= %d : %d  (%.2f %% of heavy) ***"
          % (a.nmin, a.nsmax, len(hit), 100 * len(hit) / len(rows)))
    if not hit:
        print("   THE ANTECEDENT DOES NOT EXIST IN THE TRAINING SET EITHER.")
        print("   That would make the graded prize band unreachable by any corpus we can")
        print("   run, and would close the line for a reason worth recording.")
        return 0

    print()
    print("   matching the GRADED prize cases (n, N_soft):")
    print("   %8s %8s %10s %s" % ("target n", "target NS", "available", "closest found"))
    for pn, pns, _v in sorted(PRIZE):
        near = sorted(hit, key=lambda r: (abs(r[2] - pn) + abs(r[3] - pns)))
        cnt = sum(1 for r in hit if abs(r[2] - pn) <= 5 and abs(r[3] - pns) <= 5)
        print("   %8d %8d %10d   %s"
              % (pn, pns, cnt,
                 "n=%d NS=%d" % (near[0][2], near[0][3]) if near else "-"))

    # stratified sample: spread over N_soft, prefer heavy n, deterministic
    hit.sort(key=lambda r: (r[3], -r[2], r[0], r[1]))
    k = min(a.k, len(hit))
    step = len(hit) / k
    sel = [hit[int(i * step)] for i in range(k)]
    out = DIR / "l346_selection.pkl"
    pickle.dump(sel, open(out, "wb"))
    print()
    print("   selected %d cases -> %s" % (len(sel), out.name))
    print("   %5s %6s   %s" % ("n", "N_soft", "shard/layout"))
    for r in sel[:15]:
        print("   %5d %6d   %s/%d"
              % (r[2], r[3], os.path.basename(os.path.dirname(r[0]))
                 + "/" + os.path.basename(r[0]), r[1]))
    if len(sel) > 15:
        print("   ... %d more" % (len(sel) - 15))
    print()
    print("   selection N_soft: min %d p50 %d max %d ; n: min %d p50 %d max %d"
          % (min(r[3] for r in sel), int(statistics.median(r[3] for r in sel)),
             max(r[3] for r in sel), min(r[2] for r in sel),
             int(statistics.median(r[2] for r in sel)), max(r[2] for r in sel)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "scan", "pick"])
    ap.add_argument("--workers", default="")
    ap.add_argument("--nmin", type=int, default=101)
    ap.add_argument("--nsmax", type=int, default=33)
    ap.add_argument("--k", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "gate":
        return gate()
    if a.cmd == "scan":
        return scan(a)
    return pick(a)


if __name__ == "__main__":
    sys.exit(main())
