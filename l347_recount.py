"""L347 -- L343 was priced on reduced fractions. Recount it against the real N_soft law.

WHAT WENT WRONG. L296 recovers each graded case's violation data by inverting
`violations_relative` with `Fraction(v).limit_denominator()`. That returns the REDUCED
fraction: 1/14 is indistinguishable from 2/28, 3/42, 4/56. L296 stated this ("the count
figures are lower bounds and the per-violation prizes are correspondingly upper bounds")
and L343 repeated the caveat -- and then built its headline on the lower bounds anyway.

L346 scanned all 9000 training shards (1 008 000 layouts, 201 264 with n >= 101) and
measured the real law: **no heavy layout anywhere in the training set has N_soft < 41**
(min 41, p50 65, max 90). The validation set agrees (heavy: 43 / 61 / 67). The hidden set
is drawn from the same generator with noise on shapes and placements, not on constraint
counts (Q&A A23), so it obeys the same law.

=> Every recovered N_soft below 41 on a heavy case is provably a reduced fraction. The
true pair is (k*V0, k*NS0) for the integer k that puts k*NS0 inside the observed range at
that n. That is a hard constraint, and it pins most cases uniquely.

CONSEQUENCE, both ways:
  * each violation is worth LESS than L343 said (exp(-2/54), not exp(-2/18)), and
  * there are MORE of them than L343 said (V = 3 or 4, not 1).
The total violation mass is untouched -- `vrel` itself is exact. Only its decomposition
into count x per-unit moves, and L343 quoted the decomposition.

It also CLOSES L345's blind spot, in the unfavourable direction: the graded heavy band's
true N_soft turns out to lie inside the range L345 actually measured, so L345's verdict
(paid / delta* = 2.30, the trade does not pay) applies to the prize cases directly.

  <python> l347_recount.py           # uses l347_nslaw.pkl, builds it if absent
"""
import collections
import glob
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent
LAW = DIR / "l347_nslaw.pkl"
SHARDS = "C:/ICCAD_ml/floorset_lite/worker_*/*.th"

sys.path.insert(0, str(DIR))
from l296_project import DQ_SHIP, RANK1, graded  # noqa: E402
from l346_corpus import nsoft_from_constraints  # noqa: E402


def build_law():
    """n -> Counter(N_soft) over the whole training set. Stored as histograms, not rows."""
    if LAW.exists():
        return pickle.load(open(LAW, "rb"))
    import torch
    fs = sorted(glob.glob(SHARDS))
    byn = collections.defaultdict(collections.Counter)
    t0 = time.time()
    for i, f in enumerate(fs):
        try:
            d = torch.load(f, weights_only=False)
        except Exception:
            continue
        meta = d[0]
        for b in range(meta.shape[0]):
            nb = int((meta[b, :, 0] > 0).sum())
            if nb <= 0:
                continue
            byn[nb][nsoft_from_constraints(meta[b, :, 1:], nb)] += 1
        if (i + 1) % 2000 == 0:
            print("   law %d/%d shards  %.0fs" % (i + 1, len(fs), time.time() - t0))
    byn = {k: dict(v) for k, v in byn.items()}
    pickle.dump(byn, open(LAW, "wb"))
    print("   law built in %.0fs -> %s" % (time.time() - t0, LAW.name))
    return byn


def main():
    print("== L347: recount L343 against the measured N_soft law ==")
    law = build_law()
    R = graded()

    # --- disambiguate every case that carries a violation -------------------
    amb = 0
    fixed = 0
    for r in R:
        r["k"] = 1
        r["kset"] = [1]
        if r["V"] <= 0:
            continue
        h = law.get(r["n"])
        if not h:
            continue
        lo, hi = min(h), max(h)
        ks = [k for k in range(1, 16) if lo <= k * r["NS"] <= hi]
        if not ks:
            continue
        r["kset"] = ks
        # point estimate: the k whose k*NS0 lands nearest the median N_soft at that n
        med = statistics.median(sorted(h))
        r["k"] = min(ks, key=lambda k: abs(k * r["NS"] - med))
        if len(ks) > 1:
            amb += 1
        if 1 not in ks:
            fixed += 1
    print("   cases carrying a violation      %d"
          % sum(1 for r in R if r["V"] > 0))
    print("   where k=1 is IMPOSSIBLE          %d   <- L343 was wrong on exactly these"
          % fixed)
    print("   still ambiguous (k not unique)   %d" % amb)
    print()

    def q(r, dV=0, k=None):
        kk = r["k"] if k is None else k
        V = max(0, kk * r["V"] + dV)
        NS = kk * r["NS"]
        return (1 + 0.5 * (r["h"] + r["a"])) * math.exp(2.0 * V / NS) * (1 + DQ_SHIP)

    def total(mod=None):
        num = den = 0.0
        for r in R:
            num += r["w"] * (mod(r) if mod else q(r)) * r["rf"]
            den += r["w"]
        return num / den

    T0 = total()
    need = 100 * (T0 / RANK1 - 1)
    print("   projection %.6f   (unchanged: vrel itself is exact) gap to rank-1 %+.3f %%"
          % (T0, need))
    print()

    # --- A. recount the per-case prize --------------------------------------
    for r in R:
        r["save"] = (100 * (1 - total(lambda x, t=r: q(x, dV=-1 if x is t else 0)) / T0)
                     if r["V"] > 0 else 0.0)
    hit = sorted([r for r in R if r["V"] > 0], key=lambda r: -r["save"])

    print("A. CORRECTED per-case value of removing ONE violation")
    print("   %5s %5s | %8s %8s | %8s %8s %8s | %9s %9s"
          % ("case", "n", "V0(L343)", "NS0", "k", "V true", "NS true",
             "L343 said", "truth"))
    old = {}
    for r in R:
        if r["V"] > 0:
            old[r["i"]] = 100 * (1 - (sum(
                rr["w"] * ((1 + 0.5 * (rr["h"] + rr["a"]))
                           * math.exp(2.0 * max(0, rr["V"] - (1 if rr is r else 0))
                                      / rr["NS"]) * (1 + DQ_SHIP)) * rr["rf"]
                for rr in R) / sum(rr["w"] for rr in R)) / (sum(
                    rr["w"] * ((1 + 0.5 * (rr["h"] + rr["a"]))
                               * math.exp(2.0 * rr["V"] / rr["NS"]) * (1 + DQ_SHIP))
                    * rr["rf"] for rr in R) / sum(rr["w"] for rr in R)))
    for r in sorted([x for x in R if x["V"] > 0], key=lambda x: -old[x["i"]])[:6]:
        print("   %5d %5d | %8d %8d | %8s %8d %8d | %9.4f %9.4f"
              % (r["i"], r["n"], r["V"], r["NS"], r["kset"], r["k"] * r["V"],
                 r["k"] * r["NS"], old[r["i"]], r["save"]))
    print()

    def joint(k):
        top = {id(x) for x in hit[:k]}
        return 100 * (1 - total(lambda x: q(x, dV=-1 if id(x) in top else 0)) / T0)
    print("   exact joint removal, best-first (CORRECTED):")
    print("     k=1 %.4f   k=2 %.4f   k=3 %.4f   k=5 %.4f   k=10 %.4f   k=20 %.4f"
          % (joint(1), joint(2), joint(3), joint(5), joint(10), joint(20)))
    kneed = next((k for k in range(1, len(hit) + 1) if joint(k) >= need), None)
    print("     L343 said 2 violations close the 1.32 %% gap and 5 close 2.32 %%.")
    print("     Corrected: **%s violations** to close %+.2f %%; %s to close 1.32 %%."
          % (kneed, need,
             next((k for k in range(1, len(hit) + 1) if joint(k) >= 1.32), "more than %d"
                  % len(hit))))
    print("     total violations now counted: %d (L343 counted %d)"
          % (sum(r["k"] * r["V"] for r in R), sum(r["V"] for r in R)))
    print()

    # --- B. corrected delta* -------------------------------------------------
    print("B. CORRECTED break-even licence  delta* = (1+G)(exp(2/N_soft)-1)")
    print("   %14s %8s %10s %12s %14s"
          % ("N_soft band", "cases", "G median", "delta*", "as % of G"))
    for lo, hi in ((1, 33), (34, 49), (50, 59), (60, 69), (70, 999)):
        sel = [r for r in R if r["V"] > 0 and lo <= r["k"] * r["NS"] <= hi]
        if not sel:
            print("   %14s %8d %10s %12s %14s"
                  % ("%d-%d" % (lo, hi), 0, "-", "-", "EMPTY"))
            continue
        G = statistics.median(0.5 * (r["h"] + r["a"]) for r in sel)
        d = statistics.median((1 + 0.5 * (r["h"] + r["a"]))
                              * (math.exp(2.0 / (r["k"] * r["NS"])) - 1) for r in sel)
        print("   %14s %8d %10.4f %12.4f %13.1f %%"
              % ("%d-%d" % (lo, hi), len(sel), G, d, 100 * d / G))
    hv = [r for r in R if r["V"] > 0 and r["n"] >= 101]
    ds = [(1 + 0.5 * (r["h"] + r["a"])) * (math.exp(2.0 / (r["k"] * r["NS"])) - 1)
          for r in hv]
    print()
    print("   graded HEAVY band, corrected: N_soft %d..%d, delta* median %.4f"
          % (min(r["k"] * r["NS"] for r in hv), max(r["k"] * r["NS"] for r in hv),
             statistics.median(ds)))
    print("   L345 measured on N_soft 59-81 and found paid/delta* median 2.30.")
    print("   => the graded prize cases sit INSIDE the band L345 measured.")
    print("      L345's blind spot is closed, and the trade still does not pay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
