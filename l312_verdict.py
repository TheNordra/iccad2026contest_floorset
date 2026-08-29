"""L312 -- per-case verdict for the RF-SAFE _L196_LPGATE (71 -> 83 on).

Two gates, and they check opposite things:

  48c       the graded shape. Cost may only IMPROVE, and only on the 12 block
            counts we ungated. A mover anywhere else means the edit was not
            confined to the gate table.
  default   below the >=40-core gate the whole LP lane is inert, so this must be
            BIT-IDENTICAL to the shipped default -- cost and positions both.
            This is the cheap control that catches an edit leaking out of its
            band; L158 records that a gate you can only reach through an env var
            is inert in the package, and this is the inverse check.
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).parent
SEL = {38, 40, 56, 76, 79, 81, 94, 95, 107, 108, 114, 120}
D_48C = "results_L237_post.json"
D_DEF = "l303_mixpkg_def.json"      # mix pkg @ default == shipped default (L303)


def load(p):
    return {r["test_id"]: r for r in json.load(open(DIR / p))["test_results"]}


def total(d):
    W = {i: math.exp(r["block_count"] / 12.0) for i, r in d.items()}
    return sum(W[i] * d[i]["cost"] for i in d) / sum(W.values())


def pos_equal(a, b):
    return sum(1 for i in a if a[i].get("positions") == b[i].get("positions"))


def main():
    ok = True

    # ---------- gate 1: 48c, the graded shape ----------
    print("== gate 1: 48c (graded shape) ==")
    D, A = load(D_48C), load("l312_rfsafe_c48.json")
    tD, tA = total(D), total(A)
    print(f"  D   {tD:.9f}    RF-SAFE {tA:.9f}    quality {100*(tD-tA)/tD:+.4f}%")
    feas = sum(1 for r in A.values() if r["is_feasible"])
    print(f"  feasible {feas}/100" + ("" if feas == 100 else "   <== FAIL"))
    ok &= feas == 100

    movers = [i for i in D if A[i]["cost"] != D[i]["cost"]]
    worse = [i for i in movers if A[i]["cost"] > D[i]["cost"]]
    stray = [i for i in movers if D[i]["block_count"] not in SEL]
    print(f"  cost movers {len(movers)}   better {len(movers)-len(worse)}   "
          f"worse {len(worse)}" + ("" if not worse else f"  <== FAIL {worse}"))
    print(f"  movers outside the 12 ungated block counts: {len(stray)}"
          + ("" if not stray else f"  <== FAIL {[(i, D[i]['block_count']) for i in stray]}"))
    ok &= not worse and not stray
    print(f"  block counts that moved: "
          f"{sorted({D[i]['block_count'] for i in movers})}")
    unmoved = sorted(SEL - {D[i]["block_count"] for i in movers})
    if unmoved:
        print(f"  ungated but did not move (LP found nothing): {unmoved}")

    # ---------- gate 2: default cores, must be inert ----------
    print("\n== gate 2: default cores (must be bit-identical: LP lane inert) ==")
    Dd, Ad = load(D_DEF), load("l312_rfsafe_def.json")
    td, ta = total(Dd), total(Ad)
    dc = sum(1 for i in Dd if Ad[i]["cost"] == Dd[i]["cost"])
    dp = pos_equal(Dd, Ad)
    print(f"  anchor {td:.9f}   RF-SAFE {ta:.9f}   |d| {abs(ta-td):.3e}")
    print(f"  cost identical {dc}/100      positions identical {dp}/100")
    good = dc == 100 and dp == 100
    ok &= good
    if not good:
        bad = [i for i in Dd if Ad[i]["cost"] != Dd[i]["cost"]][:5]
        print(f"  <== FAIL, first movers: {[(i, Dd[i]['block_count']) for i in bad]}")

    print(f"\nL312 VERDICT: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
