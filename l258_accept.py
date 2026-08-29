"""L258 -- what is the CEILING for a better L256 acceptance rule?

L256 accepts a shrink when csc_of improves; 105 of 259 accepted shrinks make true
cost worse, so the rule is wrong ~40% of the time. csc's exchange rate between
area and hpwl is a hardcoded `hw = (N>=116)?0.12:0.06`, while the true cost is
`1 + 0.5*(hpwl/H0 - 1) + 0.5*(area/A0 - 1)` -- i.e. each term divided by its OWN
baseline, so the correct rate is A0/H0 and nothing in the constant tracks it.

Before writing a better rule, bound what one could buy. Everything below is
computed offline from l257_cache.pkl (base and L256 cost for 40 cases x 51
profiles), no solving.

Three pools, each scored by the SHIPPED proxy, then by true cost:

  base      what we ship
  ORACLE    per profile keep whichever of {base, L256} has the lower TRUE cost
            -> the ceiling for ANY acceptance rule, needs labels, unshippable
  PROXY     per profile keep whichever has the lower SHIPPED-PROXY value
            -> implementable IF the C++ could evaluate that proxy
  TWIN      keep both (L257) -- the ceiling if you are willing to pay for slots

The gap between base and ORACLE is the entire prize for task (b). If it is small,
(b) is dead before it is written.
"""
import math
import pickle
import sys
from pathlib import Path

DIR = Path(__file__).parent
RH = 1.4


def load():
    B = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
    C = pickle.load(open(DIR / "l257_cache.pkl", "rb"))
    cases = []
    for k in sorted(C, key=lambda k: -C[k]["n"]):
        if k not in B:
            continue
        e, b = C[k], B[k]
        ids = [i for i in e["recs"] if i in b["recs"] and i in e["basecost"]]
        if len(ids) < 2:
            continue
        cases.append(dict(
            n=e["n"], sumA=e["sumA"], ids=ids,
            bm={i: b["recs"][i] for i in ids},
            bc={i: e["basecost"][i]["cost"] for i in ids},
            nm={i: e["recs"][i] for i in ids},
            nc={i: e["recs"][i]["cost"] for i in ids}))
    return cases


def prox(m, A_hat, hmin):
    return (m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])


def score(cases, mode):
    """weighted true cost of the proxy pick, under an acceptance mode."""
    tw = tc = 0.0
    picked_new = 0
    tot = 0
    for e in cases:
        A_hat = 1.035 * max(e["sumA"], 1e-9)
        cand = []
        for i in e["ids"]:
            bm, nm = e["bm"][i], e["nm"][i]
            bc, nc = e["bc"][i], e["nc"][i]
            if mode == "base":
                cand.append((bm, bc, 0))
            elif mode == "twin":
                cand.append((bm, bc, 0))
                cand.append((nm, nc, 1))
            elif mode == "oracle":
                cand.append((nm, nc, 1) if nc < bc - 1e-12 else (bm, bc, 0))
            elif mode == "proxy":
                cand.append((bm, bc, 0))
                cand.append((nm, nc, 1))
        hmin = min(c[0]["hpwl"] for c in cand) or 1.0
        if mode == "proxy":
            # per profile keep the better-by-proxy of the pair, THEN select
            keep = []
            for j in range(0, len(cand), 2):
                b, nw = cand[j], cand[j + 1]
                keep.append(nw if prox(nw[0], A_hat, hmin) < prox(b[0], A_hat, hmin) else b)
            cand = keep
            hmin = min(c[0]["hpwl"] for c in cand) or 1.0
        best, bp = None, None
        for m, c, isnew in cand:
            p = prox(m, A_hat, hmin)
            if bp is None or p < bp:
                bp, best = p, (c, isnew)
        w = math.exp(e["n"] / 12.0)
        tw += w
        tc += w * best[0]
        picked_new += best[1]
        tot += 1
    return tc / max(tw, 1e-18), picked_new, tot


def main():
    cases = load()
    print("[l258] {} cases".format(len(cases)))
    base, _, _ = score(cases, "base")
    orac, pn_o, tot = score(cases, "oracle")
    prx, pn_p, _ = score(cases, "proxy")
    twin, pn_t, _ = score(cases, "twin")

    print()
    print("=" * 70)
    print("L258 -- ceiling for an L256 acceptance rule (40 cases, true cost)")
    print("=" * 70)
    print("  base                                {:.6f}".format(base))
    print("  PROXY  accept (implementable)       {:.6f}   {:+.4f}%   L256 picked {}/{}"
          .format(prx, 100 * (prx - base) / base, pn_p, tot))
    print("  ORACLE accept (needs labels)        {:.6f}   {:+.4f}%   L256 picked {}/{}"
          .format(orac, 100 * (orac - base) / base, pn_o, tot))
    print("  TWIN   keep both (L257, costs slots){:.6f}   {:+.4f}%   L256 picked {}/{}"
          .format(twin, 100 * (twin - base) / base, pn_t, tot))
    print()
    print("  => the ENTIRE prize for a better acceptance rule is the ORACLE row,")
    print("     and it costs no extra profile slots -- only the shrink's own wall.")

    # how often would each rule change the deployed answer at all?
    print()
    ideal = []
    for e in cases:
        for i in e["ids"]:
            if e["nm"][i]["pos"] != e["bm"][i]["pos"]:
                ideal.append((e["nc"][i] < e["bc"][i] - 1e-12))
    print("  among the {} profile-layouts L256 actually changed:".format(len(ideal)))
    print("    would a perfect rule have KEPT the shrink?  {}/{}  ({:.0f}%)".format(
        sum(ideal), len(ideal), 100.0 * sum(ideal) / max(len(ideal), 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
