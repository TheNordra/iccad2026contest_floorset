"""L281 deployability: the oracle gain costs 980 LP solves per case.  What would
it cost if you only had to try the units the wire prize already points at?

`rank_units` orders units by an exact, label-free, LP-free quantity (the wire
prize of moving that unit alone to its weighted-L1-median).  The census stores
`picks` in that order.  So for each case this asks: how far down that ranking do
you have to go before you have the best relocation the full scan found, and how
many LP solves does that prefix cost?

Read-only on the cache.
"""
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
DB = pickle.load(open(_DIR / "l281_cache.pkl", "rb"))["db"]

cen = {k[1]: v for k, v in DB.items() if k[0] == "census" and len(k) > 2}
rel = [(k[1], k[2], k[3], v) for k, v in DB.items() if k[0] == "rel2"]

for ci in sorted({r[0] for r in rel}):
    picks = cen.get(ci, {}).get("picks")
    if not picks:
        print(f"case {ci}: no full-width census cached")
        continue
    order = [ku for ku, _kept in picks]
    rank = {ku: i for i, ku in enumerate(order)}
    ncand = {ku: len(kept) for ku, kept in picks}
    cands = [(ku, ic, min(v["cost"], v.get("polished", float("inf"))))
             for c, ku, ic, v in rel
             if c == ci and v.get("status") == "ok" and v.get("feas")]
    if not cands:
        print(f"case {ci}: no feasible relocation")
        continue
    cbase = [DB[("ctrl", ci)]["cost"]] if DB.get(("ctrl", ci), {}).get("feas") \
        else []
    cbase += [v["cost"] for k, v in DB.items()
              if k[0] == "ctrlp" and k[1] == ci and v.get("feas")]
    cb = min(cbase) if cbase else float("inf")
    best = min(cands, key=lambda t: t[2])
    r = rank.get(best[0], -1)
    prefix = sum(ncand[ku] for ku in order[:r + 1])
    total = sum(ncand.values())
    print(f"case {ci}: best relocation is unit {best[0]} = rank "
          f"{r + 1}/{len(order)} by wire prize   gain "
          f"{100.0 * (cb - best[2]) / cb:+.4f} %")
    print(f"          LP solves to reach it: {prefix} of {total} "
          f"({100.0 * prefix / max(total, 1):.1f} %)")
    # what the top-1 unit alone would have bought
    top = [c for c in cands if c[0] == order[0]]
    if top:
        b1 = min(top, key=lambda t: t[2])
        print(f"          rank-1 unit alone ({ncand[order[0]]} solves): "
              f"{100.0 * (cb - b1[2]) / cb:+.4f} %")
    for kmax in (1, 2, 3, 5, 10):
        sub = [c for c in cands if rank.get(c[0], 1 << 30) < kmax]
        if not sub:
            continue
        bb = min(sub, key=lambda t: t[2])[2]
        ns = sum(ncand[ku] for ku in order[:kmax])
        print(f"          top-{kmax:<2d} units ({ns:4d} solves): "
              f"{100.0 * (cb - bb) / cb:+.4f} %")
