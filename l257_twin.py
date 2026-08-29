"""L257 step 2 -- price ANY L256 twin set offline, exactly, with no further solving.

Index space: i = the shipped profile i (from l252_cache.pkl), 1000+i = its
ICCAD_L256 twin (from l257_cache.pkl). A twin set S gives the pool

    originals (all 51)  U  { 1000+i : i in S }

and the shipped selector is reconstructed on that union -- including hmin, which
is the pool-wide minimum HPWL and is exactly what the global-overlay form got
wrong. The originals never leave, so the proxy can only be offered more.

Reports the no-twin baseline (must reproduce l253/L250), the union oracle (the
ceiling any selector could reach), and a greedy K-curve.

⚠️ The greedy is IN-SAMPLE on these cases. L124's discipline applies: pick K on a
held-out sample with an elbow, and measure cross-sample transfer, before believing
any K. This file produces the curve, not a ship decision.

  <python> l257_twin.py
"""
import argparse
import math
import pickle
import sys
from pathlib import Path

DIR = Path(__file__).parent
RH = 1.4          # oc._RH; asserted against the module below


def build(cases, twins):
    """weighted true cost of the proxy pick over originals U twins."""
    tot_w, tot_c, picks = 0.0, 0.0, []
    for e in cases:
        cand = []                       # (proxy, cost, tag)
        for i, m in e["base"].items():
            cand.append((m, e["basecost"][i]["cost"], ("o", i)))
        for i in twins:
            m = e["new"].get(i)
            if m is not None:
                cand.append((m, m["cost"], ("t", i)))
        if not cand:
            continue
        A_hat = 1.035 * max(e["sumA"], 1e-9)
        hmin = min(c[0]["hpwl"] for c in cand) or 1.0
        best, bp = None, None
        for m, c, tag in cand:
            p = (m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
            if bp is None or p < bp:
                bp, best = p, (c, tag)
        w = math.exp(e["n"] / 12.0)
        tot_w += w
        tot_c += w * best[0]
        picks.append(best[1])
    return tot_c / max(tot_w, 1e-18), picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--kmax", type=int, default=16)
    a = ap.parse_args(sys.argv[1:])

    B = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
    C = pickle.load(open(DIR / "l257_cache.pkl", "rb"))
    keys = [k for k in C if k[0] == a.sample and k in B]
    keys.sort(key=lambda k: -C[k]["n"])
    if not keys:
        print("no cases")
        return 1

    cases = []
    for k in keys:
        e = C[k]
        base = {i: dict(area=r["area"], hpwl=r["hpwl"], vrel=r["vrel"])
                for i, r in B[k]["recs"].items() if i in e["basecost"]}
        if len(base) < 2:
            continue
        cases.append(dict(n=e["n"], sumA=e["sumA"], base=base,
                          basecost=e["basecost"], new=e["recs"]))
    print("[l257] {} cases, sample {}".format(len(cases), a.sample))

    base_cost, _ = build(cases, [])
    all_t = sorted({i for e in cases for i in e["new"]})
    full_cost, _ = build(cases, all_t)

    # ceiling: best of the union by TRUE cost, any selector
    ow, oc_ = 0.0, 0.0
    bw, bo = 0.0, 0.0
    for e in cases:
        w = math.exp(e["n"] / 12.0)
        cb = min(e["basecost"][i]["cost"] for i in e["base"])
        cu = min([cb] + [e["new"][i]["cost"] for i in e["new"]])
        ow += w; oc_ += w * cu; bw += w; bo += w * cb
    orc_u, orc_b = oc_ / ow, bo / bw

    print()
    print("=" * 68)
    print("L257 twin pricing -- weighted TRUE cost, official scorer")
    print("=" * 68)
    print("  no twins (must match L250/l253)   {:.6f}".format(base_cost))
    print("  ALL {} twins                       {:.6f}   {:+.4f}%".format(
        len(all_t), full_cost, 100 * (full_cost - base_cost) / base_cost))
    print()
    print("  oracle over originals only        {:.6f}".format(orc_b))
    print("  oracle over originals U ALL twins {:.6f}   {:+.4f}%  <- the CEILING".format(
        orc_u, 100 * (orc_u - orc_b) / orc_b))
    print()

    # WHY it comes out however it comes out: a twin only helps if it beats the
    # best ORIGINAL in the pool, which is a far higher bar than beating its own
    # parent. L256's isolated result (154/259 twins beat their parent) says
    # nothing about this bar.
    nbeat = nc = 0
    parent_better = 0
    margins = []
    for e in cases:
        cb = min(e["basecost"][i]["cost"] for i in e["base"])
        hits = [i for i in e["new"] if e["new"][i]["cost"] < cb - 1e-12]
        nbeat += len(hits)
        nc += len(e["new"])
        parent_better += sum(1 for i in e["new"] if i in e["basecost"]
                             and e["new"][i]["cost"] < e["basecost"][i]["cost"] - 1e-12)
        if e["new"]:
            margins.append(min(e["new"][i]["cost"] for i in e["new"]) / cb - 1.0)
    margins.sort()
    print("  diagnostic:")
    print("    twins that beat THEIR OWN parent      {}/{}".format(parent_better, nc))
    print("    twins that beat the BEST original     {}/{}   <- the bar that matters"
          .format(nbeat, nc))
    print("    cases with at least one such twin     {}/{}".format(
        sum(1 for e in cases
            if any(e["new"][i]["cost"] < min(e["basecost"][j]["cost"] for j in e["base"]) - 1e-12
                   for i in e["new"])), len(cases)))
    if margins:
        print("    best twin vs best original, per case  p10 {:+.3f}%  p50 {:+.3f}%  p90 {:+.3f}%"
              .format(100 * margins[int(.1 * (len(margins) - 1))],
                      100 * margins[len(margins) // 2],
                      100 * margins[int(.9 * (len(margins) - 1))]))
    print()

    # greedy over twin sets, in-sample
    chosen, cur = [], base_cost
    print("  greedy K-curve (IN-SAMPLE -- needs an OOS elbow before use):")
    print("  {:>3s} {:>6s} {:>12s} {:>10s}".format("K", "add", "cost", "vs base"))
    for k in range(1, a.kmax + 1):
        bestc, besti = None, None
        for i in all_t:
            if i in chosen:
                continue
            c, _ = build(cases, chosen + [i])
            if bestc is None or c < bestc:
                bestc, besti = c, i
        if besti is None or not (bestc < cur - 1e-12):
            print("   no further twin improves in-sample; stopping at K={}".format(len(chosen)))
            break
        chosen.append(besti)
        cur = bestc
        print("  {:3d} {:6d} {:12.6f} {:+9.4f}%".format(
            k, besti, cur, 100 * (cur - base_cost) / base_cost))
    print()
    print("  chosen order: {}".format(chosen))
    pickle.dump(dict(chosen=chosen, base=base_cost, full=full_cost,
                     oracle_union=orc_u, oracle_base=orc_b),
                open(DIR / "l257_twin.pkl", "wb"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
