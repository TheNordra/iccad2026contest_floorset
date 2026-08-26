"""L251 - decompose the GENERATION gap into its three cost terms.

L250 settled the partition on the shipped 51-profile pool, heavy band, OOS s1:

    proxy pick 1.511619 | oracle 1.511432 | label 1.245233
    SELECTION loss +0.0124%   GENERATION loss +17.6124%

So the pool does not contain the layouts and a better selector buys nothing.
This file asks WHICH TERM the 17.6% lives in, on the same cases, by taking our
best-in-pool layout and the label apart:

    cost = (1 + 0.5*(hpwl_gap + area_gap)) * exp(2*vrel)

and pricing three counterfactuals per case -- what the weighted total becomes if
one term is set to the label's value and the others are left alone. That is a
per-term value, not a decomposition of a sum, because the terms multiply.

L128 did this globally and got hpwl 10.15% / area 6.00% / vrel 3.57% on a tree
that predates L131, L136, L147, L223 and L231, and on a different corpus. This
re-measures it where the score actually is (n>=101 carries 81% of the weight)
and on the tree we ship.

⚠️ The label's own vrel is NOT zero (0.05037 per L128) -- it violates soft
constraints too, and we are three times better on boundary. So "vrel -> label"
can be NEGATIVE, and that is information: it says the term is already won.

Caches the captures so any follow-up analysis is free.

  <python> l251_terms.py --sample s1 --nmin 101 --limit 40
"""
import argparse
import math
import os
import pickle
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent
CACHE = DIR / "l251_cache.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--nmin", type=int, default=101)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    sys.argv = ["x"]
    import torch                                                   # noqa: F401
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    import l124_r3_scale as R
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)   # AFTER m67 strips
    import optimizer_constructive as oc

    if len(list(oc._pool_indices(120))) != 51:
        print("!! not the shipped pool -- refusing")
        return 1
    RH = oc._RH
    specs = [(ck, fk, L, n) for ck, fk, L, n in m77._specs(a.sample)
             if n >= a.nmin]
    specs.sort(key=lambda t: -t[3])
    if a.limit:
        specs = specs[:a.limit]
    C = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    byf = {}
    for ck, fk, L, n in specs:
        if (a.sample, ck) not in C:
            byf.setdefault(fk, []).append((ck, L, n))
    print("[l251] {} cases, {} to capture".format(len(specs),
                                                  sum(len(v) for v in byf.values())))
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, L, n in byf[fk]:
            lay = m67._load_case(d, L)
            lay["base"], _ = m67._baseline_official(lay)
            cap = R._capture(oc, lay, "0")
            if len(cap) < 2:
                continue
            idxs = sorted(cap)
            met = [cap[i][1] for i in idxs]
            sumA = sum(max(0.0, float(lay["at"][i])) for i in range(n))
            A_hat = 1.035 * max(sumA, 1e-9)
            hmin = min(m["hpwl"] for m in met) or 1.0
            prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin)
                    * math.exp(2.0 * m["vrel"]) for m in met]
            k = min(range(len(idxs)), key=lambda j: prox[j])
            mm = m67._mt(m67._cost(cap[idxs[k]][0], lay))
            lb = m67._mt(m67._cost(lay["tp"], lay))
            C[(a.sample, ck)] = dict(n=n, ours=mm, label=lb)
            pickle.dump(C, open(CACHE, "wb"))
    pickle.dump(C, open(CACHE, "wb"))

    rows = [v for kk, v in C.items() if kk[0] == a.sample and v["n"] >= a.nmin]
    if not rows:
        print("nothing cached")
        return 1
    W = lambda r: math.exp(r["n"] / 12.0)                          # noqa: E731
    SW = sum(W(r) for r in rows)

    def cost(hg, ag, vr):
        return (1 + 0.5 * (max(hg, 0.0) + max(ag, 0.0))) * math.exp(2 * vr)

    def tot(f):
        return sum(W(r) * f(r) for r in rows) / SW

    base = tot(lambda r: cost(r["ours"]["hgap"], r["ours"]["agap"],
                              r["ours"]["vrel"]))
    lab = tot(lambda r: cost(r["label"]["hgap"], r["label"]["agap"],
                             r["label"]["vrel"]))
    print()
    print("=" * 70)
    print("GENERATION gap by term, {} cases n>={} (sample {})"
          .format(len(rows), a.nmin, a.sample))
    print("=" * 70)
    print("  our best-in-pool, recomputed from terms  {:.6f}".format(base))
    print("  the label, recomputed from terms         {:.6f}".format(lab))
    print("  gap                                      {:+.4f}%"
          .format(100 * (base - lab) / base))
    print()
    print("  {:<28}{:>12}{:>12}{:>12}".format("term", "ours", "label", "worth"))
    print("  " + "-" * 64)
    for name, f in (
        ("hpwl_gap -> label's",
         lambda r: cost(r["label"]["hgap"], r["ours"]["agap"], r["ours"]["vrel"])),
        ("area_gap -> label's",
         lambda r: cost(r["ours"]["hgap"], r["label"]["agap"], r["ours"]["vrel"])),
        ("vrel     -> label's",
         lambda r: cost(r["ours"]["hgap"], r["ours"]["agap"], r["label"]["vrel"])),
        ("all three -> label's", lambda r: cost(r["label"]["hgap"],
                                                r["label"]["agap"],
                                                r["label"]["vrel"])),
    ):
        key = name.split()[0]
        o = st.median(r["ours"][{"hpwl_gap": "hgap", "area_gap": "agap",
                                 "vrel": "vrel"}.get(key, "vrel")]
                      for r in rows) if key != "all" else float("nan")
        l_ = st.median(r["label"][{"hpwl_gap": "hgap", "area_gap": "agap",
                                   "vrel": "vrel"}.get(key, "vrel")]
                       for r in rows) if key != "all" else float("nan")
        v = 100 * (base - tot(f)) / base
        print("  {:<28}{:>12}{:>12}{:>11.4f}%"
              .format(name,
                      "-" if key == "all" else "{:.4f}".format(o),
                      "-" if key == "all" else "{:.4f}".format(l_), v))
    print("  " + "-" * 64)
    print()
    nb = sum(1 for r in rows if r["ours"]["vrel"] < r["label"]["vrel"])
    print("  we already beat the label on vrel in {}/{} cases"
          .format(nb, len(rows)))
    print("  feasible: ours {}/{}   label {}/{}"
          .format(sum(1 for r in rows if r["ours"]["feasible"]), len(rows),
                  sum(1 for r in rows if r["label"]["feasible"]), len(rows)))
    print()
    print("  'worth' is what the weighted total gains if THAT term alone is set")
    print("  to the label's value. The terms multiply, so they do not sum to the")
    print("  total -- a negative means we are already ahead on that term.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
