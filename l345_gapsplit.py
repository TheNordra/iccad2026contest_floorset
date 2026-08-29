"""L345 -- is the violation prize a GENERATION gap or a SELECTION gap?

L343 priced the prize exactly: two soft-constraint violations on two heavy hidden cases
close the whole rank-1 gap, and the break-even licence to buy one is
delta* = (1+G)(exp(2/N_soft)-1) -- up to 87.8 % of that case's own geometry term when
N_soft <= 14. Everything in L343 prices a fix we do not have.

The first thing to know before building anything is WHERE the missing violation-free
layout is missing. Two possibilities, and they have opposite consequences:

  SELECTION gap -- the 51-profile pool already contains a candidate with fewer violations
                   whose true cost is lower, and the proxy is not picking it. Then the fix
                   is a re-weighting of `_proxy_metrics`, which is wrapper-only and cheap.
  GENERATION gap -- no pool member produces fewer violations on these cases at all. Then
                   no amount of re-weighting reaches it, and the line requires a new
                   mechanism inside the packer.

Prior belief is GENERATION: M13/M76/M77 measured the proxy to be per-case oracle-perfect,
and the proxy already carries the exact `exp(2*vrel)` factor. But those measurements were
about COST, on the in-set, and L296 measured that the in-set understates every
violation-trading mechanism by ~3x. Nobody has asked the question about VIOLATIONS on a
heavy out-of-sample corpus. That is what this does.

CORPUS. `l252_cache.pkl` -- OOS s1, n >= 101, 40 cases, all 51 shipped-pool candidates
each, with positions. The heavy band is where the prize is (L296: band 101-120 carries
6.468 of the 8.287 % total violation mass).

GATE. The proxy pick's rank in TRUE cost must come out ~0 and selection efficiency ~100 %,
independently reproducing M13/M76/M77. If it does not, the pool reconstruction is wrong
and nothing below means anything.

Offline oracle probe: reads labels for diagnosis, trains nothing, ships nothing, touches
no file on the shipping path (2026-08-05 ruling, same standing as L250-L253, L343, L344).

  <python> l345_gapsplit.py [--limit 80] [--cores 48]

MEASURED CAVEAT (see L345_REPORT sec.4): this corpus's heavy band carries
N_soft 59-81, while the GRADED heavy band goes down to 14 and has 6/19 cases
at N_soft <= 33. delta* is 3.9x larger there. This probe therefore CANNOT
vote on the band that carries the prize -- an L278 antecedent failure.
"""
import argparse
import math
import os
import pickle
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
CACHE = DIR / "l252_cache.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    sys.argv = ["x"]
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    import optimizer_constructive as oc

    if len(list(oc._pool_indices(120))) != 51:
        print("!! not the shipped pool")
        return 1
    RH = oc._RH

    C = pickle.load(open(CACHE, "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs(a.sample, verbose=False)}
    keys = [k for k in C if k[0] == a.sample]
    keys.sort(key=lambda k: -C[k]["n"])
    keys = keys[:a.limit]
    print("== L345: generation gap or selection gap? ==")
    print("   corpus %s, %d heavy cases, 51 candidates each, strict official scoring"
          % (a.sample, len(keys)))
    print()

    rows = []
    loaded = {}
    for kn, key in enumerate(keys):
        e = C[key]
        fk, L, n = spec_of[key[1]]
        if fk not in loaded:
            loaded.clear()
            loaded[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(loaded[fk], L)
        lay["base"], _ = m67._baseline_official(lay)

        idxs = sorted(e["recs"])
        met = [e["recs"][i] for i in idxs]
        A_hat = 1.035 * max(e["sumA"], 1e-9)
        hmin = min(m["hpwl"] for m in met) or 1.0
        prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
                for m in met]

        M = [m67._mt(m67._cost(m["pos"], lay)) for m in met]
        V = [x["vb"] + x["vg"] + x["vm"] for x in M]
        cost = [x["cost"] if x["feasible"] else float("inf") for x in M]
        G = [0.5 * (max(0.0, x["hgap"]) + max(0.0, x["agap"])) for x in M]
        NS = M[0]["nsoft"]

        k_sel = min(range(len(idxs)), key=lambda t: prox[t])
        k_best = min(range(len(idxs)), key=lambda t: cost[t])
        # min violations, ties broken by true cost -- the friendliest reading for "the
        # pool already has it"
        k_mv = min(range(len(idxs)), key=lambda t: (V[t], cost[t]))
        order = sorted(range(len(idxs)), key=lambda t: cost[t])

        # delta*: the licence, per violation, on THIS case at the selected geometry
        dstar = (1 + G[k_sel]) * (math.exp(2.0 / max(NS, 1)) - 1) if NS else 0.0
        dV = V[k_sel] - V[k_mv]
        # geometry actually paid by the lower-violation candidate, per violation removed
        paid = ((G[k_mv] - G[k_sel]) / dV) if dV > 0 else float("nan")

        rows.append(dict(
            n=n, NS=NS, w=math.exp(n / 12.0),
            V_sel=V[k_sel], V_min=V[k_mv], V_best=V[k_best],
            c_sel=cost[k_sel], c_best=cost[k_best], c_mv=cost[k_mv],
            G_sel=G[k_sel], G_mv=G[k_mv], dstar=dstar, paid=paid, dV=dV,
            rank_sel=order.index(k_sel), npool=len(idxs),
            vb=M[k_sel]["vb"], vg=M[k_sel]["vg"], vm=M[k_sel]["vm"],
            vb_mv=M[k_mv]["vb"], vg_mv=M[k_mv]["vg"], vm_mv=M[k_mv]["vm"],
            c_worst=max(c for c in cost if c < float("inf"))))
        if (kn + 1) % 10 == 0:
            print("   %d/%d" % (kn + 1, len(keys)))

    SW = sum(r["w"] for r in rows)

    def wm(f):
        return sum(r["w"] * f(r) for r in rows) / SW

    print()
    print("=" * 84)
    print("GATE -- does the proxy reproduce its known oracle-perfect behaviour?")
    print("=" * 84)
    eff = wm(lambda r: (r["c_worst"] - r["c_sel"]) / max(r["c_worst"] - r["c_best"], 1e-12))
    print("   proxy pick's rank in TRUE cost        %.2f / %d   (M13/M76/M77: ~0)"
          % (wm(lambda r: r["rank_sel"]), rows[0]["npool"]))
    print("   selection efficiency                  %.2f %%" % (100 * eff))
    print("   cost of proxy pick / cost of oracle   %.6f / %.6f  = %+.4f %%"
          % (wm(lambda r: r["c_sel"]), wm(lambda r: r["c_best"]),
             100 * (wm(lambda r: r["c_sel"]) / wm(lambda r: r["c_best"]) - 1)))
    print("   cases where proxy pick IS the oracle  %d/%d"
          % (sum(1 for r in rows if r["rank_sel"] == 0), len(rows)))
    print()

    print("=" * 84)
    print("A. THE GENERATION CEILING -- can the pool produce fewer violations at all?")
    print("=" * 84)
    print("   weighted violations of the proxy pick        %.3f" % wm(lambda r: r["V_sel"]))
    print("   weighted violations of the pool MINIMUM      %.3f" % wm(lambda r: r["V_min"]))
    print("   weighted violations of the true-cost oracle  %.3f" % wm(lambda r: r["V_best"]))
    nz = [r for r in rows if r["dV"] > 0]
    print()
    print("   cases where SOME pool member has fewer violations: %d/%d  (%.1f %% of weight)"
          % (len(nz), len(rows), 100 * sum(r["w"] for r in nz) / SW))
    print("   cases where the true-cost ORACLE has fewer:        %d/%d"
          % (sum(1 for r in rows if r["V_best"] < r["V_sel"]), len(rows)))
    print("   cases where the proxy pick is already pool-min V:  %d/%d"
          % (sum(1 for r in rows if r["dV"] == 0), len(rows)))
    print()
    print("   violation type of the proxy pick    bnd %.3f  grp %.3f  mib %.3f"
          % (wm(lambda r: r["vb"]), wm(lambda r: r["vg"]), wm(lambda r: r["vm"])))
    print("   ... of the pool-minimum candidate   bnd %.3f  grp %.3f  mib %.3f"
          % (wm(lambda r: r["vb_mv"]), wm(lambda r: r["vg_mv"]), wm(lambda r: r["vm_mv"])))
    print()

    print("=" * 84)
    print("B. IF WE FORCED THE MINIMUM-VIOLATION CANDIDATE, WHAT WOULD IT COST?")
    print("=" * 84)
    print("   proxy pick        %.6f" % wm(lambda r: r["c_sel"]))
    print("   pool-min-V pick   %.6f   %+.4f %%"
          % (wm(lambda r: r["c_mv"]),
             100 * (wm(lambda r: r["c_mv"]) / wm(lambda r: r["c_sel"]) - 1)))
    print("   true-cost oracle  %.6f   %+.4f %%"
          % (wm(lambda r: r["c_best"]),
             100 * (wm(lambda r: r["c_best"]) / wm(lambda r: r["c_sel"]) - 1)))
    print()

    print("=" * 84)
    print("C. THE delta* TEST -- on the cases that DO have a lower-violation candidate,")
    print("   is the geometry it charges below the break-even licence?")
    print("=" * 84)
    if nz:
        print("   %5s %5s %6s %5s %5s %9s %9s %9s %7s"
              % ("n", "NS", "dV", "Gsel", "Gmv", "paid/viol", "delta*", "ratio", "pays?"))
        for r in sorted(nz, key=lambda r: -r["w"])[:12]:
            ok = r["paid"] < r["dstar"]
            print("   %5d %5d %6d %5.3f %5.3f %9.4f %9.4f %9.2f %7s"
                  % (r["n"], r["NS"], r["dV"], r["G_sel"], r["G_mv"], r["paid"],
                     r["dstar"], r["paid"] / max(r["dstar"], 1e-12),
                     "YES" if ok else "no"))
        if len(nz) > 12:
            print("   ... %d more" % (len(nz) - 12))
        print()
        print("   BY N_soft BAND -- delta* varies 3.7x with N_soft (L343), and this")
        print("   corpus is n >= 101 so it may not carry the prize band at all:")
        print("   %10s %7s %7s %10s %10s %8s"
              % ("N_soft", "cases", "w-frac", "delta*", "paid/viol", "ratio"))
        for lo, hi in ((1, 24), (25, 34), (35, 49), (50, 999)):
            sel = [r for r in nz if lo <= r["NS"] <= hi]
            allb = [r for r in rows if lo <= r["NS"] <= hi]
            if not allb:
                continue
            if sel:
                print("   %10s %7d %6.1f%% %10.4f %10.4f %8.2f"
                      % ("%d-%d" % (lo, hi), len(allb),
                         100 * sum(r["w"] for r in allb) / SW,
                         statistics.median(r["dstar"] for r in sel),
                         statistics.median(r["paid"] for r in sel),
                         statistics.median(r["paid"] / max(r["dstar"], 1e-12)
                                           for r in sel)))
            else:
                print("   %10s %7d %6.1f%% %10s %10s %8s"
                      % ("%d-%d" % (lo, hi), len(allb),
                         100 * sum(r["w"] for r in allb) / SW,
                         "-", "no lower-V candidate", "-"))
        nsl = [r["NS"] for r in rows]
        print()
        print("   ** corpus caveat: N_soft here is min %d / p50 %d / max %d; the GRADED"
              % (min(nsl), int(statistics.median(nsl)), max(nsl)))
        print("      corpus is 9 / 37 / 65 and the top-5 prize cases sit at 14-53. **")
        good = [r for r in nz if r["paid"] < r["dstar"]]
        print()
        print("   pays for itself in %d of %d cases (%.1f %% of the weight that has any"
              " lower-violation candidate at all)"
              % (len(good), len(nz),
                 100 * sum(r["w"] for r in good) / max(sum(r["w"] for r in nz), 1e-12)))
        print("   median paid/delta* ratio  %.2f   (<1 means the trade is profitable)"
              % statistics.median(r["paid"] / max(r["dstar"], 1e-12) for r in nz))
    else:
        print("   NO CASE in this corpus has a pool member with fewer violations.")
        print("   => pure GENERATION gap; re-weighting selection cannot reach it.")
    print()

    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    dsel = 100 * (wm(lambda r: r["c_sel"]) / wm(lambda r: r["c_best"]) - 1)
    print("   selection leaves %.4f %% on the table (cost of proxy pick vs oracle)" % dsel)
    print("   the pool's own violation floor is %.3f against the %.3f we select"
          % (wm(lambda r: r["V_min"]), wm(lambda r: r["V_sel"])))
    print()
    print("   Read: if the two numbers on the second line are close, the pool cannot")
    print("   produce violation-free layouts on these cases and the gap is GENERATION.")
    print("   If they differ but section B shows forcing min-V costs MORE, then the")
    print("   pool's lower-violation candidates are geometrically too expensive -- which")
    print("   is a generation gap wearing a selection gap's clothes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
