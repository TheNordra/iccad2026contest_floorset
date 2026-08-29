"""L342 -- how much of L340's quality win survives exp(2*V) and the hard constraints?

L344 closed the quality-axis question (the good trees head for the label) and named the
one place all the remaining risk sits: L340 scores `quality = 1 + 0.5(hpwl_gap+area_gap)`
and NOTHING ELSE. No violation term, and the SA does not honour preplaced blocks at all
(a B*-tree cannot express a fixed coordinate -- L340 limit #2). M52's zero-tolerance
cliff lives exactly there: one near-miss token -> wR 1.232, driven by 27 boundary touches
per case and cluster abutment going to zero under any geometric error, multiplied up by
exp(2*vrel).

So this probe scores the SAME layouts three ways and reports the decomposition:

  [Q]      quality only          1 + 0.5(hg + ag)          <- what L340 reported
  [SOFT]   + exp(2*vrel)         target_positions=None     <- hard checks OFF
  [STRICT] + hard constraints    target_positions=fp_sol   <- what the harness does

The middle row is the one that answers the question M52 asked, because it isolates the
violation cliff from the preplaced blocker. Passing target_positions=None makes the
problem measurably easier (l308: ~1.7x) -- that is the POINT of the row, and it is
labelled, not hidden.

NO-OP GATE. Our shipped packer's positions are read back out of the shipping run's own
results json and re-scored through this reconstruction. hpwl_gap / area_gap / vrel / cost
must come back BIT-IDENTICAL to what the json already recorded. If they do not, the
harness reconstruction is wrong and every number below is meaningless.

Offline oracle probe: reads labels for diagnosis, trains nothing, ships nothing, touches
no file on the shipping path (2026-08-05 ruling, same standing as L250-L253, L344).

Usage: cd ship_final
       <python> l342_strictcost.py [--ns 40,80,120] [--seeds 5] [--hw 2]
                                   [--iters 10000,100000,2000000]
"""
import argparse
import json
import statistics
import sys

import torch

sys.path.insert(0, "iccad2026contest")
from iccad2026_evaluate import evaluate_solution  # noqa: E402

from l340_run import DAT, LAB, load, run  # noqa: E402
from l344_treedist import OURS_JSON, OURS_JSON_ALT  # noqa: E402


def harness_label_rects(polygons, nb):
    """Byte-for-byte the harness's own conversion (_extract_baseline, eval:806-819)."""
    out = []
    for i in range(nb):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        if len(valid) > 0:
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            out.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            out.append((0, 0, 1, 1))
    return out


def case(n):
    d = torch.load(DAT[n], weights_only=False)[0]
    meta, b2b, p2b, pins = d[0], d[1], d[2], d[3]
    lab = torch.load(LAB[n], weights_only=False)[0]
    m8, polys = lab[0], lab[1]
    nb = int((meta[:, 0] > 0).sum())
    tp = harness_label_rects(polys, nb)
    # harness baseline: stored metrics win when valid (eval:826-833)
    base = {"hpwl_baseline": float(m8[-2]) + float(m8[-1]),
            "area_baseline": float(m8[0])}
    return dict(n=nb, meta=meta[:nb], cons=meta[:nb, 1:], b2b=b2b, p2b=p2b,
                pins=pins, at=meta[:nb, 0], tp=tp, base=base)


def score(pos, c, strict):
    m = evaluate_solution({"positions": pos, "runtime": 1.0}, c["base"], c["cons"],
                          c["b2b"], c["p2b"], c["pins"], c["at"],
                          target_positions=c["tp"] if strict else None,
                          median_runtime=1.0)
    return m


def row(tag, m):
    q = 1 + 0.5 * (max(0.0, m.hpwl_gap) + max(0.0, m.area_gap))
    return ("   %-22s %8.4f %8.4f %8.4f %8.4f %9.4f %5s  b%-3d g%-3d m%-3d d%-3d o%-3d"
            % (tag, max(0.0, m.hpwl_gap), max(0.0, m.area_gap), q,
               m.violations_relative, m.cost, "yes" if m.is_feasible else "NO",
               m.boundary_violations, m.grouping_violations, m.mib_violations,
               m.dimension_violations, m.overlap_violations))


HDR = ("   %-22s %8s %8s %8s %8s %9s %5s  %s"
       % ("layout", "hpwl_g", "area_g", "[Q]", "vrel", "cost", "feas",
          "b=bnd g=grp m=mib d=dim o=ovl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="40,80,120")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hw", type=float, default=2.0)
    ap.add_argument("--iters", default="10000,100000,2000000")
    a = ap.parse_args()
    NS = [int(v) for v in a.ns.split(",")]
    ITS = [int(v) for v in a.iters.split(",")]

    print("== L342: L340's layouts under the FULL official cost ==")
    print("   [Q] = 1+0.5(hg+ag) is what L340 reported. cost adds exp(2*vrel) and,")
    print("   in the STRICT block, the preplaced/fixed hard constraints.")
    print()

    summ = []
    for n in NS:
        c = case(n)
        nb = c["n"]
        pre = int((c["cons"][:, 1] != 0).sum())
        fix = int((c["cons"][:, 0] != 0).sum())
        ourpos = None
        ourrec = None
        for t in json.load(open(OURS_JSON))["test_results"]:
            if t.get("block_count") == nb and t.get("positions"):
                ourpos = [tuple(map(float, r)) for r in t["positions"]]
                ourrec = t
                break

        print("=" * 112)
        print("n = %d   preplaced %d   fixed %d" % (nb, pre, fix))
        print("=" * 112)

        # ---- no-op gate ---------------------------------------------------
        mo = score(ourpos, c, strict=True)
        d = max(abs(mo.hpwl_gap - ourrec["hpwl_gap"]),
                abs(mo.area_gap - ourrec["area_gap"]),
                abs(mo.violations_relative - ourrec["violations_relative"]),
                abs(mo.cost - ourrec["cost"]))
        ok = d < 1e-12 and bool(mo.is_feasible) == bool(ourrec["is_feasible"])
        print("   NO-OP GATE  re-score the shipped run's own positions: max|delta| "
              "= %.3e   %s" % (d, "PASS" if ok else "*** FAIL ***"))
        if not ok:
            print("   the harness reconstruction is wrong; everything below is void.")
            return 1
        print()

        print(HDR)
        ml = score(c["tp"], c, strict=True)
        print(row("LABEL (fp_sol)", ml))
        print(row("ours RF-SAFE  STRICT", mo))
        print(row("ours RF-SAFE  soft-only", score(ourpos, c, strict=False)))
        print()

        best = {}
        for it in ITS:
            qs, cs_soft, cs_str, vs, feas = [], [], [], [], 0
            first = None
            for s in range(1, a.seeds + 1):
                r = run(n, (c["base"]["area_baseline"] / c["base"]["hpwl_baseline"])
                        * a.hw, it, seed=s)
                ms = score(r["pos"], c, strict=False)
                mt = score(r["pos"], c, strict=True)
                if first is None:
                    first = (ms, mt)
                qs.append(1 + 0.5 * (r["hg"] + r["ag"]))
                cs_soft.append(float(ms.cost))
                cs_str.append(float(mt.cost))
                vs.append(float(ms.violations_relative))
                feas += 1 if mt.is_feasible else 0
            print(row("SA %-8d soft-only" % it, first[0]))
            print(row("SA %-8d STRICT" % it, first[1]))
            print("   %-22s [Q] med %.4f | soft cost med %.4f | STRICT cost med %.4f"
                  "  feasible %d/%d  vrel med %.4f"
                  % ("SA %d  (%d seeds)" % (it, a.seeds), statistics.median(qs),
                     statistics.median(cs_soft), statistics.median(cs_str),
                     feas, a.seeds, statistics.median(vs)))
            print()
            best[it] = dict(q=statistics.median(qs),
                            soft=statistics.median(cs_soft),
                            strict=statistics.median(cs_str),
                            vrel=statistics.median(vs), feas=feas)
        summ.append(dict(n=nb, best=best, top=max(ITS), our=mo, lab=ml,
                         nseed=a.seeds, our_soft=score(ourpos, c, strict=False)))

    print("=" * 112)
    print("DECOMPOSITION  --  does the quality win survive the violation term?")
    print("=" * 112)
    print("  %5s | %-28s | %-28s | %s"
          % ("n", "[Q]  quality only", "SOFT  +exp(2*vrel)", "STRICT +hard"))
    print("  %5s | %8s %8s %9s | %8s %8s %9s | %s"
          % ("", "ours", "SA", "delta", "ours", "SA", "delta",
             "SA feasible / label cost"))
    for s in summ:
        b = s["best"][s["top"]]
        qo = 1 + 0.5 * (max(0.0, s["our"].hpwl_gap) + max(0.0, s["our"].area_gap))
        so = float(s["our_soft"].cost)
        print("  %5d | %8.4f %8.4f %+9.4f | %8.4f %8.4f %+9.4f | %d/%d   label %.4f"
              % (s["n"], qo, b["q"], b["q"] - qo, so, b["soft"], b["soft"] - so,
                 b["feas"], s["nseed"], float(s["lab"].cost)))
    print()
    print("  READ IT LIKE THIS")
    print("    [Q] delta negative, SOFT delta negative  -> the win survives the")
    print("        violation term; only the preplaced blocker is left.")
    print("    [Q] negative, SOFT positive              -> M52's cliff eats it; the")
    print("        B*-tree manifold buys geometry and pays for it in violations,")
    print("        which is the same trade [[density-is-paid-in-violations]] found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
