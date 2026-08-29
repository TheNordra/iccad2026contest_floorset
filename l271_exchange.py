"""L271 -- the exchange rate, as a predictor rather than a post-hoc story.

L267_L269_REPORT 2.3 observed that

    Cost = (1 + 0.5*(hpwl_gap + area_gap)) * exp(2*vrel)

prices hpwl and area IDENTICALLY, so a mechanism only pays while the wire it
costs is smaller than the area it buys. Reconstructing each arm's cost from its
three gaps reproduced every measured arm to <=0.15 pp, which makes the identity
an INSTRUMENT: the three proxy gaps are enough to predict the score.

Two things it is good for:

  * screening. An arm's gaps come out of the same run as its cost, but a
    mechanism can be triaged from gaps alone before spending a scorer pass.
  * attribution. `ratio` = the wire paid per unit of area bought. Below 1.0 the
    trade pays; above 1.0 it cannot, however good the density looks. That single
    number is what closed big-first ordering (ratio 1.24) while keeping the
    adaptive frame search (ratio -0.78, it improves both).

🚨 The residual column is the honest part. If prediction and measurement diverge
on an arm, the gaps are NOT the whole story for it -- usually because the
portfolio SELECTOR changed which profile wins, which is a different mechanism
from the layout getting better. Read a large residual as "this arm's gain is
selection, not geometry".

  <python> l271_exchange.py l271_q40.pkl [more.pkl ...]
"""
import math
import pickle
import sys
from pathlib import Path

DIR = Path(__file__).parent


def cost_of(hg, ag, vr):
    return (1.0 + 0.5 * (hg + ag)) * math.exp(2.0 * vr)


def main():
    pkls = sys.argv[1:] or ["l271_q40.pkl"]
    print("{:>12s} {:>9s} {:>9s} {:>9s} {:>8s} {:>10s} {:>10s} {:>9s}".format(
        "arm", "d_hpwl", "d_area", "d_vrel", "ratio", "predicted", "measured", "resid"))
    print("-" * 84)
    for pk in pkls:
        p = DIR / pk
        if not p.exists():
            print("  (missing {})".format(pk))
            continue
        D = pickle.load(open(p, "rb"))
        rows, arms = D["rows"], D["arms"]
        if not rows:
            continue

        def wm(f):
            sw = sum(math.exp(r["n"] / 12.0) for r in rows)
            return sum(math.exp(r["n"] / 12.0) * f(r) for r in rows) / max(sw, 1e-18)

        wb = wm(lambda r: r["base"])
        # the base gaps come from the `ship` arm when present (it is byte-identical
        # to the shipped placer, verified in-run), otherwise from the arm list's
        # own reference -- say which, because it changes what "d_" means.
        ref = "ship" if "ship" in arms else None
        if ref is None:
            print("  [{}] no `ship` arm: gaps are absolute, not deltas".format(pk))
            bh = ba = bv = 0.0
        else:
            bh = wm(lambda r: r[ref + "_hg"])
            ba = wm(lambda r: r[ref + "_ag"])
            bv = wm(lambda r: r[ref + "_vr"])
        print("  [{}]  base cost {:.6f}   gaps hpwl {:.4f} area {:.4f} vrel {:.4f}"
              .format(pk, wb, bh, ba, bv))
        cb = cost_of(bh, ba, bv)
        for nm in arms:
            if nm == ref:
                continue
            h = wm(lambda r: r[nm + "_hg"])
            a = wm(lambda r: r[nm + "_ag"])
            v = wm(lambda r: r[nm + "_vr"])
            dh, da, dv = h - bh, a - ba, v - bv
            pred = 100.0 * (cost_of(h, a, v) / cb - 1.0)
            meas = 100.0 * (wm(lambda r: r[nm]) - wb) / wb
            ratio = (dh / -da) if abs(da) > 1e-9 else float("nan")
            print("{:>12s} {:+9.4f} {:+9.4f} {:+9.4f} {:8.2f} {:+9.3f}% {:+9.3f}% "
                  "{:+8.3f}".format(nm, dh, da, dv, ratio, pred, meas, meas - pred))
        print()
    print("  ratio < 1.0 pays; ratio > 1.0 cannot, however good the density looks.")
    print("  a large |resid| means the arm's delta is SELECTION, not geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
