"""OFFLINE (never shipped): price the L122 tangent-cut LP on the RF grid.

Quality is not the deliverable -- score is.  The depth ladder died here, not on
quality: k=12 buys +5.4688% of quality and still prices RED because runtime
enters the official cost as max(0.7, R^0.3) and 91.8% of the weight already sits
on the floor, so the gain has to clear the wall it adds.

Mirrors `l114_combined_price.main` exactly where it matters -- same RF model
(l86.FLOOR / l86.GAMMA), same alpha-calibrated per-case M, same weights, same
`t = tnow + tLP` composition, same speed grid -- so the row it prints is
comparable with every LP number on record.  It differs only in taking the arm's
per-case quality and MIN-OF-3 tLP from an `l122_area_tangent.py reps` dump
instead of the k-ladder's.

GRID WORST is the decision number: the worst gain over the machine-speed grid,
because s is not known and the shipped configuration must not lose anywhere on
it.  s=1 is the calibrated point; SPEEDS is its uncertainty band.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

import l86_cap_grid as l86                                  # noqa: E402

ANCHOR = _DIR / "results_M80_anchor_48c.json"
REPS = _DIR / "results_L122_reps.json"


def total(qs, ts, speed, mvals, weights):
    tot = w = 0.0
    for ci in qs:
        rf = max(l86.FLOOR, (speed * ts[ci] / mvals[ci]) ** l86.GAMMA)
        tot += weights[ci] * qs[ci] * rf
        w += weights[ci]
    return tot / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-json", default=str(REPS))
    ap.add_argument("--route-a", type=float, default=1.0,
                    help="wall multiplier for a route-A style solve; 1.0 prices "
                         "the LP lane alone")
    a = ap.parse_args()

    d = json.loads(Path(a.reps_json).read_text(encoding="utf-8"))
    anch = {int(t["test_id"]): t for t in
            json.loads(ANCHOR.read_text(encoding="utf-8"))["test_results"]}
    arms = d["arms"]
    cases = sorted(int(c) for c in arms["ship"])

    rows = {ci: {"n": arms["ship"][str(ci)]["n"]} for ci in cases}
    weights, _ = l86._weights(rows)
    mvals = l86._alpha_M()
    tnow = l86._load_tnow(rows)
    qship = {ci: float(anch[ci]["cost"]) for ci in cases}

    def gain(qs, ts, s):
        return 100.0 * (1.0 - total(qs, ts, s, mvals, weights)
                        / total(qship, tnow, s, mvals, weights))

    print(f"[inputs] {len(cases)} cases, min-of-{d['reps']} tLP, "
          f"area_price={d['area_price']}, ROUTE_A={a.route_a}")
    wq = sum(weights[c] * qship[c] for c in cases) / sum(weights.values())
    print(f"[anchor] weighted quality {wq:.9f}")

    lanes = {}
    for lab, per in arms.items():
        q = {ci: float(per[str(ci)]["q"]) for ci in cases}
        t = {ci: a.route_a * tnow[ci] + float(per[str(ci)]["t_lp"])
             for ci in cases}
        lanes[lab] = (q, t)
        gq = 100 * (1 - sum(weights[c] * q[c] for c in cases)
                    / sum(weights[c] * qship[c] for c in cases))
        print(f"[{lab:>7}] quality {gq:+.4f}%  weighted tLP "
              f"{sum(weights[c] * float(per[str(c)]['t_lp']) for c in cases) / sum(weights.values()):.4f}s")

    print(f"\n  {'arm':>8} {'s=1':>9} {'s=1.5':>9} {'s=2':>9} {'s=2.5':>9} "
          f"{'grid worst':>11}")
    best = None
    for lab, (q, t) in lanes.items():
        pts = [gain(q, t, s) for s in l86.SPEEDS]
        row = [gain(q, t, s) for s in (1.0, 1.5, 2.0, 2.5)]
        gw = min(pts)
        print(f"  {lab:>8} " + " ".join(f"{v:+8.3f}%" for v in row)
              + f" {gw:+10.3f}%")
        if lab == "ship":
            best = gw
    if best is not None:
        print(f"\n  bar to beat = the shipped LP's own grid worst "
              f"{best:+.3f}%")
        for lab, (q, t) in lanes.items():
            if lab == "ship":
                continue
            gw = min(gain(q, t, s) for s in l86.SPEEDS)
            verdict = "GREEN" if gw > best else "RED"
            print(f"  {lab}: grid worst {gw:+.3f}% vs {best:+.3f}%  -> "
                  f"{verdict} ({gw - best:+.3f}pp)")


if __name__ == "__main__":
    main()
