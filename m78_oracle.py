"""M78: 2-way per-case oracle ceiling on an m67_oos_probe A/B dump. OFFLINE.

Answers "would the pool-TIER form of this knob be worth building?".  A global
overlay forces every profile to carry the mechanism, so a case it hurts has no
escape; a tier adds knob-ON twins and lets the proxy arbitrate per case.  M76/M77
measured the proxy to be oracle-perfect on heterogeneous candidates, so the 2-way
oracle is the realistic ceiling of the tier form -- before its wall cost.

Usage:  python m78_oracle.py results_M72_ab_<arm>_0_inf.json
"""
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import compute_total_score            # noqa: E402


def per_n_total(rows, field):
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r[field])
    ns = sorted(by_n)
    means = [sum(by_n[n]) / len(by_n[n]) for n in ns]
    return compute_total_score(means, ns)


def main():
    d = json.load(open(_DIR / sys.argv[1]))
    rows = d["rows"]
    for r in rows:
        r["O"] = min(r["S"], r["R"])
    S = per_n_total(rows, "S")
    R = per_n_total(rows, "R")
    O = per_n_total(rows, "O")
    print(f"arm {d['arm']}   {d['cases']} cases   sel {d['sel']}")
    print(f"  shipped              {S:.6f}")
    print(f"  arm (global overlay) {R:.6f}   {100 * (S - R) / S:+.4f}%")
    print(f"  2-way per-case ORACLE{O:.6f}   {100 * (S - O) / S:+.4f}%"
          "   <- ceiling of the pool-tier form")
    print(f"\n  realized by the overlay: "
          f"{100 * (S - R) / (S - O) if S != O else 0:.1f}% of the ceiling")
    print(f"  movers {d['better']} better / {d['worse']} worse")
    print(f"\n  bar = NET (quality - dRF@48c) >= 0.30%")
    print(f"  wall: {d['tS']:.2f}s -> {d['tR']:.2f}s ({d['tR'] / d['tS']:.2f}x) "
          f"@12c; RF sign-check upper bound {d['rf_sign_pct']:+.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
