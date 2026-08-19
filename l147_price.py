"""L147 Gate 3 -- price the measured per-case added time against the beta medians.

Inputs: three timed reps per arm (l147_gate3.sh). Per case we take the MIN over
reps for each arm; the difference is the deployed added time. That vector is then
priced with l146_rf_price.price_seconds, which applies it case by case rather
than as an average -- the whole point, since the added time is right-tailed and
lands on the big-n cases, which have the least slack.

Two joins are reported:
  identity   the in-set case with block_count n is priced onto the beta case
             with the same n (the pessimistic, structure-preserving join)
  permuted   the same dt values shuffled across cases, 300x -- separates
             "this mechanism is expensive" from "expensive ON THE CASES THAT
             MATTER". If identity is far below the permuted p05, the cost is
             concentrated exactly where we cannot afford it.

  <python> l147_price.py --quality 2.4995
"""
import argparse
import json
import statistics as st
from pathlib import Path

import l146_rf_price as L

_DIR = Path(__file__).parent


def per_case_min(tags):
    """{n: min runtime over reps}, plus the per-case list for diagnostics."""
    acc = {}
    for t in tags:
        f = _DIR / f"results_L147_{t}.json"
        if not f.exists():
            raise SystemExit(f"missing {f.name} -- run l147_gate3.sh first")
        for r in json.load(open(f))["test_results"]:
            acc.setdefault(r["block_count"], []).append(r["runtime_seconds"])
    return {n: min(v) for n, v in acc.items()}, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=float, required=True,
                    help="in-set quality gain of the arm, %% (Gate 2)")
    ap.add_argument("--arm", default="r15g")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    reps = range(1, a.reps + 1)
    ctrl, ctrl_all = per_case_min([f"t{i}_ctrl" for i in reps])
    arm, arm_all = per_case_min([f"t{i}_{a.arm}" for i in reps])
    dt = {n: arm[n] - ctrl[n] for n in ctrl}

    sp = sorted(dt.values())
    print(f"\n=== L147 Gate 3: {a.arm} vs ctrl, min of {a.reps} ===\n")
    print(f"per-case added time (s):  min {sp[0]:+.3f}  p50 {st.median(sp):+.3f}  "
          f"p90 {sp[int(0.9 * len(sp))]:+.3f}  max {sp[-1]:+.3f}  "
          f"sum {sum(sp):+.2f}")
    print(f"wall ratio (sum arm / sum ctrl):  "
          f"{sum(arm.values()) / sum(ctrl.values()):.4f}x")
    # run-to-run spread of the control itself -- the honest noise floor
    spread = [(max(v) - min(v)) / max(min(v), 1e-9) for v in ctrl_all.values()]
    print(f"control's own run-to-run spread:  p50 {100 * st.median(spread):.1f}%"
          f"  max {100 * max(spread):.1f}%   (this is why min-of-{a.reps})")

    print(f"\nquality (in-set, Gate 2)   {a.quality:+.4f}%")
    for lo, lab in ((0, "all cases"), (100, "n>100 only")):
        r = L.price_seconds(lambda n, lo=lo: dt.get(n, 0.0) if n > lo else 0.0,
                            quality_delta_pct=a.quality, perm=300)
        print(f"  {lab:<12} RF {r['rf_cost']:+.4f}%   "
              f"permuted p50 {r['perm_p50']:+.4f}% / p05 {r['perm_p05']:+.4f}%"
              f"   ->  NET {r['net']:+.4f}%")
    r = L.price_seconds(lambda n: dt.get(n, 0.0), quality_delta_pct=a.quality)
    print(f"\nGATE 3 (identity join, all cases): NET {r['net']:+.4f}%  "
          f"-- bar is +0.80%  ->  {'PASS' if r['net'] >= 0.8 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
