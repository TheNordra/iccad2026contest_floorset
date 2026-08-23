"""L146 - price a runtime change against the MEASURED beta medians.

Until 2026-08-19 the runtime factor was priced against a model
(`m67e_rf48.py`: `M_i ~= 3.161 * t_i^alpha`, anchored on the M10 pool). The
contest published the per-case medians of the beta hidden set
(`C_median_runtimes_beta_hidden.csv`), so RF is now exactly computable:

    RF_i = max(0.7, (t_i / M_i) ** 0.3)          iccad2026_evaluate.py:552
    total = sum_i w_i * quality_i * RF_i / sum_i w_i,     w_i = exp(n_i / 12)

Validation. ⚠️ Reconstructing `raw_score` (1.3206649447461245) proves NOTHING:
`load()` divides each case's cost by its own RF and `_total(1.0)` multiplies it
straight back, so that number is an algebraic identity, not a measurement. The
real check is against the graded `total_score`, and it passes:

    sum w*cost/sum w     = 1.3206649447461245   (xlsx raw_score,   identity)
    sum w*cost*RF/sum w  = 0.9245185859205597   (xlsx total_score  0.9245183669982832)

i.e. 2.4e-7 relative — exactly what rounding the published medians to three
decimals predicts. THAT is the evidence RF is now exactly computable.

WHY THIS MATTERS. The beta result put our cost-weighted RF at 0.7000400 -- the
floor -- which was read as "runtime is free". Per case that is only true up to
each case's own slack, `0.3046 * M_i / t_i`: median 1.74x but p10 1.24x and min
0.96x. So a UNIFORM slowdown costs

    1.25x  -0.09%      1.5x  -0.70%      2x  -3.72%      3x  -15.44%

i.e. roughly +20-25% of wall is free and it degrades fast after that. Every
mechanism this project shelved on runtime grounds (L125 beam, the REFINE
band-cut, LP depth k>1, L129) was priced against the old model and needs to be
re-read against this one.

USAGE

  as a CLI, for the budget curves:
    <python> l146_rf_price.py curve
    <python> l146_rf_price.py band --lo 100 --mult 1.5

  as a library, to price a measured A/B:
    from l146_rf_price import price
    price({case_key: (n, t_off, t_on)}, quality_delta_pct)
  or, when you only have a per-case runtime RATIO and a quality delta:
    price_ratio({n: ratio}, quality_delta_pct)
"""
import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

_DIR = Path(__file__).parent
CSV = Path("C:/Users/.01/Downloads/C_median_runtimes_beta_hidden.csv")
BETA = _DIR / "beta_2026-08-16" / "beta_evaluation_results.json"
THR = 0.7 ** (1 / 0.3)            # t/M at which RF leaves the floor: 0.3045515...


def load():
    """[(n, our_beta_runtime, published_median, quality_cost_without_RF)] x100.

    🚨 `cost` in the results json is ALREADY RF-free -- the harness forces
    RF=1.0 (iccad2026_evaluate.py:924-940) and the grader applies the factor
    when it aggregates. So q = cost, NOT cost/rf0: dividing here would remove a
    factor that was never applied, and would make `_total(1.0)` reproduce
    raw_score as an algebraic identity instead of reproducing the graded
    total_score, which is the only number that tests anything."""
    B = {r["test_id"]: r
         for r in json.load(open(BETA))["test_results"]}
    M = {}
    with open(CSV) as f:
        for row in csv.DictReader(f):
            M[int(row[list(row)[0]])] = float(row["median_runtime_s"])
    out = []
    for i in sorted(B):
        r = B[i]
        out.append(dict(i=i, n=r["block_count"], t=r["runtime_seconds"],
                        med=M[i], q=r["cost"],
                        w=math.exp(r["block_count"] / 12.0),
                        slack=THR * M[i] / r["runtime_seconds"]))
    return out


def _total(rows, mult):
    num = den = 0.0
    for r in rows:
        s = mult(r)
        num += r["w"] * r["q"] * max(0.7, (s * r["t"] / r["med"]) ** 0.3)
        den += r["w"]
    return num / den


def price_seconds(dt_of, quality_delta_pct=0.0, rows=None, perm=0, seed=0):
    """NET % of a mechanism that ADDS `dt_of(n)` seconds per case.

    🚨 Use this, not price_ratio, for anything whose cost is measured in seconds.
    An added-time distribution is not a multiplier: L122's tangent cut adds
    p50 0.040s but max 0.962s, and the expensive cases are the big-n ones, which
    have the LEAST slack. Priced as a flat +0.34s/case it reads -0.34%; priced
    with the measured per-case vector it reads -1.05%, a factor of 3 from
    concentration alone. Band-gating cannot fix that -- restricting the same
    vector to n>100 only moves -1.048% to -0.981%, because the cost and the
    weight sit on the same cases.

    `perm` > 0 also returns the distribution over `perm` random re-assignments
    of the same dt values to cases, which separates "this mechanism is expensive"
    from "this mechanism is expensive ON THE CASES THAT MATTER"."""
    rows = rows or load()
    base = _total(rows, lambda r: 1.0)

    def with_dt(fn):
        num = den = 0.0
        for r in rows:
            t = r["t"] + max(0.0, fn(r))
            num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
            den += r["w"]
        return num / den

    slow = with_dt(lambda r: dt_of(r["n"]))
    rf_cost = 100 * (base - slow) / base
    out = dict(rf_cost=rf_cost, net=quality_delta_pct + rf_cost, base=base)
    if perm:
        import random
        rnd = random.Random(seed)
        dts = [max(0.0, dt_of(r["n"])) for r in rows]
        vals = []
        for _ in range(perm):
            rnd.shuffle(dts)
            m = {id(r): dts[i] for i, r in enumerate(rows)}
            vals.append(100 * (base - with_dt(lambda r: m[id(r)])) / base)
        vals.sort()
        out["perm_p50"] = vals[len(vals) // 2]
        out["perm_p05"] = vals[max(0, int(0.05 * len(vals)) - 1)]
    return out


def price_ratio(ratio_of_n, quality_delta_pct=0.0, rows=None):
    """NET % of a mechanism that multiplies runtime by `ratio_of_n(n)` and
    improves quality by `quality_delta_pct` (positive = better).

    Returns (rf_cost_pct, net_pct). Both are percentages of the weighted total,
    positive = better, matching every other number in this project."""
    rows = rows or load()
    base = _total(rows, lambda r: 1.0)
    slow = _total(rows, lambda r: ratio_of_n(r["n"]))
    rf_cost = 100 * (base - slow) / base          # negative when it costs
    return rf_cost, quality_delta_pct + rf_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["curve", "band", "slack"])
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--mult", type=float, default=1.5)
    ap.add_argument("--quality", type=float, default=0.0,
                    help="quality gain in %, to report the NET")
    a = ap.parse_args()
    rows = load()
    base = _total(rows, lambda r: 1.0)

    if a.mode == "curve":
        print(f"\nreconstructed graded total {base:.10f}  "
              f"(official total_score 0.9245183669982832, rel 2.4e-7)\n")
        print(f"{'uniform':>9} {'total':>12} {'RF cost':>10} {'off floor':>10}")
        for s in (1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2, 2.5, 3):
            t = _total(rows, lambda r, s=s: s)
            off = sum(1 for r in rows if (s * r["t"] / r["med"]) ** 0.3 > 0.7)
            print(f"{s:>8.2f}x {t:>12.6f} {100 * (base - t) / base:>+9.4f}% "
                  f"{off:>7}/100")
        return 0

    if a.mode == "band":
        t = _total(rows, lambda r: a.mult if r["n"] > a.lo else 1.0)
        c = 100 * (base - t) / base
        w = sum(r["w"] for r in rows if r["n"] > a.lo) / sum(r["w"] for r in rows)
        print(f"\nn>{a.lo} ({100 * w:.1f}% of weight) at {a.mult:.2f}x:"
              f"  RF cost {c:+.4f}%"
              + (f"   quality {a.quality:+.4f}%  ->  NET {a.quality + c:+.4f}%"
                 if a.quality else ""))
        return 0

    sl = sorted(r["slack"] for r in rows)
    print(f"\nper-case slack (how much slower, still at the floor):")
    print(f"  min {sl[0]:.2f}x  p10 {sl[9]:.2f}x  p50 {st.median(sl):.2f}x  "
          f"p90 {sl[89]:.2f}x  max {sl[-1]:.2f}x")
    for lo, hi, lab in ((0, 60, "n<=60"), (60, 100, "60<n<=100"),
                        (100, 110, "100<n<=110"), (110, 999, "n>110")):
        seg = [r for r in rows if lo < r["n"] <= hi]
        w = sum(r["w"] for r in seg) / sum(r["w"] for r in rows)
        print(f"  {lab:<11} {100 * w:>5.1f}% of weight | "
              f"slack p50 {st.median([r['slack'] for r in seg]):.2f}x  "
              f"min {min(r['slack'] for r in seg):.2f}x | "
              f"free {sum(THR * r['med'] - r['t'] for r in seg):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
