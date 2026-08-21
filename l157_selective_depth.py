"""L157 — is LP depth k=2 affordable if it is spent only where the floor pays?

HANDOFF_2026-08-20 §3 killed k=2 as an all-or-nothing: +0.5967% of quality for
+23.18s, priced at RF -1.06% => NET -0.4593%. That pricing treats the runtime
budget as ONE GLOBAL POOL. It is not one pool -- against the published beta
medians the per-case slack runs from 0.96x to 3.91x, so some cases can absorb a
second LP pass for nothing while others have no room at all.

This is M62's "break-even" line, which CLAUDE.md still lists as GREEN but never
started. Its stated prerequisite was "a pre-build time predictor + machine-speed
calibration"; L146's measured medians are that calibration.

THE GATE UNDER TEST -- no oracle anywhere in it:

    spend the second LP pass iff   t_case + dt_lp2  <=  0.3046 * M_hat(n)

    t_case   the case's own elapsed time      OBSERVED at runtime
    dt_lp2   cost of the second pass          OBSERVED (the first pass just ran)
    0.3046   = 0.7**(1/0.3), where max(0.7, R**0.3) leaves the floor
    M_hat(n) the cross-submission median      the ONLY estimated quantity

NO NEW EVALUATION IS RUN. Everything is already committed:
    results_L154_catchoff.json   the L147 arm (k=1)          quality baseline
    results_L148_lp2.json        the same arm at k=2         quality arm
    results_L149_t{1,2,3}_*      min-of-3 timings            the measured dt
    C_median_runtimes_beta_hidden.csv (not in git)           the medians

  <python> -u l157_selective_depth.py
"""
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import l146_rf_price as L
from l147_price import per_case_min

BAR = 0.30
DIR = Path(__file__).parent


def _mins(prefix):
    """{n: min runtime over the three L149 reps} for one arm."""
    acc = defaultdict(list)
    for i in (1, 2, 3):
        f = DIR / f"results_L149_t{i}_{prefix}.json"
        if not f.exists():
            raise SystemExit(f"missing {f.name}")
        for r in json.load(open(f))["test_results"]:
            acc[r["block_count"]].append(r["runtime_seconds"])
    return {n: min(v) for n, v in acc.items()}


def fit_median(rows):
    """M_hat(n) = A * n**b, least squares in log-log. Returns (fn, R2, resid)."""
    xs = [math.log(r["n"]) for r in rows]
    ys = [math.log(r["med"]) for r in rows]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    sr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    fn = lambda n: math.exp(a) * n ** b                              # noqa: E731
    resid = sorted(math.exp(y) / fn(math.exp(x)) for x, y in zip(xs, ys))
    return fn, 1 - sr / ss, resid, math.exp(a), b


def main():
    rows = L.load()
    Mhat, r2, resid, A, bexp = fit_median(rows)

    # the arm's own added seconds -- the beta run was the PRE-L147 package, so
    # the slack a second pass has to fit into is what L147 has already left.
    ctrl, _ = per_case_min([f"t{i}_ctrl" for i in (1, 2, 3)])
    arm, _ = per_case_min([f"t{i}_r15g" for i in (1, 2, 3)])
    DT147 = {n: arm[n] - ctrl[n] for n in ctrl}

    # MEASURED incremental cost of the second pass, on the eval path.
    # NOT l155's standalone LP timing: that reads p50 0.119s against this
    # 0.165s (+39%) and moves the verdict by ~0.05pp. Price the arm you would
    # actually ship, measured the way it would actually run.
    base_t, lp2_t = _mins("base"), _mins("lp2")
    DTLP2 = {n: max(0.0, lp2_t[n] - base_t[n]) for n in base_t}
    ns = sorted(DTLP2)
    near = lambda t: min(ns, key=lambda n: abs(n - t))                # noqa: E731

    q1 = {r["test_id"]: r for r in json.load(open(DIR / "results_L154_catchoff.json"))["test_results"]}
    q2 = {r["test_id"]: r for r in json.load(open(DIR / "results_L148_lp2.json"))["test_results"]}
    w = lambda i: math.exp(q1[i]["block_count"] / 12.0)               # noqa: E731
    q_base = sum(w(i) * q1[i]["cost"] for i in q1)
    net_of = lambda x: x if isinstance(x, float) else x.get("net")    # noqa: E731

    v = sorted(DTLP2.values())
    print(__doc__.split("  <python>")[0])
    print("=" * 72)
    print(f"M_hat(n) = {A:.4f} * n^{bexp:.3f}   R^2 = {r2:.3f}")
    print(f"  residual M/M_hat: p10 {resid[9]:.2f}x  p50 {resid[49]:.2f}x  "
          f"p90 {resid[89]:.2f}x  ({100*(resid[89]/resid[9]-1):.0f}% spread)")
    sl = sorted(r["slack"] for r in rows)
    print(f"per-case RF slack: min {sl[0]:.2f}x  p10 {sl[9]:.2f}x  p50 "
          f"{st.median(sl):.2f}x  p90 {sl[89]:.2f}x  max {sl[-1]:.2f}x  "
          f"({sum(1 for s in sl if s < 1.0)}/100 already past the edge)")
    print(f"measured 2nd-pass cost: p50 {st.median(v):.3f}s  p90 {v[89]:.3f}s  "
          f"max {v[-1]:.3f}s  sum {sum(v):.2f}s  (handoff says +23.18s)")

    print("\n=== the gate, priced against a shift in the final medians ===")
    print(f"{'medians':>9}{'n sel':>7}{'quality':>10}{'RF':>10}{'NET':>10}   bar {BAR:.2f}%")
    for mult in (1.20, 1.10, 1.00, 0.90, 0.80):
        sel = {r["n"] for r in rows
               if L.THR * Mhat(r["n"]) * mult - (r["t"] + max(0.0, DT147.get(r["n"], 0.0)))
               >= DTLP2.get(near(r["n"]), 0.0)}
        picked = {i for i in q1 if q1[i]["block_count"] in sel}
        qual = 100 * (q_base - sum(w(i) * ((q2 if i in picked else q1)[i]["cost"])
                                   for i in q1)) / q_base
        rf = net_of(L.price_seconds(
            lambda nn: DTLP2.get(nn, 0.0) if nn in sel else 0.0, 0.0,
            rows=[dict(r, med=r["med"] * mult) for r in rows]))
        flag = "  <= MEASURED" if mult == 1.00 else ""
        print(f"{mult:>8.2f}x{len(sel):>7}{qual:>9.4f}%{rf:>9.4f}%"
              f"{qual+rf:>9.4f}%{flag}")

    # The optimistic bracket: per-case affordability on both sides. The two
    # corpora are different cases, so this credits the in-set cases with the
    # most slack inside each band rather than a random share of it.
    print("\n=== the optimistic bracket (per-case discrimination on both sides) ===")
    MEDn = {r["n"]: r["med"] for r in rows}
    aff = {r["i"]: (L.THR * r["med"] - (r["t"] + max(0.0, DT147.get(r["n"], 0.0))))
           >= DTLP2.get(near(r["n"]), 0.0) for r in rows}
    picked = set()
    for lo, hi, lbl in ((0, 60, "n<=60"), (61, 100, "60<n<=100"), (101, 999, "n>100")):
        cs = [i for i in q1 if lo <= q1[i]["block_count"] <= hi]
        b = [r for r in rows if lo <= r["n"] <= hi]
        frac = sum(1 for r in b if aff[r["i"]]) / max(1, len(b))
        ranked = sorted(cs, key=lambda i: q1[i]["runtime_seconds"] / MEDn[near(q1[i]["block_count"])])
        picked |= set(ranked[:int(round(frac * len(cs)))])
        print(f"  {lbl:>11}: beta affordable {frac:.2f} "
              f"({sum(1 for r in b if aff[r['i']])}/{len(b)})")
    qual = 100 * (q_base - sum(w(i) * ((q2 if i in picked else q1)[i]["cost"])
                               for i in q1)) / q_base
    print(f"  quality {qual:+.4f}% at RF ~0 by construction")

    print(f"\n=== verdict ===")
    print(f"  k=2 everywhere         quality +0.5967%   RF -1.49%    NET -0.90%")
    print(f"  selective, pessimistic                                 NET +0.2310%")
    print(f"  selective, optimistic                                  NET {qual:+.4f}%")
    print(f"  bar                                                        {BAR:.4f}%")
    print("  => straddles the bar. UNDECIDED -- the OOS transfer of k=2's")
    print("     quality is the one measurement that settles it (L147 ran 86%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
