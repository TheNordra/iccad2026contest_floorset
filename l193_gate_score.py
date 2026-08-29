"""L193 verdict - does the per-case LP gate survive out of sample?

THE CANDIDATE. Thin pool (M80/twins/L137 off) whose own wall is ~M73's, so the
whole 14.72 s budget stays free, then spend it on the shape LP only where the
RF floor allows:

    run the LP on block count n  iff   t_pool(n) + dt_lp(n) <= 0.3046 * M(n) * s

In set that fires on 45 of 100 cases and reads +2.551% vs beta at route-A
neutral -- the only configuration measured today that clearly beats beta
without betting on route A.

WHAT IS ACTUALLY UNDER TEST. The RF side is already measured (real walls,
transported calibration-free) and does not depend on the corpus. What is in-set
is WHICH CASES the LP helps. That is a per-case selection rule, the form that
has failed out of sample most often here: L127's tally fitting transferred at
15-25%, and today the twins (0.00% in set, +0.67% OOS), L171 (+0.077% -> -0.051%)
and the thin pool (-1.39% in set, -2.66% OOS) all moved against their in-set
reading.

METHOD. Arm-mixing, which the ledger records as exact -- mixing flat arms
reproduces a really-run gated arm 100/100 on cost AND positions. Per case, the
gate picks the LP-off arm or the LP-k=1 arm:

    l193_{s}_thinoff.json   thin pool, LP OFF
    l192_{s}_thin.json      thin pool, LP at k=1

No new mechanism is run, so nothing here can be a no-op of the flag kind: the
two arms differ by construction, and the mix is arithmetic over them.
"""
import json
import math
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
THR = 0.7 ** (1 / 0.3)


def ld(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return {r["test_id"]: r for r in d}


def inset(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


def main():
    # the gate, derived exactly as l190 does, on the THIN pool
    wm, _ = inset("_l181_m73.json")
    wo, _ = inset("_l181_nohint.json")
    wk, _ = inset("_l191_thinpool_lp.json")
    rows = M.rows_new()
    gate = {}
    for r in rows:
        n = r["n"]
        if not wm.get(n):
            continue
        k = r["t"] / wm[n]
        pool = wo[n] * k
        dtlp = max(0.0, (wk[n] - wo[n]) * k)
        gate[n] = 1 if pool + dtlp <= THR * r["med"] else 0
    print(__doc__)
    print("=" * 78)
    print("gate fires on {} of {} block counts".format(sum(gate.values()), len(gate)))

    print("\n{:<6}{:>12}{:>12}{:>12}{:>11}"
          .format("sample", "LP off", "LP on", "GATED", "gate vs off"))
    got = []
    for s in ("s1", "s2"):
        try:
            off = ld("l193_{}_thinoff.json".format(s))
            on = ld("l192_{}_thin.json".format(s))
        except FileNotFoundError:
            print("{:<6}  not finished".format(s))
            continue
        ids = sorted(set(off) & set(on))
        w = lambda i: math.exp(off[i]["n"] / 12.0)                 # noqa: E731
        sw = sum(w(i) for i in ids)
        qo = sum(w(i) * off[i]["cost"] for i in ids) / sw
        qn = sum(w(i) * on[i]["cost"] for i in ids) / sw
        qg = sum(w(i) * (on if gate.get(off[i]["n"], 0) else off)[i]["cost"]
                 for i in ids) / sw
        got.append((qo, qn, qg))
        print("{:<6}{:>12.6f}{:>12.6f}{:>12.6f}{:>+10.4f}%"
              .format(s, qo, qn, qg, 100 * (qo - qg) / qo))
    if len(got) < 2:
        return 1
    lp_full = sum(100 * (o - n) / o for o, n, _ in got) / 2
    lp_gate = sum(100 * (o - g) / o for o, _, g in got) / 2
    print("\nthe LP everywhere is worth  {:+.4f}%   out of sample".format(lp_full))
    print("the GATED LP is worth       {:+.4f}%   out of sample".format(lp_gate))
    print("capture rate                {:.0f}%   of the LP's OOS quality"
          .format(100 * lp_gate / lp_full if lp_full else 0))
    print("\nIN SET the same gate captured 45/100 cases; compare the capture")
    print("rate above against that to see whether the SELECTION transferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
