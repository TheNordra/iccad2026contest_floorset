"""L220 - do the two runtime knobs compose, and what does the joint table give?

Per-n composition across BLOCK COUNTS was verified exact: mixing k per block
count reproduced the directly-measured shape on 0/100 mismatches. That is a
statement about separability in n, and it holds because the drop at block count
n only affects the case with that block count.

Composition across KNOBS is a different claim and is NOT implied by it. Dropping
profiles changes WHICH profile wins; lowering REFINE changes WHAT each profile
produces. The two can interact through the argmin. So it is tested here before
any joint table is built on it:

    q(k, R)  ?=  q(k, ship) + q(0, R) - q(0, ship)     per block count

If it holds, the 6 drop arms and 4 REFINE arms already measured span the whole
grid and the joint optimum can be read off them. If it fails, the marginals are
not usable for mixing and the honest move is the full 6x4 grid.

  <python> l220_additivity.py
"""
import json
import math
from pathlib import Path

DIR = Path(__file__).parent
TOL = 1e-9


def load(fn):
    f = DIR / fn
    if not f.exists():
        return None
    return {r["block_count"]: r for r in json.load(open(f))["test_results"]}


def main():
    base = load("results_L219_r4.json") or load("results_L209_det1.json")
    if not base:
        print("no reference arm")
        return 1
    W = lambda n: math.exp(n / 12.0)                           # noqa: E731
    SW = sum(W(n) for n in base)

    drops = {}
    for k in (3, 8, 12, 16, 20):
        d = load("results_L211_k{}.json".format(k))
        if d:
            drops[k] = d
    refs = {}
    for r in (3, 2, 1):
        d = load("results_L219_r{}.json".format(r))
        if d:
            refs[r] = d

    print("=" * 78)
    print("ADDITIVITY of the two runtime knobs, per block count")
    print("=" * 78)
    if not drops or not refs:
        print("marginal arms missing -- run L211 and L219 first")
        return 1
    print("{:>6}{:>5}{:>12}{:>12}{:>12}{:>10}"
          .format("k", "R", "predicted", "measured", "error", "n mismatch"))
    print("-" * 78)
    ok_all = True
    for k, r in ((8, 2), (16, 2), (8, 1)):
        j = load("results_L220_k{}r{}.json".format(k, r))
        if not j or k not in drops or r not in refs:
            continue
        bad = 0
        pe = me = 0.0
        for n in base:
            pred = drops[k][n]["cost"] + refs[r][n]["cost"] - base[n]["cost"]
            meas = j[n]["cost"]
            if abs(pred - meas) > 1e-9:
                bad += 1
            pe += W(n) * pred
            me += W(n) * meas
        pe /= SW
        me /= SW
        err = 100 * (pe - me) / me
        ok = bad == 0
        ok_all &= ok
        print("{:>6}{:>5}{:>12.6f}{:>12.6f}{:>+11.4f}%{:>10}"
              .format(k, r, pe, me, err, "{}/100".format(bad)))
    print("-" * 78)
    if ok_all:
        print("ADDITIVE: the marginals span the grid. The joint table can be")
        print("built from the arms already measured -- run l220_build.py.")
    else:
        print("NOT ADDITIVE. The marginals cannot be mixed across knobs, so a")
        print("joint table built from them would be fiction. Either run the")
        print("full 6x4 grid (l220_grid.sh, ~66 min) or restrict the joint")
        print("search to the (k, R) pairs actually measured.")
        print()
        print("Note the weighted error column: a small weighted error with many")
        print("per-case mismatches means the interaction exists but is not")
        print("weighted where it matters -- that is a REASON TO MEASURE the")
        print("chosen pair, not a licence to mix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
