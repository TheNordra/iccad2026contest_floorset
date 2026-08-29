"""L172g - the depth rule is AFFORDABILITY-ONLY. Can a quality-aware one beat it?

`_depth_ok` spends the deepest k each case can afford and never asks whether
that case gets anything for it. On the OOS samples the old map moved 220/240
cases and 37 of them got WORSE; the x0.90 map moves 116 and 18 get worse. So
roughly one in six of the passes we buy is actively harmful.

Choosing k per CASE from its own OOS cost is an oracle and is not allowed.
Choosing k per BLOCK COUNT from one sample and testing on the other is not:
the samples are disjoint, so it is a fit on 240 cases scored on a different
240. That is the experiment here.

    RULE      for each n, take the largest k that (a) the x0.90 affordability
              rule allows and (b) did not make things worse on the TRAINING
              sample; ties broken toward the shallower k.
    TRAIN s1 -> TEST s2, and TRAIN s2 -> TEST s1, reported separately.
    A rule that only works in the direction it was fitted is noise.

⚠️ Read the transfer, not the training number. This ledger records tally-fitted
rules whose OOS transfer was 15-25% (L127) and one whose in-sample curve said
win while held-out said lose (M80 R=512).
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M
import l172_grid as G

DIR = Path(__file__).parent
ARMS = {1: "l147_oos_{}_r15g.json", 2: "l157_oos_{}_k2.json",
        3: "l165_oos_{}_k3.json"}


def arms(sample):
    Q = {k: {r["test_id"]: r for r in json.load(open(DIR / fn.format(sample)))
             ["test_results"]} for k, fn in ARMS.items()}
    ids = sorted(set(Q[1]) & set(Q[2]) & set(Q[3]))
    return Q, ids


def score(Q, ids, dmap):
    w = lambda i: math.exp(Q[1][i]["n"] / 12.0)                    # noqa: E731
    base = sum(w(i) * Q[1][i]["cost"] for i in ids)
    mix = sum(w(i) * Q[dmap.get(Q[1][i]["n"], 1)][i]["cost"] for i in ids)
    worse = sum(1 for i in ids
                if Q[dmap.get(Q[1][i]["n"], 1)][i]["cost"] > Q[1][i]["cost"] + 1e-12)
    moved = sum(1 for i in ids if dmap.get(Q[1][i]["n"], 1) != 1)
    return 100 * (base - mix) / base, moved, worse


def per_n_delta(Q, ids):
    """{n: {k: weighted cost delta vs k=1, positive = better}} on this sample."""
    out = defaultdict(dict)
    byn = defaultdict(list)
    for i in ids:
        byn[Q[1][i]["n"]].append(i)
    for n, cs in byn.items():
        w = math.exp(n / 12.0)
        b = sum(w * Q[1][i]["cost"] for i in cs)
        for k in (2, 3):
            out[n][k] = b - sum(w * Q[k][i]["cost"] for i in cs)
    return out


def main():
    cap = {int(k): v for k, v in
           json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    k1 = {n: 1 for n in cap}
    print(__doc__)
    print("=" * 76)

    A = {s: arms(s) for s in ("s1", "s2")}
    D = {s: per_n_delta(*A[s]) for s in ("s1", "s2")}

    for train, test in (("s1", "s2"), ("s2", "s1")):
        d = D[train]
        rule = {}
        for n, kmax in cap.items():
            best = 1
            for k in range(2, kmax + 1):
                if d.get(n, {}).get(k, 0.0) > 0:
                    best = k
            rule[n] = best
        qt, mt, wt = score(*A[train], rule)
        qs, ms, ws = score(*A[test], rule)
        qc_t, _, _ = score(*A[train], cap)
        qc_s, _, _ = score(*A[test], cap)
        rf = G.rf_on(rows, rule, dtan, dpass, near, 1.00)
        rfb = G.rf_on(rows, k1, dtan, dpass, near, 1.00)
        rf90 = G.rf_on(rows, rule, dtan, dpass, near, 0.90)
        rfb90 = G.rf_on(rows, k1, dtan, dpass, near, 0.90)
        print("\nTRAIN {} -> TEST {}    depths {}"
              .format(train, test, dict(sorted(Counter(rule.values()).items()))))
        print("   on {} (fitted)   {:+.4f}%   vs x0.90 rule {:+.4f}%   "
              "{} moved / {} worse".format(train, qt, qc_t, mt, wt))
        print("   on {} (HELD OUT) {:+.4f}%   vs x0.90 rule {:+.4f}%   "
              "{} moved / {} worse".format(test, qs, qc_s, ms, ws))
        print("   transfer {:.0f}%   RF vs k=1: {:+.4f}% at x1.00, "
              "{:+.4f}% at x0.90"
              .format(100 * qs / qt if qt else 0.0,
                      100 * (rfb - rf) / rfb, 100 * (rfb90 - rf90) / rfb90))
        print("   NET on the held-out sample {:+.4f}%   (x0.90 rule {:+.4f}%)"
              .format(qs + 100 * (rfb90 - rf90) / rfb90,
                      qc_s + 100 * (rfb90 - G.rf_on(rows, cap, dtan, dpass,
                                                    near, 0.90)) / rfb90))

    print("\n--- for reference, the affordability-only rules on both samples ---")
    for lbl, m in (("k=1", k1), ("x0.90 (shipped now)", cap),
                   ("old L165 map", M.SHIPPED)):
        q1, _, w1 = score(*A["s1"], m)
        q2, _, w2 = score(*A["s2"], m)
        print("   {:<22} s1 {:+.4f}% ({} worse)   s2 {:+.4f}% ({} worse)"
              .format(lbl, q1, w1, q2, w2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
