"""L172 - the shipped LP depth map was built from medians that have been REPLACED.

The contest published an UPDATED per-case median table on 2026-08-23
(`beta_2026-08-23/C_median_runtimes_beta_hidden_update.csv`) together with an
updated leaderboard. Both are authoritative: feeding the new medians through
the published cost formula reproduces our new graded total_score
0.9265861161320369 to 6e-7, exactly as the old table reproduced the old one.

WHAT CHANGED:  every one of the 100 medians FELL.  p50 ratio 0.742,
sum 295.72s -> 216.13s.  The field got faster (two submissions were replaced
between the two leaderboards), so the runtime budget the RF floor grants us
shrank by ~26% on every single case.

WHY THAT MATTERS: `_L157_DEPTH` in optimizer_constructive.py -- the shipped
per-case LP depth map -- was derived from the OLD table by
    largest k with  t_beta(n) + (dt_tan(n) + (k-1)*dt_pass(n))/f <= 0.3046*M(n)
It is now spending against a budget that no longer exists.  This is the same
stale-baseline failure the ledger already records three times (M_hat, the
L152 wall re-price, the L154 pricing basis) -- and this time it is live in the
package.

This script re-derives the map on the new table and prices both maps, on the
new table, with quality taken by ARM-MIXING the committed flat k=1/2/3 OOS
arms (mixing is exact: it reproduces a really-run gated arm 100/100 on cost
AND positions).
"""
import json
import math
import csv
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import l146_rf_price as L
from l147_price import per_case_min

DIR = Path(__file__).parent
NEW = DIR / "beta_2026-08-23" / "C_median_runtimes_beta_hidden_update.csv"
F = 3.17                       # dev-box LP second -> grader second (L161)
SHIPPED = {
    21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3,
    31: 3, 32: 3, 33: 3, 34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 3,
    41: 3, 42: 3, 43: 3, 44: 2, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3,
    51: 3, 52: 3, 53: 3, 54: 2, 55: 3, 56: 3, 57: 3, 58: 3, 59: 3, 60: 2,
    61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3,
    71: 3, 72: 3, 73: 3, 74: 2, 75: 3, 76: 3, 77: 3, 78: 3, 79: 1, 80: 2,
    81: 3, 82: 3, 83: 1, 84: 3, 85: 1, 86: 3, 87: 1, 88: 2, 89: 3, 90: 3,
    91: 3, 92: 1, 93: 3, 94: 1, 95: 3, 96: 3, 97: 3, 98: 3, 99: 3, 100: 3,
    101: 3, 102: 3, 103: 3, 104: 3, 105: 3, 106: 2, 107: 3, 108: 3, 109: 3,
    110: 3, 111: 3, 112: 1, 113: 3, 114: 3, 115: 3, 116: 3, 117: 2, 118: 1,
    119: 3, 120: 3,
}


def med_new():
    m = {}
    for r in csv.DictReader(open(NEW)):
        m[int(r["test_id"])] = float(r["median_runtime_s"])
    return m


def rows_new():
    """L.load() but with the 2026-08-23 medians."""
    M = med_new()
    return [dict(r, med=M[r["i"]], slack=L.THR * M[r["i"]] / r["t"])
            for r in L.load()]


def _mins(prefix):
    acc = defaultdict(list)
    for i in (1, 2, 3):
        f = DIR / "results_L149_t{}_{}.json".format(i, prefix)
        for r in json.load(open(f))["test_results"]:
            acc[r["block_count"]].append(r["runtime_seconds"])
    return {n: min(v) for n, v in acc.items()}


def costs():
    """(dt_tangent by n, dt_per_pass by n, nearest-n lookup) -- all MEASURED."""
    ctrl, _ = per_case_min(["t{}_ctrl".format(i) for i in (1, 2, 3)])
    arm, _ = per_case_min(["t{}_r15g".format(i) for i in (1, 2, 3)])
    dtan = {n: max(0.0, arm[n] - ctrl[n]) for n in ctrl}
    b, l2 = _mins("base"), _mins("lp2")
    dpass = {n: max(0.0, l2[n] - b[n]) for n in b}
    ns = sorted(dpass)
    return dtan, dpass, (lambda t: min(ns, key=lambda n: abs(n - t)))


def build(rows, dtan, dpass, near, f=F, scale=1.0):
    """largest k in 1..3 that still fits under 0.3046 * med * scale."""
    out = {}
    for r in rows:
        n = r["n"]
        budget = L.THR * r["med"] * scale - r["t"] - dtan.get(near(n), 0.0) / f
        k = 1
        for kk in (2, 3):
            if (kk - 1) * dpass.get(near(n), 0.0) / f <= budget:
                k = kk
        out[n] = k
    return out


def rf_of(rows, dmap, dtan, dpass, near, f=F):
    """RF-adjusted total when each case pays tangent + (k-1) extra passes."""
    num = den = 0.0
    for r in rows:
        n = r["n"]
        k = dmap.get(n, 1)
        t = r["t"] + (dtan.get(near(n), 0.0)
                      + (k - 1) * dpass.get(near(n), 0.0)) / f
        num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        den += r["w"]
    return num / den


def quality(dmap, sample):
    """Arm-mix the committed flat OOS arms under `dmap`; % better than k=1."""
    A = {1: "l147_oos_{}_r15g.json".format(sample),
         2: "l157_oos_{}_k2.json".format(sample),
         3: "l165_oos_{}_k3.json".format(sample)}
    Q = {k: {r["test_id"]: r
             for r in json.load(open(DIR / fn))["test_results"]}
         for k, fn in A.items()}
    ids = sorted(set(Q[1]) & set(Q[2]) & set(Q[3]))
    # the l140 OOS dumps key block count as "n", the eval dumps as "block_count"
    nof = lambda i: Q[1][i].get("n", Q[1][i].get("block_count"))   # noqa: E731
    w = lambda i: math.exp(nof(i) / 12.0)                          # noqa: E731
    pick = lambda i: Q[dmap.get(nof(i), 1)][i]                     # noqa: E731
    base = sum(w(i) * Q[1][i]["cost"] for i in ids)
    mix = sum(w(i) * pick(i)["cost"] for i in ids)
    moved = sum(1 for i in ids if dmap.get(nof(i), 1) != 1)
    worse = sum(1 for i in ids if pick(i)["cost"] > Q[1][i]["cost"] + 1e-12)
    return 100 * (base - mix) / base, moved, worse, len(ids)


def main():
    old_rows, new_rows = L.load(), rows_new()
    dtan, dpass, near = costs()
    print(__doc__)
    print("=" * 74)
    slo = sorted(r["slack"] for r in old_rows)
    sl = sorted(r["slack"] for r in new_rows)
    for lbl, s in (("OLD", slo), ("NEW", sl)):
        print("per-case slack  {} medians  min {:.2f} p10 {:.2f} p50 {:.2f} "
              "p90 {:.2f} max {:.2f}   ({:>2}/100 already past the edge)"
              .format(lbl, s[0], s[9], st.median(s), s[89], s[-1],
                      sum(1 for x in s if x < 1)))

    old_map = build(old_rows, dtan, dpass, near)
    new_map = build(new_rows, dtan, dpass, near)
    bad = sum(1 for n in SHIPPED if old_map.get(n) != SHIPPED[n])
    print("\nre-derivation on the OLD medians reproduces the shipped map: {}"
          "   (differs on {} of 100)".format(old_map == SHIPPED, bad))
    print("shipped map depths    {}"
          .format(dict(sorted(Counter(SHIPPED.values()).items()))))
    print("re-derived on NEW     {}"
          .format(dict(sorted(Counter(new_map.values()).items()))))
    ch = [(n, SHIPPED[n], new_map[n]) for n in sorted(SHIPPED)
          if new_map.get(n, 1) != SHIPPED[n]]
    print("depths that change: {} of 100".format(len(ch)))
    print("   " + ", ".join("n{}:{}->{}".format(n, a, b) for n, a, b in ch))

    print("\n=== priced on the NEW medians; quality arm-mixed on the OOS arms ===")
    k1 = {n: 1 for n in SHIPPED}
    base_rf = rf_of(new_rows, k1, dtan, dpass, near)
    print("{:>12}{:>12}{:>11}{:>10}{:>10}{:>10}{:>10}"
          .format("map", "RF total", "RF vs k=1", "qual s1", "qual s2",
                  "NET s1", "NET s2"))
    for lbl, m in (("k=1 anchor", k1), ("SHIPPED", SHIPPED),
                   ("re-derived", new_map)):
        rf = rf_of(new_rows, m, dtan, dpass, near)
        drf = 100 * (base_rf - rf) / base_rf
        q1, mv1, w1, _ = quality(m, "s1")
        q2, mv2, w2, _ = quality(m, "s2")
        print("{:>12}{:>12.6f}{:>+10.4f}%{:>+9.4f}%{:>+9.4f}%{:>+9.4f}%{:>+9.4f}%"
              .format(lbl, rf, drf, q1, q2, q1 + drf, q2 + drf))
    for lbl, m in (("SHIPPED", SHIPPED), ("re-derived", new_map)):
        a, b = quality(m, "s1"), quality(m, "s2")
        print("  {:>10}  moved/worse  s1 {}/{}   s2 {}/{}   of {} cases"
              .format(lbl, a[1], a[2], b[1], b[2], a[3]))

    print("\n=== sensitivity: the FINAL medians will move again ===")
    print("{:>11}{:>22}{:>11}{:>10}{:>10}"
          .format("medians x", "depths", "RF vs k=1", "NET s1", "NET s2"))
    for sc in (1.30, 1.15, 1.00, 0.90, 0.80, 0.70):
        m = build(new_rows, dtan, dpass, near, scale=sc)
        rf = rf_of(new_rows, m, dtan, dpass, near)
        drf = 100 * (base_rf - rf) / base_rf
        q1, _, _, _ = quality(m, "s1")
        q2, _, _, _ = quality(m, "s2")
        print("{:>10.2f}x{:>22}{:>+10.4f}%{:>+9.4f}%{:>+9.4f}%"
              .format(sc, str(dict(sorted(Counter(m.values()).items()))),
                      drf, q1 + drf, q2 + drf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
