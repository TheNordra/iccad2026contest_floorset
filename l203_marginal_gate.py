"""L203 - the one-parameter gate family is provably the wrong SHAPE.

L196/L196b swept `gate_s(n) = 1 iff t_pool(n) + dt_lp(n) <= 0.3046*M(n)*s` and
cross-validated s. That search was over the wrong set. Both halves of the
objective are SEPARABLE in n:

    qual(g) = SUM_n  QGAIN(n) * g[n]                      (affine in g)
    rf(g)   = rf(0) - SUM_n  RFCOST(n) * g[n]             (affine in g)
    NET(g)  = const + SUM_n [QGAIN(n) - RFCOST(n)] * g[n]

There is no shared budget -- the runtime factor is applied per case, not to a
pool -- so this is not a knapsack and needs no ordering. The optimal gate is
the sign test

    g[n] = 1  iff  QGAIN(n) > RFCOST(n)

and any single-threshold family in `t/M` can only reach it by accident. s=1.2
is the best member of a family that does not contain the answer.

THE CATCH, and it is the whole reason this file cross-validates: RFCOST(n) is
known exactly (measured seconds, published medians), but QGAIN(n) has to be
ESTIMATED, from 2-4 OOS cases per block count. A sign test on a noisy estimate
selects for noise -- it turns on exactly the n where the estimate is
optimistically wrong. That is L127's tally fitting with a different label, and
this ledger has been four-for-four on offline advantages shrinking or reversing
out of sample.

So every row here is FIT ON ONE SAMPLE AND SCORED ON THE OTHER, both
directions, reported separately. The same-sample row is printed too, labelled
ORACLE, purely as the ceiling neither honest row may be confused with.

Three estimators of QGAIN(n), increasingly willing to trade resolution for
variance:

    raw        the per-n mean. Maximum resolution, maximum noise.
    smooth-K   mean over the +-K neighbouring block counts. n is a smooth axis
               for both cost and value, so a neighbour is evidence.
    sign-K     smooth-K, but only allowed to flip a block count ON when the
               raw and smoothed estimates AGREE on the sign.

  <python> l203_marginal_gate.py
"""
import json
import math
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
THR = 0.7 ** (1 / 0.3)
BETA = 0.9265861161320369
Q_POOL_FULL = 0.3976 + 2.6588
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333),
         (6, 0.9552271810705998)]


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t - 1e-9) + 1


def ins(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


def oos(fn):
    return {r["test_id"]: r for r in json.load(open(DIR / fn))["test_results"]}


# ---- the measured side: seconds, medians, weights ---------------------------
wm, _ = ins("_l181_m73.json")
wo, _ = ins("_l181_cur.json")
wk, _ = ins("_l189_k1.json")
ROWS = M.rows_new()
W = sum(r["w"] for r in ROWS)
BETA_NUM = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in ROWS) / W

POOL, DT, MED, ROW = {}, {}, {}, {}
for r in ROWS:
    n = r["n"]
    if not wm.get(n):
        continue
    k = r["t"] / wm[n]                       # this box -> grader, machine-free
    POOL[n] = wo[n] * k
    DT[n] = max(0.0, (wk[n] - wo[n]) * k)
    MED[n] = r["med"]
    ROW[n] = r
NS = sorted(POOL)

A = {s: (oos("l194_{}_fulloff.json".format(s)),
         oos("l192_{}_full.json".format(s))) for s in ("s1", "s2")}


def rfcost(ra):
    """RFCOST(n) -- what giving the LP to block count n costs, PER CASE AT n.

    ⚠️ THE UNITS ARE THE WHOLE PROBLEM, and the ORACLE row is what caught it.
    Under exact separability the same-sample sign test cannot lose to any other
    gate; the first version of this file had ORACLE scoring BELOW the shipped
    s=1.2, which is arithmetically impossible and meant the two sides were
    being compared across different corpora. RFCOST lives on the 100-row beta
    corpus (exactly one case per block count, W = sum of its weights); QGAIN
    was being read off a 240-case sample carrying 2-4 replicates per block
    count. Dividing one by 240-case sums and the other by 100-row sums makes
    the comparison depend on how many replicates a block count happened to get.

    Put both per-case-at-n and the weights cancel exactly:

        turn n on  iff  meangain(n) / qo  >  q_n * delta_n / BETA_NUM

    w_n and W appear on both sides and drop out. That is the test below."""
    out = {}
    for n in NS:
        r = ROW[n]
        base = max(0.7, (POOL[n] * ra / r["med"]) ** 0.3)
        with_lp = max(0.7, ((POOL[n] * ra + DT[n]) / r["med"]) ** 0.3)
        out[n] = 100.0 * r["q"] * (with_lp - base) / BETA_NUM
    return out


def rf_at(g, ra):
    num = sum(ROW[n]["w"] * ROW[n]["q"]
              * max(0.7, ((POOL[n] * ra + (DT[n] if g.get(n, 0) else 0.0))
                          / ROW[n]["med"]) ** 0.3) for n in NS)
    # the rows without a wall measurement keep their beta contribution
    num += sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in ROWS if r["n"] not in POOL)
    return 100 * (BETA_NUM - num / W) / BETA_NUM


def qgain(sample):
    """QGAIN(n) -- the LP's quality gain PER CASE at block count n, in the same
    per-case units as rfcost(): 100 * meangain(n) / qo."""
    off, on = A[sample]
    ids = sorted(set(off) & set(on))
    w = lambda i: math.exp(off[i]["n"] / 12.0)                     # noqa: E731
    sw = sum(w(i) for i in ids)
    qo = sum(w(i) * off[i]["cost"] for i in ids) / sw
    acc, cnt = {}, {}
    for i in ids:
        n = off[i]["n"]
        acc[n] = acc.get(n, 0.0) + (off[i]["cost"] - on[i]["cost"])
        cnt[n] = cnt.get(n, 0) + 1
    return {n: 100.0 * (acc[n] / cnt[n]) / qo for n in acc}, qo


def qual_at(g, sample):
    """L196's scorer, kept verbatim so every number here is comparable to the
    handoff table: every one of the 240 OOS cases weighted exp(n/12)."""
    off, on = A[sample]
    ids = sorted(set(off) & set(on))
    w = lambda i: math.exp(off[i]["n"] / 12.0)                     # noqa: E731
    sw = sum(w(i) for i in ids)
    qo = sum(w(i) * off[i]["cost"] for i in ids) / sw
    qg = sum(w(i) * (on if g.get(off[i]["n"], 0) else off)[i]["cost"]
             for i in ids) / sw
    return 100 * (qo - qg) / qo


def qual_pern(g, sample):
    """The scorer that matches the corpus the RUNTIME side is defined on.

    The graded set is 100 cases, one per block count 21..120 -- that is what
    ROWS is, what the published medians are indexed by, and what RFCOST is
    computed over. The 240-case OOS samples carry 2-4 replicates per block
    count, so L196's scorer silently gives a block count that drew 4 cases
    twice the influence of one that drew 2. Here the replicates are used for
    what they are -- repeated measurements of the same n -- and the corpus
    weighting comes from the grader's structure instead of the sample's."""
    off, on = A[sample]
    ids = sorted(set(off) & set(on))
    so, sn, cnt = {}, {}, {}
    for i in ids:
        n = off[i]["n"]
        so[n] = so.get(n, 0.0) + off[i]["cost"]
        sn[n] = sn.get(n, 0.0) + on[i]["cost"]
        cnt[n] = cnt.get(n, 0) + 1
    ns = [n for n in cnt if n in ROW]
    sw = sum(ROW[n]["w"] for n in ns)
    qo = sum(ROW[n]["w"] * so[n] / cnt[n] for n in ns) / sw
    qg = sum(ROW[n]["w"] * (sn[n] if g.get(n, 0) else so[n]) / cnt[n]
             for n in ns) / sw
    return 100 * (qo - qg) / qo


def smooth(v, k):
    out = {}
    for n in NS:
        vals = [v[m] for m in range(n - k, n + k + 1) if m in v]
        out[n] = sum(vals) / len(vals) if vals else 0.0
    return out


def time_gate(s):
    return {n: 1 if POOL[n] + DT[n] <= THR * MED[n] * s else 0 for n in NS}


def marginal_gate(v, cost, raw=None):
    g = {}
    for n in NS:
        ok = v.get(n, 0.0) > cost[n]
        if raw is not None:                 # sign-agreement variant
            ok = ok and raw.get(n, 0.0) > 0.0
        g[n] = 1 if ok else 0
    return g


def report(label, g, ra, sample, out):
    r_ = rf_at(g, ra)
    q = qual_pern(g, sample)                 # primary: the grader's corpus
    q240 = qual_at(g, sample)                # L196's scorer, for comparability
    net = q + Q_POOL_FULL + r_
    gr = BETA * (1 - net / 100.0)
    out.append((label, int(100 * sum(g.values()) / len(g)), q, q240, r_, net,
                gr, rank_of(gr)))


def main():
    print(__doc__)
    RC = {1.0: rfcost(1.0), 0.68: rfcost(0.68)}
    QG = {s: qgain(s)[0] for s in ("s1", "s2")}
    ship = time_gate(1.2)

    for ra, tag in ((1.0, "route A NEUTRAL (the downside)"),
                    (0.68, "route A 0.68x (the bet)")):
        cost = RC[ra]
        print("=" * 86)
        print("=== {} ===".format(tag))
        print("{:<34}{:>6}{:>10}{:>10}{:>10}{:>10}{:>10}{:>6}"
              .format("gate", "on", "qual/n", "q(L196)", "RF", "NET",
                      "graded", "rank"))
        rows = []
        for fit, test in (("s1", "s2"), ("s2", "s1")):
            report("shipped time gate s=1.2  [{}]".format(test), ship, ra,
                   test, rows)
            raw = QG[fit]
            report("marginal raw       fit {}->{}".format(fit, test),
                   marginal_gate(raw, cost), ra, test, rows)
            for k in (2, 4, 8):
                report("marginal smooth-{} fit {}->{}".format(k, fit, test),
                       marginal_gate(smooth(raw, k), cost), ra, test, rows)
            report("marginal sign-4    fit {}->{}".format(fit, test),
                   marginal_gate(smooth(raw, 4), cost, raw), ra, test, rows)
            report("ORACLE marginal    fit {}->{}".format(test, test),
                   marginal_gate(QG[test], cost), ra, test, rows)
            rows.append(None)
        for r in rows:
            if r is None:
                print("-" * 86)
                continue
            lbl, on, q, q240, rf_, net, gr, rk = r
            print("{:<34}{:>5}%{:>+9.3f}%{:>+9.3f}%{:>+9.3f}%{:>+9.3f}%"
                  "{:>10.5f}{:>6}".format(lbl, on, q, q240, rf_, net, gr, rk))
        print()

    # ---- THE DECISION TABLE -------------------------------------------------
    # Everything above builds a gate in the regime it is scored in. That is not
    # a shippable object: the package carries ONE table and cannot know at
    # runtime whether route A delivered. The regime-matched rows are a ceiling,
    # not a candidate. Below, each candidate table is built ONCE and scored in
    # BOTH regimes, so the columns are what actually happens on the grader.
    print("=" * 86)
    print("THE DECISION TABLE -- one table, scored in both regimes")
    print("  (built on the FIT sample, scored on the OTHER; both directions)")
    print("  beta as graded {:.5f}.  r3 {:.5f}   r2 {:.5f}   r1 {:.5f}"
          .format(BETA, RANKS[2][1], RANKS[1][1], RANKS[0][1]))
    print()
    cands = [("time gate s=1.2  (SHIPPED)", lambda f: time_gate(1.2)),
             ("time gate s=1.0", lambda f: time_gate(1.0)),
             ("time gate s=1.5", lambda f: time_gate(1.5)),
             ("marginal @ra=1.00 smooth-4",
              lambda f: marginal_gate(smooth(QG[f], 4), RC[1.0])),
             ("marginal @ra=0.68 smooth-4",
              lambda f: marginal_gate(smooth(QG[f], 4), RC[0.68])),
             ("marginal @ra=0.84 smooth-4",
              lambda f: marginal_gate(smooth(QG[f], 4), rfcost(0.84)))]
    print("{:<28}{:>5}{:>10}{:>9}{:>6}{:>10}{:>9}{:>6}"
          .format("candidate", "on", "NEUTRAL", "graded", "rk",
                  "BET .68", "graded", "rk"))
    print("-" * 86)
    for lbl, mk in cands:
        acc = {}
        ons = []
        for fit, test in (("s1", "s2"), ("s2", "s1")):
            g = mk(fit)
            ons.append(sum(g.values()))
            for ra in (1.0, 0.68):
                net = qual_pern(g, test) + Q_POOL_FULL + rf_at(g, ra)
                acc.setdefault(ra, []).append(net)
        m = {ra: sum(v) / len(v) for ra, v in acc.items()}
        g1, g0 = BETA * (1 - m[1.0] / 100), BETA * (1 - m[0.68] / 100)
        print("{:<28}{:>4}{:>+10.3f}%{:>9.5f}{:>6}{:>+10.3f}%{:>9.5f}{:>6}"
              .format(lbl, int(sum(ons) / len(ons)), m[1.0], g1, rank_of(g1),
                      m[0.68], g0, rank_of(g0)))
    print("-" * 86)
    print("  NEUTRAL is the downside that must stay BELOW beta {:.5f}; the"
          .format(BETA))
    print("  margin there is the whole reason this package is a hedged bet and")
    print("  not an open one. BET .68 is the upside route A is supposed to buy.")

    print()
    print("WHAT THE ra=1.0 MARGINAL RULE CHANGES vs the shipped 63")
    cost = RC[1.0]
    for fit in ("s1", "s2"):
        g = marginal_gate(smooth(QG[fit], 4), cost)
        add = sorted(n for n in NS if g[n] and not ship[n])
        drop = sorted(n for n in NS if ship[n] and not g[n])
        print("  smooth-4 fit {}: on {}  (+{} added, -{} dropped)"
              .format(fit, sum(g.values()), len(add), len(drop)))
        print("      adds  {}".format(add[:18]))
        print("      drops {}".format(drop[:18]))
    print("\n  AGREEMENT between the two fits is the transfer test that matters:")
    a = marginal_gate(smooth(QG["s1"], 4), cost)
    b = marginal_gate(smooth(QG["s2"], 4), cost)
    agree = sum(1 for n in NS if a[n] == b[n])
    print("  the two independently-fitted ra=1.0 tables agree on {}/{} block "
          "counts.".format(agree, len(NS)))
    print("  A table fitted to noise would not: the s1 and s2 samples share no "
          "cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
