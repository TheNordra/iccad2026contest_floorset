"""L230 - rebuild _L196_LPGATE on the POST-REFINE tree, reproducibly.

HANDOFF_2026-08-26 §4 claims the rebuilt gate fires on 71 instead of 63 for
+4.748% against +4.265%, and ships the table in l228_gate_new.txt -- but the
derivation was inline and is not on disk. This file re-derives it, and repairs
a units problem the inline version could not have avoided.

  l203_marginal_gate.py builds POOL[n] from _l181_cur.json (full pool, LP off)
  and the box->grader factor k from _l181_m73.json, BOTH from the 2026-08-24
  batch. The 2026-08-25 arms run ~17.5% SLOWER than that batch in the n<=100
  control band -- where nothing changed. So a post-REFINE pool time read off an
  Aug-25 arm cannot be divided by an Aug-24 k; the offset lands straight in a
  threshold test.

  The fix is this ledger's own rule (memory: "正解是同機比值"): every new number
  enters as a RATIO measured in ONE batch, so the machine factor cancels.

      POOL_new[n] = POOL_old[n] * A[n]/B[n]          A REFINE=2, B REFINE=4
      DT_new[n]   = DT_old[n]   * (C[n]-A[n])/(D[n]-B[n])

  min-of-3 on both sides of both ratios. n<=100 is the control: REFINE is
  untouched there, so A/B must read 1.00 and whatever it does read is the
  estimator's own bias, printed next to every result.

  <python> l230_gate.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import l203_marginal_gate as G

DIR = Path(__file__).parent
REPS = (1, 2, 3)


def minof(tag):
    acc = {}
    for i in REPS:
        f = DIR / "results_L230_{}{}.json".format(tag, i)
        if not f.exists():
            continue
        for r in json.load(open(f))["test_results"]:
            acc.setdefault(r["block_count"], []).append(r["runtime_seconds"])
    return {n: min(v) for n, v in acc.items()}, (len(acc) and len(next(iter(acc.values()))))


def band_stat(d, lo, hi):
    v = [d[n] for n in d if lo < n <= hi]
    return (st.median(v), len(v)) if v else (float("nan"), 0)


def main():
    print(__doc__)
    A, nA = minof("A")
    B, nB = minof("B")
    C, nC = minof("C")
    D, nD = minof("D")
    print("arms: A(refine2,lpoff) {} reps | B(refine4,lpoff) {} | "
          "C(refine2,lp-all) {} | D(refine4,lp-all) {}".format(nA, nB, nC, nD))
    if min(len(A), len(B), len(C), len(D)) < 100:
        print("!! incomplete batch: |A|={} |B|={} |C|={} |D|={}"
              .format(len(A), len(B), len(C), len(D)))
        return 1

    rho = {n: A[n] / B[n] for n in A if n in B}
    global dtA, dtB
    dtA = {n: max(1e-6, C[n] - A[n]) for n in A if n in C}
    dtB = {n: max(1e-6, D[n] - B[n]) for n in B if n in D}
    sig = {n: dtA[n] / dtB[n] for n in dtA if n in dtB}

    print()
    print("=== the ratios, with their own control band ===")
    for lbl, d in (("rho  = pool  REFINE2/REFINE4", rho),
                   ("sig  = dt_LP REFINE2/REFINE4", sig)):
        m1, c1 = band_stat(d, 20, 100)
        m2, c2 = band_stat(d, 100, 121)
        print("  {:30s} n<=100 (CONTROL, must be 1.00) {:.3f} [{}]"
              "   n>100 {:.3f} [{}]".format(lbl, m1, c1, m2, c2))
    print("  the n<=100 column is the estimator's bias; the n>100 column is the")
    print("  effect. A cut that does not clear its own control band is noise.")

    # ---- BAND-level, not per-n ---------------------------------------------
    # REFINE is a BAND constant: every case with n>100 gets the same change, so
    # the effect has no per-n structure to estimate. Applying a per-n ratio
    # would inject this box's ~17% per-case noise into a threshold test that
    # has 0.12pp of rank margin. The dispersion printed below is the evidence:
    # if the treated band's spread is the same size as the untouched control
    # band's, the per-n detail IS the noise.
    #
    # dt_LP is worse still. It is a DIFFERENCE of two ~1s numbers, so its ratio
    # has near-zero denominators: 4 block counts came out with dtB <= 0 (the
    # LP-everywhere arm timed FASTER than LP-off) and the per-n ratio went to
    # 1e4, which is what a quotient of noise does. Band level or nothing.
    cr = band_stat(rho, 20, 100)[0]
    for lbl, d in (("rho", rho), ("sig", sig)):
        for lo, hi in ((20, 100), (100, 121)):
            v = sorted(d[n] for n in d if lo < n <= hi)
            if not v:
                continue
            print("  {} {:>3}-{:<3} dispersion p10 {:.3f}  p50 {:.3f}  p90 {:.3f}"
                  .format(lbl, lo + 1, hi, v[len(v) // 10], v[len(v) // 2],
                          v[-max(1, len(v) // 10)]))
    ok = [n for n in sig if n > 100 and dtB[n] > 0.05 * B[n] and dtA[n] > 0.05 * A[n]]
    sb = st.median(sig[n] for n in ok) if ok else 1.0
    rb = band_stat(rho, 100, 121)[0] / cr
    print("  BAND estimates: pool x{:.4f} on n>100 (control de-bias /{:.3f}),"
          " dt_LP x{:.4f} on {} well-conditioned block counts"
          .format(rb, cr, sb, len(ok)))
    POOL = {n: (G.POOL[n] * rb if n > 100 else G.POOL[n]) for n in G.NS}
    DT = {n: (G.DT[n] * sb if n > 100 else G.DT[n]) for n in G.NS}

    old_pool, old_dt = G.POOL, G.DT
    json.dump({"POOL": {str(k): v for k, v in POOL.items()},
               "DT": {str(k): v for k, v in DT.items()},
               "rho_control": cr, "pool_band": rb, "dt_band": sb},
              open(DIR / "l230_pool_new.json", "w"), indent=0)
    print("  wrote l230_pool_new.json (the post-REFINE inputs, de-biased)")
    print()
    print("=== the free budget, in grader seconds, where the score lives ===")
    print("{:>6}{:>10}{:>10}{:>10}{:>10}{:>10}"
          .format("band", "sum w%", "pool_old", "pool_new", "median", "slack"))
    for lo, hi in ((20, 60), (60, 100), (100, 121)):
        ns = [n for n in G.NS if lo < n <= hi]
        wtot = sum(G.ROW[n]["w"] for n in G.NS)
        w = 100 * sum(G.ROW[n]["w"] for n in ns) / wtot
        po = sum(old_pool[n] for n in ns)
        pn = sum(POOL[n] for n in ns)
        md = sum(G.MED[n] for n in ns)
        sl = st.median(G.THR * G.MED[n] / POOL[n] for n in ns)
        print("{:>6}{:>9.1f}%{:>10.1f}{:>10.1f}{:>10.1f}{:>9.2f}x"
              .format("{}-{}".format(lo + 1, hi), w, po, pn, md, sl))
    print("  slack = 0.3046*M/pool = how much MORE wall a case can spend at")
    print("  ZERO RF cost. >1 means the case sits on the floor with room.")

    # ---- score every candidate table, both samples, both directions ---------
    def score(g):
        acc = {}
        for fit, test in (("s1", "s2"), ("s2", "s1")):
            gg = g(fit) if callable(g) else g
            net = G.qual_pern(gg, test) + G.Q_POOL_FULL + G.rf_at(gg, 1.0)
            acc[test] = (net, sum(gg.values()))
        m = sum(v[0] for v in acc.values()) / 2
        on = sum(v[1] for v in acc.values()) // 2
        gr = G.BETA * (1 - m / 100.0)
        return on, m, gr, G.rank_of(gr)

    ship = dict(SHIPPED)
    l228 = dict(L228)
    QG = {s: G.qgain(s)[0] for s in ("s1", "s2")}

    for tag, P, T in (("OLD pool times (l203 inputs, pre-REFINE)", old_pool, old_dt),
                      ("NEW pool times (post-REFINE, this batch)", POOL, DT)):
        G.POOL, G.DT = P, T
        cost = G.rfcost(1.0)
        print()
        print("=" * 84)
        print("=== scored at {} ===".format(tag))
        print("{:<40}{:>5}{:>11}{:>10}{:>6}".format(
            "table", "on", "NET", "graded", "rank"))
        print("-" * 84)
        rows = [("shipped _L196_LPGATE", ship),
                ("l228_gate_new.txt (handoff §4)", l228),
                ("time gate s=1.2 recomputed", G.time_gate(1.2))]
        for s in (1.0, 1.1, 1.15, 1.25, 1.3, 1.4, 1.5):
            rows.append(("time gate s={}".format(s), G.time_gate(s)))
        rows.append(("marginal smooth-4 (fit->other)",
                     lambda f: G.marginal_gate(G.smooth(QG[f], 4), cost)))
        for lbl, g in rows:
            on, m, gr, rk = score(g)
            print("{:<40}{:>5}{:>+11.3f}%{:>10.5f}{:>6}".format(lbl, on, m, gr, rk))
    G.POOL, G.DT = old_pool, old_dt

    # ---- what the recomputed table actually changes -------------------------
    G.POOL, G.DT = POOL, DT
    new = G.time_gate(1.2)
    add = sorted(n for n in G.NS if new[n] and not ship.get(n, 1))
    drop = sorted(n for n in G.NS if ship.get(n, 1) and not new[n])
    print()
    print("recomputed s=1.2 vs shipped: +{} added {}  -{} dropped {}"
          .format(len(add), add, len(drop), drop))
    a2 = sorted(n for n in G.NS if l228.get(n) and not ship.get(n, 1))
    d2 = sorted(n for n in G.NS if ship.get(n, 1) and not l228.get(n))
    print("l228 (handoff)      vs shipped: +{} added {}  -{} dropped {}"
          .format(len(a2), a2, len(d2), d2))
    agree = sum(1 for n in G.NS if bool(new[n]) == bool(l228.get(n)))
    print("this derivation agrees with l228_gate_new.txt on {}/{} block counts"
          .format(agree, len(G.NS)))
    G.POOL, G.DT = old_pool, old_dt
    return 0


SHIPPED = {}
L228 = {}


def _load_tables():
    import re
    src = (DIR / "optimizer_constructive.py").read_text(encoding="utf-8")
    SHIPPED.update(eval(re.search(r"^_L196_LPGATE = \{.*?^\}", src,
                                  re.S | re.M).group(0).split("=", 1)[1]))
    L228.update(eval("{" + (DIR / "l228_gate_new.txt").read_text() + "}"))


if __name__ == "__main__":
    _load_tables()
    raise SystemExit(main())
