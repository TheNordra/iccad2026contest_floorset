"""L342 -- re-run L230's rb table with RF-SAFE as a candidate.

WHY. HANDOFF_2026-08-27 §5(a) forbids re-widening the LP gate: "every widening
candidate is negative at rb = 0.82, inside the honest interval". RF-SAFE widens
71 -> 83 and its derivation (l311/l312) never mentions rb -- it prices on f_eff
(machine speed x median drift) instead. Nobody ran L230's instrument against it.

Worse, L230 §3 named {90, 107, 114, 120} as "what turns it negative above
rb = 0.80", and RF-SAFE ungates 107, 114 and 120 -- three of those four. L313
then independently found n=114 to be the single case that loses 0.2255 pp on
Linux. Two methods, two failure modes, same case.

CONTROL FIRST. The published L230 rows are reproduced before the new row is
believed. If they do not match L230_REPORT.md the new number means nothing.

Run:  <python> l342_rb_rfsafe.py
"""
import ast
import re
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l203_marginal_gate as G                                    # noqa: E402
import l230_gate as L                                             # noqa: E402

RBS = (0.72, 0.7682, 0.80, 0.82)


def gate_from(path, pattern=r"_L196_LPGATE\s*=\s*(\{.*?\n\})"):
    s = Path(path).read_text(encoding="utf-8")
    return ast.literal_eval(re.search(pattern, s, re.S).group(1))


def score_at(g, rb, dt_band):
    """L230's score(), with the n>100 pool band scaled by rb."""
    old_pool, old_dt = G.POOL, G.DT
    G.POOL = {n: (BASE_POOL[n] * rb if n > 100 else BASE_POOL[n]) for n in G.NS}
    G.DT = {n: (BASE_DT[n] * dt_band if n > 100 else BASE_DT[n]) for n in G.NS}
    acc = []
    for fit, test in (("s1", "s2"), ("s2", "s1")):
        acc.append(G.qual_pern(g, test) + G.Q_POOL_FULL + G.rf_at(g, 1.0))
    G.POOL, G.DT = old_pool, old_dt
    return sum(acc) / 2


if __name__ == "__main__":
    print(__doc__)
    L._load_tables()

    # the de-biased band factors L230 derived from its own batch
    A, _ = L.minof("A"); B, _ = L.minof("B")
    C, _ = L.minof("C"); D, _ = L.minof("D")
    rho = {n: A[n] / B[n] for n in A if n in B}
    dtA = {n: max(1e-6, C[n] - A[n]) for n in A if n in C}
    dtB = {n: max(1e-6, D[n] - B[n]) for n in B if n in D}
    sig = {n: dtA[n] / dtB[n] for n in dtA if n in dtB}
    cr = L.band_stat(rho, 20, 100)[0]
    ok = [n for n in sig if n > 100 and dtB[n] > 0.05 * B[n] and dtA[n] > 0.05 * A[n]]
    sb = st.median(sig[n] for n in ok) if ok else 1.0
    rb_meas = L.band_stat(rho, 100, 121)[0] / cr
    print("measured rb = %.4f   dt band = %.4f   (L230 reported rb 0.7682)\n"
          % (rb_meas, sb))

    BASE_POOL = dict(G.POOL)
    BASE_DT = dict(G.DT)

    SHIPPED_L230 = dict(L.SHIPPED)          # live source = D's 71
    L228 = dict(L.L228)
    RFSAFE = gate_from(DIR / "build_submission.RFSAFE" / "cadc1075" / "op_wrapper.py")

    print("table identities")
    print("  live source (= D)      %3d on" % sum(SHIPPED_L230.values()))
    print("  l228_gate_new.txt      %3d on" % sum(L228.values()))
    print("  RF-SAFE (uploaded)     %3d on" % sum(RFSAFE.values()))
    same = sum(1 for n in SHIPPED_L230 if bool(SHIPPED_L230[n]) == bool(L228.get(n)))
    print("  live vs l228 agree on  %3d / %d block counts" % (same, len(SHIPPED_L230)))
    addl = sorted(n for n in RFSAFE if RFSAFE[n] and not SHIPPED_L230.get(n))
    print("  RF-SAFE adds: %s" % addl)
    flagged = {90, 107, 114, 120}
    print("  of L230's flagged {90,107,114,120}: RF-SAFE turns ON %s"
          % sorted(flagged & set(addl)))
    print("  ... and D already has ON: %s"
          % sorted(n for n in flagged if SHIPPED_L230.get(n)))

    print()
    print("=== NET (%), mean of both OOS directions, vs pool ratio rb ===")
    hdr = "{:<34}{:>5}".format("table", "on") + "".join("%11s" % ("rb=%.4g" % r) for r in RBS)
    print(hdr); print("-" * len(hdr))
    rows = [("live _L196_LPGATE (= D, uploaded->replaced)", SHIPPED_L230),
            ("l228_gate_new.txt", L228),
            ("RF-SAFE (what is on the Drive now)", RFSAFE)]
    res = {}
    for lbl, g in rows:
        vals = [score_at(g, r, sb) for r in RBS]
        res[lbl] = vals
        print("{:<34}{:>5}".format(lbl[:34], sum(g.values()))
              + "".join("%+11.3f" % v for v in vals))

    print()
    print("=== delta vs the live table (what widening 71->83 buys/costs) ===")
    base = res["live _L196_LPGATE (= D, uploaded->replaced)"]
    for lbl in ("l228_gate_new.txt", "RF-SAFE (what is on the Drive now)"):
        print("{:<34}{:>5}".format(lbl[:34], "")
              + "".join("%+11.3f" % (a - b) for a, b in zip(res[lbl], base)))
    print()
    print("§5(a)'s bar: must not be negative anywhere in the honest interval")
    print("[0.72, 0.82]. The rightmost column is the test.")
