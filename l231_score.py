"""L231 score - the mid band and the two never-priced post-processing budgets.

Same discipline as l223_timing.py:
  * min-of-3 wall per (arm, block count); wall = max(max d_p, sum d_p / 48)
  * n<=60 is the CONTROL band for the mid-band arms (REFINE is untouched there),
    so whatever it reads after min-of-3 is the noise floor the verdict clears
  * quality here is IN SET and in-set quality has been optimistic every single
    time in this ledger, so a negative in-set delta is amplified by the
    MEASURED OOS/in-set ratio (L223: 2.41x) before it enters NET

HARD GATE. Arm m6 carries the shipped mid band (6) and the shipped heavy band
(2) through L219_REFINE_TABLE, so it must reproduce results_L227_det1.json
100/100 on cost. If it does not, the probe tree and the shipping tree have
drifted and every number below is measuring something that is not shipping.

  <python> l231_score.py
"""
import collections
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent
ARMS = ("m6", "m4", "m3", "m2", "mpc")
BASE = "m6"
REPS = (1, 2, 3)
CORES = 48
AMP = 2.41                      # L223: measured OOS / in-set quality-cost ratio


def load_prof(fn):
    f = DIR / fn
    if not f.exists():
        return None
    per = collections.defaultdict(dict)
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) == 3:
            per[int(p[0])][int(p[1])] = float(p[2])
    return per


def wall(d):
    v = list(d.values())
    return max(max(v), sum(v) / CORES)


def main():
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l204_routea_risk as m

    # ---- the pool the shipped package actually has today --------------------
    pf = DIR / "l230_pool_new.json"
    if pf.exists():
        d = json.load(open(pf))
        POOL0 = {int(k): v for k, v in d["POOL"].items()}
        DT = {int(k): v for k, v in d["DT"].items()}
        src = "l230_pool_new.json (post-REFINE, de-biased)"
    else:
        POOL0, DT = dict(m.POOL), dict(m.DT)
        src = "!! l204 PRE-REFINE pool -- run l230_gate.py first"
    print("pool inputs: {}".format(src))
    TOTW = sum(r["w"] for r in m.ROWS)
    THR = 0.7 ** (1 / 0.3)
    gate = eval_shipped_gate()
    print("LP gate: {} of {} block counts on".format(
        sum(1 for n in m.NS if gate.get(n, 1)), len(m.NS)))

    # ---- min-of-3 walls -----------------------------------------------------
    best, got = {}, collections.Counter()
    for a in ARMS:
        for i in REPS:
            per = load_prof("l231_prof_{}_{}.txt".format(a, i))
            if not per:
                continue
            got[a] += 1
            for n in per:
                w = wall(per[n])
                if (a, n) not in best or w < best[(a, n)]:
                    best[(a, n)] = w
    print("repeats: " + "  ".join("{}={}".format(a, got[a]) for a in ARMS))
    if got[BASE] == 0:
        print("no baseline arm yet")
        return 1

    # ---- PHI, recomputed on THIS tree --------------------------------------
    PHI = {}
    f = DIR / "results_L231_{}_1.json".format(BASE)
    per = load_prof("l231_prof_{}_1.txt".format(BASE))
    if f.exists() and per:
        d = {r["block_count"]: r for r in json.load(open(f))["test_results"]}
        for n in per:
            if n in d and d[n]["runtime_seconds"] > 0:
                PHI[n] = min(1.0, wall(per[n]) / d[n]["runtime_seconds"])
    phid = st.median(PHI.values()) if PHI else 0.878
    print("PHI (profile phase share of case wall) median {:.3f} on {} cases"
          .format(phid, len(PHI)))

    # ---- hard gate ----------------------------------------------------------
    ref = DIR / "results_L227_det1.json"
    if f.exists() and ref.exists():
        a = {r["block_count"]: r["cost"] for r in json.load(open(f))["test_results"]}
        b = {r["block_count"]: r["cost"] for r in json.load(open(ref))["test_results"]}
        same = sum(1 for n in a if n in b and abs(a[n] - b[n]) < 1e-12)
        print("HARD GATE  m6 vs results_L227_det1.json: {}/{} identical   {}"
              .format(same, len(a), "PASS" if same == len(a) else "FAIL"))
        if same != len(a):
            print("  !! the probe tree is not the shipping tree -- STOP.")
            bad = sorted(n for n in a if n in b and abs(a[n] - b[n]) >= 1e-12)
            print("  differing block counts:", bad[:20])

    # ---- score --------------------------------------------------------------
    res = {}
    for a in ARMS:
        p = DIR / "results_L231_{}_1.json".format(a)
        if p.exists():
            res[a] = {r["block_count"]: r for r in json.load(open(p))["test_results"]}
    W = lambda n: math.exp(n / 12.0)                              # noqa: E731
    SW = sum(W(n) for n in res[BASE])
    q0 = sum(W(n) * res[BASE][n]["cost"] for n in res[BASE]) / SW
    qbase = sum(m.qual_pern(gate, s) for s in ("s1", "s2")) / 2

    print()
    print("{:<6}{:>11}{:>11}{:>11}{:>11}{:>10}{:>10}{:>6}"
          .format("arm", "wall 21-60", "61-100", "101-120", "quality",
                  "NET", "graded", "rank"))
    print("-" * 80)
    out = {}
    for a in ARMS:
        if (a, 21) not in best or a not in res:
            continue
        cuts = {n: 1 - best[(a, n)] / best[(BASE, n)]
                for n in m.NS if (a, n) in best and (BASE, n) in best}
        bands = []
        for lo, hi in ((20, 60), (60, 100), (100, 121)):
            v = [c for n, c in cuts.items() if lo < n <= hi]
            bands.append(100 * sum(v) / len(v) if v else 0.0)
        q = 100 * (q0 - sum(W(n) * res[a][n]["cost"] for n in res[BASE]) / SW) / q0
        num = 0.0
        for n in m.NS:
            t = POOL0[n] * (1 - cuts.get(n, 0.0) * PHI.get(n, phid)) \
                + (DT[n] if gate.get(n, 1) else 0.0)
            r = m.ROW[n]
            num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        num += sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
                   for r in m.ROWS if r["n"] not in POOL0)
        rf = 100 * (m.BETA_NUM - num / TOTW) / m.BETA_NUM
        net = qbase + (q * AMP if q < 0 else q) + m.Q_POOL_FULL + rf
        gr = m.BETA * (1 - net / 100)
        out[a] = (net, gr, q, rf)
        print("{:<6}{:>+10.2f}%{:>+10.2f}%{:>+10.2f}%{:>+10.4f}%{:>+9.3f}%"
              "{:>10.5f}{:>6}".format(a, bands[0], bands[1], bands[2], q, net,
                                      gr, m.rank_of(gr)))
    print("-" * 80)
    print("wall columns are the CUT (positive = faster). For the m-arms 21-60 is")
    print("the control and must read ~0; 101-120 must also read ~0 (all arms")
    print("carry heavy=2). For pc every band moves by design.")
    print("quality is IN SET; a negative one is amplified x{} into NET.".format(AMP))
    print("r2 = 0.88819   r3 = 0.89933")
    return 0


def eval_shipped_gate():
    import re
    src = (DIR / "optimizer_constructive.py").read_text(encoding="utf-8")
    return eval(re.search(r"^_L196_LPGATE = \{.*?^\}", src, re.S | re.M)
                .group(0).split("=", 1)[1])


if __name__ == "__main__":
    raise SystemExit(main())
