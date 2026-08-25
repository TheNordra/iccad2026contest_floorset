"""L223 timing - min-of-N on both sides, with the control band as the noise floor.

A single run pair put REFINE 4->2 at a 21.5% profile-wall cut on n>100 and the
joint arm at NET +4.233%, 0.09% past the rank-2 threshold. The same pair put the
UNTOUCHED n<=100 control band at -2.9%, where it must read 0. So the noise is
comparable to the margin, and a rank claim from one pair would be a claim about
this box's scheduler.

Two disciplines, both from the ledger:

  min-of-N     the minimum over repeats is the least-contended observation, and
               contention only ever adds time. Applied to BOTH sides of the
               ratio: a quotient of two noisy numbers is noisier than either, so
               re-running only the treated arm would bias the result in its own
               favour.
  control band n<=100 never changes REFINE, so whatever it reads after the
               min-of-N is the residual noise the verdict has to clear.

  <python> l223_timing.py
"""
import collections
import json
import math
import os
import sys
from pathlib import Path

DIR = Path(__file__).parent
ARMS = ("r4", "r2", "k8r2")
REPS = (1, 2, 3)


def load(fn):
    f = DIR / fn
    if not f.exists():
        return None
    per = collections.defaultdict(dict)
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) == 3:
            per[int(p[0])][int(p[1])] = float(p[2])
    return per


def wall(per, n):
    v = list(per[n].values())
    return max(max(v), sum(v) / 48)


def main():
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l204_routea_risk as m
    POOL0 = dict(m.POOL)
    TOTW = sum(r["w"] for r in m.ROWS)
    THR = 0.7 ** (1 / 0.3)
    BETA, QP = m.BETA, m.Q_POOL_FULL
    AMP = 2.41
    gate = {n: 1 if POOL0[n] + m.DT[n] <= THR * m.MED[n] * 1.2 else 0
            for n in m.NS}

    PHI = {}
    for tag in ("r1", "r2"):
        pf = load("l205_prof_{}.txt".format(tag))
        rf = DIR / "results_L205_{}.json".format(tag)
        if not (pf and rf.exists()):
            continue
        d = {r["block_count"]: r
             for r in json.load(open(rf))["test_results"]}
        for n in pf:
            if n in d and d[n]["runtime_seconds"] > 0:
                PHI.setdefault(n, []).append(
                    min(1.0, max(pf[n].values()) / d[n]["runtime_seconds"]))
    PHI = {n: sum(v) / len(v) for n, v in PHI.items()}

    # min over repeats, per (arm, block count)
    best = {}
    got = {}
    for a in ARMS:
        got[a] = 0
        for i in REPS:
            per = load("l223_prof_{}_{}.txt".format(a, i))
            if not per:
                continue
            got[a] += 1
            for n in per:
                w = wall(per, n)
                key = (a, n)
                if key not in best or w < best[key]:
                    best[key] = w
    print("=" * 78)
    print("repeats found: " + "  ".join("{}={}".format(a, got[a]) for a in ARMS))
    if min(got.values()) < 2:
        print("!! fewer than 2 repeats on some arm -- min-of-N is not yet "
              "meaningful")
    print("=" * 78)

    W = lambda n: math.exp(n / 12.0)                           # noqa: E731
    res = {}
    for a in ARMS:
        f = DIR / "results_L223_{}_1.json".format(a)
        if f.exists():
            res[a] = {r["block_count"]: r
                      for r in json.load(open(f))["test_results"]}
    if "r4" not in res:
        print("no baseline results")
        return 1
    SW = sum(W(n) for n in res["r4"])
    q0 = sum(W(n) * res["r4"][n]["cost"] for n in res["r4"]) / SW

    print("{:<10}{:>13}{:>15}{:>11}{:>11}{:>10}{:>6}"
          .format("arm", "wall n>100", "CONTROL n<=100", "quality", "NET",
                  "graded", "rank"))
    print("-" * 78)
    for a in ARMS:
        if ("r4", 21) not in best or (a, 21) not in best:
            continue
        cuts = {n: 1 - best[(a, n)] / best[("r4", n)]
                for n in m.NS if (a, n) in best and ("r4", n) in best}
        hv = [c for n, c in cuts.items() if n > 100]
        lo = [c for n, c in cuts.items() if n <= 100]
        q = 100 * (q0 - sum(W(n) * res[a][n]["cost"]
                            for n in res["r4"]) / SW) / q0 if a in res else 0.0
        num = 0.0
        newpool = {}
        for n in m.NS:
            newpool[n] = POOL0[n] * (1 - cuts.get(n, 0.0) * PHI.get(n, 0.878))
            t = newpool[n] + (m.DT[n] if gate.get(n, 0) else 0.0)
            r = m.ROW[n]
            num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        num += sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
                   for r in m.ROWS if r["n"] not in m.POOL)
        for n in m.NS:
            m.POOL[n] = newpool[n]
        b = sum(m.qual_pern(gate, s) for s in ("s1", "s2")) / 2
        for n in m.NS:
            m.POOL[n] = POOL0[n]
        net = (b + (q * AMP if q < 0 else q) + QP
               + 100 * (m.BETA_NUM - num / TOTW) / m.BETA_NUM)
        gr = BETA * (1 - net / 100)
        print("{:<10}{:>+12.2f}%{:>+14.2f}%{:>+10.4f}%{:>+10.3f}%{:>10.5f}{:>6}"
              .format(a, 100 * sum(hv) / len(hv) if hv else 0.0,
                      100 * sum(lo) / len(lo) if lo else 0.0, q, net, gr,
                      m.rank_of(gr)))
    print("-" * 78)
    print("CONTROL is the residual noise after min-of-N: REFINE never changes")
    print("below n=101, so anything it reads is the floor the verdict must")
    print("clear. r3 = 0.89933   r2 = 0.88819")
    print()
    print("!! quality is IN SET. The OOS runs are what decide this; M49 picked")
    print("   REFINE=4 by a strictly selection-preserving derivation, so going")
    print("   below it changes selections BY CONSTRUCTION and '2 movers in set'")
    print("   is exactly the number this ledger has watched grow out of sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
