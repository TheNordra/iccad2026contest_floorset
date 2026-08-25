"""L232 score - what does buying quality BACK cost, now that REFINE freed wall?

Every arm here ADDS candidates or ADDS refinement. Two things follow that do not
hold for the cut-side experiments:

  * quality can only improve. Selection is over a superset, and M76/M77 measured
    the proxy as oracle-perfect on heterogeneous candidates, so an added profile
    either wins or is ignored. An in-set quality GAIN is therefore not the
    optimistic-in-set trap; it is a floor. (L167 transferred at 75-122% OOS.)
  * the wall model has to include the SERIAL PROXY. `wall = max(max d_p, sum
    d_p/48)` covers only the C++ pool; L167's kill was the ~71 ms/profile of
    pure-Python proxy on the main thread, which that model cannot see. So the
    runtime side is taken from the WHOLE-CASE runtime ratio, and the gap between
    the two is printed -- that gap IS the serial cost.

  <python> l232_score.py
"""
import collections
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent
ARMS = ("base", "full", "hint", "lensd")
BASE = "base"
REPS = (1, 2, 3)
CORES = 48


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
    import re

    pf = DIR / "l230_pool_new.json"
    if not pf.exists():
        print("run l230_gate.py first"); return 1
    d = json.load(open(pf))
    POOL = {int(k): v for k, v in d["POOL"].items()}
    DT = {int(k): v for k, v in d["DT"].items()}
    gate = eval(re.search(r"^_L196_LPGATE = \{.*?^\}",
                          (DIR / "optimizer_constructive.py").read_text(encoding="utf-8"),
                          re.S | re.M).group(0).split("=", 1)[1])
    TOTW = sum(r["w"] for r in m.ROWS)
    T0 = {n: POOL[n] + (DT[n] if gate.get(n, 1) else 0.0) for n in m.NS}

    # ---- min-of-3, whole case AND profile phase ----------------------------
    tmin, pmin, npro, got = {}, {}, {}, collections.Counter()
    for a in ARMS:
        for i in REPS:
            f = DIR / "results_L232_{}_{}.json".format(a, i)
            if not f.exists():
                continue
            got[a] += 1
            for r in json.load(open(f))["test_results"]:
                n = r["block_count"]
                k = (a, n)
                if k not in tmin or r["runtime_seconds"] < tmin[k]:
                    tmin[k] = r["runtime_seconds"]
            per = load_prof("l232_prof_{}_{}.txt".format(a, i))
            if per:
                for n in per:
                    w = wall(per[n])
                    if (a, n) not in pmin or w < pmin[(a, n)]:
                        pmin[(a, n)] = w
                    npro[(a, n)] = len(per[n])
    print("repeats: " + "  ".join("{}={}".format(a, got[a]) for a in ARMS))
    if got[BASE] == 0:
        print("no baseline"); return 1

    res = {}
    for a in ARMS:
        f = DIR / "results_L232_{}_1.json".format(a)
        if f.exists():
            res[a] = {r["block_count"]: r for r in json.load(open(f))["test_results"]}
    W = lambda n: math.exp(n / 12.0)                             # noqa: E731
    SW = sum(W(n) for n in res[BASE])
    q0 = sum(W(n) * res[BASE][n]["cost"] for n in res[BASE]) / SW
    qbase = sum(m.qual_pern(gate, s) for s in ("s1", "s2")) / 2

    print()
    print("{:<7}{:>7}{:>10}{:>11}{:>11}{:>10}{:>10}{:>10}{:>6}"
          .format("arm", "pool", "C++ wall", "case wall", "serial", "quality",
                  "NET", "graded", "rank"))
    print("-" * 84)
    for a in ARMS:
        if a not in res or (a, 21) not in tmin:
            continue
        hv = [n for n in m.NS if n > 100 and (a, n) in tmin and (BASE, n) in tmin]
        cw = st.median(pmin[(a, n)] / pmin[(BASE, n)]
                       for n in hv if (a, n) in pmin and (BASE, n) in pmin) \
            if any((a, n) in pmin for n in hv) else float("nan")
        tw = st.median(tmin[(a, n)] / tmin[(BASE, n)] for n in hv)
        pool = st.median(npro[(a, n)] for n in hv) if hv and (a, hv[0]) in npro else 0
        q = 100 * (q0 - sum(W(n) * res[a][n]["cost"] for n in res[BASE]) / SW) / q0
        num = 0.0
        for n in m.NS:
            rr = tmin.get((a, n), 0) / tmin[(BASE, n)] if (BASE, n) in tmin else 1.0
            r = m.ROW[n]
            num += r["w"] * r["q"] * max(0.7, (T0[n] * rr / r["med"]) ** 0.3)
        num += sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
                   for r in m.ROWS if r["n"] not in POOL)
        rf = 100 * (m.BETA_NUM - num / TOTW) / m.BETA_NUM
        net = qbase + q + m.Q_POOL_FULL + rf
        gr = m.BETA * (1 - net / 100)
        print("{:<7}{:>7}{:>9.3f}x{:>10.3f}x{:>10.3f}x{:>+9.4f}%{:>+9.3f}%"
              "{:>10.5f}{:>6}".format(a, pool, cw, tw, tw / cw if cw else 0,
                                      q, net, gr, m.rank_of(gr)))
    print("-" * 84)
    print("pool      = profiles actually run per heavy case (liveness: an arm")
    print("            with the same count as base changed nothing)")
    print("C++ wall  = max(max d_p, sum d_p/48) ratio vs base, n>100")
    print("case wall = whole-case runtime ratio vs base, n>100  <- this is what")
    print("            RF is priced on")
    print("serial    = case wall / C++ wall = the part the pool model CANNOT")
    print("            see. L167 died on exactly this term.")
    print("quality   = IN SET, and for ADD-only arms that is a floor, not an")
    print("            optimistic estimate (superset selection + oracle proxy).")
    print("r2 = 0.88819   r3 = 0.89933")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
