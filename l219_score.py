"""L219 score - is REFINE a cheaper way to buy wall than dropping profiles?

The pool drop closed at a proven ceiling of +2.773% NET (per-n oracle, measured
phi), 0.17pp short of rank 3. It buys wall by removing whole profiles. The other
way to shorten a max-setter-bound wall is to make the profiles cheaper, and
REFINE_ITERS is exactly that knob -- a C++ post-processing budget the wrapper
already sets per band.

Two reasons this is not a re-run of M49:

  * M49 derived its bands STRICTLY SELECTION-PRESERVING: it took the part that
    costs no quality and stopped. Everything past that point was never priced,
    only assumed unprofitable.
  * the assumption was recorded as "do not stack more wall cuts, the floor is
    saturated" -- and that premise is false in the current package. 45 of 100
    cases sit ABOVE the RF floor, and 10 of them carry 86% of the deficit.

Reported the same way as the pool drop so the two are directly comparable:
quality is MEASURED in set, the wall is MEASURED from the instrumented
durations, and phi is the measured per-block-count share rather than M47's
pre-fix constant.

  <python> l219_score.py
"""
import collections
import json
import math
import os
import sys
from pathlib import Path

DIR = Path(__file__).parent
AMP = 2.41          # OOS amplification measured for the pool drop at k=8
RS = (4, 3, 2, 1)


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


def load_res(fn):
    f = DIR / fn
    if not f.exists():
        return None
    return {r["block_count"]: r for r in json.load(open(f))["test_results"]}


def main():
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l204_routea_risk as m
    POOL0 = dict(m.POOL)
    TOTW = sum(r["w"] for r in m.ROWS)
    THR = 0.7 ** (1 / 0.3)
    BETA, QP = m.BETA, m.Q_POOL_FULL
    gate = {n: 1 if POOL0[n] + m.DT[n] <= THR * m.MED[n] * 1.2 else 0
            for n in m.NS}

    # measured phi, per block count
    PHI = {}
    for tag in ("r1", "r2"):
        pf, d = load_prof("l205_prof_{}.txt".format(tag)), \
            load_res("results_L205_{}.json".format(tag))
        if not (pf and d):
            continue
        for n in pf:
            if n in d and d[n]["runtime_seconds"] > 0:
                PHI.setdefault(n, []).append(
                    min(1.0, max(pf[n].values()) / d[n]["runtime_seconds"]))
    PHI = {n: sum(v) / len(v) for n, v in PHI.items()}

    base_prof = load_prof("l219_prof_r4.txt")
    base_res = load_res("results_L219_r4.json")
    if not (base_prof and base_res):
        print("the REFINE=4 control has not run")
        return 1
    W = lambda n: math.exp(n / 12.0)                           # noqa: E731
    SW = sum(W(n) for n in base_res)
    q0 = sum(W(n) * base_res[n]["cost"] for n in base_res) / SW

    print("=" * 82)
    print("REFINE on n>100: quality MEASURED, wall MEASURED, phi MEASURED")
    print("=" * 82)
    print("{:>8}{:>11}{:>8}{:>13}{:>11}{:>10}{:>6}"
          .format("REFINE", "quality", "moved", "prof wall n>100", "NET",
                  "graded", "rank"))
    print("-" * 82)
    for R in RS:
        pf, rs = load_prof("l219_prof_r{}.txt".format(R)), \
            load_res("results_L219_r{}.json".format(R))
        if not (pf and rs):
            continue
        q = 100 * (q0 - sum(W(n) * rs[n]["cost"] for n in base_res) / SW) / q0
        mv = sum(1 for n in base_res if rs[n]["cost"] != base_res[n]["cost"])
        nf = sum(1 for n in base_res if not rs[n].get("is_feasible", True))
        cuts, hv = {}, []
        for n in m.NS:
            if n not in pf or n not in base_prof:
                cuts[n] = 0.0
                continue
            w1 = max(max(pf[n].values()), sum(pf[n].values()) / 48)
            w0 = max(max(base_prof[n].values()),
                     sum(base_prof[n].values()) / 48)
            cuts[n] = 1 - w1 / w0
            if n > 100:
                hv.append(cuts[n])
        num = 0.0
        newpool = {}
        for n in m.NS:
            newpool[n] = POOL0[n] * (1 - cuts[n] * PHI.get(n, 0.878))
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
        # A POSITIVE in-set quality delta must NOT be amplified: AMP models a
        # COST growing out of sample (measured 2.41x for the pool drop), and
        # applying it to a gain manufactures score out of an OOS penalty. An
        # unamplified positive is still optimistic -- it is an in-set reading --
        # which is why L221 measures this arm on both OOS samples.
        net = (b + (q * AMP if q < 0 else q) + QP
               + 100 * (m.BETA_NUM - num / TOTW) / m.BETA_NUM)
        gr = BETA * (1 - net / 100)
        print("{:>8}{:>+10.4f}%{:>8}{:>+12.2f}%{:>+10.3f}%{:>10.5f}{:>6}{}"
              .format(R, (q * AMP if q < 0 else q), mv,
                      100 * (sum(hv) / len(hv) if hv else 0.0), net, gr,
                      m.rank_of(gr),
                      "   !! {} INFEASIBLE".format(nf) if nf else ""))
    print("-" * 82)
    print("quality: in-set delta, x{:.2f} only when NEGATIVE (see the code).".format(AMP))
    print("REFINE=4 is the shipped band, so its row is the reference (0 by "
          "construction).")
    print("For comparison: pool drop shipped +1.838%, targeted +2.415%, "
          "oracle ceiling +2.773%.")
    print("r3 = 0.89933 needs NET >= 2.942%")
    print()
    print("!! The two knobs are NOT additive -- both shorten the same "
          "max-setter.")
    print("   A positive row here licenses a JOINT re-derivation, not a sum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
