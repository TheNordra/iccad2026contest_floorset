"""L224 - the REFINE verdict from uncontended durations.

Everything else about the REFINE finding is already measured: quality in set
(2 movers, net slightly positive), quality out of sample (L223), and the fact
that 81% of heavy-band profiles get faster while the untouched control band does
not move. What was NOT solid is the SIZE of the cut per block count, and the NET
is computed from exactly that.

The problem, measured directly rather than assumed: two identical REFINE=4 runs
disagree per block count by a median of 17% (range 0.269-1.861). Aggregated over
the band that falls to ~3%, which is why "the band cut is 21.5%" is safe and
"case n=112 got 24% faster" is not -- and NET is dominated by ~10 heavy cases,
so it inherits the per-case number, not the band one.

Uncontended durations have no scheduler in them, so the R2/R4 ratio is the
workload property it was always supposed to be. Same instrument, same argument
as L205b's route A verdict.

Reported against the contended estimate so the correction is visible rather
than silently substituted.

  <python> l224_seq_score.py
"""
import collections
import json
import math
import os
import sys
from pathlib import Path

DIR = Path(__file__).parent
AMP = 2.41


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


def wall(per, n, drop=frozenset()):
    v = [t for i, t in per[n].items() if i not in drop]
    return max(max(v), sum(v) / 48)


def main():
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l204_routea_risk as m
    import optimizer_constructive as O
    POOL0 = dict(m.POOL)
    TOTW = sum(r["w"] for r in m.ROWS)
    THR = 0.7 ** (1 / 0.3)
    BETA, QP = m.BETA, m.Q_POOL_FULL
    gate = {n: 1 if POOL0[n] + m.DT[n] <= THR * m.MED[n] * 1.2 else 0
            for n in m.NS}

    s4 = load("l205b_prof_seq.txt")
    s2 = load("l224_prof_seq_r2.txt")
    if not (s4 and s2):
        print("the uncontended pair is incomplete")
        return 1

    PHI = {}
    for tag in ("r1", "r2"):
        pf = load("l205_prof_{}.txt".format(tag))
        rf = DIR / "results_L205_{}.json".format(tag)
        if not (pf and rf.exists()):
            continue
        d = {r["block_count"]: r for r in json.load(open(rf))["test_results"]}
        for n in pf:
            if n in d and d[n]["runtime_seconds"] > 0:
                PHI.setdefault(n, []).append(
                    min(1.0, max(pf[n].values()) / d[n]["runtime_seconds"]))
    PHI = {n: sum(v) / len(v) for n, v in PHI.items()}

    hv = [1 - wall(s2, n) / wall(s4, n) for n in s4 if n in s2 and n > 100]
    lo = [1 - wall(s2, n) / wall(s4, n) for n in s4 if n in s2 and n <= 100]
    print("=" * 78)
    print("UNCONTENDED REFINE 4 -> 2")
    print("  band cut n>100 : {:+.2f}%   (contended single pair read +21.52%)"
          .format(100 * sum(hv) / len(hv)))
    print("  CONTROL n<=100 : {:+.2f}%   (must be ~0; REFINE untouched there)"
          .format(100 * sum(lo) / len(lo)))
    per = sorted(hv)
    print("  per block count: p10 {:+.2f}%  median {:+.2f}%  p90 {:+.2f}%"
          .format(100 * per[len(per) // 10], 100 * per[len(per) // 2],
                  100 * per[9 * len(per) // 10]))
    print("=" * 78)

    # quality: in set from the L219 arms, OOS from L223 when present
    J = (lambda f: {r["block_count"]: r
                    for r in json.load(open(DIR / f))["test_results"]}
         if (DIR / "results_L219_r4.json").exists() else None)
    b = J("results_L219_r4.json")
    W = lambda n: math.exp(n / 12.0)                           # noqa: E731
    SW = sum(W(n) for n in b)
    q0 = sum(W(n) * b[n]["cost"] for n in b) / SW
    r2 = J("results_L219_r2.json")
    q_r2 = 100 * (q0 - sum(W(n) * r2[n]["cost"] for n in b) / SW) / q0

    drops = {n: frozenset(O._L211_POOLDROP.get(n, ())) for n in m.NS}
    os.environ["ICCAD_L211_POOLDROP"] = "0"
    FULL = {n: O._pool_indices(n) for n in m.NS}
    os.environ.pop("ICCAD_L211_POOLDROP")
    dpos = {n: {i for i, g in enumerate(FULL[n]) if g in drops[n]}
            for n in m.NS}

    def net(cuts, q):
        num = 0.0
        np_ = {}
        for n in m.NS:
            np_[n] = POOL0[n] * (1 - cuts.get(n, 0.0) * PHI.get(n, 0.878))
            t = np_[n] + (m.DT[n] if gate.get(n, 0) else 0.0)
            r = m.ROW[n]
            num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        num += sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
                   for r in m.ROWS if r["n"] not in m.POOL)
        for n in m.NS:
            m.POOL[n] = np_[n]
        bq = sum(m.qual_pern(gate, s) for s in ("s1", "s2")) / 2
        for n in m.NS:
            m.POOL[n] = POOL0[n]
        x = bq + (q * AMP if q < 0 else q) + QP \
            + 100 * (m.BETA_NUM - num / TOTW) / m.BETA_NUM
        return x, BETA * (1 - x / 100)

    cut_r2 = {n: (1 - wall(s2, n) / wall(s4, n)) if n in s2 else 0.0
              for n in m.NS}
    cut_k8 = {n: 1 - wall(s4, n, dpos[n]) / wall(s4, n) for n in m.NS}
    cut_j = {n: 1 - wall(s2, n, dpos[n]) / wall(s4, n) if n in s2 else 0.0
             for n in m.NS}
    print("{:<28}{:>11}{:>11}{:>10}{:>6}"
          .format("configuration", "quality", "NET", "graded", "rank"))
    print("-" * 70)
    for lbl, c, q in (("today (k=0, R=4)", {}, 0.0),
                      ("pool drop k=8 only", cut_k8, -0.2989),
                      ("REFINE=2 only", cut_r2, q_r2),
                      ("k=8 + REFINE=2", cut_j, None)):
        if q is None:
            j = J("results_L220_k8r2.json")
            q = 100 * (q0 - sum(W(n) * j[n]["cost"] for n in b) / SW) / q0
        x, g = net(c, q)
        print("{:<28}{:>+10.4f}%{:>+10.3f}%{:>10.5f}{:>6}"
              .format(lbl, q, x, g, m.rank_of(g)))
    print("-" * 70)
    print("r3 = 0.89933   r2 = 0.88819")
    print()
    print("!! quality for the two REFINE rows is still IN SET. L223's OOS runs")
    print("   are what decide them; this file only fixes the WALL half.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
