"""L212 - the pool-drop curve, scored with the REAL RF model rather than 0.3x.

`l211_score.py` prices the wall with the linear approximation `RF = -0.3 * dt`,
which is the derivative of `t^0.3` and therefore only valid for small moves and
only away from the 0.7 floor. At k=16-20 the move is 9-12% of case wall and
several cases sit ON the floor, where the derivative is zero and the linear form
overstates the gain.

This scores every k the way l204 does -- rebuilding the LP gate at the reduced
pool time, because the two levers COMPOUND: a cheaper pool makes more block
counts affordable under `t_pool + dt_lp <= 0.3046*M*s`, so the gate widens for
free and collects more LP quality. Ignoring that undercounts; using the linear
RF overcounts. Both are fixed here.

Quality is the MEASURED in-set delta from the L211 arms, not a model.

  <python> l212_curve.py
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).parent
PHI = 0.71
CORES = 48
BASE = "results_L209_det1.json"
KS = (3, 8, 12, 16, 20, 24)


def load(fn):
    f = DIR / fn
    if not f.exists():
        return None
    return {r["test_id"]: r for r in json.load(open(f))["test_results"]}


def wq(d, ids):
    w = lambda i: math.exp(d[i]["block_count"] / 12.0)         # noqa: E731
    return sum(w(i) * d[i]["cost"] for i in ids) / sum(w(i) for i in ids)


def prof_cut(k, O):
    """profile-phase wall saving of table k on the uncontended run."""
    import collections
    per = collections.defaultdict(dict)
    f = DIR / "l205b_prof_seq.txt"
    t = DIR / "l211_drop_k{}.json".format(k)
    if not (f.exists() and t.exists()):
        return None
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) == 3:
            per[int(p[0])][int(p[1])] = float(p[2])
    tab = {int(a): set(b) for a, b in json.loads(t.read_text()).items()}
    base = tot = 0.0
    for n, d in per.items():
        idx = O._pool_indices(n)
        keep = [v for i, v in d.items() if idx[i] not in tab.get(n, set())]
        if not keep:
            continue
        base += max(max(d.values()), sum(d.values()) / CORES)
        tot += max(max(keep), sum(keep) / CORES)
    return 100 * (tot / base - 1)


def main():
    base = load(BASE)
    if not base:
        print("no baseline")
        return 1
    sys.argv = ["x"]
    import os
    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    import optimizer_constructive as O
    import l204_routea_risk as m
    POOL0 = dict(m.POOL)
    BETA, QP = m.BETA, m.Q_POOL_FULL

    def model(cut_frac):
        for n in m.NS:
            m.POOL[n] = POOL0[n] * (1 - cut_frac)
        g = m.time_gate(1.2)
        q = sum(m.qual_pern(g, s) for s in ("s1", "s2")) / 2
        net = q + QP + m.rf_at(g, 1.0)
        for n in m.NS:
            m.POOL[n] = POOL0[n]
        return net, sum(g.values())

    net0, on0 = model(0.0)
    print("=" * 84)
    print("POOL DROP CURVE -- measured quality, modelled wall+gate (l204 RF, "
          "with the floor)")
    print("=" * 84)
    print("{:>4}{:>6}{:>10}{:>8}{:>11}{:>7}{:>11}{:>10}{:>6}"
          .format("k", "pool", "quality", "moved", "case wall", "gate",
                  "NET", "graded", "rank"))
    print("{:>4}{:>6}{:>10}{:>8}{:>11}{:>7}{:>+11.3f}%{:>10.5f}{:>6}"
          .format(0, 51, "-", "-", "-", on0, net0, BETA * (1 - net0 / 100),
                  m.rank_of(BETA * (1 - net0 / 100))))
    print("-" * 84)
    for k in KS:
        d = load("results_L211_k{}.json".format(k))
        pc = prof_cut(k, O)
        if d is None or pc is None:
            continue
        ids = sorted(set(base) & set(d))
        q = 100 * (wq(base, ids) - wq(d, ids)) / wq(base, ids)
        mv = sum(1 for i in ids if d[i]["cost"] != base[i]["cost"])
        ws = sum(1 for i in ids if d[i]["cost"] > base[i]["cost"] + 1e-12)
        nf = sum(1 for i in ids if not d[i].get("is_feasible", True))
        cut = -pc * PHI / 100.0
        net_m, on = model(cut)
        net = net_m + q            # measured quality delta rides on the model
        gr = BETA * (1 - net / 100)
        flag = "  !! {} INFEASIBLE".format(nf) if nf else ""
        print("{:>4}{:>6}{:>+9.4f}%{:>5}/{:<2}{:>+10.2f}%{:>7}{:>+11.3f}%"
              "{:>10.5f}{:>6}{}"
              .format(k, 51 - k, q, mv, ws, -100 * cut, on, net, gr,
                      m.rank_of(gr), flag))
    print("-" * 84)
    print("quality: MEASURED in set, + is better.  case wall: profile-phase "
          "cut x{:.2f} (M47 tail).".format(PHI))
    print("gate: how many block counts the LP gate fires on once the cheaper "
          "pool widens it.")
    print("r3 = 0.89933   r2 = 0.88819   beta = {:.5f}".format(BETA))
    print()
    print("!! IN-SET quality only. Pool pruning is the exact mechanism that")
    print("   reversed out of sample twice (L138/L139: 12 of 22 held-out")
    print("   winners removed). Nothing here ships without the two OOS 240s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
