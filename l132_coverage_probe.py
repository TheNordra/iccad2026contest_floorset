"""L132 — why does `legalise` fail on 37 of 100 cases?

L130 item 2. Coverage is where the gate lives: this candidate's whole portfolio
value is ~3 cases, so dropping one heavy case costs more than a 5% solo
improvement earns. The two stage-A forms drop five cases each, DISJOINTLY, which
already says the failures are not a property of the instance alone.

`legalise` has exactly two ways to give up:

  A. `compact` hits a pinned (preplaced) unit whose longest-path lower bound
     already exceeds where the pin must sit, and the unit has NO predecessor to
     flip -- nothing to repair, so it returns (None, None);
  B. the repair loop runs `rounds` times without both axes clearing in the same
     iteration -- it is thrashing, flipping pairs back and forth.

These want completely different fixes, so the first question is which one is
happening. This classifies every failure and reports how far B gets.

  <python> -u l132_coverage_probe.py --rounds 40
  <python> -u l132_coverage_probe.py --rounds 200   # does more budget help?
"""
import argparse
import collections
import math
import time

import numpy as np

import l129_global_placer as L


def classify(v, units, cx, cy, rounds):
    """Re-implements legalise's control flow to record WHY it stopped."""
    U = len(units)
    W = np.array([u["w"] for u in units])
    H = np.array([u["h"] for u in units])
    pre = np.array([u["pre"] for u in units])
    px, py = cx.copy(), cy.copy()
    need = [u["need"] for u in units]

    axis = {}
    for a in range(U):
        for b in range(a + 1, U):
            ox = (W[a] + W[b]) / 2.0 - abs(cx[b] - cx[a])
            oy = (H[a] + H[b]) / 2.0 - abs(cy[b] - cy[a])
            cheap = 0 if ox <= oy else 1
            lo_x, hi_x = (a, b) if (cx[a], a) < (cx[b], b) else (b, a)
            lo_y, hi_y = (a, b) if (cy[a], a) < (cy[b], b) else (b, a)
            okx = not (need[lo_x]["R"] or need[hi_x]["L"])
            oky = not (need[lo_y]["T"] or need[hi_y]["B"])
            axis[(a, b)] = cheap if (okx and oky) else (0 if okx else
                                                       (1 if oky else cheap))

    def compact(ax, C, size):
        order = sorted(range(U), key=lambda k: (C[k], k))
        rank = {k: r for r, k in enumerate(order)}
        pred = [[] for _ in range(U)]
        for (a, b), s in axis.items():
            if s != ax:
                continue
            lo, hi = (a, b) if rank[a] < rank[b] else (b, a)
            pred[hi].append(lo)
        want_of = {k: C[k] - size[k] / 2.0 for k in range(U) if pre[k]}
        low0 = min([0.0] + list(want_of.values()))
        low = np.full(U, low0)
        for k in order:
            base = low0
            for p in pred[k]:
                base = max(base, low[p] + size[p])
            if pre[k]:
                want = want_of[k]
                if base > want + 1e-6:
                    if not pred[k]:
                        return None, None, (k, base - want)
                    return None, (k, sorted(pred[k],
                                            key=lambda p: -(low[p] + size[p]))), \
                        (k, base - want)
                low[k] = want
            else:
                low[k] = base
        return low, None, None

    flips = collections.Counter()
    locked = set()

    def pick(k, cand):
        for p in cand:
            if not L.LEGAL_LOCK or (min(k, p), max(k, p)) not in locked:
                return p
        return None

    for it in range(rounds):
        for ax, C, size in ((0, px, W), (1, py, H)):
            low, bad, info = compact(ax, C, size)
            if low is not None:
                if ax == 1:
                    return dict(ok=True, it=it, flips=len(flips))
                continue
            if bad is None:
                return dict(ok=False, why="A_no_pred", axis="xy"[ax], it=it,
                            over=info[1], unit=info[0], flips=len(flips))
            k, cand = bad
            p = pick(k, cand)
            if p is None:
                return dict(ok=False, why="C_moves_out", axis="xy"[ax], it=it,
                            over=info[1], unit=info[0], flips=len(flips))
            pair = (min(k, p), max(k, p))
            axis[pair] = 1 - ax
            locked.add(pair)
            flips[pair] += 1
            break
    rep = flips.most_common(1)[0][1] if flips else 0
    return dict(ok=False, why="B_rounds", it=rounds, flips=len(flips),
                max_repeat=rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--gordian", action="store_true")
    a = ap.parse_args()
    L.GORDIAN = a.gordian

    kinds = collections.Counter()
    rows = []
    t0 = time.time()
    for c in L.CASES:
        v = L.case_view(c)
        DIMS = {i: L.choose_dims(v, i) for i in range(v["n"])}
        if L.MIB_UNIFY:
            L.unify_mib(v, DIMS)
        units = L.build_units(v, DIMS)
        if L.GORDIAN:
            cx, cy, _ = L.gordian(v, units)
        else:
            cx, cy, _ = L.global_place(v, units)
            cx, cy = L.spread(v, units, cx, cy)
        r = classify(v, units, cx, cy, a.rounds)
        npre = sum(1 for u in units if u["pre"])
        kinds[r.get("why", "ok")] += 1
        rows.append((c["idx"], c["n"], len(units), npre, r))

    print(f"\n=== L132: legalise outcome, rounds={a.rounds}, "
          f"gordian={int(a.gordian)} ===\n")
    for k, n in kinds.most_common():
        print(f"  {k:<12} {n}")
    print(f"\n  (first-pass only; the real placer retries from compacted "
          f"centres up to 6 times, so its coverage is >= 'ok' here)")

    fails = [r for r in rows if not r[4]["ok"]]
    print(f"\n{'case':>5} {'n':>4} {'units':>6} {'pre':>4} {'why':>10} "
          f"{'iters':>6} {'flips':>6} {'overrun':>12}")
    for idx, n, nu, npre, r in fails[:40]:
        print(f"{idx:>5} {n:>4} {nu:>6} {npre:>4} {r.get('why', ''):>10} "
              f"{r.get('it', 0):>6} {r.get('flips', 0):>6} "
              f"{r.get('over', float('nan')):>12.4g}")

    npre_fail = sum(1 for r in fails if r[3] > 0)
    npre_ok = sum(1 for r in rows if r[4]["ok"] and r[3] > 0)
    print(f"\ncases WITH preplaced units:  fail {npre_fail}, ok {npre_ok}")
    print(f"cases WITHOUT preplaced:     fail {len(fails) - npre_fail}, "
          f"ok {sum(1 for r in rows if r[4]['ok'] and r[3] == 0)}")
    print(f"\nwall {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
