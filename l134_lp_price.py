"""L134 — is there a cheap form of lp_polish?

Handoff 08-16 §7.4: runtime is the first problem on this line, not quality. The
candidate costs 30-38% of total score because at 48 cores the wall is the
max-setter and it runs mean 3.48s / max 35.92s against an incumbent wall of
1.0-1.8s. `lp_polish` is **95.8%** of that time: six `oc.lp_pass` calls per case.

So the question is narrow and measurable: what does each pass BUY, and what does
each pass COST? If the curve is front-loaded, `iters` is a free lever. If it is
flat, the polish has to be made cheaper per pass instead (or abandoned).

Three axes, measured independently on the same layouts:
  * iters      -- cost and cumulative time after each of the 6 passes
  * PRUNE_B    -- lp_polish deliberately sets it to None for exactness;
                  08-15 §4 says the shipped settings are worth 1.37x
  * sep_trim   -- ditto, off for exactness

Everything is judged against the number that actually matters: **the per-case
wall**, not the mean. A pass that costs 0.2s on a small case is free; the same
pass costing 20s on n=99 is what sets dRF.

  <python> -u l134_lp_price.py --minn 80          # the 96%-of-weight cases
  <python> -u l134_lp_price.py --limit 100
"""
import argparse
import math
import time

import l129_global_placer as L
import optimizer_constructive as oc

_l3 = oc.l3


def lp_trace(c, P, iters, prune_b, sep_trim):
    """lp_polish, but recording official cost and elapsed time after EACH pass.

    Mirrors l129_global_placer.lp_polish exactly (same key, same objective, same
    hard_ok guard) so the numbers are comparable to the shipped measurements.
    """
    key = "l134"
    n = c["n"]
    sumA = sum(max(0.0, float(c["at"][i])) for i in range(n))
    P0 = [tuple(float(x) for x in q) for q in P]
    try:
        hp = oc._proxy_metrics(P0, c["at"], c["b2b"], c["p2b"], c["pins"],
                               c["cons"], n)["hpwl"]
    except Exception:
        return None
    base = {"hpwl_baseline": max(float(hp), 1e-6),
            "area_baseline": max(sumA / oc._LP_UTIL, 1e-6)}
    _l3.CASES[key] = oc._lp_build_case(n, c["at"], c["b2b"], c["p2b"],
                                       c["pins"], c["cons"], base)
    saved = oc.PRUNE_B
    oc.PRUNE_B = saved if prune_b else None
    Q = P0
    out = []
    t0 = time.perf_counter()
    try:
        for it in range(iters):
            newQ, _tele, _B = oc.lp_pass(key, Q, 0.06, sep_trim=sep_trim)
            if newQ is None or not oc.hard_ok(P0, newQ, key):
                break
            Q = newQ
            m = L.official(c, Q)
            out.append((it + 1, time.perf_counter() - t0, float(m.cost),
                        bool(m.is_feasible)))
    except Exception:
        pass
    finally:
        oc.PRUNE_B = saved
        _l3.CASES.pop(key, None)
        oc._HARD_MASKS.pop(key, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--minn", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()

    cases = L.CASES[:a.limit] if a.limit else L.CASES
    if a.minn:
        cases = [c for c in cases if c["n"] >= a.minn]
    print(f"[l134] {len(cases)} cases, DENSITY={L.DENSITY} "
          f"ABUT={int(L.EXACT_ABUT)} GORDIAN={int(L.GORDIAN)}\n")

    variants = [("base", True, False, False),
                ("prune_b", True, True, False),
                ("sep_trim", True, False, True),
                ("both", True, True, True)]

    # per variant: per-iteration weighted cost, cumulative time, and the WALL
    agg = {v[0]: {} for v in variants}
    walls = {v[0]: {} for v in variants}
    t_place_tot = 0.0
    wsum = 0.0
    npre = 0
    for c in cases:
        t0 = time.perf_counter()
        P = L.place(c)
        t_place = time.perf_counter() - t0
        if P is None:
            continue
        w = math.exp(c["n"] / 12.0)
        wsum += w
        t_place_tot += w * t_place
        npre += 1
        m0 = L.official(c, P)
        for name, _on, pb, st in variants:
            tr = lp_trace(c, P, a.iters, pb, st)
            if tr is None:
                continue
            # iteration 0 = the un-polished layout
            rec = agg[name].setdefault(0, [0.0, 0.0])
            rec[0] += w * float(m0.cost)
            rec[1] += w * 0.0
            wl = walls[name].setdefault(0, [])
            wl.append((c["n"], t_place))
            last = (0.0, float(m0.cost))
            for it, el, cost, _fe in tr:
                r = agg[name].setdefault(it, [0.0, 0.0])
                r[0] += w * cost
                r[1] += w * el
                walls[name].setdefault(it, []).append((c["n"], t_place + el))
                last = (el, cost)
            # carry the last value forward so every variant sums over the same
            # weight, otherwise a case that broke early flatters the tail
            for it in range(len(tr) + 1, a.iters + 1):
                r = agg[name].setdefault(it, [0.0, 0.0])
                r[0] += w * last[1]
                r[1] += w * last[0]
                walls[name].setdefault(it, []).append((c["n"], t_place + last[0]))

    print(f"place() alone: weighted {t_place_tot / max(wsum, 1e-9):.3f}s   "
          f"({npre} cases)\n")
    for name, _on, pb, st in variants:
        print(f"--- {name}  (PRUNE_B={'shipped' if pb else 'None'}, "
              f"sep_trim={st}) ---")
        print(f"{'iters':>6} {'w.cost':>10} {'w.time':>9} {'max case s':>11} "
              f"{'cases>1.5s':>11}")
        for it in sorted(agg[name]):
            cost, el = agg[name][it]
            wl = walls[name].get(it, [])
            mx = max((t for _n, t in wl), default=0.0)
            over = sum(1 for _n, t in wl if t > 1.5)
            print(f"{it:>6} {cost / max(wsum, 1e-9):>10.5f} "
                  f"{el / max(wsum, 1e-9):>9.3f} {mx:>11.2f} "
                  f"{str(over) + '/' + str(len(wl)):>11}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
