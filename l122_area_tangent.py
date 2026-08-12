"""OFFLINE (never shipped): exact area by TANGENT CUTS, replacing the band.

WHERE THIS CAME FROM.  L120 set out to make the shape step exact so that depth
would not be needed, and route C (L121) tried to do it by pinning closed-form
shapes into the LP.  That went infeasible on every pilot case at a 0.05% shape
change -- four orders below the trust region it was trying to escape -- and the
elastic phase-1 blamed SEPARATION rows, not the area constraint: a block wedged
between two members of the same rigid unit has a hard width ceiling that no
reformulation lifts.  Pinning is therefore the wrong shape of answer; the LP has
to be allowed to back off where the topology says no.

Two measurements from that detour set up this file:

  1. rho IS binding.  44.7% of shape columns sit exactly on the trust region
     bound, and widening rho 0.06 -> 0.088 moves the LP objective by +0.13% to
     +0.71% per case.  So the shape step really is capped, and the cap is worth
     money.
  2. The cap is ARITHMETIC, not geometric.  The band rows carry
     `slack = rho^2 * p` to cover the second-order term dw*dh, so the band on the
     linearised area is [A(1-TOL)+slack, A(1+TOL)-slack], which is EMPTY once
     rho^2 >= TOL = 0.008, i.e. rho >= 0.0894.  The record's "rho=0.10 keeps
     0/100 passes" is not a rejected geometry -- the LP is infeasible and every
     pass fails to build.

And the band's own asymmetry points at the fix.  The LOWER row (w*h >= A) binds
on 259 of 338 reshapeable units on the pilot; the UPPER on 9.  The lower one
bounds a CONVEX region, so tangents represent it exactly, with no linearisation
error and no trust region at all.  The upper one is the non-convex side -- and
it is the actual barrier to a large aspect change, because an exact-area
widening by r has true area p but LINEARISED area p*(r + 1/r - 1): at r=1.5 that
reads as +16.7% against a band of +/-0.8%.  So: tangents for the lower side, drop
the upper side, and let `hard_ok`'s 1% area test be the verifier, which is the
same solve-then-verify contract solve_pruned already runs for HPWL pruning.

Correctness does not rest on the verification passing.  The guard rejects any
pass hard_ok fails, so a blown area costs a gain, never a wrong answer.

⚠️ SCOPE.  Every number here is on the offline LABEL-derived baseline
(c["base"] carries the evaluator's hpwl_baseline / area_baseline).  The shipping
path substitutes a baseline-free one, and the two do not transfer -- the offline
+2.36% became +2.1817% deployed.  Arm-vs-arm comparisons on the SAME baseline
are the thing this file is for; the absolute gain has to be re-measured through
the official eval before anything is claimed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import l100_lp_speed as L                                   # noqa: E402
import m53_l3_probe as l3                                   # noqa: E402

DUMP = Path(__file__).parent / "results_L122_area_tangent.json"

SHIP_RHO = 0.06
SHIP_PRUNE = 8.0


def _anchor():
    return {t["test_id"]: t for t in
            json.loads(L.ANCHOR.read_text(encoding="utf-8"))["test_results"]}


def area_report(ci, P0, P):
    """Worst relative area error over the blocks the LP was allowed to reshape.

    This is the quantity the dropped upper bound put at risk, so it is reported
    per case rather than folded into a pass/fail -- `hard_ok` already decides
    the pass, and what is wanted here is how much margin is left.
    """
    c = l3.CASES[ci]
    cn, at, n = c["cn"], c["at"], c["n"]
    worst = 0.0
    over = 0
    for i in range(n):
        if cn[i][0] != 0 or cn[i][1] != 0:
            continue
        A = float(at[i])
        if A <= 0:
            continue
        e = abs(P[i][2] * P[i][3] - A) / A
        worst = max(worst, e)
        if e > 0.01:
            over += 1
    return worst, over


def one_pass(ci, P, area_R=None, area_g=1.05, sep_trim=True, area_price=0.0):
    """One LP pass, mirroring lp_pass's three-attempt cluster re-freeze loop."""
    c = l3.CASES[ci]
    freeze = set()
    for attempt in range(3):
        t0 = time.perf_counter()
        # solve_pruned, not build_and_solve: the HPWL pruning is only EXACT
        # because of the sign re-check it runs afterwards.  Calling the builder
        # directly with prune_B leaves the pruned program a lower bound whose
        # optimum need not be the real one, which would quietly bias every
        # number in this file.
        kw = {} if area_R is None else dict(area_R=area_R, area_g=area_g)
        B, _ = L.solve_pruned(ci, P, freeze, rho=SHIP_RHO, prune_B=L.PRUNE_B,
                              sep_trim=sep_trim, area_price=area_price, **kw)
        t = time.perf_counter() - t0
        if B["res"].status != 0:
            return None, dict(status=f"lp_status_{B['res'].status}", t=t,
                              lp_obj=None, rows=B["rows_ub"] + B["rows_eq"])
        newP = L.apply_all(P, B, B["res"].x)
        broken = [g for g, c0 in B["group_comp0"].items()
                  if len(l3.comp_split(newP, [i for i in range(c["n"])
                                              if c["cn"][i][3] == g])) > c0]
        if not broken:
            return newP, dict(status="ok", t=t,
                              lp_obj=float(B["res"].fun) + B.get("prune_const", 0.0),
                              rows=B["rows_ub"] + B["rows_eq"])
        for g in broken:
            freeze |= B["group_units"][g]
    return None, dict(status="cluster_break", t=0.0, lp_obj=None, rows=0)


def arm_case(ci, anch, area_R=None, area_g=1.05, iters=1, area_price=0.0):
    """`dep_case`'s accept guard, verbatim, so the numbers stay comparable."""
    P0 = [tuple(p) for p in anch[ci]["positions"]]
    q_ship = anch[ci]["cost"]
    m_prev = L.proxy_m(ci, P0)
    P, kept, t_lp, rows = P0, 0, 0.0, 0
    status = "ok"
    for _ in range(iters):
        newP, tele = one_pass(ci, P, area_R=area_R, area_g=area_g,
                              area_price=area_price)
        t_lp += tele["t"]
        rows = max(rows, tele["rows"])
        status = tele["status"]
        if newP is None:
            break
        m_new = L.proxy_m(ci, newP)
        better = (m_new["hpwl"] < m_prev["hpwl"] * (1 - 1e-12)
                  or m_new["area"] < m_prev["area"] * (1 - 1e-12))
        worse = (m_new["hpwl"] > m_prev["hpwl"] * (1 + 1e-12)
                 or m_new["area"] > m_prev["area"] * (1 + 1e-12)
                 or m_new["vrel"] > m_prev["vrel"] + 1e-12)
        hard = L.hard_ok(P0, newP, ci)
        if not (better and not worse and hard):
            status = ("reject_hard" if not hard else
                      "reject_worse" if worse else "reject_flat")
            break
        P, m_prev = newP, m_new
        kept += 1
    worst_area, over = area_report(ci, P0, P)
    mm = l3.cost_eval(ci, P)
    return dict(ci=ci, n=l3.CASES[ci]["n"], q_ship=q_ship, q_new=float(mm.cost),
                feas=bool(mm.is_feasible), kept=kept, status=status,
                t_lp=t_lp, rows=rows, worst_area=worst_area, area_over=over)


def mode_gate0(cases=(60, 95, 99), area_g=1.05):
    ok = True
    anch = _anchor()

    P = [tuple(p) for p in anch[99]["positions"]]
    v = float(L.build_and_solve(99, P, set(), rho=0.06)["res"].fun)
    hit = abs(v - 1.747467759676) < 1e-11
    ok &= hit
    print(f"  G1 shipped build untouched: case 99 obj {v:.12f} -> "
          f"{'PASS' if hit else 'FAIL'}")

    # G2 the tangent envelope's guarantee, checked numerically rather than
    # trusted from the algebra: over the whole width range the minimum h the
    # cuts allow must keep w*h above A(1-TOL)*(1-(sqrt(g)-1)^2).
    import math
    worst = 0.0
    rng = np.random.default_rng(122)
    for _ in range(400):
        A = float(rng.uniform(1.0, 5000.0))
        r0 = float(rng.uniform(0.3, 3.0))
        w0 = (A * r0) ** 0.5
        h0 = A / w0
        R = float(rng.choice([1.2, 1.5, 2.0, 3.0]))
        steps = max(1, int(math.ceil(2.0 * math.log(R) / math.log(area_g))))
        Ap = A * (1.0 - L.AREA_TOL)
        wk = [(w0 / R) * (R * R) ** (s / steps) for s in range(steps + 1)]
        for w in np.geomspace(w0 / R, w0 * R, 257):
            hmin = max(2.0 * Ap / k - (Ap / (k * k)) * w for k in wk)
            worst = max(worst, 1.0 - (w * hmin) / Ap)
        del h0
    bound = (area_g ** 0.5 - 1.0) ** 2
    p2 = worst <= bound * 1.02
    ok &= p2
    print(f"  G2 tangent envelope deficit {100 * worst:.4f}% vs bound "
          f"{100 * bound:.4f}% -> {'PASS' if p2 else 'FAIL'}")
    tot = (1.0 - L.AREA_TOL) * (1.0 - worst)
    p3 = tot >= 0.99
    ok &= p3
    print(f"  G3 worst true area {100 * tot:.4f}% of A vs the 99% hard limit "
          f"-> {'PASS' if p3 else 'FAIL'}")

    # G4 degenerate case: R -> 1 must move no shape and must not break the solve
    bad = []
    for ci in cases:
        P = [tuple(p) for p in anch[ci]["positions"]]
        B = L.build_and_solve(ci, P, set(), rho=0.0, area_R=1.0 + 1e-9,
                              area_g=area_g)
        if B["res"].status != 0:
            bad.append((ci, f"status{B['res'].status}"))
            continue
        newP = L.apply_all(P, B, B["res"].x)
        d = max(max(abs(newP[i][2] - P[i][2]), abs(newP[i][3] - P[i][3]))
                for i in range(l3.CASES[ci]["n"]))
        if d > 1e-6:
            bad.append((ci, d))
    ok &= not bad
    print(f"  G4 R->1 freezes every shape on {len(cases)} cases: "
          f"{'PASS' if not bad else f'FAIL {bad}'}")
    print(f"gate0 -> {'ALL PASS' if ok else 'FAIL'}")
    return ok


def mode_ab(cases, Rs, area_g, iters, area_price=0.0):
    anch = _anchor()
    W = l3.W
    cov = sum(W[c] for c in cases)
    L.PRUNE_B = SHIP_PRUNE

    arms = {}
    for label, R in [("ship", None)] + [(f"R={r:g}", r) for r in Rs]:
        rows = {}
        t0 = time.perf_counter()
        for ci in cases:
            rows[ci] = arm_case(ci, anch, area_R=R, area_g=area_g, iters=iters,
                                area_price=area_price)
        arms[label] = rows
        ship = sum(W[c] * rows[c]["q_ship"] for c in cases) / cov
        after = sum(W[c] * rows[c]["q_new"] for c in cases) / cov
        tw = sum(W[c] * rows[c]["t_lp"] for c in cases) / cov
        kept = sum(r["kept"] > 0 for r in rows.values())
        regr = [c for c in cases if rows[c]["q_new"] > rows[c]["q_ship"] + 1e-12]
        infe = [c for c in cases if not rows[c]["feas"]]
        wa = max(r["worst_area"] for r in rows.values())
        over = sum(r["area_over"] for r in rows.values())
        rej = {}
        for r in rows.values():
            rej[r["status"]] = rej.get(r["status"], 0) + 1
        print(f"\n--- arm {label} ({time.perf_counter() - t0:.1f}s wall) ---")
        print(f"  weighted quality  {after:.9f}   gain vs anchor "
              f"{100.0 * (1 - after / ship):+.4f}%")
        print(f"  kept              {kept}/{len(cases)}   regressions "
              f"{len(regr)}   infeasible {len(infe)}")
        print(f"  weighted tLP      {tw:.4f}s   rows(max) "
              f"{max(r['rows'] for r in rows.values())}")
        print(f"  worst area err    {100 * wa:.4f}%   blocks over 1%: {over}")
        print(f"  status            {rej}")

    base = arms["ship"]
    ship = sum(W[c] * base[c]["q_ship"] for c in cases) / cov
    sq = sum(W[c] * base[c]["q_new"] for c in cases) / cov
    st = sum(W[c] * base[c]["t_lp"] for c in cases) / cov
    print("\n=== TANGENT vs SHIPPED-EQUIVALENT ===")
    for label, rows in arms.items():
        if label == "ship":
            continue
        q = sum(W[c] * rows[c]["q_new"] for c in cases) / cov
        t = sum(W[c] * rows[c]["t_lp"] for c in cases) / cov
        print(f"  {label:>8}: quality {100.0 * (1 - q / sq):+.4f}pp vs ship-LP, "
              f"tLP x{t / max(st, 1e-9):.2f}  (abs {q:.9f} / {t:.4f}s)")
    DUMP.write_text(json.dumps(
        dict(cases=len(cases), Rs=list(Rs), area_g=area_g, iters=iters,
             anchor=ship, arms={k: {str(c): v for c, v in r.items()}
                                for k, r in arms.items()}), indent=1),
        encoding="utf-8")
    print(f"\njson -> {DUMP}")


def mode_reps(cases, Rs, area_g, area_price, reps=3):
    """Per-case MIN over `reps` runs, per L100's rule that a single timing
    cannot rank solvers.  Quality is deterministic, so only the clock repeats."""
    anch = _anchor()
    W = l3.W
    cov = sum(W[c] for c in cases)
    L.PRUNE_B = SHIP_PRUNE
    arms = [("ship", None)] + [(f"R={r:g}", r) for r in Rs]
    best = {lab: {} for lab, _ in arms}
    qual = {}
    for rep in range(reps):
        for lab, R in arms:
            t0 = time.perf_counter()
            for ci in cases:
                r = arm_case(ci, anch, area_R=R, area_g=area_g, iters=1,
                             area_price=area_price)
                prev = best[lab].get(ci)
                best[lab][ci] = r["t_lp"] if prev is None else min(prev, r["t_lp"])
                qual.setdefault(lab, {})[ci] = r["q_new"]
            print(f"  rep {rep + 1}/{reps} arm {lab:>7}: "
                  f"{time.perf_counter() - t0:.1f}s wall")
    base = sum(W[c] * best["ship"][c] for c in cases) / cov
    print(f"\n=== min-of-{reps} weighted tLP ===")
    for lab, _ in arms:
        t = sum(W[c] * best[lab][c] for c in cases) / cov
        q = sum(W[c] * qual[lab][c] for c in cases) / cov
        print(f"  {lab:>7}: tLP {t:.4f}s  x{t / base:.2f}   quality {q:.9f}")
    out = Path(__file__).parent / "results_L122_reps.json"
    out.write_text(json.dumps(
        dict(reps=reps, area_price=area_price, area_g=area_g,
             arms={lab: {str(c): dict(t_lp=best[lab][c], q=qual[lab][c],
                                      n=l3.CASES[c]["n"]) for c in cases}
                   for lab, _ in arms}), indent=1), encoding="utf-8")
    print(f"json -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate0", "ab", "reps"])
    ap.add_argument("--cases", default="0-99")
    ap.add_argument("--R", default="1.2,1.5,2.0")
    ap.add_argument("--g", type=float, default=1.05)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--area-price", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.mode == "gate0":
        sys.exit(0 if mode_gate0(area_g=a.g) else 1)
    sel = []
    for part in a.cases.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            sel += list(range(int(lo), int(hi) + 1))
        else:
            sel.append(int(part))
    if a.mode == "reps":
        mode_reps(sel, [float(x) for x in a.R.split(",")], a.g, a.area_price,
                  a.reps)
        return
    mode_ab(sel, [float(x) for x in a.R.split(",")], a.g, a.iters,
            a.area_price)


if __name__ == "__main__":
    main()
