"""OFFLINE (never shipped): L120 stage L-4, route C.

ROUTE C = alternate.  Positions come from the LP (HiGHS, HPWL objective kept);
shapes come from L-3's closed form, which holds w*h = A EXACTLY.  Route A
(relax HPWL into the multipliers) needs two rows per b2b edge and case 99 has
7056 of them; route B (min-cost flow) is correct but a pure-Python MCF hits the
same constant wall L-2 just did.  C attacks the actual bottleneck -- the rho
trust region -- instead of the position step, which the budget analysis says is
only ~0.3x.

WHY THE GRADIENT HAS TO COME FROM A SHAPE-PINNED SOLVE.  At the shipped
rho=0.06 optimum the shape columns are mostly INTERIOR, so their reduced costs
are zero and there is no gradient to read.  Worse, where the area band is tight
and the column is interior, stationarity on that column reads

    g_w - h*mu = 0,   g_h - w*mu = 0     =>     g_w * w == g_h * h

which is EXACTLY the closed form's KKT condition.  So feeding the linearised
LP's own duals into the closed form returns the shape it already has, up to the
second-order area error -- which is precisely the recorded 0/100 of the cheap LR
attempt.  Pinning the shape columns and dropping the area band (`fix_dsize`)
removes that stationarity condition, and the reduced costs then are the true
one-sided derivatives of the objective with positions optimised out.

  a_u = d(obj)/d(w_u) = obj[kw] + sum of the multipliers of every row dw_u
                                  appears in (all `<=`, all coefficient +1)

so a_u >= 0 whenever the HPWL half is not pruned into the objective -- except
through the |z| aux rows, where the two rows carry +0.5s and -0.5s and their
difference CAN be negative.  A block genuinely can want to be wider because that
moves its centre towards its net.  a <= 0 means "nothing pushes back", the
minimiser runs to infinity, and the caller must clamp; that is a real answer and
CAP below is where it is dealt with, not a place to hide it.

ROUND STRUCTURE.  Each LP does double duty:

    B_r = LP(positions | shapes pinned at S_r)   -> positions AND the gradient at S_r
    S_{r+1} = closed form from that gradient

so K shape updates cost K+1 LP passes (the last shape still needs positions).
K=1 is therefore 2x one pass, already over the <1.8x budget on record -- this
file measures QUALITY first, because if the quality is not there the cost does
not matter, and if it is there the last legalisation can be moved to L-2's
shortest path at 0.3x.

SELF-GATE (handoff note 4: every probe needs a degenerate case that must
reproduce a known value).  `gate0` checks three things:
  1. the shipped build is untouched -- case 99 objective 1.747467759676;
  2. CAP=0 (no shape may move) reproduces the rho=0 LP objective case by case,
     so the pinning/area-band-dropping machinery is a no-op when it should be;
  3. the gradient is real: computed two independent ways it must agree, and a
     finite-difference re-solve must move the objective by a_u * eps.
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

DUMP = Path(__file__).parent / "results_L121_route_c.json"


# ---------------------------------------------------------------------------
# the gradient

def shape_gradient(B):
    """{uid: (a, b)} = (d obj / d w, d obj / d h) as reduced costs.

    Two independent expressions are returned by `gradient_check`; this one is
    `c - A' lambda`, which is the definition rather than a solver convenience
    field, so it is the one the route uses.
    """
    res = B["res"]
    rc = np.asarray(B["obj"], dtype=float).copy()
    rc -= B["A_ub"].T @ np.asarray(res.ineqlin.marginals, dtype=float)
    if B["A_eq"] is not None:
        rc -= B["A_eq"].T @ np.asarray(res.eqlin.marginals, dtype=float)
    return {uid: (float(rc[kw]), float(rc[kh])) for uid, (kw, kh) in B["sv"].items()}


def closed_form_shapes(ci, P, B, grad, cap=2.0):
    """{uid: (dw, dh)} from `minimise a*w + b*h  s.t.  w*h = A`.

    Returns deltas against the shapes in P, so the caller can hand them straight
    to `fix_dsize`.  `cap` bounds the width ratio w'/w; a <= 0 or b <= 0 means
    the linear model is unbounded in that direction and the cap is the answer,
    which is recorded in the returned stats rather than silently applied.
    """
    at = l3.CASES[ci]["at"]
    resh = L.reshapeable(ci, B["units"])
    out, stats = {}, dict(n=0, capped=0, unbounded=0, degenerate=0, moved=0)
    for uid, (a, b) in grad.items():
        i = resh[uid]
        w, h = P[i][2], P[i][3]
        A = float(at[i]) if float(at[i]) > 0 else w * h
        stats["n"] += 1
        if a <= 0.0 and b <= 0.0:
            # both axes want to grow and the area is fixed: the linear model has
            # no interior answer, and neither cap direction is justified
            stats["degenerate"] += 1
            out[uid] = (0.0, 0.0)
            continue
        if a <= 0.0 or b <= 0.0:
            stats["unbounded"] += 1
            r = cap if a <= 0.0 else 1.0 / cap
        else:
            r = ((A * b / a) ** 0.5) / w
        rc = min(max(r, 1.0 / cap), cap)
        if rc != r:
            stats["capped"] += 1
        w2 = w * rc
        h2 = A / w2
        if abs(w2 - w) > 1e-12 or abs(h2 - h) > 1e-12:
            stats["moved"] += 1
        out[uid] = (w2 - w, h2 - h)
    return out, stats


def _rho_for(fix, P, B):
    """A bound on |dsize| / dim over the pinned shapes.

    `rho` no longer sizes a trust region here, but it still feeds the separation
    reduction mask and the HPWL prune slack, both of which only need an upper
    bound on how much a size can move.  Over-estimating is the safe side.
    """
    r = 0.0
    for uid, (dw, dh) in fix.items():
        i = B["units"][uid][0]
        r = max(r, abs(dw) / max(P[i][2], 1e-12), abs(dh) / max(P[i][3], 1e-12))
    return r


# ---------------------------------------------------------------------------
# one route-C pass, mirroring lp_pass's cluster re-freeze loop

def pinned_pass(ci, P, fix, rho, sep_trim=False, bbox_slack=None):
    """LP with the shape columns pinned; returns (newP, telemetry, B).

    Mirrors `lp_pass`: the same three-attempt loop that re-freezes any cluster
    the solution split.  `hard_ok` does not test contiguity, so a probe that
    skips this loop scores cases that the real pipeline would reject.
    """
    c = l3.CASES[ci]
    freeze = set()
    for attempt in range(3):
        t0 = time.perf_counter()
        B = L.build_and_solve(ci, P, freeze, rho=rho, sep_trim=sep_trim,
                              fix_dsize=fix, bbox_slack=bbox_slack)
        if B["res"].status != 0:
            return None, dict(status=f"lp_status_{B['res'].status}",
                              t=time.perf_counter() - t0, attempts=attempt + 1,
                              lp_obj=None), None
        newP = L.apply_all(P, B, B["res"].x)
        broken = [g for g, c0 in B["group_comp0"].items()
                  if len(l3.comp_split(newP, [i for i in range(c["n"])
                                              if c["cn"][i][3] == g])) > c0]
        if not broken:
            return newP, dict(status="ok", t=time.perf_counter() - t0,
                              attempts=attempt + 1,
                              lp_obj=float(B["res"].fun)), B
        for g in broken:
            freeze |= B["group_units"][g]
    return None, dict(status="cluster_break", t=0.0, attempts=3, lp_obj=None), None


def route_c_case(ci, anch, rounds=1, cap=2.0, sep_trim=False, bbox_slack=None):
    """K = `rounds` shape updates, K+1 LP passes.

    The accept guard is `dep_case`'s, verbatim, so the number this produces is
    comparable with every other LP measurement on record: keep only if hpwl or
    area improved, nothing got worse, and the layout is hard-feasible.
    """
    P0 = [tuple(p) for p in anch[ci]["positions"]]
    q_ship = anch[ci]["cost"]
    m_prev = L.proxy_m(ci, P0)

    P = P0
    fix = {}
    rho = 0.0
    t_lp = 0.0
    trace = []
    shape_stats = None

    for r in range(rounds + 1):
        # round 0 pins every shape at zero, so it needs no room the shipped
        # layout does not already have; the slack only exists for the reshaped
        # rounds, and giving it to round 0 would let the very first pass buy
        # HPWL with area before a single shape has moved.
        newP, tele, B = pinned_pass(ci, P, fix, rho, sep_trim=sep_trim,
                                    bbox_slack=None if r == 0 else bbox_slack)
        t_lp += tele["t"]
        trace.append(dict(round=r, status=tele["status"], t=tele["t"],
                          lp_obj=tele["lp_obj"], attempts=tele["attempts"]))
        if newP is None:
            # A failed round leaves the PREVIOUS layout, which for r>0 is the
            # round-0 re-placement -- reporting that as route C's score would
            # credit the route with a gain it did not produce, so the score
            # falls back to the shipped one and the failure is what is reported.
            return dict(ci=ci, n=l3.CASES[ci]["n"], q_ship=q_ship,
                        q_new=q_ship, q_raw=float(l3.cost_eval(ci, P).cost),
                        feas=bool(l3.cost_eval(ci, P).is_feasible), kept=0,
                        status=tele["status"], failed_round=r, t_lp=t_lp,
                        trace=trace, shape_stats=shape_stats)
        P = newP
        if r == rounds:
            break
        grad = shape_gradient(B)
        fix, shape_stats = closed_form_shapes(ci, P, B, grad, cap=cap)
        rho = _rho_for(fix, P, B)

    m_new = L.proxy_m(ci, P)
    better = (m_new["hpwl"] < m_prev["hpwl"] * (1 - 1e-12)
              or m_new["area"] < m_prev["area"] * (1 - 1e-12))
    worse = (m_new["hpwl"] > m_prev["hpwl"] * (1 + 1e-12)
             or m_new["area"] > m_prev["area"] * (1 + 1e-12)
             or m_new["vrel"] > m_prev["vrel"] + 1e-12)
    keep = bool(better and not worse and L.hard_ok(P0, P, ci))
    mm = l3.cost_eval(ci, P)
    return dict(ci=ci, n=l3.CASES[ci]["n"], q_ship=q_ship,
                q_new=float(mm.cost) if keep else q_ship,
                q_raw=float(mm.cost), feas=bool(mm.is_feasible),
                kept=int(keep), status="ok", t_lp=t_lp, trace=trace,
                shape_stats=shape_stats,
                guard=dict(better=bool(better), worse=bool(worse),
                           hard=bool(L.hard_ok(P0, P, ci))))


# ---------------------------------------------------------------------------
# gates

def gate0(cases=(60, 95, 99), eps=1e-4):
    anch = _anchor()
    ok = True

    P = [tuple(p) for p in anch[99]["positions"]]
    B = L.build_and_solve(99, P, set(), rho=0.06)
    v = float(B["res"].fun)
    hit = abs(v - 1.747467759676) < 1e-11
    ok &= hit
    print(f"  G1 shipped build untouched: case 99 obj {v:.12f} -> "
          f"{'PASS' if hit else 'FAIL'}")

    bad = []
    for ci in cases:
        P = [tuple(p) for p in anch[ci]["positions"]]
        a = L.build_and_solve(ci, P, set(), rho=0.0)
        b = L.build_and_solve(ci, P, set(), rho=0.0, fix_dsize={})
        if a["res"].status != 0 or b["res"].status != 0:
            bad.append((ci, "status"))
            continue
        rel = abs(a["res"].fun - b["res"].fun) / max(abs(a["res"].fun), 1e-30)
        if rel > 1e-12:
            bad.append((ci, rel))
    ok &= not bad
    print(f"  G2 pinned-at-zero == rho=0 LP on {len(cases)} cases: "
          f"{'PASS' if not bad else f'FAIL {bad}'}")

    worst_pair = worst_fd = 0.0
    for ci in cases:
        P = [tuple(p) for p in anch[ci]["positions"]]
        B = L.build_and_solve(ci, P, set(), rho=0.0, fix_dsize={})
        res = B["res"]
        rc1 = shape_gradient(B)
        lo = np.asarray(res.lower.marginals, dtype=float)
        up = np.asarray(res.upper.marginals, dtype=float)
        f0 = float(res.fun)
        for uid, (kw, kh) in B["sv"].items():
            for k, g in ((kw, rc1[uid][0]), (kh, rc1[uid][1])):
                worst_pair = max(worst_pair, abs(g - (lo[k] + up[k]))
                                 / max(abs(g), 1.0))
        # finite difference on the largest-gradient unit of this case
        uid = max(rc1, key=lambda u: abs(rc1[u][0]))
        a = rc1[uid][0]
        i = B["units"][uid][0]
        step = eps * P[i][2]
        B2 = L.build_and_solve(ci, P, set(), rho=eps, fix_dsize={uid: (step, 0.0)})
        if B2["res"].status == 0:
            pred, got = a * step, float(B2["res"].fun) - f0
            worst_fd = max(worst_fd, abs(pred - got) / max(abs(got), 1e-12))
    p1 = worst_pair <= 1e-6
    p2 = worst_fd <= 1e-6
    ok &= p1 and p2
    print(f"  G3 gradient two ways agree, worst rel {worst_pair:.2e} -> "
          f"{'PASS' if p1 else 'FAIL'}")
    print(f"  G3 finite difference matches a*eps, worst rel {worst_fd:.2e} -> "
          f"{'PASS' if p2 else 'FAIL'}")
    print(f"gate0 -> {'ALL PASS' if ok else 'FAIL'}")
    return ok


def elastic_blame(B, tol=1e-7):
    """Which row family makes an infeasible build infeasible.

    Phase-1 on the SAME matrix: give every row its own nonnegative slack, price
    each at 1, and minimise the total.  Rows that take slack are the ones that
    cannot all hold at once, so the origins with nonzero slack name the blocking
    family.  This is not a minimal IIS -- it is the cheapest honest answer, and
    it needs no solver features scipy does not have.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    A_ub, A_eq = B["A_ub"], B["A_eq"]
    m_ub = A_ub.shape[0]
    m_eq = A_eq.shape[0] if A_eq is not None else 0
    nx = A_ub.shape[1]
    ns = m_ub + 2 * m_eq
    obj = np.r_[np.zeros(nx), np.ones(ns)]
    Aub = sparse.hstack([A_ub,
                         -sparse.eye(m_ub, format="csr"),
                         sparse.csr_matrix((m_ub, 2 * m_eq))], format="csr")
    Aeq = (sparse.hstack([A_eq, sparse.csr_matrix((m_eq, m_ub)),
                          sparse.eye(m_eq, format="csr"),
                          -sparse.eye(m_eq, format="csr")], format="csr")
           if m_eq else None)
    res = linprog(obj, A_ub=Aub, b_ub=np.asarray(B["b_ub"]),
                  A_eq=Aeq, b_eq=np.asarray(B["b_eq"]) if m_eq else None,
                  bounds=list(B["bounds"]) + [(0.0, None)] * ns, method="highs")
    if res.status != 0:
        return None, res.status
    s = res.x[nx:]
    blame = {}
    for i, o in enumerate(B["origins_ub"]):
        if s[i] > tol:
            blame[o] = blame.get(o, 0) + 1
    for j, o in enumerate(B["origins_eq"]):
        if s[m_ub + j] > tol or s[m_ub + m_eq + j] > tol:
            blame[o] = blame.get(o, 0) + 1
    return blame, float(res.fun)


def mode_blame(cases, cap, bbox_slack):
    """Reproduce round 1 on each case and, if it is infeasible, name the rows."""
    anch = _anchor()
    for ci in cases:
        P0 = [tuple(p) for p in anch[ci]["positions"]]
        P, tele, B = pinned_pass(ci, P0, {}, 0.0)
        if P is None:
            print(f"case {ci:3d} round 0 already {tele['status']}")
            continue
        fix, st = closed_form_shapes(ci, P, B, shape_gradient(B), cap=cap)
        rho = _rho_for(fix, P, B)
        B2 = L.build_and_solve(ci, P, set(), rho=rho, fix_dsize=fix,
                               bbox_slack=bbox_slack)
        if B2["res"].status == 0:
            print(f"case {ci:3d} round 1 FEASIBLE (obj {B2['res'].fun:.6f})")
            continue
        blame, tot = elastic_blame(B2)
        print(f"case {ci:3d} round 1 status={B2['res'].status} "
              f"rho={rho:.4f} moved={st['moved']}/{st['n']} "
              f"| phase-1 residual {tot:.6g} blame {blame}")


def mode_rho(cases, rhos=(0.06, 0.12, 0.24, 0.48)):
    """Is the trust region what caps the shape step?

    The whole route exists because rho=0.06 is assumed to be the thing standing
    between us and a large aspect change.  That is testable in one line: widen
    rho and see whether the LP objective moves.  If it does not, the shape step
    is capped by the fixed topology -- a block wedged between two members of the
    same rigid cluster has a hard width ceiling no reformulation can lift -- and
    making the area constraint exact cannot buy anything, because the LP was
    never sitting on the area constraint in the first place.

    Also reports where the shape columns actually sit at rho=0.06: on the trust
    region bound (rho binds), at zero (the LP does not want to reshape at all),
    or interior (the LP found a stationary point well inside).
    """
    anch = _anchor()
    print(f"{'case':>5} {'n':>4} " + " ".join(f"obj@{r:<6}" for r in rhos)
          + "  d(obj) 0.06->max   at-bound  at-zero  interior")
    agg = [0, 0, 0]
    worst = 0.0
    for ci in cases:
        P = [tuple(p) for p in anch[ci]["positions"]]
        objs, at_b, at_z, inter = [], 0, 0, 0
        for rho in rhos:
            B = L.build_and_solve(ci, P, set(), rho=rho)
            if B["res"].status != 0:
                objs.append(float("nan"))
                continue
            objs.append(float(B["res"].fun))
            if rho != rhos[0]:
                continue
            x, resh = B["res"].x, L.reshapeable(ci, B["units"])
            for uid, (kw, kh) in B["sv"].items():
                i = resh[uid]
                for k, dim in ((kw, P[i][2]), (kh, P[i][3])):
                    lim = rho * dim
                    if abs(abs(x[k]) - lim) <= 1e-9 * max(lim, 1.0):
                        at_b += 1
                    elif abs(x[k]) <= 1e-9 * max(dim, 1.0):
                        at_z += 1
                    else:
                        inter += 1
        rel = (objs[0] - min(objs)) / max(abs(objs[0]), 1e-30)
        worst = max(worst, rel)
        agg = [agg[0] + at_b, agg[1] + at_z, agg[2] + inter]
        print(f"{ci:5d} {l3.CASES[ci]['n']:4d} "
              + " ".join(f"{o:10.6f}" for o in objs)
              + f"   {100 * rel:+8.4f}%   {at_b:6d}  {at_z:6d}  {inter:6d}")
    tot = sum(agg) or 1
    print(f"\n  shape columns at rho={rhos[0]}: on-bound {agg[0]} "
          f"({100 * agg[0] / tot:.1f}%)  at-zero {agg[1]} "
          f"({100 * agg[1] / tot:.1f}%)  interior {agg[2]} "
          f"({100 * agg[2] / tot:.1f}%)")
    print(f"  worst objective gain from widening rho to {max(rhos)}: "
          f"{100 * worst:+.4f}%")


def _anchor():
    return {t["test_id"]: t for t in
            json.loads(L.ANCHOR.read_text(encoding="utf-8"))["test_results"]}


def mode_run(cases, rounds, cap, sep_trim, bbox_slack=None):
    anch = _anchor()
    W, TOTW = l3.W, l3.TOTW
    rows = {}
    for ci in cases:
        r = route_c_case(ci, anch, rounds=rounds, cap=cap, sep_trim=sep_trim,
                         bbox_slack=bbox_slack)
        rows[ci] = r
        ss = r.get("shape_stats") or {}
        print(f"case {ci:3d} n={r['n']:3d} kept={r['kept']} feas={int(r['feas'])} "
              f"q {r['q_ship']:.6f} -> {r.get('q_raw', float('nan')):.6f} "
              f"t={r['t_lp']:.3f}s moved={ss.get('moved', 0)}/{ss.get('n', 0)} "
              f"cap={ss.get('capped', 0)} unb={ss.get('unbounded', 0)} "
              f"status={r['status']}")

    cov = sum(W[c] for c in cases)
    ship = sum(W[c] * rows[c]["q_ship"] for c in cases) / cov
    after = sum(W[c] * rows[c]["q_new"] for c in cases) / cov
    print("\n=== ROUTE C ===")
    print(f"  cases            = {len(cases)}  rounds={rounds} cap={cap}")
    print(f"  shipped weighted = {ship:.9f}")
    print(f"  route C weighted = {after:.9f}   gain {100.0 * (1 - after / ship):+.4f}%")
    print(f"  kept             = {sum(r['kept'] for r in rows.values())}/{len(cases)}")
    print(f"  infeasible       = {[c for c in cases if not rows[c]['feas']]}")
    DUMP.write_text(json.dumps(
        dict(mode="run", rounds=rounds, cap=cap, sep_trim=sep_trim,
             shipped=ship, after=after,
             gain=100.0 * (1 - after / ship),
             cases={str(c): rows[c] for c in cases}), indent=1), encoding="utf-8")
    print(f"json -> {DUMP}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate0", "run", "blame", "rho"])
    ap.add_argument("--cases", default="0-99")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--sep-trim", action="store_true")
    ap.add_argument("--bbox-slack", type=float, default=None)
    a = ap.parse_args()
    if a.mode == "gate0":
        sys.exit(0 if gate0() else 1)
    sel = []
    for part in a.cases.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            sel += list(range(int(lo), int(hi) + 1))
        else:
            sel.append(int(part))
    if a.mode == "rho":
        mode_rho(sel)
        return
    if a.mode == "blame":
        mode_blame(sel, a.cap, a.bbox_slack)
        return
    mode_run(sel, a.rounds, a.cap, a.sep_trim, a.bbox_slack)


if __name__ == "__main__":
    main()
