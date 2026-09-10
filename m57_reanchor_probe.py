"""M57 probe (OFFLINE, never shipped): frozen-violator bbox re-anchor LP.

Copy of m53_l3_probe.py (M57_PLAN.md rule: original files untouched) with ONE
semantic change in build_and_solve's force_bnd handling plus a new `reanchor`
driver mode. M53's repair mode moved a violating block TO the bbox edge and
returned None for frozen violators; M57 tests the mirrored semantics the dead-
end ledger never measured: the frozen violator stays put and the bbox edge is
PINNED onto it (the evaluator recomputes the bbox from the solution, so every
other movable extreme must retreat inward). Bounds already forbid the bbox var
crossing any frozen block, and a frozen satisfied boundary block on the same
side pins the var at the old extreme -> both conflicts surface as LP
infeasibility = correct geometric death (no special-casing).

Original M53 header follows.

M53 L3 probe (OFFLINE, never shipped): constraint-graph LP post-processing.

Takes the final chosen layout per case from an anchor results json (default
results_L1_final.json = the L1 quality anchor 1.3176), freezes all dims and
the pairwise left-of/below separation topology, and re-optimizes every movable
position SIMULTANEOUSLY with an LP whose objective is the exact evaluator
HPWL (per-edge center-to-center weighted Manhattan -> linear in (x,y) when
dims are fixed).

Model (delta formulation, per case):
  vars    (dx,dy) per mobile unit; bbox (Xmin,Xmax,Ymin,Ymax); 2 aux per edge
  unit    free block | cluster connected component (RIGID: one delta for all
          members -> grouping union preserved up to FP) ; preplaced blocks and
          components containing one are frozen constants (original bytes kept)
  edge    aux t >= +/-(Ce + dx_i - dx_j), objective sum w*(tx+ty); edges with
          both ends in one unit are constants (rigid motion cancels)
  topo    per pair (skip intra-unit): keep the currently separating axis,
          dx_left - dx_right <= current gap (gap may be ~0: abutment stays legal,
          eval overlap needs BOTH axes > 1e-6 while LP tol ~1e-7 on one axis)
  bnd     eval checks |edge - solution_bbox_edge| < 1e-6 with bbox recomputed
          from the solution -> tie every currently-satisfied boundary block to
          the bbox var preserving its offset (dx = Xmin - xmin0 form), anchor
          the current extreme-defining block to the var, envelope rows keep
          everything inside; currently-violating boundary blocks left free
          (they stay violated; guard catches surprises). Frozen boundary block
          on a side pins that side's var.
  bbox    Xmax-Xmin <= W0, Ymax-Ymin <= H0 (may shrink, never grow)

Safety ladder: after LP, every cluster group's shapely component count must
not exceed the original (M10 zero-tolerance wall); if a group breaks (ULP),
re-solve with that group's units frozen (up to 3 attempts). Final arbiter is
the official strict evaluate_solution (target_positions passed): a case is
kept only if feasible AND cost strictly improves, else revert -> zero
regression by construction.

modes:
  gate0            pass-through anchor positions -> per-case evaluated cost
                   must equal the anchor json cost exactly (loader/scorer gate)
  probe15          LP on cases 85..99 (68% of hgap headroom), report
  full             LP on all 100 cases, report + results_L3_lp.json
flags:
  --anchor PATH    anchor results json (default results_L1_final.json)
  --iters K        LP passes per case, topology rebuilt between passes (def 2)
  --cases A-B      override case range for probe mode
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from shapely.geometry import box as _sbox  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

EPS_BND = 1e-6

# ── dataset (all 100 cases: LP needs full constraint metadata) ───────────────
print("[load] dataset ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
CASES, W = {}, {}
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _inp, _lab = _s["input"], _s["label"]
    _at, _b2b, _p2b, _pins, _cons = _inp
    _n = int((_at != -1).sum().item())
    W[_idx] = math.exp(_n / 12.0)
    _base, _tp = _ev._extract_baseline(_idx, _lab, _b2b, _p2b, _pins, _n)
    CASES[_idx] = dict(
        idx=_idx, n=_n, base=_base, tp=_tp, at=_at, b2b=_b2b, p2b=_p2b,
        pins=_pins, cons=_cons,
        b2l=[(int(e[0]), int(e[1]), float(e[2]))
             for e in _b2b.tolist() if int(e[0]) != -1],
        p2l=[(int(e[0]), int(e[1]), float(e[2]))
             for e in _p2b.tolist() if int(e[0]) != -1],
        pin=[(float(p[0]), float(p[1])) for p in _pins.tolist()],
        cn=[[int(v) for v in _cons[i].tolist()] for i in range(_n)],
    )
TOTW = sum(W.values())
print(f"[load] 100 cases ready", flush=True)


def cost_eval(ci, ps):
    """Official strict scoring (target_positions passed -> hard checks on)."""
    c = CASES[ci]
    return evaluate_solution(
        {"positions": ps, "runtime": 1.0}, c["base"], c["cons"][: c["n"]],
        c["b2b"], c["p2b"], c["pins"], c["at"][: c["n"]],
        target_positions=c["tp"][: c["n"]], median_runtime=1.0)


def comp_split(P, mem):
    """Cluster members -> connected components exactly as the evaluator sees
    them (unary_union geoms; zero tolerance)."""
    boxes = {i: _sbox(P[i][0], P[i][1], P[i][0] + P[i][2], P[i][1] + P[i][3])
             for i in mem}
    u = unary_union(list(boxes.values()))
    geoms = [u] if u.geom_type == "Polygon" else list(u.geoms)
    comps = [[] for _ in geoms]
    for i in mem:
        b = boxes[i]
        k = max(range(len(geoms)), key=lambda t: geoms[t].intersection(b).area)
        comps[k].append(i)
    return comps


# ── LP build + solve (one pass) ──────────────────────────────────────────────
def build_and_solve(ci, P, freeze_units, area_obj=False, force_bnd=()):
    c = CASES[ci]
    n, cn = c["n"], c["cn"]

    # units: rigid cluster components / free singles; preplaced (and any
    # component containing one) frozen
    frozen_blk = {i for i in range(n) if cn[i][1] != 0}
    unit_of = [None] * n
    units = []
    group_units, group_comp0 = {}, {}
    for g in sorted({cn[i][3] for i in range(n) if cn[i][3] > 0}):
        mem = [i for i in range(n) if cn[i][3] == g]
        comps = comp_split(P, mem)
        group_comp0[g] = len(comps)
        gset = set()
        for cm in comps:
            if any(i in frozen_blk for i in cm):
                frozen_blk.update(cm)
            else:
                uid = len(units)
                units.append(cm)
                for i in cm:
                    unit_of[i] = uid
                gset.add(uid)
        group_units[g] = gset
    for i in range(n):
        if unit_of[i] is None and i not in frozen_blk:
            uid = len(units)
            units.append([i])
            unit_of[i] = uid

    U = len(units)
    XMIN, XMAX, YMIN, YMAX = 2 * U, 2 * U + 1, 2 * U + 2, 2 * U + 3
    nv = 2 * U + 4
    obj = [0.0] * nv
    rub, cub, vub, bub = [], [], [], []
    req, ceq, veq, beq = [], [], [], []

    def add_ub(terms, rhs):
        r = len(bub)
        bub.append(rhs)
        for col, coef in terms:
            rub.append(r), cub.append(col), vub.append(coef)

    def add_eq(terms, rhs):
        r = len(beq)
        beq.append(rhs)
        for col, coef in terms:
            req.append(r), ceq.append(col), veq.append(coef)

    def new_aux(w):
        nonlocal nv
        obj.append(w)
        nv += 1
        return nv - 1

    # HPWL objective (aux linearization); intra-unit / all-frozen edges const.
    # cost-aligned scale so an optional area term can join the same objective:
    # quality_factor = 1 + 0.5*(dH/h_base + dArea/a_base)
    h_base = max(float(c["base"].get("hpwl_baseline", 1.0)), 1e-6)
    hw_scale = 0.5 / h_base
    cx = [P[i][0] + P[i][2] / 2.0 for i in range(n)]
    cy = [P[i][1] + P[i][3] / 2.0 for i in range(n)]
    const_h = 0.0
    obj0 = 0.0

    def edge_axis(t, ui, uj, off, dC):
        # t >= dC + d_i - d_j   and   t >= -(dC + d_i - d_j)
        t1 = [(t, -1.0)]
        t2 = [(t, -1.0)]
        if ui is not None:
            t1.append((off + ui, 1.0)), t2.append((off + ui, -1.0))
        if uj is not None:
            t1.append((off + uj, -1.0)), t2.append((off + uj, 1.0))
        add_ub(t1, -dC), add_ub(t2, dC)

    for i, j, w in c["b2l"]:
        ui, uj = unit_of[i], unit_of[j]
        dCx, dCy = cx[i] - cx[j], cy[i] - cy[j]
        if w <= 0.0 or ui == uj:
            const_h += w * (abs(dCx) + abs(dCy))
            continue
        edge_axis(new_aux(w * hw_scale), ui, uj, 0, dCx)
        edge_axis(new_aux(w * hw_scale), ui, uj, U, dCy)
        obj0 += w * (abs(dCx) + abs(dCy))
    for p, i, w in c["p2l"]:
        ui = unit_of[i]
        px, py = c["pin"][p]
        dCx, dCy = cx[i] - px, cy[i] - py
        if w <= 0.0 or ui is None:
            const_h += w * (abs(dCx) + abs(dCy))
            continue
        edge_axis(new_aux(w * hw_scale), ui, None, 0, dCx)
        edge_axis(new_aux(w * hw_scale), ui, None, U, dCy)
        obj0 += w * (abs(dCx) + abs(dCy))

    # pairwise separation: keep the currently separating axis
    for i in range(n):
        xi, yi, wi, hi = P[i]
        for j in range(i + 1, n):
            ui, uj = unit_of[i], unit_of[j]
            if ui == uj:  # same unit, or both frozen (None==None)
                continue
            xj, yj, wj, hj = P[j]
            cands = ((xj - (xi + wi), ui, uj, 0),   # i left of j
                     (xi - (xj + wj), uj, ui, 0),   # j left of i
                     (yj - (yi + hi), ui, uj, U),   # i below j
                     (yi - (yj + hj), uj, ui, U))   # j below i
            gap, ul, ur, off = max(cands, key=lambda t: t[0])
            terms = []
            if ul is not None:
                terms.append((off + ul, 1.0))
            if ur is not None:
                terms.append((off + ur, -1.0))
            add_ub(terms, gap)

    # bbox / boundary
    xmin0 = min(P[i][0] for i in range(n))
    xmax0 = max(P[i][0] + P[i][2] for i in range(n))
    ymin0 = min(P[i][1] for i in range(n))
    ymax0 = max(P[i][1] + P[i][3] for i in range(n))
    W0, H0 = xmax0 - xmin0, ymax0 - ymin0

    def touch_ok(i, code):
        x, y, w, h = P[i]
        return ((not code & 1 or abs(x - xmin0) < EPS_BND)
                and (not code & 2 or abs(x + w - xmax0) < EPS_BND)
                and (not code & 4 or abs(y + h - ymax0) < EPS_BND)
                and (not code & 8 or abs(y - ymin0) < EPS_BND))

    bnd = [(i, cn[i][4]) for i in range(n) if cn[i][4] != 0]
    sat = [(i, code) for i, code in bnd if touch_ok(i, code)]
    bnd_skip = len(bnd) - len(sat)
    # M57: sides about to be pinned onto a frozen force_bnd violator -- the
    # frozen pin itself is the anchor there (envelope keeps everyone inside,
    # the frozen block sits exactly on the edge), so the old extreme-definer
    # anchor would over-constrain (forcing it to stop exactly on the new edge)
    # force_bnd entries: block index (pin ALL its code bits) or (block, mask)
    # (pin only the masked bits -- diag2 forensics)
    fb = [(e, cn[e][4]) if not isinstance(e, tuple) else e for e in force_bnd]
    pin_bits = 0
    for i, mask in fb:
        if unit_of[i] is None:
            pin_bits |= mask
    # side spec: (bit, bboxvar, axis-offset, current extreme, extreme-definer)
    sides = ((1, XMIN, 0, xmin0, min(range(n), key=lambda i: P[i][0])),
             (2, XMAX, 0, xmax0, max(range(n), key=lambda i: P[i][0] + P[i][2])),
             (4, YMAX, U, ymax0, max(range(n), key=lambda i: P[i][1] + P[i][3])),
             (8, YMIN, U, ymin0, min(range(n), key=lambda i: P[i][1])))
    for bit, bv, off, ext0, mdef in sides:
        tied = {unit_of[i] for i, code in sat if code & bit}
        if not tied:
            continue
        # every satisfied boundary block's unit moves with the edge:
        # d = BV - ext0  ->  d - BV = -ext0 ; frozen unit pins BV = ext0
        for u in tied:
            if u is None:
                add_eq([(bv, -1.0)], -ext0)
            else:
                add_eq([(off + u, 1.0), (bv, -1.0)], -ext0)
        # anchor: current extreme-definer keeps offset 0 so the bbox var IS
        # the recomputed extreme the evaluator will see
        um = unit_of[mdef]
        if um not in tied and not (pin_bits & bit):
            if um is None:
                add_eq([(bv, -1.0)], -ext0)
            else:
                add_eq([(off + um, 1.0), (bv, -1.0)], -ext0)

    # opportunistic repair: force a currently-VIOLATING boundary block onto
    # its edge(s) with exact-touch equalities (self-anchoring: the forced
    # block sits ON the bbox var, envelope makes the var the true extreme)
    for i, code in fb:
        ui = unit_of[i]
        x, y, w, h = P[i]
        if ui is None:
            # M57 reanchor: the frozen violator cannot move, so pin the bbox
            # edge onto it and let every mobile extreme retreat. Bounds above
            # already forbid the var crossing any frozen block (a more-outside
            # frozen block, or a frozen satisfied boundary block tied at the
            # old extreme, makes the LP infeasible = correct geometric death).
            if code & 1:
                add_eq([(XMIN, 1.0)], x)
            if code & 2:
                add_eq([(XMAX, 1.0)], x + w)
            if code & 4:
                add_eq([(YMAX, 1.0)], y + h)
            if code & 8:
                add_eq([(YMIN, 1.0)], y)
            continue
        if code & 1:
            add_eq([(ui, 1.0), (XMIN, -1.0)], -x)
        if code & 2:
            add_eq([(ui, 1.0), (XMAX, -1.0)], -(x + w))
        if code & 4:
            add_eq([(U + ui, 1.0), (YMAX, -1.0)], -(y + h))
        if code & 8:
            add_eq([(U + ui, 1.0), (YMIN, -1.0)], -y)

    # envelope (mobile blocks stay inside the bbox vars)
    for i in range(n):
        ui = unit_of[i]
        if ui is None:
            continue
        x, y, w, h = P[i]
        add_ub([(XMIN, 1.0), (ui, -1.0)], x)
        add_ub([(ui, 1.0), (XMAX, -1.0)], -(x + w))
        add_ub([(YMIN, 1.0), (U + ui, -1.0)], y)
        add_ub([(U + ui, 1.0), (YMAX, -1.0)], -(y + h))
    # bbox never grows
    add_ub([(XMAX, 1.0), (XMIN, -1.0)], W0)
    add_ub([(YMAX, 1.0), (YMIN, -1.0)], H0)

    # optional joint area term: Area ~= H0*W + W0*H - W0*H0 (first-order upper
    # bound for W<=W0,H<=H0) -> cost-aligned coefficient 0.5/a_base; only worth
    # it while the current bbox is above baseline (agap clamps at 0). The area
    # pressure also keeps the bbox vars tight on the envelope.
    a_base = max(float(c["base"].get("area_baseline", W0 * H0)), 1e-6)
    if area_obj and W0 * H0 > a_base:
        bA = 0.5 / a_base
        obj[XMIN] -= bA * H0
        obj[XMAX] += bA * H0
        obj[YMIN] -= bA * W0
        obj[YMAX] += bA * W0

    # ladder re-solve: freeze broken groups' units
    for u in freeze_units:
        add_eq([(u, 1.0)], 0.0)
        add_eq([(U + u, 1.0)], 0.0)

    # bounds: frozen blocks are constants -> bbox vars may not cross them
    D = W0 + H0 + 1.0
    fro = [i for i in range(n) if unit_of[i] is None]
    bounds = [(-D, D)] * (2 * U)
    bounds.append((xmin0 - D, min((P[i][0] for i in fro), default=xmin0 + D)))
    bounds.append((max((P[i][0] + P[i][2] for i in fro), default=xmax0 - D),
                   xmax0 + D))
    bounds.append((ymin0 - D, min((P[i][1] for i in fro), default=ymin0 + D)))
    bounds.append((max((P[i][1] + P[i][3] for i in fro), default=ymax0 - D),
                   ymax0 + D))
    bounds += [(0.0, None)] * (nv - 2 * U - 4)

    A_ub = sparse.csr_matrix((vub, (rub, cub)), shape=(len(bub), nv))
    A_eq = (sparse.csr_matrix((veq, (req, ceq)), shape=(len(beq), nv))
            if beq else None)
    res = linprog(np.asarray(obj), A_ub=A_ub, b_ub=np.asarray(bub),
                  A_eq=A_eq, b_eq=np.asarray(beq) if beq else None,
                  bounds=bounds, method="highs")
    return dict(res=res, units=units, unit_of=unit_of, U=U,
                group_units=group_units, group_comp0=group_comp0,
                const_h=const_h, obj0=obj0 + const_h, bnd_skip=bnd_skip)


def hpwl_of(ci, P):
    """Exact evaluator HPWL recomputed in python (self-check of the model)."""
    c = CASES[ci]
    tot = 0.0
    for i, j, w in c["b2l"]:
        tot += w * (abs((P[i][0] + P[i][2] / 2) - (P[j][0] + P[j][2] / 2))
                    + abs((P[i][1] + P[i][3] / 2) - (P[j][1] + P[j][3] / 2)))
    for p, i, w in c["p2l"]:
        px, py = c["pin"][p]
        tot += w * (abs((P[i][0] + P[i][2] / 2) - px)
                    + abs((P[i][1] + P[i][3] / 2) - py))
    return tot


def apply_deltas(P, units, dx, dy):
    newP = [tuple(p) for p in P]
    for u, mem in enumerate(units):
        ddx = 0.0 if abs(dx[u]) < 1e-12 else float(dx[u])
        ddy = 0.0 if abs(dy[u]) < 1e-12 else float(dy[u])
        if ddx == 0.0 and ddy == 0.0:
            continue
        for i in mem:
            x, y, w, h = P[i]
            newP[i] = (x + ddx, y + ddy, w, h)
    return newP


def lp_pass(ci, P, area_obj=False, force_bnd=()):
    """One LP pass with the cluster-precision re-solve ladder."""
    freeze = set()
    for attempt in range(3):
        B = build_and_solve(ci, P, freeze, area_obj=area_obj,
                            force_bnd=force_bnd)
        if B is None:
            return None, dict(status="force_frozen")
        if B["res"].status != 0:
            return None, dict(status=f"lp_status_{B['res'].status}")
        U = B["U"]
        x = B["res"].x
        newP = apply_deltas(P, B["units"], x[:U], x[U:2 * U])
        broken = [g for g, c0 in B["group_comp0"].items()
                  if len(comp_split(newP, [i for i in range(CASES[ci]["n"])
                                           if CASES[ci]["cn"][i][3] == g])) > c0]
        if not broken:
            return newP, dict(status="ok", U=U, bnd_skip=B["bnd_skip"],
                              pred_h=hpwl_of(ci, newP),
                              obj0=B["obj0"], attempts=attempt + 1,
                              frozen=len(freeze))
        for g in broken:
            freeze |= B["group_units"][g]
    return None, dict(status="cluster_break")


def run_case(ci, iters, area_obj=False):
    c0 = ANCH[ci]["cost"]
    P = [tuple(p) for p in ANCH[ci]["positions"]]
    best_P, best_cost, best_m = P, c0, None
    lines = []
    for it in range(iters):
        t0 = time.perf_counter()
        newP, tele = lp_pass(ci, P, area_obj=area_obj)
        dt = time.perf_counter() - t0
        if newP is None:
            lines.append(f"it{it}:{tele['status']}")
            break
        m = cost_eval(ci, newP)
        kept = bool(m.is_feasible) and m.cost < best_cost - 1e-12
        hh = m.hpwl_total
        pe = abs(hh - tele["pred_h"]) / max(hh, 1e-9)
        lines.append(
            f"it{it}: h {tele['obj0']:.0f}->{hh:.0f} (pred err {pe:.1e}) "
            f"cost {m.cost:.6f} vrel {m.violations_relative:.4f} "
            f"feas {int(m.is_feasible)} {'KEPT' if kept else 'rej'} "
            f"[U={tele['U']} bndskip={tele['bnd_skip']} "
            f"att={tele['attempts']} {dt:.1f}s]")
        if not kept:
            break
        best_P, best_cost, best_m = newP, m.cost, m
        P = newP
    return dict(ci=ci, cost0=c0, cost1=best_cost, P=best_P, m=best_m,
                lines=lines)


def mode_port15(k_top, iters, area_obj):
    """L3 x portfolio: LP the top-k proxy candidates per heavy case (positions
    from m53_l2_cache.pkl, sigma=0 L1-pool runs, byte-identical to shipped exe
    per the L2 gate) and re-select. Answers: does LP change the winner?"""
    import pickle
    db = pickle.load(open(_DIR / "m53_l2_cache.pkl", "rb"))["db"]
    RH = 1.4
    rows = []
    for ci in range(85, 100):
        c = CASES[ci]
        A_hat = 1.035 * max(sum(max(0.0, float(c["at"][i]))
                                for i in range(c["n"])), 1e-9)
        cand = [k for k in range(84) if ("run", ci, k, 0.0, 0.0, 0) in db
                and ("pm", ci, k, 0.0, 0.0, 0) in db]
        pm = {k: db[("pm", ci, k, 0.0, 0.0, 0)] for k in cand}
        hmin = min(v[1] for v in pm.values()) or 1.0
        prox = {k: (pm[k][0] / A_hat + RH * pm[k][1] / hmin)
                * math.exp(2 * pm[k][2]) for k in cand}
        top = sorted(cand, key=lambda k: prox[k])[:k_top]
        # consistency: rank-1 candidate positions must equal the anchor json
        w0 = top[0]
        p0 = [tuple(p) for p in db[("run", ci, w0, 0.0, 0.0, 0)][0]]
        pa = [tuple(p) for p in ANCH[ci]["positions"]]
        assert p0 == pa, f"case {ci}: cache winner != anchor json"
        res = {}
        for k in top:
            P = [tuple(p) for p in db[("run", ci, k, 0.0, 0.0, 0)][0]]
            m0 = cost_eval(ci, P)
            bestP, bestc, bestm = P, m0.cost, m0
            for it in range(iters):
                newP, tele = lp_pass(ci, P, area_obj=area_obj)
                if newP is None:
                    break
                m = cost_eval(ci, newP)
                if m.is_feasible and m.cost < bestc - 1e-12:
                    bestP, bestc, bestm = newP, m.cost, m
                    P = newP
                else:
                    break
            res[k] = (bestc, bestm)
        # post-LP proxy re-selection (deployable rule)
        h2 = {k: res[k][1].hpwl_total for k in top}
        hmin2 = min(h2.values()) or 1.0
        prox2 = {k: ((1 + res[k][1].area_gap) * float(c["base"].get(
            "area_baseline", 1.0)) / A_hat + RH * h2[k] / hmin2)
            * math.exp(2 * res[k][1].violations_relative) for k in top}
        ksel = min(top, key=lambda k: prox2[k])
        korc = min(top, key=lambda k: res[k][0])
        winlp, sel, orc = res[w0][0], res[ksel][0], res[korc][0]
        rows.append((ci, ANCH[ci]["cost"], winlp, sel, orc, ksel, korc, w0))
        print(f"case {ci:3d} anchor {ANCH[ci]['cost']:.6f} winnerLP {winlp:.6f}"
              f" selLP {sel:.6f} (k={ksel}) oracleLP {orc:.6f} (k={korc})"
              f"{' *WINNER-CHANGED*' if korc != w0 else ''}", flush=True)
    for name, col in (("winnerLP", 2), ("realizable", 3), ("oracle", 4)):
        g = sum(W[r[0]] * (r[1] - r[col]) for r in rows)
        print(f"port15 {name:10s}: weighted gain {g / TOTW / ANCHOR_TOTAL * 100:+.4f}%"
              f" of anchor (15 heavy cases)")


def mode_portfull(k_top, iters, area_obj):
    """L3 x portfolio on ALL 100 cases: run the 84-profile L1 pool where the
    L2 cache has no positions (cases 0..84), LP the top-k proxy candidates,
    re-select by the post-LP proxy. Pool runs cached in m53_l3_cache.pkl."""
    import concurrent.futures
    import hashlib
    import os
    import pickle
    import subprocess
    os.environ["ICCAD_L1_POOL"] = "1"        # BEFORE oc import
    os.environ["ICCAD_ADAPTIVE_POOL"] = "0"
    import optimizer_constructive as oc
    from optimizer_claude import _serialize_input, _parse_output
    from proxy_analysis import build_opt_target_pos
    PROFILES = list(oc._PROFILES)
    assert len(PROFILES) == 84, f"L1 pool expected 84, got {len(PROFILES)}"
    EXE = str(_DIR / "constructive.exe")
    RH = 1.4

    l2p = _DIR / "m53_l2_cache.pkl"
    l2db = pickle.load(open(l2p, "rb"))["db"] if l2p.exists() else {}
    cachep = _DIR / "m53_l3_cache.pkl"
    sig = repr((repr(PROFILES),
                hashlib.md5(open(EXE, "rb").read()).hexdigest()))
    db = {}
    if cachep.exists():
        try:
            _c = pickle.load(open(cachep, "rb"))
            if _c.get("sig") == sig:
                db = _c["db"]
            else:
                print("[cache] signature mismatch -> reset")
        except Exception:
            print("[cache] unreadable -> reset")

    def save():
        tmp = cachep.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"sig": sig, "db": db}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cachep)

    # 1) positions: adopt L2-cache runs (byte-identical exe, sigma=0), run the
    #    rest with the shipped exe
    for key in list(l2db.keys()):
        if key[0] == "run" and key[3] == 0.0 and key[4] == 0.0 and key[5] == 0:
            db.setdefault(("run", key[1], key[2]),
                          [tuple(p) for p in l2db[key][0]])
        if key[0] == "pm" and key[3] == 0.0 and key[4] == 0.0 and key[5] == 0:
            db.setdefault(("pm", key[1], key[2]), l2db[key])
    jobs = [(ci, k) for ci in range(100) for k in range(84)
            if ("run", ci, k) not in db]
    if jobs:
        txts = {}
        for ci in {j[0] for j in jobs}:
            c = CASES[ci]
            otp = build_opt_target_pos(c["tp"], c["cons"], c["n"])
            txts[ci] = _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"],
                                        c["pins"], c["cons"], otp,
                                        gnn_hint=None)

        def run_one(ci, k):
            env = dict(os.environ)
            env.update(PROFILES[k])
            r = subprocess.run([EXE], input=txts[ci], capture_output=True,
                               text=True, env=env, timeout=600)
            return _parse_output(r.stdout, CASES[ci]["n"])

        t0, done = time.perf_counter(), 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as ex:
            futs = {ex.submit(run_one, *j): j for j in jobs}
            for fut in concurrent.futures.as_completed(futs):
                ci, k = futs[fut]
                db[("run", ci, k)] = fut.result()
                done += 1
                if done % 200 == 0:
                    save()
                    print(f"  [pool] {done}/{len(jobs)} "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
        save()
        print(f"  [pool] done {len(jobs)} runs "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # 2) deployed shapely proxy metrics (serial: GIL)
    todo = [(ci, k) for ci in range(100) for k in range(84)
            if ("pm", ci, k) not in db]
    t0 = time.perf_counter()
    for cnt, (ci, k) in enumerate(todo, 1):
        c = CASES[ci]
        m = oc._proxy_metrics(db[("run", ci, k)], c["at"], c["b2b"], c["p2b"],
                              c["pins"], c["cons"], c["n"])
        db[("pm", ci, k)] = (m["area"], m["hpwl"], m["vrel"])
        if cnt % 500 == 0:
            save()
            print(f"  [pm] {cnt}/{len(todo)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    save()

    # 3) per case: top-k by deployed proxy -> LP each -> post-LP proxy select
    rows = []
    t0 = time.perf_counter()
    for ci in range(100):
        c = CASES[ci]
        A_hat = 1.035 * max(sum(max(0.0, float(c["at"][i]))
                                for i in range(c["n"])), 1e-9)
        pm = {k: db[("pm", ci, k)] for k in range(84)}
        hmin = min(v[1] for v in pm.values()) or 1.0
        prox = {k: (pm[k][0] / A_hat + RH * pm[k][1] / hmin)
                * math.exp(2 * pm[k][2]) for k in pm}
        top = sorted(pm, key=lambda k: prox[k])[:k_top]
        res = {}
        for k in top:
            kk = ("lp", ci, k, iters, area_obj)
            if kk in db:
                res[k] = db[kk]
                continue
            P = [tuple(p) for p in db[("run", ci, k)]]
            m0 = cost_eval(ci, P)
            bestP, bestc, bestm = P, m0.cost, m0
            for _ in range(iters):
                newP, tele = lp_pass(ci, P, area_obj=area_obj)
                if newP is None:
                    break
                m = cost_eval(ci, newP)
                if m.is_feasible and m.cost < bestc - 1e-12:
                    bestP, bestc, bestm = newP, m.cost, m
                    P = newP
                else:
                    break
            res[k] = (bestc, bestm.hpwl_total, bestm.area_gap,
                      bestm.violations_relative, bool(bestm.is_feasible),
                      bestm.hpwl_gap, bestP)
            db[kk] = res[k]
        save()
        h2 = {k: res[k][1] for k in top}
        hmin2 = min(h2.values()) or 1.0
        ab = float(c["base"].get("area_baseline", 1.0))
        prox2 = {k: ((1 + res[k][2]) * ab / A_hat + RH * h2[k] / hmin2)
                 * math.exp(2 * res[k][3]) for k in top}
        ksel = min(top, key=lambda k: prox2[k])
        korc = min(top, key=lambda k: res[k][0])
        rows.append((ci, ANCH[ci]["cost"], res[ksel], res[korc], ksel, korc,
                     top[0]))
        print(f"case {ci:3d} n={CASES[ci]['n']:3d} anchor "
              f"{ANCH[ci]['cost']:.6f} sel {res[ksel][0]:.6f} (k={ksel}) "
              f"orc {res[korc][0]:.6f} (k={korc})"
              f"{' *CHG*' if korc != top[0] else ''} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    for name, get in (("realizable", lambda r: r[2][0]),
                      ("oracle", lambda r: r[3][0])):
        tot = sum(W[r[0]] * get(r) for r in rows) / TOTW
        print(f"portfull {name:10s}: total {tot:.10f}  "
              f"({(ANCHOR_TOTAL - tot) / ANCHOR_TOTAL * 100:+.4f}% vs anchor)")
    # dump realizable selection as a results json
    trs = []
    for r in rows:
        ci, sel = r[0], r[2]
        trs.append(dict(test_id=ci, block_count=CASES[ci]["n"],
                        is_feasible=sel[4], hpwl_gap=sel[5], area_gap=sel[2],
                        violations_relative=sel[3], runtime_seconds=0.0,
                        cost=sel[0], positions=[list(p) for p in sel[6]],
                        error=None))
    tot = sum(W[t["test_id"]] * t["cost"] for t in trs) / TOTW
    out = dict(submission_name="L3_port",
               timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
               total_score=tot, test_results=trs,
               summary=dict(num_tests=100,
                            num_feasible=sum(t["is_feasible"] for t in trs),
                            avg_cost=sum(t["cost"] for t in trs) / 100,
                            avg_runtime=0.0))
    dump = str(_DIR / f"results_L3_port_top{k_top}"
                      f"{'_area' if area_obj else ''}.json")
    json.dump(out, open(dump, "w"))
    print(f"[dump] {dump}")


def mode_repair(src_json, iters, area_obj):
    """Greedy boundary-violation repair on top of a results json: for each
    currently-violating boundary block, try an LP with exact-touch equalities
    forced; keep if the official cost improves. (dbg_vio_stats' 0/202 verdict
    was single-block-move semantics; the LP moves everything simultaneously.)"""
    rj = json.load(open(src_json))
    rows = []
    t0 = time.perf_counter()
    fixed_blocks = 0
    for t in rj["test_results"]:
        ci = t["test_id"]
        c = CASES[ci]
        n, cn = c["n"], c["cn"]
        P = [tuple(p) for p in t["positions"]]
        cost0 = t["cost"]
        xmin0 = min(P[i][0] for i in range(n))
        xmax0 = max(P[i][0] + P[i][2] for i in range(n))
        ymin0 = min(P[i][1] for i in range(n))
        ymax0 = max(P[i][1] + P[i][3] for i in range(n))

        def _ok(i, code):
            x, y, w, h = P[i]
            return ((not code & 1 or abs(x - xmin0) < EPS_BND)
                    and (not code & 2 or abs(x + w - xmax0) < EPS_BND)
                    and (not code & 4 or abs(y + h - ymax0) < EPS_BND)
                    and (not code & 8 or abs(y - ymin0) < EPS_BND))

        viol = [i for i in range(n) if cn[i][4] != 0 and not _ok(i, cn[i][4])]
        best_P, best_cost = P, cost0
        accepted = []
        for b in viol:
            trial = accepted + [b]
            Pt = best_P
            got = None
            for _ in range(max(2, iters // 2)):
                newP, tele = lp_pass(ci, Pt, area_obj=area_obj,
                                     force_bnd=tuple(trial))
                if newP is None:
                    break
                m = cost_eval(ci, newP)
                if m.is_feasible and m.cost < (got[1] if got else best_cost) - 1e-12:
                    got = (newP, m.cost)
                    Pt = newP
                else:
                    break
            if got is not None:
                best_P, best_cost = got
                accepted.append(b)
        if accepted:
            fixed_blocks += len(accepted)
            print(f"case {ci:3d} n={n:3d} viol={len(viol)} repaired="
                  f"{len(accepted)} cost {cost0:.6f}->{best_cost:.6f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        rows.append((ci, cost0, best_cost, best_P))
    gain = sum(W[r[0]] * (r[1] - r[2]) for r in rows)
    tot = sum(W[r[0]] * r[2] for r in rows) / TOTW
    print(f"\nrepair: {fixed_blocks} boundary blocks fixed, weighted gain "
          f"{gain / TOTW / ANCHOR_TOTAL * 100:+.4f}%  new total {tot:.10f}")
    out = dict(rj)
    out["submission_name"] = "L3_port_repair"
    out["total_score"] = tot
    for (ci, c0, c1, Pb), t in zip(rows, out["test_results"]):
        if c1 < c0:
            m = cost_eval(ci, Pb)
            t.update(cost=m.cost, hpwl_gap=m.hpwl_gap, area_gap=m.area_gap,
                     violations_relative=m.violations_relative,
                     is_feasible=bool(m.is_feasible),
                     positions=[list(p) for p in Pb])
    dump = str(Path(src_json).with_name(Path(src_json).stem + "_repair.json"))
    json.dump(out, open(dump, "w"))
    print(f"[dump] {dump}")


def mode_reanchor(src_json, iters, area_obj, cases=None):
    """M57: frozen-violator bbox re-anchor scan. For every boundary violator
    living in a FROZEN unit (preplaced block, or cluster component containing
    one) -- exactly the blocks repair mode skipped with return None -- try an
    LP that pins the bbox edge(s) onto it. Greedy accumulation within a case;
    keep iff official strict cost strictly improves AND feasible. Telemetry
    records every trial: LP infeasibility = correct geometric death, and any
    feasible vrel drop below the anchor's vrel counts for the kill gate even
    when the cost guard rejects."""
    from collections import Counter
    rj = json.load(open(src_json))
    SRC_TOTAL = rj["total_score"]
    by_id = {t["test_id"]: t for t in rj["test_results"]}
    order = [89, 85, 62] + [ci for ci in range(100) if ci not in (89, 85, 62)]
    if cases is not None:
        order = [ci for ci in order if ci in cases]
    rows = []
    lpstat = Counter()
    n_fviol = n_kept = n_feas_rej = n_vdrop = 0
    t0 = time.perf_counter()
    for ci in order:
        t = by_id[ci]
        c = CASES[ci]
        n, cn = c["n"], c["cn"]
        P = [tuple(p) for p in t["positions"]]
        cost0, vrel0 = t["cost"], t["violations_relative"]
        m0 = cost_eval(ci, P)
        assert m0.cost == cost0, \
            f"case {ci}: eval {m0.cost!r} != json {cost0!r} (loader gate)"

        # frozen set exactly as build_and_solve sees it
        frozen = {i for i in range(n) if cn[i][1] != 0}
        for g in sorted({cn[i][3] for i in range(n) if cn[i][3] > 0}):
            mem = [i for i in range(n) if cn[i][3] == g]
            for cm in comp_split(P, mem):
                if any(i in frozen for i in cm):
                    frozen.update(cm)

        xmin0 = min(P[i][0] for i in range(n))
        xmax0 = max(P[i][0] + P[i][2] for i in range(n))
        ymin0 = min(P[i][1] for i in range(n))
        ymax0 = max(P[i][1] + P[i][3] for i in range(n))

        def _ok(i, code):
            x, y, w, h = P[i]
            return ((not code & 1 or abs(x - xmin0) < EPS_BND)
                    and (not code & 2 or abs(x + w - xmax0) < EPS_BND)
                    and (not code & 4 or abs(y + h - ymax0) < EPS_BND)
                    and (not code & 8 or abs(y - ymin0) < EPS_BND))

        viol = [i for i in range(n) if cn[i][4] != 0 and not _ok(i, cn[i][4])]
        fviol = [i for i in viol if i in frozen]
        n_fviol += len(fviol)
        best_P, best_cost, best_m = P, cost0, None
        accepted = []
        lines = []
        vdrop = False
        for b in fviol:
            trial = accepted + [b]
            Pt = best_P
            got = None
            for it in range(max(2, iters // 2)):
                newP, tele = lp_pass(ci, Pt, area_obj=area_obj,
                                     force_bnd=tuple(trial))
                if newP is None:
                    lpstat[tele["status"]] += 1
                    lines.append(f"blk {b:3d} code={cn[b][4]:2d} it{it}: "
                                 f"{tele['status']}")
                    break
                m = cost_eval(ci, newP)
                dv = m.violations_relative - vrel0
                if m.is_feasible and dv < -1e-12:
                    vdrop = True
                    n_vdrop += 1
                ok = (m.is_feasible
                      and m.cost < (got[1] if got else best_cost) - 1e-12)
                lines.append(
                    f"blk {b:3d} code={cn[b][4]:2d} it{it}: "
                    f"cost {m.cost:.6f} vrel {m.violations_relative:.4f} "
                    f"(dv={dv:+.4f}) feas {int(m.is_feasible)} "
                    f"{'KEPT' if ok else 'rej'}")
                if not ok:
                    n_feas_rej += int(bool(m.is_feasible))
                    break
                got = (newP, m.cost, m)
                Pt = newP
            if got is not None:
                best_P, best_cost, best_m = got
                accepted.append(b)
                n_kept += 1
        vrel1 = best_m.violations_relative if best_m is not None else vrel0
        rows.append((ci, cost0, best_cost, best_P, len(viol), len(fviol),
                     len(accepted), vrel0, vrel1, vdrop))
        if fviol:
            print(f"case {ci:3d} n={n:3d} viol={len(viol)} "
                  f"frozen={len(fviol)} kept={len(accepted)} "
                  f"cost {cost0:.6f}->{best_cost:.6f} "
                  f"vrel {vrel0:.4f}->{vrel1:.4f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            for ln in lines:
                print(f"    {ln}")

    gain = sum(W[r[0]] * (r[1] - r[2]) for r in rows)
    pct = gain / TOTW / SRC_TOTAL * 100
    tot = (SRC_TOTAL * TOTW - gain) / TOTW if cases is not None else \
        sum(W[r[0]] * r[2] for r in rows) / TOTW
    any_drop = any(r[9] for r in rows)
    print(f"\nreanchor: frozen violators tried {n_fviol}, kept {n_kept}, "
          f"feasible-but-rejected {n_feas_rej}, "
          f"feasible vrel-drop trials {n_vdrop}")
    print(f"LP dead: {dict(lpstat)}")
    print(f"weighted gain {pct:+.4f}% of src {SRC_TOTAL:.10f} "
          f"-> new total {tot:.10f}")
    print("KILL GATE: " + ("GREEN (feasible vrel drop AND >=0.05%)"
                           if any_drop and pct >= 0.05 else
                           f"RED (vrel_drop={any_drop}, gain {pct:+.4f}%)"))
    if cases is None:
        rowmap = {r[0]: r for r in rows}
        out = dict(rj)
        out["submission_name"] = "M57_reanchor"
        out["total_score"] = tot
        for t in out["test_results"]:
            r = rowmap[t["test_id"]]
            if r[2] < r[1]:
                m = cost_eval(r[0], r[3])
                t.update(cost=m.cost, hpwl_gap=m.hpwl_gap,
                         area_gap=m.area_gap,
                         violations_relative=m.violations_relative,
                         is_feasible=bool(m.is_feasible),
                         positions=[list(p) for p in r[3]])
        dump = str(_DIR / "results_M57_reanchor.json")
        json.dump(out, open(dump, "w"))
        print(f"[dump] {dump}")


def mode_diag(src_json):
    """M57 forensics (static, no LP): classify WHY each frozen violator's
    re-anchor LP dies. Per violated side bit:
      (b) same-side FROZEN satisfied boundary block ties the bbox var to the
          old extreme -> equality conflict with the pin
      (c) another frozen block extends beyond the pin value -> bounds conflict
          (the recomputed edge could never sit on the violator)
      chain = neither -> infeasibility comes from separation/envelope chains
    'touch' bits (violator already on that edge) pin the var at its current
    value = never a conflict source."""
    from collections import Counter
    rj = json.load(open(src_json))
    by_id = {t["test_id"]: t for t in rj["test_results"]}
    cls_count = Counter()
    print(f"{'case':>4} {'blk':>4} {'bit':>4} {'side':>4} {'dist':>12} "
          f"causes")
    for ci in range(100):
        t = by_id[ci]
        c = CASES[ci]
        n, cn = c["n"], c["cn"]
        P = [tuple(p) for p in t["positions"]]
        frozen = {i for i in range(n) if cn[i][1] != 0}
        for g in sorted({cn[i][3] for i in range(n) if cn[i][3] > 0}):
            mem = [i for i in range(n) if cn[i][3] == g]
            for cm in comp_split(P, mem):
                if any(i in frozen for i in cm):
                    frozen.update(cm)
        xmin0 = min(P[i][0] for i in range(n))
        xmax0 = max(P[i][0] + P[i][2] for i in range(n))
        ymin0 = min(P[i][1] for i in range(n))
        ymax0 = max(P[i][1] + P[i][3] for i in range(n))

        def _ok(i, code):
            x, y, w, h = P[i]
            return ((not code & 1 or abs(x - xmin0) < EPS_BND)
                    and (not code & 2 or abs(x + w - xmax0) < EPS_BND)
                    and (not code & 4 or abs(y + h - ymax0) < EPS_BND)
                    and (not code & 8 or abs(y - ymin0) < EPS_BND))

        viol = [i for i in range(n) if cn[i][4] != 0 and not _ok(i, cn[i][4])]
        fviol = [i for i in viol if i in frozen]
        if not fviol:
            continue
        fro_sat = [(i, cn[i][4]) for i in range(n)
                   if cn[i][4] != 0 and i in frozen and _ok(i, cn[i][4])]
        for b in fviol:
            code = cn[b][4]
            x, y, w, h = P[b]
            spec = ((1, "XMIN", x, xmin0,
                     min(P[i][0] for i in frozen), -1),
                    (2, "XMAX", x + w, xmax0,
                     max(P[i][0] + P[i][2] for i in frozen), +1),
                    (4, "YMAX", y + h, ymax0,
                     max(P[i][1] + P[i][3] for i in frozen), +1),
                    (8, "YMIN", y, ymin0,
                     min(P[i][1] for i in frozen), -1))
            for bit, name, pin, ext0, fro_ext, sgn in spec:
                if not code & bit:
                    continue
                if abs(pin - ext0) < EPS_BND:
                    cls_count["touch"] += 1
                    continue
                causes = []
                if any(cb & bit for _, cb in fro_sat):
                    causes.append("b_frozen_sat_tie")
                # (c): some frozen block extends beyond the pin on this side
                if sgn * (fro_ext - pin) > 1e-9:
                    causes.append("c_frozen_beyond")
                if not causes:
                    causes.append("chain")
                for cc in causes:
                    cls_count[cc] += 1
                print(f"{ci:4d} {b:4d} {bit:4d} {name:>4} "
                      f"{abs(pin - ext0):12.4f} {'+'.join(causes)}")
    print(f"\ndiag summary: {dict(cls_count)}")


def mode_diag2(src_json, area_obj):
    """M57 LP-level forensics on the violated bits only: pin the violated
    bit(s) WITHOUT the touch-bit pins, and each violated bit alone, to locate
    which equality kills the LP (rules out the touch pins as an artifact and
    separates joint-death from per-bit death)."""
    rj = json.load(open(src_json))
    by_id = {t["test_id"]: t for t in rj["test_results"]}
    names = {1: "XMIN", 2: "XMAX", 4: "YMAX", 8: "YMIN"}
    for ci in range(100):
        t = by_id[ci]
        c = CASES[ci]
        n, cn = c["n"], c["cn"]
        P = [tuple(p) for p in t["positions"]]
        frozen = {i for i in range(n) if cn[i][1] != 0}
        for g in sorted({cn[i][3] for i in range(n) if cn[i][3] > 0}):
            mem = [i for i in range(n) if cn[i][3] == g]
            for cm in comp_split(P, mem):
                if any(i in frozen for i in cm):
                    frozen.update(cm)
        xmin0 = min(P[i][0] for i in range(n))
        xmax0 = max(P[i][0] + P[i][2] for i in range(n))
        ymin0 = min(P[i][1] for i in range(n))
        ymax0 = max(P[i][1] + P[i][3] for i in range(n))

        def _touch(i, bit):
            x, y, w, h = P[i]
            return ((bit == 1 and abs(x - xmin0) < EPS_BND)
                    or (bit == 2 and abs(x + w - xmax0) < EPS_BND)
                    or (bit == 4 and abs(y + h - ymax0) < EPS_BND)
                    or (bit == 8 and abs(y - ymin0) < EPS_BND))

        def _ok(i, code):
            return all(_touch(i, b) for b in (1, 2, 4, 8) if code & b)

        viol = [i for i in range(n) if cn[i][4] != 0 and not _ok(i, cn[i][4])]
        fviol = [i for i in viol if i in frozen]
        for b in fviol:
            vbits = [bb for bb in (1, 2, 4, 8)
                     if cn[b][4] & bb and not _touch(b, bb)]
            vmask = sum(vbits)
            runs = [("violbits", vmask)]
            if len(vbits) > 1:
                runs += [(names[bb], bb) for bb in vbits]
            outs = []
            for tag, mask in runs:
                newP, tele = lp_pass(ci, P, area_obj=area_obj,
                                     force_bnd=((b, mask),))
                if newP is None:
                    outs.append(f"{tag}:{tele['status'].replace('lp_status_2', 'INFEAS')}")
                else:
                    m = cost_eval(ci, newP)
                    outs.append(f"{tag}:FEAS cost={m.cost:.6f} "
                                f"vrel={m.violations_relative:.4f} "
                                f"feas={int(m.is_feasible)}")
            print(f"case {ci:3d} blk {b:3d} code={cn[b][4]:2d} "
                  f"| {' | '.join(outs)}", flush=True)


def mode_l2stack(src_json, iters, area_obj, nsamp=3):
    """L2 x L3 stack: LP the best noisy samples (M53 L2 cache) of the 15 heavy
    cases and compare against the current L3-portfolio selection."""
    import pickle
    db2 = pickle.load(open(_DIR / "m53_l2_cache.pkl", "rb"))["db"]
    rj = json.load(open(src_json))
    cur = {t["test_id"]: t for t in rj["test_results"]}
    tot_gain = 0.0
    for ci in range(85, 100):
        ents = sorted((v, kk) for kk, v in db2.items()
                      if kk[0] == "cost" and kk[1] == ci
                      and (kk[3] > 0 or kk[4] > 0))[:nsamp]
        best = cur[ci]["cost"]
        src = None
        for c0, kk in ents:
            P = [tuple(p) for p in db2[("run",) + kk[1:]][0]]
            m0 = cost_eval(ci, P)
            bc, bP = m0.cost, P
            for _ in range(iters):
                newP, tele = lp_pass(ci, bP, area_obj=area_obj)
                if newP is None:
                    break
                m = cost_eval(ci, newP)
                if m.is_feasible and m.cost < bc - 1e-12:
                    bc, bP = m.cost, newP
                else:
                    break
            if bc < best - 1e-12:
                best, src = bc, kk
        d = cur[ci]["cost"] - best
        tot_gain += W[ci] * d
        if src is not None:
            print(f"case {ci:3d} L3sel {cur[ci]['cost']:.6f} -> L2+LP "
                  f"{best:.6f} d={d:+.6f} (sample {src[2:]},"
                  f" preLP {ents[0][0]:.6f})", flush=True)
    print(f"\nl2stack: weighted gain {tot_gain / TOTW / ANCHOR_TOTAL * 100:+.4f}%"
          f" on top of {Path(src_json).name}")


# ── modes ────────────────────────────────────────────────────────────────────
def mode_gate0():
    bad = 0
    t0 = time.perf_counter()
    for ci in range(100):
        m = cost_eval(ci, [tuple(p) for p in ANCH[ci]["positions"]])
        if m.cost != ANCH[ci]["cost"]:
            bad += 1
            print(f"  FAIL case {ci}: eval {m.cost!r} != json "
                  f"{ANCH[ci]['cost']!r}")
    print(f"gate0: {100 - bad}/100 exact cost match "
          f"({time.perf_counter() - t0:.0f}s)")
    return bad == 0


def run_probe(case_ids, iters, dump=None, area_obj=False):
    results = []
    t0 = time.perf_counter()
    for ci in case_ids:
        r = run_case(ci, iters, area_obj=area_obj)
        results.append(r)
        d = r["cost0"] - r["cost1"]
        wc = W[ci] * d / TOTW / ANCHOR_TOTAL * 100
        print(f"case {ci:3d} n={CASES[ci]['n']:3d} "
              f"{r['cost0']:.6f}->{r['cost1']:.6f} d={d:+.6f} "
              f"wContr={wc:+.4f}%")
        for ln in r["lines"]:
            print(f"    {ln}")
    gain = sum(W[r['ci']] * (r["cost0"] - r["cost1"]) for r in results)
    pct = gain / TOTW / ANCHOR_TOTAL * 100
    nimp = sum(1 for r in results if r["cost1"] < r["cost0"] - 1e-12)
    print(f"\n== L3 LP: {nimp}/{len(results)} cases improved, weighted gain "
          f"{pct:+.4f}% of anchor {ANCHOR_TOTAL:.6f} "
          f"({time.perf_counter() - t0:.0f}s) ==")
    if dump:
        trs = []
        for ci in range(100):
            r = next((x for x in results if x["ci"] == ci), None)
            if r is not None and r["m"] is not None:
                m = r["m"]
                trs.append(dict(test_id=ci, block_count=CASES[ci]["n"],
                                is_feasible=bool(m.is_feasible),
                                hpwl_gap=m.hpwl_gap, area_gap=m.area_gap,
                                violations_relative=m.violations_relative,
                                runtime_seconds=0.0, cost=m.cost,
                                positions=[list(p) for p in r["P"]],
                                error=None))
            else:
                trs.append(dict(ANCH[ci]))
        tot = sum(W[t["test_id"]] * t["cost"] for t in trs) / TOTW
        out = dict(submission_name="L3_lp",
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                   total_score=tot, test_results=trs,
                   summary=dict(num_tests=100,
                                num_feasible=sum(t["is_feasible"] for t in trs),
                                avg_cost=sum(t["cost"] for t in trs) / 100,
                                avg_runtime=0.0))
        json.dump(out, open(dump, "w"))
        print(f"[dump] {dump}  total_score={tot:.10f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode",
                    choices=["gate0", "probe15", "full", "port15", "portfull",
                             "repair", "l2stack", "reanchor", "diag",
                             "diag2"])
    ap.add_argument("--src", default=None,
                    help="repair: source results json to repair on top of")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--anchor", default=str(_DIR / "results_L1_final.json"))
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--area", action="store_true",
                    help="joint objective: 0.5*dH/h_base + 0.5*dArea/a_base")
    args = ap.parse_args()

    _aj = json.load(open(args.anchor))
    ANCHOR_TOTAL = _aj["total_score"]
    ANCH = {t["test_id"]: t for t in _aj["test_results"]}
    print(f"[anchor] {args.anchor} total={ANCHOR_TOTAL:.10f}")

    if args.mode == "gate0":
        sys.exit(0 if mode_gate0() else 1)
    elif args.mode == "port15":
        mode_port15(args.topk, args.iters, args.area)
    elif args.mode == "portfull":
        mode_portfull(args.topk, args.iters, args.area)
    elif args.mode == "repair":
        mode_repair(args.src or str(_DIR / "results_L3_port_top8_area.json"),
                    args.iters, args.area)
    elif args.mode == "l2stack":
        mode_l2stack(args.src or str(_DIR / "results_L3_port_top32_area.json"),
                     args.iters, args.area)
    elif args.mode == "reanchor":
        ids = None
        if args.cases:
            a, b = args.cases.split("-")
            ids = set(range(int(a), int(b) + 1))
        mode_reanchor(args.src
                      or str(_DIR / "results_L3_port_top32_area.json"),
                      args.iters, args.area, cases=ids)
    elif args.mode == "diag":
        mode_diag(args.src or str(_DIR / "results_L3_port_top32_area.json"))
    elif args.mode == "diag2":
        mode_diag2(args.src or str(_DIR / "results_L3_port_top32_area.json"),
                   args.area)
    elif args.mode == "probe15":
        ids = list(range(85, 100))
        if args.cases:
            a, b = args.cases.split("-")
            ids = list(range(int(a), int(b) + 1))
        run_probe(ids, args.iters, area_obj=args.area)
    else:
        stem = Path(args.anchor).stem.replace("results_", "")
        tag = "_area" if args.area else ""
        run_probe(list(range(100)), args.iters, area_obj=args.area,
                  dump=str(_DIR / f"results_L3_lp_{stem}{tag}.json"))
