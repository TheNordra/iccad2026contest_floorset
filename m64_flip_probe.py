"""M64 probe (OFFLINE, never shipped): adjacent-topology LP — flip ONE unit
pair's separation relation and re-solve.

M53 L3 keeps, per block pair, only the currently-max-gap separation disjunct
(fixed-disjunct, m53_l3_probe.py:213-231) -> the 1.2978 offline anchor is the
continuous optimum of the topology cell its seeds landed in; adjacent cells
were never measured.  M64 is the formal measurement of M59 stage-2 (relation
flip), restarted from the anchor's own cell (results_L3_port_top32_area.json
winner positions), not from M59's far-basin c2 seeds.

Flip semantics (unit-pair level): a target is a UNIT pair (rigid cluster
component / free single / per-block frozen pseudo-unit) plus a canonical
direction k in {0:A-left-of-B, 1:B-left-of-A, 2:A-below-B, 3:B-below-A} where
A = the lexicographically smaller unit key.  ALL block pairs spanning the two
units get their separation row REPLACED by direction k (each with its own gap
constant; direction mirrored 0<->1 / 2<->3 when the (i<j) block order maps to
(B,A)).  Single-block-pair flips between multi-member units are near-certainly
contradictory (sibling pairs keep the old disjunct) so they are not probed.
HONEST-SCOPE note: forcing one k on every member pair means "slide all of A
past all of B on that axis" — stronger than the evaluator's pairwise
requirement (a mixed per-pair topology could be feasible where this is not).

Per (pair, dir): sound extent prefilter (provably-infeasible skips only) ->
variant A (strict: bbox never grows) -> variant B (bbox rows relaxed to
sqrt(1.005) per side => area growth <= 0.5%) when A is infeasible or near
break-even -> official strict cost_eval arbitration; movers get an
m53.lp_pass fixpoint polish (area_obj=True, matching the anchor provenance).

modes:
  selfcheck  wiring proof: forcing the CURRENT direction of a homogeneous pair
             must reproduce the unforced LP bit-exactly
  pilot      cases 85,88,91 (slow case 91 last so the gate can fire early);
             gate: zero movers (official improvement > 1e-6) -> RED stop
  heavy      cases 85..99 (only after pilot found a mover)
  full       all 100 cases (only after heavy union-oracle >= 0.15%)
  l2base     regenerate the l2stack per-case bests from m53_l2_cache.pkl
             (honest 1.2978-chain baseline: min(port_top32, l2b))
  report     aggregate cache -> status histograms, tax decomposition, gate
             verdicts
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import m53_l3_probe as m53  # noqa: E402  (loads the dataset once)
from scipy import sparse  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

CASES, W, TOTW = m53.CASES, m53.W, m53.TOTW
cost_eval, comp_split, apply_deltas = m53.cost_eval, m53.comp_split, m53.apply_deltas
EPS_BND = m53.EPS_BND

PROBE_VERSION = "m64-v1"
RELAX = math.sqrt(1.005)          # per-side cap factor -> area growth <= 0.5%
MIRROR = (1, 0, 3, 2)             # canonical<->local dir map when block order swaps
DIRN = ("A<B", "B<A", "AvB", "BvA")

# globals set in __main__
ANCH64, SIG, DB = None, None, None
_dirty = 0


# ── units (must match m53.build_and_solve lines 123-146 exactly) ─────────────
def build_units(ci, P):
    c = CASES[ci]
    n, cn = c["n"], c["cn"]
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
    # canonical unit keys: frozen blocks are per-block pseudo-units ('F', blk)
    # (unit_of None is shared by ALL frozen blocks and must not be one key)
    ukey = [("F", i) if unit_of[i] is None else ("U", min(units[unit_of[i]]))
            for i in range(n)]
    return units, unit_of, group_units, group_comp0, frozen_blk, ukey


# ── LP build+solve: copy of m53.build_and_solve with force_rel /
#    skip_bnd_ties / bbox_relax; force_bnd (unused here) removed ──────────────
def build_and_solve_flip(ci, P, freeze_units, area_obj=False, force_rel=None,
                         skip_bnd_ties=False, bbox_relax=1.0):
    c = CASES[ci]
    n, cn = c["n"], c["cn"]
    units, unit_of, group_units, group_comp0, _fro, ukey = build_units(ci, P)

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

    h_base = max(float(c["base"].get("hpwl_baseline", 1.0)), 1e-6)
    hw_scale = 0.5 / h_base
    cx = [P[i][0] + P[i][2] / 2.0 for i in range(n)]
    cy = [P[i][1] + P[i][3] / 2.0 for i in range(n)]
    const_h = 0.0
    obj0 = 0.0

    def edge_axis(t, ui, uj, off, dC):
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

    # pairwise separation: keep the currently separating axis, except pairs
    # spanning a forced unit pair, whose row is REPLACED by the forced dir
    force_rel = force_rel or {}
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
            ki, kj = ukey[i], ukey[j]
            pk = (ki, kj) if ki <= kj else (kj, ki)
            kc = force_rel.get(pk)
            if kc is None:
                gap, ul, ur, off = max(cands, key=lambda t: t[0])
            else:
                gap, ul, ur, off = cands[kc if ki <= kj else MIRROR[kc]]
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
    sides = ((1, XMIN, 0, xmin0, min(range(n), key=lambda i: P[i][0])),
             (2, XMAX, 0, xmax0, max(range(n), key=lambda i: P[i][0] + P[i][2])),
             (4, YMAX, U, ymax0, max(range(n), key=lambda i: P[i][1] + P[i][3])),
             (8, YMIN, U, ymin0, min(range(n), key=lambda i: P[i][1])))
    if not skip_bnd_ties:
        for bit, bv, off, ext0, mdef in sides:
            tied = {unit_of[i] for i, code in sat if code & bit}
            if not tied:
                continue
            for u in tied:
                if u is None:
                    add_eq([(bv, -1.0)], -ext0)
                else:
                    add_eq([(off + u, 1.0), (bv, -1.0)], -ext0)
            um = unit_of[mdef]
            if um not in tied:
                if um is None:
                    add_eq([(bv, -1.0)], -ext0)
                else:
                    add_eq([(off + um, 1.0), (bv, -1.0)], -ext0)

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
    # bbox: never grows (variant A) / grows <= sqrt(1.005) per side (variant B)
    add_ub([(XMAX, 1.0), (XMIN, -1.0)], W0 * bbox_relax)
    add_ub([(YMAX, 1.0), (YMIN, -1.0)], H0 * bbox_relax)

    a_base = max(float(c["base"].get("area_baseline", W0 * H0)), 1e-6)
    if area_obj and W0 * H0 > a_base:
        bA = 0.5 / a_base
        obj[XMIN] -= bA * H0
        obj[XMAX] += bA * H0
        obj[YMIN] -= bA * W0
        obj[YMAX] += bA * W0

    for u in freeze_units:
        add_eq([(u, 1.0)], 0.0)
        add_eq([(U + u, 1.0)], 0.0)

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


def lp_pass_flip(ci, P, area_obj=False, force_rel=None, skip_bnd_ties=False,
                 bbox_relax=1.0):
    """One LP pass with the cluster-precision re-solve ladder; failure tele
    carries the attempt index (attempt>1 = ladder freeze kill, not geometry)."""
    freeze = set()
    for attempt in range(3):
        B = build_and_solve_flip(ci, P, freeze, area_obj=area_obj,
                                 force_rel=force_rel,
                                 skip_bnd_ties=skip_bnd_ties,
                                 bbox_relax=bbox_relax)
        if B["res"].status != 0:
            return None, dict(status=f"lp_status_{B['res'].status}",
                              attempt=attempt + 1)
        U = B["U"]
        x = B["res"].x
        newP = apply_deltas(P, B["units"], x[:U], x[U:2 * U])
        broken = [g for g, c0 in B["group_comp0"].items()
                  if len(comp_split(newP, [i for i in range(CASES[ci]["n"])
                                           if CASES[ci]["cn"][i][3] == g])) > c0]
        if not broken:
            return newP, dict(status="ok", U=U, bnd_skip=B["bnd_skip"],
                              attempts=attempt + 1, frozen=len(freeze))
        for g in broken:
            freeze |= B["group_units"][g]
    return None, dict(status="cluster_break", attempt=3)


# ── candidate ranking ────────────────────────────────────────────────────────
def pinned_sides(ci, P, unit_of):
    """Side bits whose bbox var is pinned by an equality to a constant
    (frozen satisfied-boundary block, or frozen extreme-definer)."""
    c = CASES[ci]
    n, cn = c["n"], c["cn"]
    xmin0 = min(P[i][0] for i in range(n))
    xmax0 = max(P[i][0] + P[i][2] for i in range(n))
    ymin0 = min(P[i][1] for i in range(n))
    ymax0 = max(P[i][1] + P[i][3] for i in range(n))

    def touch_ok(i, code):
        x, y, w, h = P[i]
        return ((not code & 1 or abs(x - xmin0) < EPS_BND)
                and (not code & 2 or abs(x + w - xmax0) < EPS_BND)
                and (not code & 4 or abs(y + h - ymax0) < EPS_BND)
                and (not code & 8 or abs(y - ymin0) < EPS_BND))

    sat = [(i, code) for i, code in ((i, cn[i][4]) for i in range(n)
                                     if cn[i][4] != 0) if touch_ok(i, code)]
    mdefs = ((1, min(range(n), key=lambda i: P[i][0])),
             (2, max(range(n), key=lambda i: P[i][0] + P[i][2])),
             (4, max(range(n), key=lambda i: P[i][1] + P[i][3])),
             (8, min(range(n), key=lambda i: P[i][1])))
    pinned = set()
    for bit, mdef in mdefs:
        tied = {unit_of[i] for i, code in sat if code & bit}
        if not tied:
            continue
        if None in tied or unit_of[mdef] is None:
            pinned.add(bit)
    return pinned


def rank_pairs(ci, P, top):
    """Top unit pairs by tightness x endpoint net gradient; per pair, the
    directions worth forcing (3 non-current, or all 4 when the member pairs'
    argmax directions are heterogeneous = already a topology change)."""
    c = CASES[ci]
    n = c["n"]
    units, unit_of, _gu, _gc0, _fro, ukey = build_units(ci, P)
    xmin0 = min(P[i][0] for i in range(n))
    xmax0 = max(P[i][0] + P[i][2] for i in range(n))
    ymin0 = min(P[i][1] for i in range(n))
    ymax0 = max(P[i][1] + P[i][3] for i in range(n))
    W0, H0 = xmax0 - xmin0, ymax0 - ymin0
    flo = 1e-3 * (W0 + H0) / 2.0

    # live-edge incident / cross weights (mirror build's live conditions)
    inc, cross = {}, {}
    for i, j, w in c["b2l"]:
        if w <= 0.0 or unit_of[i] == unit_of[j]:
            continue
        ki, kj = ukey[i], ukey[j]
        inc[ki] = inc.get(ki, 0.0) + w
        inc[kj] = inc.get(kj, 0.0) + w
        pk = (ki, kj) if ki <= kj else (kj, ki)
        cross[pk] = cross.get(pk, 0.0) + w
    for p, i, w in c["p2l"]:
        if w <= 0.0 or unit_of[i] is None:
            continue
        inc[ukey[i]] = inc.get(ukey[i], 0.0) + w

    # per unit-pair canonical min-gaps + member-argmax bookkeeping
    mins, allm, amset = {}, {}, {}
    for i in range(n):
        xi, yi, wi, hi = P[i]
        for j in range(i + 1, n):
            ui, uj = unit_of[i], unit_of[j]
            if ui == uj:  # same movable unit or both frozen -> no LP row
                continue
            xj, yj, wj, hj = P[j]
            g = (xj - (xi + wi), xi - (xj + wj),
                 yj - (yi + hi), yi - (yj + hj))
            am = 0  # first-wins argmax, mirrors max(cands, key=...)
            for t in range(1, 4):
                if g[t] > g[am]:
                    am = t
            ki, kj = ukey[i], ukey[j]
            if ki <= kj:
                pk, m2c = (ki, kj), (0, 1, 2, 3)
            else:
                pk, m2c = (kj, ki), MIRROR
            e = mins.get(pk)
            if e is None:
                mins[pk] = [g[m2c[0]], g[m2c[1]], g[m2c[2]], g[m2c[3]]]
                allm[pk] = [m2c[k] == am for k in range(4)]
                amset[pk] = {m2c[am]}
            else:
                a = allm[pk]
                for k in range(4):
                    gk = g[m2c[k]]
                    if gk < e[k]:
                        e[k] = gk
                    if a[k] and m2c[k] != am:
                        a[k] = False
                amset[pk].add(m2c[am])

    # extents per unit key (rigid, position-independent width/height demand)
    ext = {}
    for i in range(n):
        k = ukey[i]
        x, y, w_, h_ = P[i]
        e = ext.get(k)
        if e is None:
            ext[k] = [x, x + w_, y, y + h_]
        else:
            e[0], e[1] = min(e[0], x), max(e[1], x + w_)
            e[2], e[3] = min(e[2], y), max(e[3], y + h_)
    extX = {k: v[1] - v[0] for k, v in ext.items()}
    extY = {k: v[3] - v[2] for k, v in ext.items()}

    ranked = []
    for pk, mg in mins.items():
        cur = max(mg)
        het = len(amset[pk]) > 1
        if het:
            dirs = [0, 1, 2, 3]
        else:
            curk = next(iter(amset[pk]))
            dirs = [k for k in range(4) if k != curk]
        dirs = tuple(k for k in dirs if not allm[pk][k])  # exact no-ops out
        if not dirs:
            continue
        grad = cross.get(pk, 0.0) + inc.get(pk[0], 0.0) + inc.get(pk[1], 0.0)
        score = grad / (max(cur, 0.0) + flo)
        ranked.append((score, pk, dirs, cur, het))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    return dict(ranked=ranked[:top], extX=extX, extY=extY, W0=W0, H0=H0,
                pinned=pinned_sides(ci, P, unit_of),
                nunits=len(units), npk=len(mins))


# ── cache ────────────────────────────────────────────────────────────────────
def save_db(force=True):
    global _dirty
    tmp = _DIR / "m64_cache.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dict(sig=SIG, db=DB), f)
    os.replace(tmp, _DIR / "m64_cache.pkl")
    _dirty = 0


def put(key, val):
    global _dirty
    DB[key] = val
    _dirty += 1
    if _dirty >= 20:
        save_db()


# ── flip driver ──────────────────────────────────────────────────────────────
def flip_solve(ci, P0, pk, k, relax, skip_bnd=False):
    t0 = time.perf_counter()
    newP, tele = lp_pass_flip(ci, P0, area_obj=True, force_rel={pk: k},
                              skip_bnd_ties=skip_bnd, bbox_relax=relax)
    dt = time.perf_counter() - t0
    if newP is None:
        return dict(status=tele["status"], attempt=tele.get("attempt"), dt=dt)
    m = cost_eval(ci, newP)
    return dict(status="ok", dt=dt, cost=m.cost, feas=bool(m.is_feasible),
                hgap=m.hpwl_gap, agap=m.area_gap,
                vrel=m.violations_relative, attempt=tele["attempts"],
                newP=newP)


def run_one(ci, P0, c0, R, pk, k, variant, polish_iters):
    relax = RELAX if variant == "B" else 1.0
    KA, KB = pk
    if k in (0, 1):
        need, cap = R["extX"][KA] + R["extX"][KB], R["W0"] * relax
    else:
        need, cap = R["extY"][KA] + R["extY"][KB], R["H0"] * relax
    if need > cap + 1e-9:  # provably infeasible (extents vs bbox row), sound
        return dict(status="prefilter_infeasible", dt=0.0, need=need, cap=cap)
    r = flip_solve(ci, P0, pk, k, relax)
    if r["status"] != "ok":
        return r
    r["delta"] = c0 - r["cost"]
    if r["feas"] and r["delta"] > 1e-6:  # mover: polish to realizable value
        bP, bc = r.pop("newP"), r["cost"]
        for _ in range(polish_iters):
            nP, _tele = m53.lp_pass(ci, bP, area_obj=True)
            if nP is None:
                break
            mm = cost_eval(ci, nP)
            if mm.is_feasible and mm.cost < bc - 1e-12:
                bc, bP = mm.cost, nP
            else:
                break
        r["polished"] = bc
        r["positions"] = [list(p) for p in bP]
    else:
        r.pop("newP", None)
    return r


def probe_case(ci, top, near_eps, polish_iters, diag_n):
    ent = ANCH64[ci]
    P0 = [tuple(p) for p in ent["positions"]]
    c0 = ent["cost"]
    gk = ("gate", ci)
    if gk not in DB:  # gate0-style: anchor positions must reproduce json cost
        m = cost_eval(ci, P0)
        if m.cost != c0:
            raise SystemExit(f"anchor gate FAIL case {ci}: "
                             f"eval {m.cost!r} != json {c0!r}")
        put(gk, True)
    rk = ("rank", ci, top)
    if rk not in DB:
        put(rk, rank_pairs(ci, P0, top))
        save_db()
    R = DB[rk]
    pinned_all = R["pinned"] == {1, 2, 4, 8}
    t_case = time.perf_counter()
    nsolve, movers = 0, []
    for score, pk, dirs, cur, het in R["ranked"]:
        for k in dirs:
            keyA = ("flip", ci, pk, k, "A")
            if keyA not in DB:
                put(keyA, run_one(ci, P0, c0, R, pk, k, "A", polish_iters))
                nsolve += 1
            rA = DB[keyA]
            _plog(ci, pk, k, "A", rA, c0)
            trigB = (rA["status"] != "ok"
                     or abs(rA["cost"] - c0) < near_eps * c0)
            if trigB:
                keyB = ("flip", ci, pk, k, "B")
                if keyB not in DB:
                    if pinned_all:  # all four sides equality-pinned: B == A
                        put(keyB, dict(status="bnd_pinned", dt=0.0))
                    else:
                        put(keyB, run_one(ci, P0, c0, R, pk, k, "B",
                                          polish_iters))
                        nsolve += 1
                rB = DB[keyB]
                _plog(ci, pk, k, "B", rB, c0)
            for v in ("A", "B"):
                r = DB.get(("flip", ci, pk, k, v))
                if r and r.get("status") == "ok" and r.get("feas") \
                        and r.get("delta", 0.0) > 1e-6:
                    movers.append((pk, k, v, r["delta"],
                                   r.get("polished", r["cost"])))
    # boundary-lock diagnosis on a sample of pure attempt-1 LP infeasibles
    ndiag = 0
    for score, pk, dirs, cur, het in R["ranked"]:
        if ndiag >= diag_n:
            break
        for k in dirs:
            if ndiag >= diag_n:
                break
            rA = DB.get(("flip", ci, pk, k, "A"))
            if (not rA or not str(rA["status"]).startswith("lp_status")
                    or rA.get("attempt") != 1):
                continue
            dk = ("diag", ci, pk, k)
            if dk not in DB:
                r = flip_solve(ci, P0, pk, k, 1.0, skip_bnd=True)
                put(dk, dict(status=r["status"],
                             feas_after=r["status"] == "ok"))
            ndiag += 1
    save_db()
    print(f"case {ci:3d} n={CASES[ci]['n']:3d} done: {nsolve} new solves, "
          f"{len(movers)} movers, pinned={sorted(R['pinned'])} "
          f"({time.perf_counter() - t_case:.0f}s)", flush=True)
    for pk, k, v, d, pc in movers:
        print(f"    MOVER {pk} {DIRN[k]} [{v}] d={d:+.6f} "
              f"polished {pc:.6f} (base {c0:.6f})", flush=True)
    return movers


def _plog(ci, pk, k, v, r, c0):
    if r["status"] != "ok":
        extra = f" att={r['attempt']}" if r.get("attempt") else ""
        print(f"  c{ci} {pk} {DIRN[k]} [{v}] {r['status']}{extra} "
              f"({r.get('dt', 0.0):.1f}s)", flush=True)
    else:
        print(f"  c{ci} {pk} {DIRN[k]} [{v}] cost {r['cost']:.6f} "
              f"d={c0 - r['cost']:+.2e} feas={int(r['feas'])} "
              f"({r['dt']:.1f}s)", flush=True)


# ── l2base: honest 1.2978-chain per-case baseline ────────────────────────────
def mode_l2base(iters=2, nsamp=3):
    """Replicates m53.mode_l2stack's inner loop (m53_l3_probe.py:747-767) but
    standalone per case (no src-json seeding of `best`); mode_l2stack itself
    is not import-callable (ANCHOR_TOTAL is __main__-only there)."""
    dbl = pickle.load(open(_DIR / "m53_l2_cache.pkl", "rb"))["db"]
    for ci in range(85, 100):
        key = ("l2b", ci)
        if key in DB:
            e = DB[key]
            print(f"case {ci}: l2b {e['best']:.6f} (cached)", flush=True)
            continue
        ents = sorted((v, kk) for kk, v in dbl.items()
                      if kk[0] == "cost" and kk[1] == ci
                      and (kk[3] > 0 or kk[4] > 0))[:nsamp]
        best, src = float("inf"), None
        for _c0s, kk in ents:
            P = [tuple(p) for p in dbl[("run",) + kk[1:]][0]]
            m0 = cost_eval(ci, P)
            bc, bP = m0.cost, P
            for _ in range(iters):
                nP, _tele = m53.lp_pass(ci, bP, area_obj=True)
                if nP is None:
                    break
                m = cost_eval(ci, nP)
                if m.is_feasible and m.cost < bc - 1e-12:
                    bc, bP = m.cost, nP
                else:
                    break
            if bc < best:
                best, src = bc, kk
        put(key, dict(best=best, src=src))
        save_db()
        port = ANCH64[ci]["cost"]
        tag = "  L2-WIN" if best < port - 1e-12 else ""
        print(f"case {ci}: l2b {best:.6f} vs port32 {port:.6f}{tag}",
              flush=True)


# ── report / gates ───────────────────────────────────────────────────────────
def mode_report():
    hist, taxes, best_flip = {}, [], {}
    for key, r in DB.items():
        if key[0] != "flip":
            continue
        _t, ci, pk, k, v = key
        st = r["status"]
        if st == "ok":
            c0 = ANCH64[ci]["cost"]
            d = r["delta"]
            if r["feas"] and d > 1e-6:
                st = "mover"
            elif r["feas"] and d > 1e-12:
                st = "tiny_win"
            elif r["feas"]:
                st = "worse"
                e0 = ANCH64[ci]
                taxes.append((r["hgap"] - e0["hpwl_gap"],
                              r["agap"] - e0["area_gap"],
                              r["vrel"] - e0["violations_relative"]))
            else:
                st = "ok_infeasible"
            cbest = r.get("polished", r["cost"])
            if cbest < best_flip.get(ci, float("inf")):
                best_flip[ci] = cbest
        elif str(st).startswith("lp_status") and r.get("attempt", 1) > 1:
            st = "ladder_kill"
        hist[st] = hist.get(st, 0) + 1
    print("== status histogram (per (pair,dir,variant)) ==")
    for st, cnt in sorted(hist.items(), key=lambda t: -t[1]):
        print(f"  {st:22s} {cnt}")
    if taxes:
        ah = sum(t[0] for t in taxes) / len(taxes)
        aa = sum(t[1] for t in taxes) / len(taxes)
        av = sum(t[2] for t in taxes) / len(taxes)
        print(f"== feasible-but-worse tax (mean over {len(taxes)}): "
              f"d_hgap {ah:+.5f}  d_agap {aa:+.5f}  d_vrel {av:+.5f} ==")
    diag = [(k, v) for k, v in DB.items() if k[0] == "diag"]
    if diag:
        nf = sum(1 for _k, v in diag if v["feas_after"])
        print(f"== diag (skip boundary ties on attempt-1 LP-infeasible): "
              f"{nf}/{len(diag)} flip to feasible = boundary-preplaced lock ==")

    cases = sorted({k[1] for k in DB if k[0] == "flip"})
    movers = []
    for ci in cases:
        c0 = ANCH64[ci]["cost"]
        bf = best_flip.get(ci, float("inf"))
        if bf < c0 - 1e-6:
            movers.append((ci, c0 - bf))
    print(f"\n== pilot/stage gate: movers(>1e-6) on {len(cases)} probed cases: "
          f"{len(movers)} ==")
    for ci, d in movers:
        print(f"  case {ci}: best flip d={d:+.6f}")

    # union-oracle vs honest 1.2978-chain baseline min(port32, l2b)
    l2b = {k[1]: v["best"] for k, v in DB.items() if k[0] == "l2b"}
    base = {ci: min(ANCH64[ci]["cost"], l2b.get(ci, float("inf")))
            for ci in range(100)}
    base_total = sum(W[ci] * base[ci] for ci in range(100)) / TOTW
    gain = sum(W[ci] * max(0.0, base[ci] - best_flip.get(ci, float("inf")))
               for ci in cases)
    pct = gain / TOTW / base_total * 100
    tag = "(l2b missing -> baseline = port32 only)" if not l2b else \
          f"(l2b for {len(l2b)} cases)"
    print(f"\n== union-oracle: weighted gain {pct:+.4f}% of chain baseline "
          f"{base_total:.6f} {tag} ==")
    print("   thresholds: <0.15% stop | 0.15-0.3% extend to full-100 | "
          "<0.3% overall = axis RED | >=0.3% GREEN")


# ── selfcheck: forcing the current direction must be a bit-exact no-op ───────
def mode_selfcheck(ci=85):
    ent = ANCH64[ci]
    P0 = [tuple(p) for p in ent["positions"]]
    m = cost_eval(ci, P0)
    assert m.cost == ent["cost"], f"anchor gate FAIL: {m.cost!r}"
    print(f"[selfcheck] case {ci} anchor reproduces json cost exactly")
    R = rank_pairs(ci, P0, 200)
    pick = None
    for score, pk, dirs, cur, het in R["ranked"]:
        if not het:  # homogeneous pair: current dir is well-defined
            curk = next(k for k in range(4) if k not in dirs)
            pick = (pk, curk)
            break
    assert pick, "no homogeneous pair found"
    pk, curk = pick
    a, ta = lp_pass_flip(ci, P0, area_obj=True)
    b, tb = lp_pass_flip(ci, P0, area_obj=True, force_rel={pk: curk})
    assert a is not None and b is not None, (ta, tb)
    same = all(ax == bx for pa, pb in zip(a, b) for ax, bx in zip(pa, pb))
    print(f"[selfcheck] force current dir {DIRN[curk]} on {pk}: "
          f"positions bit-identical = {same}")
    if not same:
        raise SystemExit("selfcheck FAIL: forced-current != unforced")
    print("[selfcheck] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selfcheck", "pilot", "heavy", "full",
                                     "l2base", "report"])
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--anchor",
                    default=str(_DIR / "results_L3_port_top32_area.json"))
    ap.add_argument("--near-eps", type=float, default=0.002)
    ap.add_argument("--polish-iters", type=int, default=3)
    ap.add_argument("--diag", type=int, default=5)
    ap.add_argument("--cases", default=None,
                    help="comma list overriding the mode's case set")
    args = ap.parse_args()

    raw = open(args.anchor, "rb").read()
    SIG = (hashlib.md5(raw).hexdigest(), PROBE_VERSION)
    _aj = json.loads(raw)
    ANCH64 = {t["test_id"]: t for t in _aj["test_results"]}
    print(f"[anchor] {Path(args.anchor).name} "
          f"total={_aj['total_score']:.10f}", flush=True)

    DB = {}
    _cp = _DIR / "m64_cache.pkl"
    if _cp.exists():
        try:
            _d = pickle.load(open(_cp, "rb"))
        except Exception:
            _d = None
        if _d and _d.get("sig") == SIG:
            DB = _d["db"]
            print(f"[cache] resume {len(DB)} entries", flush=True)
        else:
            print("[cache] sig mismatch -> reset", flush=True)

    if args.mode == "selfcheck":
        mode_selfcheck()
    elif args.mode == "l2base":
        mode_l2base()
    elif args.mode == "report":
        mode_report()
    else:
        ids = {"pilot": [85, 88, 91], "heavy": list(range(85, 100)),
               "full": list(range(100))}[args.mode]
        if args.cases:
            ids = [int(x) for x in args.cases.split(",")]
        allmov = []
        for ci in ids:
            allmov += probe_case(ci, args.top, args.near_eps,
                                 args.polish_iters, args.diag)
        save_db()
        print(f"\n== {args.mode}: {len(allmov)} movers across {len(ids)} "
              f"cases ==", flush=True)
        if args.mode == "pilot" and not allmov:
            print("PILOT GATE: RED (zero movers > 1e-6) -> stop per spec")
