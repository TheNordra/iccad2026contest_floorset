"""L100: LP speed probe for L97 shape-LP."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in sorted(k for k in os.environ if k.startswith("ICCAD_")):
    del os.environ[_k]

import m53_l3_probe as l3
import optimizer_constructive as oc
import l97_shape_lp as l97

ANCHOR = _DIR / "results_M80_anchor_48c.json"
BASELINE = _DIR / "results_L97_shape_lp.json"
DUMP = _DIR / "results_L100_lp_speed.json"

EPS_BND = 1e-6
EPS_OVL = 1e-6
AREA_TOL = 0.008


def decompose(ci, P):
    c = l3.CASES[ci]
    n, cn = c["n"], c["cn"]
    frozen_blk = {i for i in range(n) if cn[i][1] != 0}
    unit_of = [None] * n
    units = []
    group_units, group_comp0 = {}, {}
    for g in sorted({cn[i][3] for i in range(n) if cn[i][3] > 0}):
        mem = [i for i in range(n) if cn[i][3] == g]
        comps = l3.comp_split(P, mem)
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
            unit_of[i] = len(units)
            units.append([i])
    return units, unit_of, group_units, group_comp0


def reshapeable(ci, units):
    cn = l3.CASES[ci]["cn"]
    out = {}
    for uid, mem in enumerate(units):
        if len(mem) != 1:
            continue
        i = mem[0]
        if cn[i][0] != 0 or cn[i][1] != 0 or cn[i][2] > 0:
            continue
        code = cn[i][4]
        if (code & 1 and code & 2) or (code & 4 and code & 8):
            continue
        out[uid] = i
    return out


def _aggregate_pairwise_edges(c, unit_of):
    b2l = []
    b2l_agg = {}
    for i, j, w in c["b2l"]:
        if w <= 0.0:
            b2l.append((i, j, w))
            continue

        ui, uj = unit_of[i], unit_of[j]
        if ui == uj or ui is None or uj is None:
            b2l.append((i, j, w))
            continue

        key = (min(i, j), max(i, j))
        if key not in b2l_agg:
            b2l_agg[key] = (i, j, w)
        else:
            oi, oj, ow = b2l_agg[key]
            b2l_agg[key] = (oi, oj, ow + w)
    b2l.extend((i, j, w) for i, j, w in b2l_agg.values())

    p2l = []
    p2l_agg = {}
    for p, i, w in c["p2l"]:
        if w <= 0.0:
            p2l.append((p, i, w))
            continue

        ui = unit_of[i]
        if ui is None:
            p2l.append((p, i, w))
            continue

        px, py = c["pin"][p]
        key = (i, float(px), float(py))
        if key not in p2l_agg:
            p2l_agg[key] = [p, i, w]
        else:
            p2l_agg[key][2] += w
    p2l.extend((p, i, w) for p, i, w in p2l_agg.values())
    return b2l, p2l


def _sep_reduction_mask(rows, n, P, unit_of, sv, resh, rho):
    """Transitive reduction of the same-axis separation rows. EXACT.

    A row for the ordered block pair (a, c) on the x axis reads

        d_u(a) + dsize_u(a) - d_u(c)  <=  gap_ac ,   gap_ac = x_c - (x_a + w_a)

    Adding the rows along a path a -> b -> c on the same axis and substituting
    the anchor identity `gap_ab + gap_bc = gap_ac - w_b` telescopes to

        d_u(a) + dsize_u(a) - d_u(c)  <=  gap_ac - (w_b + dsize_u(b))

    which implies the direct row as soon as `w_b + dsize_u(b) >= 0`, and the
    same collapse holds for any path length and for the y axis.  Only
    single-block units are reshapeable (`reshapeable()` skips `len(mem) != 1`),
    so that quantity is `(1 - rho) * w_b > 0` when b is reshapeable and `w_b > 0`
    otherwise -- every block qualifies as an intermediate today.  The test is
    still written out per block, so the reduction stays sound by construction if
    reshaping is ever widened to rigid multi-block units.

    The graph is per BLOCK and must stay that way.  A rigid cluster maps many
    blocks onto one unit, so a unit-level chain U1 -> U3 -> U2 is in general
    built from rows about *different* block pairs, and the identity above does
    not hold across them.  Reducing at unit level is what made the first cut of
    this drop non-redundant rows and move the LP objective by 12%.

    Removals are computed against the original edge set and applied in one shot.
    That is safe: take a longest path between the endpoints of a removed row
    (finite, since a same-axis edge forces `x_a < x_b`, so the graph is a DAG);
    none of its edges can itself be removable, or splicing in its replacement
    would yield a longer path.  So every removed row stays implied by rows that
    survive.
    """
    keep = [True] * len(rows)
    for axis in (0, 1):
        idxs = [k for k, r in enumerate(rows) if r["axis"] == axis]
        if len(idxs) <= 1:
            continue
        size = [P[b][2 + axis] for b in range(n)]
        usable = [False] * n
        for b in range(n):
            u = unit_of[b]
            lb = -rho * size[b] if (u is not None and u in sv) else 0.0
            usable[b] = size[b] + lb >= 0.0

        # chain only through non-negative gaps, which is what makes it a DAG
        adj = [0] * n
        for k in idxs:
            if rows[k]["rhs"] >= 0.0:
                adj[rows[k]["bi"]] |= 1 << rows[k]["bj"]

        # reverse topological order: an edge a -> b has x_a < x_b on this axis
        via = [0] * n          # reachable from a by a path of length >= 2
        thru = [0] * n         # reachable from a at all, usable intermediates
        for a in sorted(range(n), key=lambda b: P[b][axis], reverse=True):
            v, m_bits = 0, adj[a]
            while m_bits:
                low = m_bits & -m_bits
                m = low.bit_length() - 1
                m_bits ^= low
                if usable[m]:
                    v |= thru[m]
            via[a], thru[a] = v, adj[a] | v

        for k in idxs:
            if via[rows[k]["bi"]] >> rows[k]["bj"] & 1:
                keep[k] = False

    for k, r in enumerate(rows):
        if not r["terms"] and r["rhs"] >= 0.0:
            keep[k] = False        # `0 <= nonneg`, true without the row
    return keep


def build_and_solve(ci, P, freeze_units, rho=0.06, sep_trim=False, prune_B=None,
                    force_keep=frozenset(), hpwl_w=1.0, fix_dsize=None,
                    bbox_slack=None, area_R=None, area_g=1.05,
                    area_price=0.0):
    """prune_B (L112): displacement bound used to drop HPWL terms that provably
    cannot change sign.  None = off = the shipped formulation, bit-for-bit.

    fix_dsize (L121 route C): {uid: (dw, dh)} pins the shape columns instead of
    letting the LP pick them inside a rho trust region, and DROPS the area band
    rows.  Both halves are needed together: the caller supplies shapes that hold
    w*h = A exactly, whereas the band rows constrain the LINEARISED area
    p + h*dw + w*dh = A - dw*dh, which for any real aspect change overshoots the
    band and makes the program infeasible for no geometric reason.  Units in `sv`
    but absent from the dict are pinned at (0, 0).  `rho` still controls the two
    size-bound arguments (the separation reduction mask and the HPWL prune
    slack), so callers must pass a rho that upper-bounds their own |dsize| /
    dim -- both uses only need a bound, and an over-estimate is the safe side.

    Each (edge, axis) contributes `t >= |dC + delta|` as one aux column plus two
    rows -- 80.2% of all rows and ~98% of all columns (L112 S1: 15,572 cols /
    38,063 rows on case 99).  But |dC + delta| is only nonlinear where the sign
    can flip.  With |d_u| <= prune_B and |dsize| <= rho*dim, delta is bounded, so
    any edge with |dC| > max|delta| has a FIXED sign: the absolute value is
    linear there and its column+rows are exactly redundant -- fold the linear
    part into the objective instead.  Measured prunable share: 71.5-84.4%.

    Exactness is NOT assumed from "displacements look small": the caller must
    enforce |d_u| <= prune_B (bounds below) and then verify (a) no d_u sits on
    that bound and (b) no dropped term actually flipped.  build returns
    `prune_const` (the folded constant) and `prune_dropped/kept` for that check.
    """
    c = l3.CASES[ci]
    n, cn = c["n"], c["cn"]
    units, unit_of, group_units, group_comp0 = decompose(ci, P)
    U = len(units)
    resh = reshapeable(ci, units)
    if area_R is not None:
        # rho stops being a trust region under area_R and becomes only what its
        # two remaining users need -- an upper bound on |dsize| / dim for the
        # separation reduction mask and the HPWL prune slack.
        rho = max(rho, area_R - 1.0)

    XMIN, XMAX, YMIN, YMAX = 2 * U, 2 * U + 1, 2 * U + 2, 2 * U + 3
    nv = 2 * U + 4
    sv = {}
    for uid in sorted(resh):
        sv[uid] = (nv, nv + 1)
        nv += 2

    obj = [0.0] * nv
    rub, cub, vub = [], [], []
    req, ceq, veq, beq = [], [], [], []
    rows_by_origin = Counter()

    bub = []

    # per-row origin, so an infeasible program can be told WHICH family blocked
    # it (L121) instead of only how many rows each family contributed
    orig_ub, orig_eq = [], []

    def add_ub(terms, rhs, origin):
        r = len(bub)
        bub.append(rhs)
        orig_ub.append(origin)
        for col, coef in terms:
            rub.append(r), cub.append(col), vub.append(coef)
        rows_by_origin[origin] += 1

    def add_eq(terms, rhs, origin):
        r = len(beq)
        beq.append(rhs)
        orig_eq.append(origin)
        for col, coef in terms:
            req.append(r), ceq.append(col), veq.append(coef)
        rows_by_origin[origin] += 1

    def new_aux(w):
        nonlocal nv
        obj.append(w)
        nv += 1
        return nv - 1

    def dsize(u, axis):
        if u is None or u not in sv:
            return None
        return sv[u][axis]

    prune_const = 0.0
    prune_stat = [0, 0]                       # [dropped, kept]
    dropped = []                              # (term_id, lin, dC, wsc)
    term_id = [0]

    def add_hpwl_rows(wsc, ui, uj, off, dC, axis):
        # L120 L-2: hpwl_w=0 drops the wirelength half of the objective
        # entirely (no aux columns, no rows), leaving the pure bbox-area
        # program. That is the only form a longest-path solver can be
        # gated against -- minimising HPWL under difference constraints is
        # a min-cost-flow problem, not a path problem. Default 1.0 keeps
        # the shipped build bit-for-bit.
        if not hpwl_w:
            return
        nonlocal prune_const
        tid = term_id[0]
        term_id[0] += 1
        lin = []
        slack = 0.0
        for u, s in ((ui, 1.0), (uj, -1.0)):
            if u is None:
                continue
            lin.append((off + u, s))
            slack += prune_B or 0.0
            k = dsize(u, axis)
            if k is not None:
                lin.append((k, 0.5 * s))
                slack += 0.5 * rho * P[resh[u]][2 + axis]
        if prune_B is not None and abs(dC) > slack and tid not in force_keep:
            # Assume sign(dC + delta) == sign(dC) and fold the term in linearly.
            # |z| >= s*z for either s, so the pruned objective is a LOWER BOUND
            # on the true one everywhere.  Hence if the pruned optimum happens to
            # satisfy every assumed sign, it is optimal for the true problem too
            # -- which is why prune_B needs no bound clamp and is only a
            # HEURISTIC for which terms to try dropping.  solve_pruned() below
            # does that check and regenerates any term whose sign came out wrong.
            sgn = 1.0 if dC > 0.0 else -1.0
            for col, coef in lin:
                obj[col] += wsc * sgn * coef
            prune_const += wsc * sgn * dC
            prune_stat[0] += 1
            dropped.append((tid, tuple(lin), dC, wsc))
            return
        prune_stat[1] += 1
        t = new_aux(wsc)
        t1 = [(t, -1.0)] + lin
        t2 = [(t, -1.0)] + [(col, -coef) for col, coef in lin]
        add_ub(t1, -dC, "hpwl")
        add_ub(t2, dC, "hpwl")

    h_base = max(float(c["base"].get("hpwl_baseline", 1.0)), 1e-6)
    hw_scale = 0.5 / h_base
    cx = [P[i][0] + P[i][2] / 2.0 for i in range(n)]
    cy = [P[i][1] + P[i][3] / 2.0 for i in range(n)]
    const_h = 0.0
    obj0 = 0.0

    b2l_items, p2l_items = _aggregate_pairwise_edges(c, unit_of)
    for i, j, w in b2l_items:
        ui, uj = unit_of[i], unit_of[j]
        dCx, dCy = cx[i] - cx[j], cy[i] - cy[j]
        if w <= 0.0 or ui == uj:
            const_h += w * (abs(dCx) + abs(dCy))
            continue
        add_hpwl_rows(w * hw_scale, ui, uj, 0, dCx, 0)
        add_hpwl_rows(w * hw_scale, ui, uj, U, dCy, 1)
        obj0 += w * (abs(dCx) + abs(dCy))

    for p, i, w in p2l_items:
        ui = unit_of[i]
        px, py = c["pin"][p]
        dCx, dCy = cx[i] - px, cy[i] - py
        if w <= 0.0 or ui is None:
            const_h += w * (abs(dCx) + abs(dCy))
            continue
        add_hpwl_rows(w * hw_scale, ui, None, 0, dCx, 0)
        add_hpwl_rows(w * hw_scale, ui, None, U, dCy, 1)
        obj0 += w * (abs(dCx) + abs(dCy))

    sep_rows = []
    for i in range(n):
        xi, yi, wi, hi = P[i]
        for j in range(i + 1, n):
            ui, uj = unit_of[i], unit_of[j]
            if ui == uj:
                continue
            xj, yj, wj, hj = P[j]
            cands = (
                (xj - (xi + wi), ui, uj, 0, 0, i, j),
                (xi - (xj + wj), uj, ui, 0, 0, j, i),
                (yj - (yi + hi), ui, uj, U, 1, i, j),
                (yi - (yj + hj), uj, ui, U, 1, j, i),
            )
            # key is t[0] only, so the extra block ids cannot change the pick
            gap, ul, ur, off, axis, bl, br = max(cands, key=lambda t: t[0])
            terms = []
            if ul is not None:
                terms.append((off + ul, 1.0))
                k = dsize(ul, axis)
                if k is not None:
                    terms.append((k, 1.0))
            if ur is not None:
                terms.append((off + ur, -1.0))
            sep_rows.append({"axis": axis, "bi": bl, "bj": br,
                             "terms": terms, "rhs": gap})

    keep_mask = (_sep_reduction_mask(sep_rows, n, P, unit_of, sv, resh, rho)
                 if sep_rows else [])
    sep_kept = sum(1 for x in keep_mask if x)
    for row, kf in zip(sep_rows, keep_mask if sep_trim else [True] * len(keep_mask)):
        if kf:
            add_ub(row["terms"], row["rhs"], "separation")

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
    sides = ((1, XMIN, 0, xmin0, min(range(n), key=lambda i: P[i][0]), False, 0),
             (2, XMAX, 0, xmax0, max(range(n), key=lambda i: P[i][0] + P[i][2]), True, 0),
             (4, YMAX, U, ymax0, max(range(n), key=lambda i: P[i][1] + P[i][3]), True, 1),
             (8, YMIN, U, ymin0, min(range(n), key=lambda i: P[i][1]), False, 1))
    for bit, bv, off, ext0, mdef, far, axis in sides:
        tied = {unit_of[i] for i, code in sat if code & bit}
        if not tied:
            continue
        for u in tied:
            if u is None:
                add_eq([(bv, -1.0)], -ext0, "boundary_eq")
            else:
                t = [(off + u, 1.0), (bv, -1.0)]
                k = dsize(u, axis) if far else None
                if k is not None:
                    t.append((k, 1.0))
                add_eq(t, -ext0, "boundary_eq")
        um = unit_of[mdef]
        if um not in tied:
            if um is None:
                add_eq([(bv, -1.0)], -ext0, "boundary_eq")
            else:
                t = [(off + um, 1.0), (bv, -1.0)]
                k = dsize(um, axis) if far else None
                if k is not None:
                    t.append((k, 1.0))
                add_eq(t, -ext0, "boundary_eq")

    for i in range(n):
        ui = unit_of[i]
        if ui is None:
            continue
        x, y, w, h = P[i]
        add_ub([(XMIN, 1.0), (ui, -1.0)], x, "envelope")
        t = [(ui, 1.0), (XMAX, -1.0)]
        k = dsize(ui, 0)
        if k is not None:
            t.append((k, 1.0))
        add_ub(t, -(x + w), "envelope")
        add_ub([(YMIN, 1.0), (U + ui, -1.0)], y, "envelope")
        t = [(U + ui, 1.0), (YMAX, -1.0)]
        k = dsize(ui, 1)
        if k is not None:
            t.append((k, 1.0))
        add_ub(t, -(y + h), "envelope")
    # bbox_slack (L121): the shipped rows cap the extent at exactly today's, and
    # on a compacted layout they are exactly tight.  With shapes PINNED rather
    # than free, any block that widens on a tight x-chain has nowhere to put the
    # extra width and the whole program goes infeasible -- which is what kills
    # route C at a 0.05% shape change, four orders below the trust region it is
    # trying to escape.  Relaxing them makes bbox growth PRICED (by the area term
    # below, which bbox_slack also has to switch on) instead of FORBIDDEN.
    bs = 0.0 if bbox_slack is None else bbox_slack
    add_ub([(XMAX, 1.0), (XMIN, -1.0)], W0 * (1.0 + bs), "bbox")
    add_ub([(YMAX, 1.0), (YMIN, -1.0)], H0 * (1.0 + bs), "bbox")

    at = c["at"]
    if area_R is not None and fix_dsize is None:
        # L121: the area band replaced by TANGENT CUTS of w*h >= A(1-TOL).
        #
        # Measured on the pilot: of the two shipped band rows the LOWER one
        # binds on 259 of 338 reshapeable units and the UPPER on 9.  The lower
        # one is the boundary of a CONVEX region (h >= A'/w, and A'/w is convex),
        # so it is exactly representable by tangents -- no linearisation error
        # and no trust region.  The upper one is the non-convex side, and it is
        # also the actual barrier to a large aspect change: an exact-area
        # widening by r has TRUE area p but LINEARISED area p*(r + 1/r - 1),
        # which at r=1.5 reads as +16.7% and trips a band of +/-0.8%.  It is
        # dropped and checked afterwards instead (area_violations), which is the
        # same solve-then-verify contract solve_pruned already uses.
        #
        # Tangent at wk:  h >= 2A'/wk - (A'/wk^2) * w.  Points are geometric with
        # ratio area_g across [w0/R, w0*R]; consecutive tangents cross at
        # wk*sqrt(g), where the envelope sits (sqrt(g)-1)^2 below the curve.  At
        # g=1.05 that is 0.061%, so with A' = A*(1-AREA_TOL) the true area can
        # never fall below A*(1-0.008)*(1-0.00061) = 0.9914*A, inside the
        # official 1% hard limit with room to spare.
        import math
        steps = max(1, int(math.ceil(2.0 * math.log(area_R) / math.log(area_g))))
        for uid, i in resh.items():
            kw, kh = sv[uid]
            w, h = P[i][2], P[i][3]
            A = float(at[i]) if float(at[i]) > 0 else w * h
            Ap = A * (1.0 - AREA_TOL)
            for s in range(steps + 1):
                wk = (w / area_R) * (area_R * area_R) ** (s / steps)
                sl = Ap / (wk * wk)
                # h0 + dh >= 2A'/wk - (A'/wk^2)(w0 + dw)   ->   -sl*dw - dh <= rhs
                add_ub([(kw, -sl), (kh, -1.0)],
                       -(2.0 * Ap / wk - sl * w - h), "area_tangent")
    elif fix_dsize is None:
        for uid, i in resh.items():
            kw, kh = sv[uid]
            w, h = P[i][2], P[i][3]
            p = w * h
            A = float(at[i]) if float(at[i]) > 0 else p
            slack = rho * rho * p
            add_ub([(kw, -h), (kh, -w)], -(A * (1.0 - AREA_TOL) - p + slack), "area_band")
            add_ub([(kw, h), (kh, w)], A * (1.0 + AREA_TOL) - p - slack, "area_band")

    a_base = max(float(c["base"].get("area_baseline", W0 * H0)), 1e-6)
    # Without the area term the extra room is free and the LP spends all of it,
    # so bbox_slack has to switch the term on even below the area baseline.
    if W0 * H0 > a_base or bbox_slack is not None:
        bA = 0.5 / a_base
        obj[XMIN] -= bA * H0
        obj[XMAX] += bA * H0
        obj[YMIN] -= bA * W0
        obj[YMAX] += bA * W0

    # Deliberately NOT gated on area_R: the price has to be applicable to the
    # shipped band too, or there is no control arm separating "wider aspect
    # range" from "shrink every block to its minimum legal area".
    if area_price and fix_dsize is None:
        # The dropped upper area bound has exactly one failure mode, and it is
        # visible in the numbers: blocks with NO pressure on them run to the far
        # corner of the shape box, landing at area A*R^2 (measured worst errors
        # 44% at R=1.2, 125% at R=1.5, 300% at R=2 -- i.e. R^2-1 to the digit).
        # Growth is simply free for them.  A tiny price on the block's own area
        # removes the incentive without competing with any real term: the
        # feasible set {w*h >= A'} is convex, so a positive linear cost pushes
        # each such block ONTO its boundary, where the true area is A' exactly.
        # Blocks the rest of the objective genuinely wants larger still grow --
        # those are the ones hard_ok is left to adjudicate.
        # No upper cut can do this exactly: with the lower side held by tangents,
        # the true band is a thin NON-convex sliver and no LP represents it.
        pw = area_price * 0.5 / a_base
        for uid, i in resh.items():
            kw, kh = sv[uid]
            obj[kw] += pw * P[i][3]
            obj[kh] += pw * P[i][2]

    # L119: boundary REPAIR. The `sides` loop above only pins blocks that ALREADY
    # touch their required edge -- `bnd_skip` counts the ones that do not, and the
    # LP has never tried to fix them. It only avoids breaking what is already
    # right. That matters because violations enter the official cost as an
    # EXPONENT, exp(2.0 * V_rel), and boundary is 83% of all soft violations
    # in-set (78 of 94) and 19% held-out (109 of 561).
    #
    # A hard equality would make the LP infeasible wherever the room is not
    # already there, which measures nothing, so each unmet side gets a slack
    # column: the LP pays `lam * distance-from-the-edge` and decides for itself
    # whether pushing the neighbours aside is worth it. BND_W is dimensionless --
    # 1.0 prices a unit of boundary distance like a unit of bbox growth on that
    # axis -- and is None by default, leaving the shipped formulation untouched.
    if BND_W is not None:
        bAb = 0.5 / a_base
        for bit, bv, off, ext0, _mdef, far, axis in sides:
            for i, code in bnd:
                if not code & bit or touch_ok(i, code):
                    continue
                u = unit_of[i]
                # BND_W == 0 drops the slack column entirely, which turns the
                # row into a hard "touch" (the envelope supplies the other
                # inequality). That is the only way an LP can actually CLOSE a
                # boundary violation: touching is a 0/1 predicate with a 1e-6
                # threshold, and a linear slack price only ever shortens the
                # distance -- priced from 0.5 to 32 it never cleared a single
                # violation (77 of 78 left at every price). The hard form is
                # what tells us whether the room exists at all; it can make the
                # LP infeasible, and that answer is the measurement.
                s = (new_aux(BND_W * bAb * (H0 if axis == 0 else W0))
                     if BND_W else None)
                x0 = P[i][axis] + (P[i][2 + axis] if far else 0.0)
                # A preplaced block (u is None) cannot move, but the EDGE can:
                # the bbox is a min/max over all blocks, so the requirement is
                # met by pulling the envelope in to meet the block -- everything
                # sticking out past it moves instead. 49% of the shipped
                # boundary violations (38 of 78) sit on preplaced blocks, so
                # skipping this case gives up half the lever, and shrinking the
                # bbox happens to be what the area term wants anyway.
                if far:
                    t = [(bv, 1.0)]
                    if s is not None:
                        t.append((s, -1.0))
                    if u is not None:
                        t.append((off + u, -1.0))
                        k = dsize(u, axis)
                        if k is not None:
                            t.append((k, -1.0))
                    add_ub(t, x0, "boundary_slack")
                else:
                    t = [(bv, -1.0)]
                    if s is not None:
                        t.append((s, -1.0))
                    if u is not None:
                        t.append((off + u, 1.0))
                    add_ub(t, -x0, "boundary_slack")

    for u in freeze_units:
        add_eq([(u, 1.0)], 0.0, "boundary_eq")
        add_eq([(U + u, 1.0)], 0.0, "boundary_eq")
        if u in sv:
            add_eq([(sv[u][0], 1.0)], 0.0, "boundary_eq")
            add_eq([(sv[u][1], 1.0)], 0.0, "boundary_eq")

    D = W0 + H0 + 1.0
    fro = [i for i in range(n) if unit_of[i] is None]
    # L112: NO clamp. An earlier cut restricted |d_u| <= prune_B to make the
    # pruning bound hold a priori; the lower-bound argument in add_hpwl_rows
    # makes that unnecessary, and clamping would turn the LP into a
    # restriction whose optimum can differ from the real one.
    bounds = [(-D, D)] * (2 * U)
    bounds.append((xmin0 - D, min((P[i][0] for i in fro), default=xmin0 + D)))
    bounds.append((max((P[i][0] + P[i][2] for i in fro), default=xmax0 - D), xmax0 + D))
    bounds.append((ymin0 - D, min((P[i][1] for i in fro), default=ymin0 + D)))
    bounds.append((max((P[i][1] + P[i][3] for i in fro), default=ymax0 - D), ymax0 + D))
    for uid in sorted(sv):
        i = resh[uid]
        if fix_dsize is not None:
            dw, dh = fix_dsize.get(uid, (0.0, 0.0))
            bounds.append((dw, dw))
            bounds.append((dh, dh))
            continue
        if area_R is not None:
            # a box on the SHAPE, not a trust region on the linearisation
            bounds.append((P[i][2] / area_R - P[i][2], P[i][2] * area_R - P[i][2]))
            bounds.append((P[i][3] / area_R - P[i][3], P[i][3] * area_R - P[i][3]))
            continue
        bounds.append((-rho * P[i][2], rho * P[i][2]))
        bounds.append((-rho * P[i][3], rho * P[i][3]))
    bounds += [(0.0, None)] * (nv - len(bounds))

    t_build0 = time.perf_counter()
    A_ub = sparse.csr_matrix((vub, (rub, cub)), shape=(len(bub), nv))
    A_eq = (sparse.csr_matrix((veq, (req, ceq)), shape=(len(beq), nv))
            if beq else None)
    t_build = time.perf_counter() - t_build0

    t_solve0 = time.perf_counter()
    res = linprog(np.asarray(obj), A_ub=A_ub, b_ub=np.asarray(bub),
                  A_eq=A_eq, b_eq=np.asarray(beq) if beq else None,
                  bounds=bounds, method="highs")
    t_solve = time.perf_counter() - t_solve0

    return dict(
        res=res,
        # L120 L-4 needs the constraint matrix to fold duals back onto the
        # shape columns; these are references to objects already built, so
        # carrying them costs nothing and changes no behaviour.
        A_ub=A_ub,
        A_eq=A_eq,
        obj=obj,
        b_ub=bub,
        b_eq=beq,
        bounds=bounds,
        origins_ub=orig_ub,
        origins_eq=orig_eq,
        prune_const=prune_const,
        prune_dropped=prune_stat[0],
        prune_kept=prune_stat[1],
        prune_dropped_terms=dropped,
        units=units,
        unit_of=unit_of,
        U=U,
        sv=sv,
        group_units=group_units,
        group_comp0=group_comp0,
        const_h=const_h,
        obj0=obj0,
        rows_ub=len(bub),
        rows_eq=len(beq),
        nnz=len(vub) + len(veq),
        rows_by_origin=dict(rows_by_origin),
        sep_rows_total=len(sep_rows),
        sep_rows_kept=sep_kept,
        timing=dict(t_build=t_build, t_solve=t_solve),
    )


def apply_all(P, B, x):
    U = B["U"]
    out = list(P)
    for uid, mem in enumerate(B["units"]):
        dx, dy = x[uid], x[U + uid]
        dw = dh = 0.0
        if uid in B["sv"]:
            dw, dh = x[B["sv"][uid][0]], x[B["sv"][uid][1]]
        for i in mem:
            px, py, pw, ph = P[i]
            out[i] = (px + dx, py + dy, pw + dw, ph + dh)
    return out


PRUNE_B = None      # L112: module-level so dep_case/mode_ab measure end-to-end
BND_W = None        # L119: boundary-repair slack price; None = shipped behaviour


def lp_pass(ci, P, rho, sep_trim=False):
    c = l3.CASES[ci]
    freeze = set()
    for attempt in range(3):
        t0 = time.perf_counter()
        # HPWL pruning and the separation transitive reduction used to be
        # mutually exclusive here purely because solve_pruned did not forward
        # sep_trim. They are independent reductions -- after pruning,
        # separation is the MAJORITY of the remaining rows (56-73% on the
        # heavy cases), so the combination is the interesting one.
        if PRUNE_B is not None:
            B, _rounds = solve_pruned(ci, P, freeze, rho=rho, prune_B=PRUNE_B,
                                      sep_trim=sep_trim)
        else:
            B = build_and_solve(ci, P, freeze, rho=rho, sep_trim=sep_trim)
        if B["res"].status != 0:
            return None, dict(status=f"lp_status_{B['res'].status}", t=time.perf_counter() - t0,
                              t_build=B["timing"]["t_build"], t_solve=B["timing"]["t_solve"],
                              lp_obj=None, attempts=attempt + 1), None
        newP = apply_all(P, B, B["res"].x)
        broken = [g for g, c0 in B["group_comp0"].items()
                  if len(l3.comp_split(newP, [i for i in range(c["n"]) if c["cn"][i][3] == g])) > c0]
        if not broken:
            return newP, dict(status="ok", t=time.perf_counter() - t0,
                              t_build=B["timing"]["t_build"], t_solve=B["timing"]["t_solve"],
                              lp_obj=float(B["res"].fun) + B.get("prune_const", 0.0),
                              attempts=attempt + 1), B
        for g in broken:
            freeze |= B["group_units"][g]
    return None, dict(status="cluster_break", t=0.0, t_build=0.0, t_solve=0.0, lp_obj=None, attempts=3), None


def solve_pruned(ci, P, freeze_units, rho=0.06, prune_B=None, max_rounds=3,
                 sep_trim=False, **kw):
    """build_and_solve with HPWL pruning made EXACT by verification.

    Dropping `t >= |z|` in favour of `sgn(dC) * z` can only LOWER the objective
    (|z| >= s*z for either s), so the pruned program is a lower bound on the real
    one.  If its optimum satisfies every assumed sign then it attains the real
    objective there, and being <= the real minimum it must BE the real minimum.
    So: solve, check the dropped terms' signs, and re-solve with the offenders
    forced back in.  prune_B is therefore only a guess at which terms are safe --
    a wrong guess costs a round, never correctness.  Falls back to the unpruned
    build if it has not converged in `max_rounds`.
    """
    if prune_B is None:
        return build_and_solve(ci, P, freeze_units, rho=rho,
                               sep_trim=sep_trim, **kw), 0
    keep = set()
    for rnd in range(max_rounds):
        d = build_and_solve(ci, P, freeze_units, rho=rho, prune_B=prune_B,
                            force_keep=keep, sep_trim=sep_trim, **kw)
        res = d["res"]
        if res.status != 0:
            return build_and_solve(ci, P, freeze_units, rho=rho,
                                   sep_trim=sep_trim, **kw), rnd + 1
        x = res.x
        bad = set()
        for tid, lin, dC, _w in d["prune_dropped_terms"]:
            delta = 0.0
            for col, coef in lin:
                delta += coef * x[col]
            # assumed sgn(dC); violated iff the true value took the other branch
            if (dC + delta) * (1.0 if dC > 0.0 else -1.0) < -1e-12:
                bad.add(tid)
        if not bad:
            d["prune_rounds"] = rnd + 1
            return d, rnd + 1
        keep |= bad
    return build_and_solve(ci, P, freeze_units, rho=rho,
                           sep_trim=sep_trim, **kw), max_rounds


_HARD_MASKS = {}


def _hard_masks(ci):
    """Per-case boolean masks for hard_ok, built once."""
    m = _HARD_MASKS.get(ci)
    if m is None:
        c = l3.CASES[ci]
        cn, at, n = c["cn"], c["at"], c["n"]
        fixed = np.array([cn[i][0] != 0 or cn[i][1] != 0 for i in range(n)])
        pre = np.array([cn[i][1] != 0 for i in range(n)])
        area = np.array([float(at[i]) for i in range(n)])
        m = _HARD_MASKS[ci] = (fixed, pre, area, ~fixed & (area > 0))
    return m


def hard_ok(P0, P, ci):
    """Vectorised; same accept/reject as the original nested loops.

    The old version was a pure-Python O(n^2) pair scan -- part of the ~19% of
    tLP that L100's honest-scope section never split out (L112 S1 measured it).
    Early-exit on the first violation is dropped, which only matters for the
    rejected minority; the returned verdict is identical because all the
    comparisons are the same strict float tests over the same pairs.
    """
    fixed, pre, area, soft = _hard_masks(ci)
    A = np.asarray(P, dtype=float)
    A0 = np.asarray(P0, dtype=float)
    if np.any(A[:, 2] <= 0) or np.any(A[:, 3] <= 0):
        return False
    if np.any(fixed & ((A[:, 2] != A0[:, 2]) | (A[:, 3] != A0[:, 3]))):
        return False
    if np.any(pre & ((A[:, 0] != A0[:, 0]) | (A[:, 1] != A0[:, 1]))):
        return False
    if np.any(soft & (np.abs(A[:, 2] * A[:, 3] - area) > 0.01 * area)):
        return False
    x2, y2 = A[:, 0] + A[:, 2], A[:, 1] + A[:, 3]
    ox = np.minimum(x2[:, None], x2[None, :]) - np.maximum(A[:, 0][:, None], A[:, 0][None, :])
    oy = np.minimum(y2[:, None], y2[None, :]) - np.maximum(A[:, 1][:, None], A[:, 1][None, :])
    bad = (ox > EPS_OVL) & (oy > EPS_OVL)
    np.fill_diagonal(bad, False)
    return not bool(bad.any())


def proxy_m(ci, P):
    c = l3.CASES[ci]
    return oc._proxy_metrics(P, c["at"], c["b2b"], c["p2b"], c["pins"],
                             c["cons"], c["n"])


def dep_case(ci, anch, iters, rho, sep_trim=False):
    P0 = [tuple(p) for p in anch[ci]["positions"]]
    q_ship = anch[ci]["cost"]
    m_prev = proxy_m(ci, P0)
    P, kept_n = P0, 0
    passes = []
    lp_obj0 = None

    for _ in range(iters):
        newP, tele, B = lp_pass(ci, P, rho, sep_trim=sep_trim)
        if tele["lp_obj"] is None:
            passes.append(dict(t=tele["t"], t_build=tele["t_build"], t_solve=tele["t_solve"],
                              kept=False, cost=None, feas=False, lp_obj=None))
            break
        if lp_obj0 is None:
            lp_obj0 = tele["lp_obj"]
        m_new = proxy_m(ci, newP)
        better = (m_new["hpwl"] < m_prev["hpwl"] * (1 - 1e-12)
                  or m_new["area"] < m_prev["area"] * (1 - 1e-12))
        worse = (m_new["hpwl"] > m_prev["hpwl"] * (1 + 1e-12)
                 or m_new["area"] > m_prev["area"] * (1 + 1e-12)
                 or m_new["vrel"] > m_prev["vrel"] + 1e-12)
        mm = l3.cost_eval(ci, newP)
        keep = bool(better and not worse and hard_ok(P0, newP, ci))
        passes.append(dict(t=tele["t"], t_build=tele["t_build"], t_solve=tele["t_solve"],
                          kept=keep, cost=float(mm.cost), feas=bool(mm.is_feasible),
                          lp_obj=float(tele["lp_obj"])))
        if not keep:
            break
        P = newP
        kept_n += 1
        m_prev = m_new

    m = l3.cost_eval(ci, P)
    return dict(ci=ci, n=l3.CASES[ci]["n"], q_ship=q_ship, q_new=float(m.cost),
                feas=bool(m.is_feasible), kept=kept_n, status=tele["status"] if passes else "ok",
                passes=passes, lp_obj0=lp_obj0)


def baseline_lp_obj0(ci, P, rho):
    c = l3.CASES[ci]
    freeze = set()
    for _ in range(3):
        B = l97.build_and_solve(ci, P, freeze_units=freeze, area_obj=True,
                               allow_reshape=True, rho=rho)
        if B["res"].status != 0:
            return None
        newP = l97.apply_all(P, B, B["res"].x)
        broken = [g for g, c0 in B["group_comp0"].items()
                  if len(l3.comp_split(newP, [i for i in range(c["n"]) if c["cn"][i][3] == g])) > c0]
        if not broken:
            return float(B["res"].fun)
        for g in broken:
            freeze |= B["group_units"][g]
    return None


def _after_k_cost(case_row, k):
    kept = [p for p in case_row["passes"][:k] if p.get("kept") and p.get("cost") is not None]
    return kept[-1]["cost"] if kept else case_row["q_ship"]


def census_duplicates(ci, unit_of, c):
    b2l_total = p2l_total = 0
    b2l_keys = set()
    p2l_keys = set()
    for i, j, w in c["b2l"]:
        ui, uj = unit_of[i], unit_of[j]
        if w > 0 and ui != uj:
            b2l_total += 1
            b2l_keys.add(tuple(sorted((i, j))))
    for p, i, w in c["p2l"]:
        if w <= 0:
            continue
        px, py = c["pin"][p]
        p2l_total += 1
        p2l_keys.add((i, float(px), float(py)))
    return dict(
        b2l_total=b2l_total, b2l_unique=len(b2l_keys), b2l_dup=b2l_total - len(b2l_keys),
        p2l_total=p2l_total, p2l_unique=len(p2l_keys), p2l_dup=p2l_total - len(p2l_keys),
    )


def mode_census(cases):
    anch = {t["test_id"]: t for t in json.loads(ANCHOR.read_text(encoding="utf-8"))["test_results"]}
    total_rows_ub = total_rows_eq = total_nnz = 0
    origin = Counter()
    sep_total = sep_kept = 0
    dup_b2l = dup_p2l = 0
    dup_edges = 0
    tb, ts = [], []
    print(f"{'ci':>3} {'nv':>5} {'ub':>6} {'eq':>6} {'nnz':>7} "
          f"{'build':>7} {'solve':>7} {'sep':>6} {'trim':>6} {'dup':>5} {'obj':>11}")
    for ci in range(cases):
        P = [tuple(p) for p in anch[ci]["positions"]]
        units, unit_of, _, _ = decompose(ci, P)
        dup = census_duplicates(ci, unit_of, l3.CASES[ci])
        B = build_and_solve(ci, P, set(), rho=0.06, sep_trim=False)
        if B["res"].status != 0:
            print(f"{ci:3d} status {B['res'].status}")
            continue
        tb.append(B["timing"]["t_build"])
        ts.append(B["timing"]["t_solve"])
        total_rows_ub += B["rows_ub"]
        total_rows_eq += B["rows_eq"]
        total_nnz += B["nnz"]
        sep_total += B["sep_rows_total"]
        sep_kept += B["sep_rows_kept"]
        for k, v in B["rows_by_origin"].items():
            origin[k] += v
        dup_b2l += dup["b2l_dup"]
        dup_p2l += dup["p2l_dup"]
        dup_edges += dup["b2l_total"] + dup["p2l_total"]
        print(f"{ci:3d} {B['rows_ub']+B['rows_eq']:5d} {B['rows_ub']:6d} {B['rows_eq']:6d} {B['nnz']:7d} "
              f"{B['timing']['t_build']:7.3f} {B['timing']['t_solve']:7.3f} "
              f"{B['sep_rows_total']:6d} {B['sep_rows_kept']:6d} {dup['b2l_dup'] + dup['p2l_dup']:5d} {float(B['res'].fun):11.6g}")
    print("\n=== CENSUS AGGREGATE ===")
    print(f"  cases                  = {cases}")
    print(f"  rows ub/eq/nnz          = {total_rows_ub}/{total_rows_eq}/{total_nnz}")
    if total_rows_ub + total_rows_eq:
        print("  row split by origin     = " +
              ", ".join(f"{k}:{v/(total_rows_ub + total_rows_eq):.1%}" for k, v in origin.items()))
    if sep_total:
        print(f"  separation rows         = {sep_total}, kept={sep_kept}, removable={sep_total - sep_kept} "
              f"({(sep_total - sep_kept)/sep_total:.1%})")
    print(f"  duplicates             = {dup_b2l + dup_p2l}/{dup_edges} ({(dup_b2l + dup_p2l)/dup_edges:.1%} if dup_edges>0 else 0.0)")
    print(f"  build mean             = {sum(tb)/len(tb):.3f}s")
    print(f"  solve mean             = {sum(ts)/len(ts):.3f}s")
    DUMP.write_text(json.dumps({
        "mode": "census",
        "cases": cases,
        "rows": dict(ub=total_rows_ub, eq=total_rows_eq, nnz=total_nnz, by_origin=dict(origin)),
        "separation": dict(total=sep_total, kept=sep_kept),
        "duplicates": dict(b2l_dup=dup_b2l, p2l_dup=dup_p2l, edges=dup_edges),
        "timing": dict(avg_build=sum(tb) / len(tb), avg_solve=sum(ts) / len(ts)),
    }, indent=1), encoding="utf-8")
    print(f"\njson -> {DUMP}")


def mode_ab(cases, iters, rho, sep_trim, baseline_json):
    anch = {t["test_id"]: t for t in json.loads(ANCHOR.read_text(encoding="utf-8"))["test_results"]}
    base_raw = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
    base_cases = {int(k): v for k, v in base_raw["cases"].items()}
    W, TOTW = l3.W, l3.TOTW

    rows = {}
    max_obj_rel = 0.0
    for ci in range(cases):
        P0 = [tuple(p) for p in anch[ci]["positions"]]
        base_obj = baseline_lp_obj0(ci, P0, rho)
        r = dep_case(ci, anch, iters, rho, sep_trim=sep_trim)
        rows[ci] = r
        if base_obj is not None and r["lp_obj0"] is not None and base_obj != 0.0:
            max_obj_rel = max(max_obj_rel, abs(r["lp_obj0"] - base_obj) / abs(base_obj))
        print(f"case {ci:3d} kept={r['kept']:2d}/{iters} t={sum(p['t'] for p in r['passes']):.3f}s status={r['status']}")

    cov = sum(W[c] for c in range(cases))
    after_k = {}
    base_k = {}
    for k in range(1, iters + 1):
        after_k[k] = sum(W[ci] * _after_k_cost(rows[ci], k) for ci in range(cases)) / cov
        base_k[k] = sum(W[ci] * _after_k_cost(base_cases[ci], k) for ci in range(cases)) / cov

    base_ship = sum(W[c] * rows[c]["q_ship"] for c in range(cases)) / cov
    after = sum(W[c] * rows[c]["q_new"] for c in range(cases)) / cov
    gain = 100.0 * (1.0 - after / base_ship)
    regr = [c for c in range(cases) if rows[c]["q_new"] > rows[c]["q_ship"] + 1e-12]
    infea = [c for c in range(cases) if not rows[c]["feas"]]

    tlp1 = [sum(p["t"] for p in rows[ci]["passes"][:1]) for ci in range(cases)]
    tlp4 = [sum(p["t"] for p in rows[ci]["passes"]) for ci in range(cases)]
    t_sorted1 = sorted(tlp1)
    t_sorted4 = sorted(tlp4)

    print("\n=== A/B RESULTS ===")
    print(f"  baseline k1 gain        = {100.0 * (1.0 - base_k[1] / base_ship):+8.4f}%")
    print(f"  accelerated k1 gain     = {100.0 * (1.0 - after_k[1] / base_ship):+8.4f}%")
    print(f"  baseline k{iters} gain   = {100.0 * (1.0 - base_k[iters] / base_ship):+8.4f}%")
    print(f"  accelerated k{iters} gain= {100.0 * (1.0 - after_k[iters] / base_ship):+8.4f}%")
    print(f"  objective k1 rel diff    = {max_obj_rel:.3e}")
    print(f"  regressions             = {len(regr)} {regr[:8]}")
    print(f"  infeasible              = {len(infea)} {infea[:8]}")
    print(f"  tLP k1 p50/p90/max      = {t_sorted1[len(t_sorted1)//2]:.3f}s/{t_sorted1[int(.9*len(t_sorted1))]:.3f}s/{max(t_sorted1):.3f}s")
    print(f"  tLP all p50/p90/max     = {t_sorted4[len(t_sorted4)//2]:.3f}s/{t_sorted4[int(.9*len(t_sorted4))]:.3f}s/{max(t_sorted4):.3f}s")
    print(f"  weighted-mean tLP(k1)    = {sum(W[c] * tlp1[c] for c in range(cases)) / cov:.3f}s")

    DUMP.write_text(json.dumps({
        "mode": "ab",
        "gain": gain,
        "before": base_ship,
        "after": after,
        "regressions": regr,
        "infeasible": infea,
        "max_lp_obj_rel_diff": max_obj_rel,
        "curve": {str(k): {"accelerated": after_k[k], "baseline": base_k[k],
                          "gain": 100.0 * (1.0 - after_k[k] / base_ship),
                          "delta_vs_baseline_pp": 100.0 * (1.0 - after_k[k] / base_k[k])
                          if base_k[k] != 0 else 0.0}
                  for k in range(1, iters + 1)},
        "cases": {str(ci): {k: v for k, v in rows[ci].items() if k != "lp_obj0"} for ci in range(cases)},
    }, indent=1), encoding="utf-8")
    print(f"\njson -> {DUMP}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["census", "ab"])
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.06)
    ap.add_argument("--sep-trim", action="store_true", dest="sep_trim")
    ap.add_argument("--baseline-json", default=str(BASELINE))
    ap.add_argument("--prune-b", type=float, default=None, dest="prune_b",
                    help="HPWL pruning bound B (L112). Unset = off = the "
                         "unpruned chain. The pruned program is a LOWER "
                         "bound that solve_pruned() re-solves until every "
                         "assumed sign holds, so a bad B costs a round, "
                         "never correctness. Best fixed B measured: 8.")
    a = ap.parse_args()

    global PRUNE_B
    PRUNE_B = a.prune_b
    print(f"[cfg] PRUNE_B = {PRUNE_B}")

    if a.mode == "census":
        mode_census(a.cases)
    else:
        mode_ab(a.cases, a.iters, a.rho, a.sep_trim, a.baseline_json)


if __name__ == "__main__":
    main()
