"""L129 — a global placer written from scratch, as a portfolio CANDIDATE.

OFFLINE for now, and label-free by construction: it reads only what the C++ reads
from stdin (areas, connectivity, pins, constraints, and the preplaced/fixed
placements the evaluator itself hands the optimizer via `build_opt_target_pos`).
No fp_sol anywhere. **The submission package is not touched by this file.**

WHY THIS SHAPE, given everything measured
-----------------------------------------
L128 split the deficit: hpwl_gap 0.2422 (worth +10.15%), area_gap 0.1431 (+6.00%),
vrel 0.01789 (+3.57%) -- and showed the LP's own fixpoint closes HALF the area gap
and 7.5% of the wirelength gap, because shortening wire means moving blocks PAST
each other and that is a topology change. So the residual is topological, and only
a global method reaches it.

L128 also killed the cheap way to get one: a topology cannot be transplanted (the
label's topology + a 2% aspect change is already infeasible), and **our LP is a
local repair tool, not a legaliser**. So this file brings its own legaliser, and
it is deliberately NOT an LP:

    relation selection (per pair, cheapest axis, oriented by centre order)
      -> both constraint graphs are sub-orders of the centre orderings, hence
         ACYCLIC by construction
      -> longest-path compaction, which ALWAYS yields an overlap-free layout

That is the property the LP does not have and the reason L128 saw status-2 on
98/100. Feasibility here is structural, not hoped for.

WHY PARTIAL COVERAGE IS FINE
----------------------------
The deployed form is one more candidate in a proxy-arbitrated portfolio, and the
proxy is oracle-perfect on heterogeneous candidates (M76 full-union, M77
efficiency 100.0%). A candidate does not have to win everywhere -- it has to win
SOMETIMES. So a case this placer cannot legalise simply emits nothing and the
portfolio is unchanged there. That converts "preplaced blocks make compaction
hard" from fatal into a coverage number.

MEASUREMENT PATH (already built, verified, and never yet used on a real candidate)
  in-set   m77_ml_candidate_probe.py score <json> --cores 48   bar +0.05%
  OOS      m77_oos_probe.py                                     bar NET +0.30%

Run:
  <python> -u l129_global_placer.py run  --out results_L129.json
  <python> -u l129_global_placer.py run  --limit 10 --verbose
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in sorted(k for k in os.environ if k.startswith("ICCAD_")):
    del os.environ[_k]

import numpy as np                                                  # noqa: E402
from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402
import optimizer_constructive as oc                                 # noqa: E402

# The wrapper's `l3` is an internal shim class, not the m53 module (L128).
_l3 = oc.l3

EPS = 1e-9
# Bisection into a square of side sqrt(total_area) asks for 100% density, which
# no rectangle packing reaches; the relation selection then reads "everything
# overlaps everything" and the compaction chains blow the bbox up. Give it the
# room a real packing needs.
DENSITY = float(os.environ.get("L129_DENSITY", "0.80"))
REFINE_AREA = os.environ.get("L129_REFINE_AREA", "1") != "0"
LP_POLISH = os.environ.get("L129_LP", "1") != "0"
# L134: the LP is 95.8% of a case and each pass costs ~1.5s weighted (up to 4.9s
# for ONE pass at n=99) against a 48c incumbent wall of 1.0-1.8s. These two make
# the polish affordable-by-construction instead of unconditional:
#   LP_BUDGET  stop before a pass once the case has already spent this long
#   LP_MAXN    skip the polish entirely above this block count
# 0 = off for both, i.e. the unconditional 6 passes that were measured before.
LP_BUDGET = float(os.environ.get("L129_LP_BUDGET", "0"))
LP_MAXN = int(os.environ.get("L129_LP_MAXN", "0"))
MIB_UNIFY = os.environ.get("L129_MIB", "1") != "0"
BND_SHELF = os.environ.get("L129_BND_SHELF", "1") != "0"
# IRLS=1 (plain quadratic) is the DEFAULT and the measured best: see the
# note on refine_area. Reweighting by 1/|delta| collapses connected units
# together, and the spread the quadratic term provides is what makes the
# ordering readable at all. IRLS=4 measured 1.709 -> 2.018.
IRLS = int(os.environ.get("L129_IRLS", "1"))
# GORDIAN alternation (L130). OFF by default so the v6 baseline (1.745, 64/100)
# stays reproducible from this file; L129_GORDIAN=1 replaces the
# one-solve-then-one-destructive-bisection stage A with solve/partition/re-solve.
# Priced BEFORE building (l130_gordian_price.py): stage A is 0.46% of weighted
# runtime and the added KKT solves are 0.06%, so this lever cannot move the wall.
GORDIAN = os.environ.get("L129_GORDIAN", "0") != "0"
# regions are split until they hold at most this many units; 1 reproduces
# `spread`'s leaf behaviour (a lone free unit is pinned at its region centre)
GORDIAN_MIN = int(os.environ.get("L129_GORDIAN_MIN", "1"))
GORDIAN_MAXLVL = int(os.environ.get("L129_GORDIAN_LEVELS", "16"))
# boundary-aware partitioning; without it the alternation buys hpwl and area and
# pays for both in boundary violations (see _bnd_rank)
GORDIAN_BND = os.environ.get("L129_GORDIAN_BND", "1") != "0"
# snap the final coordinates to leaf region centres, keeping only the ORDER the
# alternation found (see _snap_leaves)
GORDIAN_SNAP = os.environ.get("L129_GORDIAN_SNAP", "1") != "0"
# bit-exact abutment at emit time (see _emit_abut). OFF by default so the v6
# anchor stays reproducible; it is a separate lever and is measured as one.
EXACT_ABUT = os.environ.get("L129_EXACT_ABUT", "0") != "0"
# never flip the same pair twice inside one legalise call (see the note in
# `legalise`). OFF by default so the v6 anchor stays reproducible.
LEGAL_LOCK = os.environ.get("L129_LEGAL_LOCK", "0") != "0"
# 🚨 1=left, 2=right, 4=TOP, 8=BOTTOM. Verified against BOTH the official
# evaluator (iccad2026_evaluate.py:521, "1=left, 2=right, 4=top, 8=bottom") and
# constructive.cpp:50. The first draft of this file had 4/8 swapped; it was
# harmless only because choose_dims uses the UNION (B_TOP|B_BOTTOM), and it would
# have inverted every boundary decision below.
B_LEFT, B_RIGHT, B_TOP, B_BOTTOM = 1, 2, 4, 8

print("[l129] loading dataset ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()

CASES = []
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _at, _b2b, _p2b, _pins, _cons = _s["input"]
    _n = int((_at != -1).sum().item())
    _base, _tp = _ev._extract_baseline(_idx, _s["label"], _b2b, _p2b, _pins, _n)
    CASES.append(dict(idx=_idx, n=_n, w=math.exp(_n / 12.0), base=_base, tp=_tp,
                      at=_at, b2b=_b2b, p2b=_p2b, pins=_pins, cons=_cons,
                      otp=build_opt_target_pos(_tp, _cons, _n)))


# ── the only inputs the deployed placer may read ────────────────────────────
def case_view(c):
    """Everything below this line is what the C++ gets on stdin. No label."""
    n = c["n"]
    cons = [[int(v) for v in c["cons"][i].tolist()] for i in range(n)]
    otp = [[float(v) for v in c["otp"][i].tolist()] for i in range(n)]
    area = [max(0.0, float(c["at"][i])) for i in range(n)]
    b2b = [(int(e[0]), int(e[1]), float(e[2]))
           for e in c["b2b"].tolist() if int(e[0]) != -1]
    p2b = [(int(e[0]), int(e[1]), float(e[2]))
           for e in c["p2b"].tolist() if int(e[0]) != -1]
    pins = [(float(p[0]), float(p[1])) for p in c["pins"].tolist()]
    return dict(n=n, cons=cons, otp=otp, area=area, b2b=b2b, p2b=p2b, pins=pins)


def _is_pre(v, i):
    return v["cons"][i][1] != 0


def _is_fixedshape(v, i):
    return v["cons"][i][0] != 0 or v["cons"][i][1] != 0


# ── stage B: shapes (label-free) ────────────────────────────────────────────
def choose_dims(v, i):
    """Fixed/preplaced keep their given (w,h). Soft blocks get the shipped
    boundary-aspect rule, which is the project's highest-ROI insight and is
    label-free: LEFT/RIGHT-only 2.50, TOP/BOTTOM-only 0.40, else square."""
    if _is_fixedshape(v, i):
        w, h = v["otp"][i][2], v["otp"][i][3]
        if w > 0 and h > 0:
            return w, h
    A = v["area"][i]
    if A <= 0:
        return 0.0, 0.0
    code = v["cons"][i][4]
    lr = bool(code & (B_LEFT | B_RIGHT))
    tb = bool(code & (B_TOP | B_BOTTOM))
    r = 2.50 if (lr and not tb) else (0.40 if (tb and not lr) else 1.0)
    w = math.sqrt(A * r)
    return w, A / w


# -- stage B2: MIB shape unification ----------------------------------------
def unify_mib(v, dims):
    """Every member of a MIB group must carry an IDENTICAL (w,h).

    The official count is `distinct shapes - 1` per group
    (iccad2026_evaluate.py:508-517), and identical shape implies identical AREA
    while the +-1% area band is a HARD constraint -- so a group can only reach
    zero when its area targets span at most 1.01/0.99 = 1.0202x. Wider groups get
    BUCKETED into the fewest shape classes, which is L124's mechanism and is
    provably the locked-aware minimum.

    Construction-time and free at runtime, which is the bar L122 set for anything
    on this axis.
    """
    SPAN = 1.01 / 0.99
    groups = {}
    for i in range(v["n"]):
        m = v["cons"][i][2]
        if m:
            groups.setdefault(m, []).append(i)

    def assign(bucket):
        if not bucket:
            return
        # the common area must sit in the intersection of every member's band
        lo = max(v["area"][i] * 0.99 for i in bucket)
        hi = min(v["area"][i] * 1.01 for i in bucket)
        if lo > hi:
            return                                   # cannot unify; leave as is
        A = math.sqrt(lo * hi)
        side = math.sqrt(A)
        for i in bucket:
            dims[i] = (side, side)

    for _g, mem in groups.items():
        if len(mem) <= 1:
            continue
        pinned = [i for i in mem if _is_fixedshape(v, i)]
        free = [i for i in mem if not _is_fixedshape(v, i)]
        if pinned:
            # a fixed member cannot move, so it dictates the shape; everyone
            # whose area band admits that exact rectangle joins it, the rest
            # keep what they had (and are counted, correctly, as violations)
            w, h = dims[pinned[0]]
            A = w * h
            for i in free:
                if abs(A - v["area"][i]) <= 0.01 * v["area"][i]:
                    dims[i] = (w, h)
            continue
        order = sorted(free, key=lambda i: v["area"][i])
        cur = []
        for i in order:
            if cur and v["area"][i] > v["area"][cur[0]] * SPAN:
                assign(cur)
                cur = []
            cur.append(i)
        assign(cur)
    return dims


# ── units: a cluster is one rigid unit ──────────────────────────────────────
def build_units(v, DIMS=None):
    """Returns units, each a dict with member ids and their intra-unit offsets.

    Cluster members are shelf-packed abutting, so the unit is one touching
    component and contributes ZERO grouping fragments by construction -- the
    thing the greedy has to search for, this gets for free.
    """
    n = v["n"]
    by_cl = {}
    for i in range(n):
        cl = v["cons"][i][3]
        by_cl.setdefault(cl if cl else ("solo", i), []).append(i)
    units = []
    for key, mem in by_cl.items():
        dims = {i: DIMS[i] for i in mem}
        if len(mem) == 1:
            i = mem[0]
            w, h = dims[i]
            units.append(dict(mem=[i], off=[(0.0, 0.0)], w=w, h=h,
                              pre=_is_pre(v, i)))
            continue
        tot = sum(dims[i][0] * dims[i][1] for i in mem)
        target = math.sqrt(max(tot, EPS)) if not BND_SHELF else             math.sqrt(max(tot, EPS))
        code = {i: v["cons"][i][4] for i in mem}

        if BND_SHELF:
            # Boundary-aware shelf packing. tag_boundary only claims a side when
            # the requiring member actually sits at that extreme of the unit, and
            # the plain height-sorted shelf leaves 18% of side requirements
            # unclaimed (measured: L 85.7 / R 81.2 / T 73.7 / B 84.9 % claimed).
            # Placing them where they can reach is what closes that.
            #   BOTTOM -> first row, which starts at y=0
            #   TOP    -> last row, TOP-aligned inside it (rows are as tall as
            #             their tallest member, so a shorter one has to be
            #             lifted or it never reaches the unit's top edge)
            #   LEFT   -> first in its row
            #   RIGHT  -> last in its row, and that row right-aligned to the
            #             unit width
            wantB = [i for i in mem if code[i] & B_BOTTOM]
            wantT = [i for i in mem if code[i] & B_TOP and i not in wantB]
            rest = [i for i in mem if i not in wantB and i not in wantT]
            rest.sort(key=lambda i: -dims[i][1])

            def mkrow(seed, pool):
                """A row: LEFT-wanters first, RIGHT-wanters last."""
                row = list(seed)
                wl = sum(dims[i][0] for i in row)
                while pool and wl + dims[pool[0]][0] <= target:
                    row.append(pool.pop(0))
                    wl += dims[row[-1]][0]
                if not row and pool:
                    row.append(pool.pop(0))
                row.sort(key=lambda i: (0 if code[i] & B_LEFT else
                                        (2 if code[i] & B_RIGHT else 1)))
                return row

            rows = []
            pool = list(rest)
            if wantB:
                rows.append(mkrow(wantB, pool))
            while pool:
                rows.append(mkrow([], pool))
            if wantT:
                rows.append(mkrow(wantT, []))

            widths = [sum(dims[i][0] for i in r) for r in rows]
            uw = max(widths) if widths else 0.0
            off, y = {}, 0.0
            rowx, rowh, lift = [], [], set()
            for r, roww in zip(rows, widths):
                rh = max(dims[i][1] for i in r)
                # right-align the row when it carries a RIGHT-wanter, so that
                # member's right edge coincides with the unit's
                x = (uw - roww) if any(code[i] & B_RIGHT for i in r) else 0.0
                rowx.append(x)
                rowh.append(rh)
                last = (r is rows[-1])
                for i in r:
                    if last and code[i] & B_TOP:
                        lift.add(i)
                    yy = (y + rh - dims[i][1]) if (last and code[i] & B_TOP) else y
                    off[i] = (x, yy)
                    x += dims[i][0]
                y += rh
            order = [i for r in rows for i in r]
        else:
            order = sorted(mem, key=lambda i: -dims[i][1])
            rows, cur, curw = [], [], 0.0
            for i in order:
                w, _h = dims[i]
                if cur and curw + w > target:
                    rows.append(cur)
                    cur, curw = [], 0.0
                cur.append(i)
                curw += w
            if cur:
                rows.append(cur)
            off, y, uw = {}, 0.0, 0.0
            rowx, rowh, lift = [], [], set()
            for row in rows:
                x = 0.0
                rowx.append(0.0)
                rowh.append(max(dims[i][1] for i in row))
                for i in row:
                    off[i] = (x, y)
                    x += dims[i][0]
                uw = max(uw, x)
                y += max(dims[i][1] for i in row)
        units.append(dict(mem=order, off=[off[i] for i in order], w=uw, h=y,
                          pre=any(_is_pre(v, i) for i in mem), dims=dims,
                          rows=rows, rowx=rowx, rowh=rowh, lift=lift))
    for u in units:
        u.setdefault("dims", {u["mem"][0]: (u["w"], u["h"])})
        if "rows" not in u:
            i = u["mem"][0]
            u["rows"], u["rowx"], u["rowh"], u["lift"] = [[i]], [0.0], [u["h"]], set()
    tag_boundary(v, units)
    return units


def tag_boundary(v, units):
    """Per-unit needL/needR/needT/needB.

    A boundary block must touch the BBOX edge, so the requirement is really a
    statement about the unit's constraint-graph position:

        needL  <=>  the unit must have NO x-predecessor  (it lands at xmin)
        needR  <=>  the unit must have NO x-successor    (it can then be pushed
                    to xmax, which is safe precisely because nothing is to its
                    right on x, and anything not x-ordered with it is y-separated
                    and cannot overlap whatever x it takes)
        needB / needT are the same statement on y.

    A flag is only claimed when the requiring member actually sits at that
    extreme of its unit; a clustered block buried in the middle of the shelf
    packing cannot reach the bbox edge and the violation is accepted rather than
    silently mis-encoded.
    """
    for u in units:
        need = {"L": False, "R": False, "T": False, "B": False}
        for t, i in enumerate(u["mem"]):
            code = v["cons"][i][4]
            if not code:
                continue
            ox, oy = u["off"][t]
            w, h = u["dims"][i]
            if code & B_LEFT and ox <= EPS:
                need["L"] = True
            if code & B_RIGHT and ox + w >= u["w"] - EPS:
                need["R"] = True
            if code & B_BOTTOM and oy <= EPS:
                need["B"] = True
            if code & B_TOP and oy + h >= u["h"] - EPS:
                need["T"] = True
        u["need"] = need


# ── stage A: quadratic global placement ─────────────────────────────────────
def unit_map(units):
    uof = {}
    for k, u in enumerate(units):
        for i in u["mem"]:
            uof[i] = k
    return uof


def fixed_centres(v, units):
    """Which unit CENTRES are pinned by a preplaced member, and where."""
    U = len(units)
    fixed = np.zeros(U, dtype=bool)
    fx = np.zeros(U)
    fy = np.zeros(U)
    for k, u in enumerate(units):
        for i in u["mem"]:
            if _is_pre(v, i):
                ox, oy = u["off"][u["mem"].index(i)]
                fixed[k] = True
                fx[k] = v["otp"][i][0] - ox + u["w"] / 2.0
                fy[k] = v["otp"][i][1] - oy + u["h"] / 2.0
                break
    return fixed, fx, fy


def assemble(v, units, uof, wx=None, wy=None):
    """Assemble the weighted Laplacian for one IRLS sweep.

    The quadratic objective sum w*(du-dv)^2 is NOT HPWL, which is
    sum w*|du-dv|. Reweighting each edge by 1/|du-dv| from the previous
    solve turns the sequence of quadratic solves into a minimiser of the L1
    objective (the standard linearisation), and the ORDER it produces is the
    only thing stage A ever contributes -- the bisection keeps the order and
    throws the coordinates away, and the LP then solves exact HPWL inside
    whatever topology that order implies.

    Shared with the GORDIAN alternation, which solves this same system under
    region centre-of-gravity constraints.
    """
    U = len(units)
    wx = wx or {}
    wy = wy or {}
    L = np.zeros((U, U))
    bx = np.zeros(U)
    by = np.zeros(U)
    for i, j, w in v["b2b"]:
        if w <= 0 or i >= v["n"] or j >= v["n"]:
            continue
        a, b = uof.get(i), uof.get(j)
        if a is None or b is None or a == b:
            continue
        wa = w * wx.get((a, b), 1.0)
        L[a, a] += wa
        L[b, b] += wa
        L[a, b] -= wa
        L[b, a] -= wa
    for p, j, w in v["p2b"]:
        a = uof.get(j)
        if w <= 0 or j >= v["n"] or p >= len(v["pins"]) or a is None:
            continue
        wp = w * wy.get((p, a), 1.0)
        L[a, a] += wp
        bx[a] += wp * v["pins"][p][0]
        by[a] += wp * v["pins"][p][1]
    return L, bx, by


def global_place(v, units):
    """Minimise sum w*(du-dv)^2 over unit CENTRES, pins and preplaced pinned.

    Quadratic rather than the true linear HPWL because it is one dense solve at
    n<=120 and its optimum is the right neighbourhood; the exact-linear step
    already exists downstream (the shipped LP).
    """
    U = len(units)
    uof = unit_map(units)
    fixed, fx, fy = fixed_centres(v, units)

    def build(wx, wy):
        return assemble(v, units, uof, wx, wy)
    # anything unconnected would leave a singular row; pull it weakly to the
    # centroid of the pins (or the origin if there are none)
    px = np.mean([p[0] for p in v["pins"]]) if v["pins"] else 0.0
    py = np.mean([p[1] for p in v["pins"]]) if v["pins"] else 0.0
    free = ~fixed
    if not free.any():
        return fx, fy, uof
    def solve(L, rhs, fv):
        r = rhs[free] - L[np.ix_(free, fixed)] @ fv[fixed]
        try:
            sol = np.linalg.solve(L[np.ix_(free, free)], r)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(L[np.ix_(free, free)], r, rcond=None)[0]
        out = fv.copy()
        out[free] = sol
        return out
    ex, ey = {}, {}
    X = Y = None
    for sweep in range(max(1, IRLS)):
        L, bx, by = build(ex, ey)
        for k in range(U):
            L[k, k] += 1e-6
            bx[k] += 1e-6 * px
            by[k] += 1e-6 * py
        if not free.any():
            return fx, fy, uof
        X, Y = solve(L, bx, fx), solve(L, by, fy)
        if sweep + 1 >= IRLS:
            break
        # reweight by 1/|delta| -> the quadratic sequence converges on HPWL
        ex, ey = {}, {}
        for i, j, w in v["b2b"]:
            a, b = uof.get(i), uof.get(j)
            if a is None or b is None or a == b:
                continue
            d = abs(X[a] - X[b]) + abs(Y[a] - Y[b])
            ex[(a, b)] = 1.0 / max(d, 1e-6)
        for pi, j, w in v["p2b"]:
            a = uof.get(j)
            if a is None or pi >= len(v["pins"]):
                continue
            d = abs(X[a] - v["pins"][pi][0]) + abs(Y[a] - v["pins"][pi][1])
            ey[(pi, a)] = 1.0 / max(d, 1e-6)
    return X, Y, uof


# ── stage A2: spreading (this is the half that makes it a PLACER) ──────────
def spread(v, units, cx, cy):
    """Area-balanced recursive bisection over the quadratic positions.

    Wirelength alone is degenerate -- every unit piles onto the same point, and
    the relation selection downstream then reads off numerical noise. (First run
    of this file did exactly that: area_gap ~1.15, i.e. the layout was more than
    twice the area it needed.) Bisection keeps the ORDER the quadratic solve
    found and gives each unit room proportional to its area, which is what makes
    the constraint graph meaningful.

    Preplaced units keep their true centre; they only steer the recursion.
    """
    U = len(units)
    A = np.array([max(units[k]["w"] * units[k]["h"], EPS) for k in range(U)])
    pre = np.array([u["pre"] for u in units])
    tot = float(A.sum())
    side = math.sqrt(tot / max(DENSITY, 0.05))
    ox = float(min(cx)) if U else 0.0
    oy = float(min(cy)) if U else 0.0
    out_x, out_y = cx.copy(), cy.copy()

    def rec(idx, x0, y0, w, h, horiz):
        if len(idx) == 0:
            return
        if len(idx) == 1:
            k = idx[0]
            if not pre[k]:
                out_x[k], out_y[k] = x0 + w / 2.0, y0 + h / 2.0
            return
        key = cx if horiz else cy
        idx = sorted(idx, key=lambda k: (key[k], k))
        half, run = float(A[idx].sum()) / 2.0, 0.0
        cut = 1
        for t, k in enumerate(idx):
            run += A[k]
            if run >= half:
                cut = max(1, min(len(idx) - 1, t + 1))
                break
        left, right = idx[:cut], idx[cut:]
        fr = float(A[left].sum()) / max(float(A[idx].sum()), EPS)
        if horiz:
            rec(left, x0, y0, w * fr, h, not horiz)
            rec(right, x0 + w * fr, y0, w * (1 - fr), h, not horiz)
        else:
            rec(left, x0, y0, w, h * fr, not horiz)
            rec(right, x0, y0 + h * fr, w, h * (1 - fr), not horiz)

    rec(list(range(U)), ox, oy, side, side, True)
    return out_x, out_y


# ── stage A2': the GORDIAN alternation (L130) ───────────────────────────────
def _region_box(v, units, cx, cy):
    """Root rectangle for the recursion.

    Same convention as `spread` (origin at the min solved centre, side sized for
    DENSITY) so the two stage-A forms are comparable, plus one guard `spread`
    does not need: preplaced unit centres are HARD and the alternation's
    constraints are stated relative to this box, so the box must at least span
    them or every level asks the free units to average out an impossibility.
    """
    U = len(units)
    A = np.array([max(units[k]["w"] * units[k]["h"], EPS) for k in range(U)])
    side = math.sqrt(float(A.sum()) / max(DENSITY, 0.05))
    ox = float(min(cx)) if U else 0.0
    oy = float(min(cy)) if U else 0.0
    pre = np.array([u["pre"] for u in units])
    if pre.any():
        ox = min(ox, float(cx[pre].min()))
        oy = min(oy, float(cy[pre].min()))
        side = max(side, float(cx[pre].max()) - ox, float(cy[pre].max()) - oy)
    return A, ox, oy, side


def _bnd_rank(need, horiz):
    """Which end of a split a boundary-requiring unit must be sorted to.

    🚨 The alternation's objective knows about wire and its constraints know
    about area; NOTHING in it knows that a needL unit has to end up with no
    x-predecessor. Without this the free solve is right to pull such a unit
    inward -- and it then reads as a boundary violation downstream.

    MEASURED, 30 cases, alternation without this: hpwl_gap 0.4265 -> 0.4172 and
    area_gap 0.4297 -> 0.3851, both better, while vbnd went 2.0775 -> 2.8490 and
    the TOTAL got worse (1.7086 -> 1.7767). Third occurrence of the same lesson:
    a stage that improves one term of a multiplicative cost must re-enforce every
    constraint the stages around it rely on.

    Forcing the rank recursively is what makes it work: needL goes into the left
    half at every horizontal split it meets, so it lands in the leftmost column
    of leaves and is x-minimal in the centre order `legalise` reads.
    """
    if horiz:
        lo, hi = need["L"], need["R"]
    else:
        lo, hi = need["B"], need["T"]
    if lo and not hi:
        return 0
    if hi and not lo:
        return 2
    return 1


def _split(region, key, A, minsize, need=None):
    """Area-balanced bisection of one region along its longer side.

    Ordering comes from the CURRENT solve, which is the whole point: at level
    L the cut is made on coordinates that already know about levels 1..L-1's
    constraints, instead of on one unconstrained solve's coordinates the way a
    single destructive `spread` pass does.
    """
    idx, x0, y0, w, h = region
    if len(idx) <= minsize:
        return [region]
    horiz = w >= h
    k = key[0] if horiz else key[1]
    if need is None:
        idx = sorted(idx, key=lambda q: (k[q], q))
    else:
        idx = sorted(idx, key=lambda q: (_bnd_rank(need[q], horiz), k[q], q))
    half, run, cut = float(A[idx].sum()) / 2.0, 0.0, 1
    for t, q in enumerate(idx):
        run += A[q]
        if run >= half:
            cut = max(1, min(len(idx) - 1, t + 1))
            break
    left, right = idx[:cut], idx[cut:]
    fr = float(A[left].sum()) / max(float(A[idx].sum()), EPS)
    if horiz:
        return [(left, x0, y0, w * fr, h),
                (right, x0 + w * fr, y0, w * (1 - fr), h)]
    return [(left, x0, y0, w, h * fr),
            (right, x0, y0 + h * fr, w, h * (1 - fr))]


def _solve_cog(Lap, bx, by, fixed, fx, fy, regions, A):
    """One constrained re-solve: minimise the quadratic subject to, for every
    region, the area-weighted centre of gravity of its FREE units sitting at the
    region's geometric centre.

        min  x'Lx - 2b'x    s.t.  Ax = u
        =>   [[L, A'], [A, 0]] [x; lam] = [b; u]

    The KKT system is one dense (F+R) solve and both axes share the matrix, so
    it is factored once for two right-hand sides.

    Constraining only the FREE units (rather than free+fixed with the fixed
    contribution moved to the rhs) is deliberate: a region whose area is mostly
    preplaced would otherwise demand its handful of movable units average out to
    a point far outside the box, and the next partition would then read that
    excursion as an ordering. A region with no free unit contributes no row.

    This is the step the single `spread` pass does not have. Spreading by
    bisection assigns coordinates and throws the objective away; here the
    objective is still being minimised WHILE the units are pushed apart, so
    wirelength keeps a say at every level instead of only at level 0.
    """
    U = Lap.shape[0]
    free = ~fixed
    F = int(free.sum())
    if F == 0:
        return fx.copy(), fy.copy()
    fidx = np.flatnonzero(free)
    pos = {int(k): t for t, k in enumerate(fidx)}
    Lf = Lap[np.ix_(free, free)]
    rx = bx[free] - Lap[np.ix_(free, fixed)] @ fx[fixed]
    ry = by[free] - Lap[np.ix_(free, fixed)] @ fy[fixed]

    rows, ux, uy = [], [], []
    for idx, x0, y0, w, h in regions:
        mem = [q for q in idx if free[q]]
        if not mem:
            continue
        tot = float(sum(A[q] for q in mem))
        if tot <= EPS:
            continue
        r = np.zeros(F)
        for q in mem:
            r[pos[q]] = A[q] / tot
        rows.append(r)
        ux.append(x0 + w / 2.0)
        uy.append(y0 + h / 2.0)
    if not rows:
        try:
            sx = np.linalg.solve(Lf, rx)
            sy = np.linalg.solve(Lf, ry)
        except np.linalg.LinAlgError:
            sx = np.linalg.lstsq(Lf, rx, rcond=None)[0]
            sy = np.linalg.lstsq(Lf, ry, rcond=None)[0]
        out_x, out_y = fx.copy(), fy.copy()
        out_x[free], out_y[free] = sx, sy
        return out_x, out_y

    Am = np.array(rows)
    R = Am.shape[0]
    K = np.zeros((F + R, F + R))
    K[:F, :F] = Lf
    K[:F, F:] = Am.T
    K[F:, :F] = Am
    # regions are disjoint so the rows are independent by construction; the tiny
    # negative block is only insurance against a degenerate all-zero row
    K[F:, F:] = -1e-12 * np.eye(R)
    rhs = np.zeros((F + R, 2))
    rhs[:F, 0] = rx
    rhs[:F, 1] = ry
    rhs[F:, 0] = ux
    rhs[F:, 1] = uy
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    out_x, out_y = fx.copy(), fy.copy()
    out_x[free] = sol[:F, 0]
    out_y[free] = sol[:F, 1]
    return out_x, out_y


def _snap_leaves(regions, cx, cy, fixed, A, need, minsize):
    """Place every free unit at its LEAF region centre, discarding the solve's
    coordinates but keeping the order they produced.

    🚨 A centre-of-gravity row is an AVERAGE. A unit is free to sit far outside
    its region as long as its region-mates compensate, and the solve will happily
    use that freedom to shorten wire. `legalise` then reads those coordinates
    twice -- once to pick each pair's cheaper axis from the overlap of their real
    (w,h), and once to orient it -- and the first read is meaningless when the
    centres are excursions rather than a spread.

    MEASURED, full 100, unsnapped: for n<80 the alternation wins everywhere
    (-0.12 to -0.23 weighted cost, 32/41 cases) while for n>=80 -- which carries
    96.3% of the weight -- hpwl_gap still improves 0.480 -> 0.429 but area_gap
    blows out 0.439 -> 0.522. Excursions accumulate with unit count, so the
    damage is exactly where the weight is.

    Snapping is not a retreat to `spread`: the leaf a unit lands in was decided
    by log2(U) constrained solves that each knew about wirelength, which is the
    whole point of the alternation. What is thrown away is only the within-region
    offset, and stage A never contributed that anyway -- the bisection has always
    kept the order and discarded the coordinates.
    """
    out_x, out_y = cx.copy(), cy.copy()
    stack = list(regions)
    guard = 0
    while stack and guard < 4096:
        guard += 1
        reg = stack.pop()
        idx, x0, y0, w, h = reg
        free_mem = [q for q in idx if not fixed[q]]
        if not free_mem:
            continue
        if len(idx) > 1:
            # only reachable when the alternation hit its level cap; finish the
            # recursion geometrically, the way `spread` would
            parts = _split(reg, (cx, cy), A, 1, need)
            if len(parts) > 1:
                stack.extend(parts)
                continue
        for q in free_mem:
            out_x[q] = x0 + w / 2.0
            out_y[q] = y0 + h / 2.0
    return out_x, out_y


def gordian(v, units):
    """Stage A, alternating: solve -> partition -> re-solve under the new
    regions -> partition again, until every region holds one unit.

    L129 v6's stage A does all the spreading in ONE destructive step: an
    unconstrained solve piles every connected unit onto a point, and a single
    recursive bisection then reads an ORDER off that pile and discards the
    objective. Everything after it (relation selection, compaction, the LP) is
    confined to the topology that one order implies, which is why L128's two
    cheap hpwl routes were both null -- they were operating downstream of the
    only decision that matters.

    The alternation keeps the objective alive during the spreading. Each level
    adds one centre-of-gravity equality per region, which is weak enough that
    units still move to shorten wire, and strong enough that they cannot pile
    up again; the next partition is then made on coordinates that already
    respect every earlier level.

    Returns the same (cx, cy, uof) triple as global_place + spread.
    """
    U = len(units)
    uof = unit_map(units)
    fixed, fx, fy = fixed_centres(v, units)
    Lap, bx, by = assemble(v, units, uof)
    px = float(np.mean([p[0] for p in v["pins"]])) if v["pins"] else 0.0
    py = float(np.mean([p[1] for p in v["pins"]])) if v["pins"] else 0.0
    for k in range(U):
        Lap[k, k] += 1e-6
        bx[k] += 1e-6 * px
        by[k] += 1e-6 * py
    if not (~fixed).any():
        return fx, fy, uof

    # level 0 -- the unconstrained solve, exactly what global_place computes
    cx, cy = _solve_cog(Lap, bx, by, fixed, fx, fy, [], np.ones(U))
    A, ox, oy, side = _region_box(v, units, cx, cy)

    need = [u["need"] for u in units] if GORDIAN_BND else None
    regions = [(list(range(U)), ox, oy, side, side)]
    for _lvl in range(max(1, GORDIAN_MAXLVL)):
        nxt = []
        for reg in regions:
            nxt.extend(_split(reg, (cx, cy), A, max(1, GORDIAN_MIN), need))
        if len(nxt) == len(regions):
            break                       # every region terminal; nothing to add
        regions = nxt
        cx, cy = _solve_cog(Lap, bx, by, fixed, fx, fy, regions, A)
    if GORDIAN_SNAP:
        cx, cy = _snap_leaves(regions, cx, cy, fixed, A, need,
                              max(1, GORDIAN_MIN))
    return cx, cy, uof


# ── stage C: legalise -- relation selection + longest-path compaction ───────
def legalise(v, units, cx, cy, rounds=40, axis0=None):
    """Overlap-free by CONSTRUCTION.

    For each unit pair pick the axis whose overlap is cheaper to remove and
    orient it by the centre order, so each constraint graph is a sub-order of a
    centre ordering and is acyclic. Longest path then always yields a legal
    layout. Preplaced units are pinned; a pin that a longest path overruns is
    repaired by flipping that pair to the other axis, and a case that will not
    converge simply emits no candidate.
    """
    U = len(units)
    W = np.array([u["w"] for u in units])
    H = np.array([u["h"] for u in units])
    pre = np.array([u["pre"] for u in units])
    # preplaced units must land exactly where they are; their centre is fixed
    px, py = cx.copy(), cy.copy()

    need = [u["need"] for u in units]
    axis = dict(axis0) if axis0 is not None else {}
    for a in range(U):
        if axis0 is not None:
            break
        for b in range(a + 1, U):
            ox = (W[a] + W[b]) / 2.0 - abs(cx[b] - cx[a])
            oy = (H[a] + H[b]) / 2.0 - abs(cy[b] - cy[a])
            cheap = 0 if ox <= oy else 1
            # Which unit would become the predecessor on each axis is fixed by
            # the centre order -- the same order `compact` sorts on -- so the
            # boundary test can be made here. Only the AXIS is ever changed,
            # never the orientation, so the graphs stay acyclic.
            lo_x, hi_x = (a, b) if (cx[a], a) < (cx[b], b) else (b, a)
            lo_y, hi_y = (a, b) if (cy[a], a) < (cy[b], b) else (b, a)
            okx = not (need[lo_x]["R"] or need[hi_x]["L"])
            oky = not (need[lo_y]["T"] or need[hi_y]["B"])
            if okx and oky:
                axis[(a, b)] = cheap
            elif okx:
                axis[(a, b)] = 0
            elif oky:
                axis[(a, b)] = 1
            else:
                axis[(a, b)] = cheap        # unsatisfiable pair; take the cheap one

    def compact(ax, C, size):
        """Longest path along `ax`, honouring pinned units. Returns lows."""
        order = sorted(range(U), key=lambda k: (C[k], k))
        rank = {k: r for r, k in enumerate(order)}
        pred = [[] for _ in range(U)]
        for (a, b), s in axis.items():
            if s != ax:
                continue
            lo, hi = (a, b) if rank[a] < rank[b] else (b, a)
            pred[hi].append(lo)
        # The layout does NOT have to start at 0. A cluster carrying a preplaced
        # member is positioned by that member's offset inside the unit, so the
        # unit's own origin can legitimately be negative; anchoring free units at
        # 0 would then make the pin unreachable and read as "infeasible".
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
                        return None, None          # nothing to flip -> give up
                    # Flip only the single binding predecessor, not all of them:
                    # flipping the whole set thrashes between the two axes and
                    # burns the repair budget without converging.
                    # Returned worst-first so the caller can skip a pair it has
                    # already flipped once (LEGAL_LOCK) rather than oscillate.
                    return None, (k, sorted(pred[k],
                                            key=lambda p: -(low[p] + size[p])))
                low[k] = want
            else:
                low[k] = base
        return low, None

    # 🚨 A pair flipped x->y and then straight back to x oscillates forever, and
    # the loop just burns `rounds` and reports "did not converge".
    #
    # MEASURED (l132_coverage_probe.py): ALL 36 first-pass failures on the in-set
    # 100 are this, and NONE are the other exit (a pinned unit with no
    # predecessor to flip). Several fail having touched only ONE distinct pair in
    # 40 rounds -- the same pair, forever. Raising rounds 40 -> 400 changes the
    # outcome on exactly zero cases.
    #
    # LEGAL_LOCK flips each pair at most once and otherwise takes the next
    # binding predecessor, so the repair either makes progress or admits it is
    # out of moves.
    locked = set()

    def pick(k, cand):
        for p in cand:
            if not LEGAL_LOCK or (min(k, p), max(k, p)) not in locked:
                return p
        return None

    for _ in range(rounds):
        lx, bad = compact(0, px, W)
        if lx is None:
            if bad is None:
                return None, None, axis
            k, cand = bad
            p = pick(k, cand)
            if p is None:
                return None, None, axis
            axis[(min(k, p), max(k, p))] = 1        # flip the offender to y
            locked.add((min(k, p), max(k, p)))
            continue
        ly, bad = compact(1, py, H)
        if ly is None:
            if bad is None:
                return None, None, axis
            k, cand = bad
            p = pick(k, cand)
            if p is None:
                return None, None, axis
            axis[(min(k, p), max(k, p))] = 0
            locked.add((min(k, p), max(k, p)))
            continue
        _push_to_max(axis, px, lx, W, 0, need, pre)
        _push_to_max(axis, py, ly, H, 1, need, pre)
        return lx, ly, axis
    return None, None, axis


def _push_to_max(axis, C, low, size, ax, need, pre):
    """Slide every needR/needT unit that has no successor on `ax` out to the max.

    Longest-path compaction packs everything toward the minimum, so a unit with
    no successor is merely somewhere, not at the far edge. Pushing it out is
    provably overlap-free: anything ordered with it on `ax` is by construction on
    its low side, and anything NOT ordered with it on `ax` is ordered on the
    other axis and so cannot overlap at any coordinate it takes here.
    """
    U = len(low)
    key = "R" if ax == 0 else "T"
    rank = {k: r for r, k in enumerate(sorted(range(U), key=lambda k: (C[k], k)))}
    succ = [0] * U
    for (a, b), s in axis.items():
        if s != ax:
            continue
        succ[a if rank[a] < rank[b] else b] += 1
    hi = max(low[k] + size[k] for k in range(U))
    for k in range(U):
        if need[k][key] and not succ[k] and not pre[k]:
            low[k] = hi - size[k]


def _hpwl(v, units, lx, ly):
    """True HPWL of a legal layout, straight from block centres.

    Cheap enough to sit inside the flip search (no LP needed), which is what lets
    the local search optimise the metric the cost actually charges rather than
    bbox area alone.
    """
    cx = {}
    cy = {}
    for k, u in enumerate(units):
        for t, i in enumerate(u["mem"]):
            ox, oy = u["off"][t]
            w, h = u["dims"][i]
            cx[i] = lx[k] + ox + w / 2.0
            cy[i] = ly[k] + oy + h / 2.0
    tot = 0.0
    for i, j, w in v["b2b"]:
        if w <= 0 or i not in cx or j not in cx:
            continue
        tot += w * (abs(cx[i] - cx[j]) + abs(cy[i] - cy[j]))
    for pi, j, w in v["p2b"]:
        if w <= 0 or j not in cx or pi >= len(v["pins"]):
            continue
        tot += w * (abs(cx[j] - v["pins"][pi][0]) + abs(cy[j] - v["pins"][pi][1]))
    return tot


def _bbox(units, lx, ly):
    U = len(units)
    x0 = min(lx); y0 = min(ly)
    x1 = max(lx[k] + units[k]["w"] for k in range(U))
    y1 = max(ly[k] + units[k]["h"] for k in range(U))
    return (x1 - x0) * (y1 - y0), x1 - x0, y1 - y0


def _critical(units, axis, C, low, size, ax):
    """The chain that sets the extent on `ax`, as the list of pairs along it.

    The bbox on an axis IS the longest chain in that constraint graph, so the
    only way to shrink it is to move one of the chain's relations to the other
    axis. Anything not on the chain is slack and flipping it cannot help.
    """
    U = len(low)
    rank = {k: r for r, k in enumerate(sorted(range(U), key=lambda k: (C[k], k)))}
    pred = [[] for _ in range(U)]
    for (a, b), s in axis.items():
        if s != ax:
            continue
        lo, hi = (a, b) if rank[a] < rank[b] else (b, a)
        pred[hi].append(lo)
    k = max(range(U), key=lambda k: low[k] + size[k])
    chain = []
    seen = set()
    while k not in seen:
        seen.add(k)
        best = None
        for q in pred[k]:
            if abs(low[q] + size[q] - low[k]) <= 1e-6:
                if best is None or low[q] < low[best]:
                    best = q
        if best is None:
            break
        chain.append((min(k, best), max(k, best)))
        k = best
    return chain


def _axis_allowed(units, cx, cy, a, b, t):
    """Same boundary test legalise makes at selection time.

    🚨 refine_area must re-apply it. The first version did not, and flipping a
    pair onto an axis that a needL/needR/needB/needT unit forbids silently
    reintroduced boundary violations: area_gap improved 0.955 -> 0.892 while vrel
    went 0.1481 -> 0.1636 and the TOTAL got worse. Optimising one term of the
    cost while quietly breaking another is the failure mode to watch for here.
    """
    need = [u["need"] for u in units]
    if t == 0:
        lo, hi = (a, b) if (cx[a], a) < (cx[b], b) else (b, a)
        return not (need[lo]["R"] or need[hi]["L"])
    lo, hi = (a, b) if (cy[a], a) < (cy[b], b) else (b, a)
    return not (need[lo]["T"] or need[hi]["B"])


def refine_area(v, units, cx, cy, axis, lx, ly, budget=60):
    """Flip relations off the critical path while the bbox shrinks.

    Only the AXIS of a pair is ever changed, never its orientation, so both
    graphs stay sub-orders of the centre orderings and stay acyclic -- the same
    invariant the boundary handling relies on.
    """
    best_a, bw, bh = _bbox(units, lx, ly)
    a0 = max(best_a, EPS)
    h0 = max(_hpwl(v, units, lx, ly), EPS)

    def score(nx, ny):
        """The cost charges 0.5*area_gap + 0.5*hpwl_gap, so the search weighs
        them the same way.

        MEASURED: this changed the 30-case result by exactly NOTHING (1.708551
        to every digit, hpwl_gap 0.4265, area_gap 0.4297). Kept because it is the
        correct objective and costs nothing, but the null result is the finding:
        this search runs BEFORE lp_polish, and the LP then re-solves HPWL exactly
        inside whatever topology comes out, so any HPWL the search buys here is
        simply re-derived. Pre-LP HPWL work is not a lever."""
        a, _w, _h = _bbox(units, nx, ny)
        return 0.5 * a / a0 + 0.5 * _hpwl(v, units, nx, ny) / h0

    best_s = score(lx, ly)
    W = [u["w"] for u in units]
    H = [u["h"] for u in units]
    used = 0
    while used < budget:
        ax = 0 if bw >= bh else 1
        chain = _critical(units, axis, cx if ax == 0 else cy,
                          lx if ax == 0 else ly, W if ax == 0 else H, ax)
        if not chain:
            break
        improved = False
        for pair in chain[:12]:
            if used >= budget:
                break
            if axis.get(pair) != ax:
                continue
            if not _axis_allowed(units, cx, cy, pair[0], pair[1], 1 - ax):
                continue                      # would break a boundary requirement
            trial = dict(axis)
            trial[pair] = 1 - ax
            used += 1
            nx, ny, naxis = legalise(v, units, cx, cy, axis0=trial)
            if nx is None:
                continue
            sc = score(nx, ny)
            if sc < best_s - 1e-12:
                a, w2, h2 = _bbox(units, nx, ny)
                best_s, best_a, bw, bh = sc, a, w2, h2
                lx, ly, axis = nx, ny, naxis
                improved = True
                break
        if not improved:
            break
    return lx, ly, axis


def place(c):
    v = case_view(c)
    DIMS = {i: choose_dims(v, i) for i in range(v['n'])}
    if MIB_UNIFY:
        unify_mib(v, DIMS)
    units = build_units(v, DIMS)
    if GORDIAN:
        cx, cy, uof = gordian(v, units)
    else:
        cx, cy, uof = global_place(v, units)
        cx, cy = spread(v, units, cx, cy)
    # One compaction is not enough: the relation set is all n^2/2 pairs, so the
    # bbox is whatever the chain structure happens to be, and the choice made
    # from the SPREAD centres is a poor one (area_gap ~1.21 regardless of how
    # much room the spreading is given -- swept 1.0 down to 0.4, flat). Feeding
    # the compacted centres back in re-derives the relations from a layout that
    # is already legal, which is the direction the chains actually shorten in.
    lx = ly = None
    best = None
    W = [u["w"] for u in units]
    H = [u["h"] for u in units]
    for _ in range(6):
        nx, ny, ax0 = legalise(v, units, cx, cy)
        if nx is not None and REFINE_AREA:
            nx, ny, ax0 = refine_area(v, units, cx, cy, ax0, nx, ny)
        if nx is None:
            break
        area = ((max(nx[k] + W[k] for k in range(len(units)))
                 - min(nx)) * (max(ny[k] + H[k] for k in range(len(units)))
                               - min(ny)))
        if best is None or area < best[0] - 1e-9:
            best, lx, ly = (area, nx, ny), nx, ny
        else:
            break
        cx = np.array([nx[k] + W[k] / 2.0 for k in range(len(units))])
        cy = np.array([ny[k] + H[k] / 2.0 for k in range(len(units))])
    if lx is None:
        return None
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(v["n"])]
    if EXACT_ABUT:
        _emit_abut(units, lx, ly, out)
        return out
    for k, u in enumerate(units):
        for t, i in enumerate(u["mem"]):
            ox, oy = u["off"][t]
            w, h = u["dims"][i]
            out[i] = [lx[k] + ox, ly[k] + oy, w, h]
    return out


def _emit_abut(units, lx, ly, out):
    """Emit member coordinates by REPLAYING the shelf accumulation in absolute
    coordinates, so a shared edge is the identical float on both sides.

    🚨 `lx[k] + off_i` does not abut. The evaluator builds a block's far edge as
    `x + w` (iccad2026_evaluate.py, shapely box), and floating-point addition is
    not associative: `(lx + ox) + w` and `lx + (ox + w)` differ by an ULP. The
    packing abuts exactly in exact arithmetic and lands +-2.8e-14 off in doubles.

    MEASURED on case 66: the baseline landed on +2.84e-14 -- a GAP, so
    `unary_union` returned two geometries and the group scored
    `connected_components - 1 = 1`. The alternation landed on -2.84e-14 -- a hair
    of OVERLAP, so the union merged and the group scored 0. Same packing, same
    relative offsets to 1e-6, opposite verdicts.

    So part of `grouping_violations` in this placer has been a coin flip on the
    last bit, and any vgrp delta between two stage-A forms is that noise plus
    whatever is real. Accumulating instead of offsetting removes it: the next
    member's low edge is computed by the same `+ w` the evaluator uses for the
    previous member's high edge, so they are bit-identical and the polygons touch.
    """
    for k, u in enumerate(units):
        y = ly[k]
        for r, row in enumerate(u["rows"]):
            x = lx[k] + u["rowx"][r]
            rh = u["rowh"][r]
            for i in row:
                w, h = u["dims"][i]
                out[i] = [x, (y + rh - h) if i in u["lift"] else y, w, h]
                x = x + w
            y = y + rh


# ── evaluation ──────────────────────────────────────────────────────────────

# -- stage D: exact-HPWL polish, reusing the shipped LP ----------------------
def lp_polish(c, P, iters=6):
    """Hand the legalised layout to the SHIPPED LP.

    Not a new solver: build_and_solve already minimises EXACT linear HPWL under a
    fixed topology (an aux column and two rows per (edge,axis) give a true
    absolute value; only the area constraint is linearised). L128 showed it works
    whenever its input is already legal -- and this placer's output is legal by
    construction, which is precisely the precondition L128 found the label's
    transplanted topology could not meet.

    Objective is the shipped BASELINE-FREE one (own hpwl, structural floor for
    area), so the result stays comparable to the shipped number rather than being
    inflated by an oracle constant.
    """
    key = "l129"
    n = c["n"]
    if LP_MAXN and n > LP_MAXN:
        return P
    sumA = sum(max(0.0, float(c["at"][i])) for i in range(n))
    P0 = [tuple(float(x) for x in q) for q in P]
    try:
        hp = oc._proxy_metrics(P0, c["at"], c["b2b"], c["p2b"], c["pins"],
                               c["cons"], n)["hpwl"]
    except Exception:
        return P
    base = {"hpwl_baseline": max(float(hp), 1e-6),
            "area_baseline": max(sumA / oc._LP_UTIL, 1e-6)}
    _l3.CASES[key] = oc._lp_build_case(n, c["at"], c["b2b"], c["p2b"],
                                       c["pins"], c["cons"], base)
    saved, oc.PRUNE_B = oc.PRUNE_B, None
    Q = P0
    t0 = time.perf_counter()
    prev = 0.0
    try:
        for _ in range(iters):
            if LP_BUDGET > 0:
                # Charge the NEXT pass at the price of the last one. A pass cannot
                # be interrupted once started, so budgeting on elapsed alone would
                # always overshoot by a full pass -- and on the cases that matter
                # a single pass is 4.9s against a 1.0-1.8s wall.
                el = time.perf_counter() - t0
                if el + prev > LP_BUDGET:
                    break
            tp = time.perf_counter()
            newQ, tele, _B = oc.lp_pass(key, Q, 0.06, sep_trim=False)
            prev = time.perf_counter() - tp
            if newQ is None:
                break
            if not oc.hard_ok(P0, newQ, key):
                break
            Q = newQ
    except Exception:
        Q = P0
    finally:
        oc.PRUNE_B = saved
        _l3.CASES.pop(key, None)
        oc._HARD_MASKS.pop(key, None)
    return Q


def official(c, P):
    m = evaluate_solution({"positions": [list(p) for p in P], "runtime": 1.0},
                          c["base"], c["cons"][:c["n"]], c["b2b"], c["p2b"],
                          c["pins"], c["at"][:c["n"]],
                          target_positions=c["tp"][:c["n"]], median_runtime=1.0)
    return m


def _selected(a):
    """--minn/--maxn exist because the weight is exp(n/12): the n>=80 cases carry
    96% of it, so iterating on the first 30 (which are all small) measures the
    3.7% of the score nobody is graded on."""
    sel = CASES[:a.limit] if a.limit else CASES
    if a.minn:
        sel = [c for c in sel if c["n"] >= a.minn]
    if a.maxn:
        sel = [c for c in sel if c["n"] <= a.maxn]
    return sel


def mode_run(a):
    rows, cov, feas = [], 0, 0
    t0 = time.time()
    for c in _selected(a):
        t1 = time.perf_counter()
        P = place(c)
        if P is not None and LP_POLISH:
            P = lp_polish(c, P)
        dt = time.perf_counter() - t1
        if P is None:
            if a.verbose:
                print(f"  case {c['idx']:>3} n={c['n']:>3}  no candidate "
                      f"(legalisation did not converge)")
            continue
        cov += 1
        m = official(c, P)
        if m.is_feasible:
            feas += 1
        if a.verbose:
            print(f"  case {c['idx']:>3} n={c['n']:>3}  cost {m.cost:8.4f}  "
                  f"feas {int(m.is_feasible)}  hgap {m.hpwl_gap:6.3f}  "
                  f"agap {m.area_gap:6.3f}  vrel {m.violations_relative:.4f}  "
                  f"{dt:.3f}s")
        rows.append(dict(test_id=c["idx"], block_count=c["n"],
                         positions=[list(p) for p in P],
                         cost=float(m.cost), is_feasible=bool(m.is_feasible),
                         hpwl_gap=float(m.hpwl_gap), area_gap=float(m.area_gap),
                         violations_relative=float(m.violations_relative),
                         vbnd=float(m.boundary_violations),
                         vgrp=float(m.grouping_violations),
                         vmib=float(m.mib_violations),
                         nsoft=float(m.max_possible_violations),
                         runtime_seconds=dt, error=None))
    tot = len(_selected(a))
    print(f"\n=== L129 global placer ===")
    print(f"  cases            {tot}")
    print(f"  legalised        {cov}/{tot}   ({100 * cov / max(tot, 1):.0f}% coverage)")
    print(f"  feasible         {feas}/{cov if cov else 1}")
    if rows:
        ok = [r for r in rows if r["is_feasible"]]
        if ok:
            wsum = sum(math.exp(r["block_count"] / 12.0) for r in ok)
            wc = sum(math.exp(r["block_count"] / 12.0) * r["cost"] for r in ok) / wsum
            print(f"  weighted solo cost (feasible only)   {wc:.6f}")
            for k in ("hpwl_gap", "area_gap", "violations_relative",
                      "vbnd", "vgrp", "vmib", "nsoft"):
                m = sum(math.exp(r["block_count"] / 12.0) * r[k] for r in ok) / wsum
                print(f"    {k:<22} {m:.4f}")
        print(f"  wall             {time.time() - t0:.1f}s total")
    if a.out:
        json.dump(dict(total_score=0.0, test_results=rows), open(a.out, "w"))
        print(f"  wrote {a.out}  ({len(rows)} cases)")
    return 0




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--minn", type=int, default=0)
    ap.add_argument("--maxn", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    sys.exit(mode_run(ap.parse_args()))
