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

EPS = 1e-9
# Bisection into a square of side sqrt(total_area) asks for 100% density, which
# no rectangle packing reaches; the relation selection then reads "everything
# overlaps everything" and the compaction chains blow the bbox up. Give it the
# room a real packing needs.
DENSITY = float(os.environ.get("L129_DENSITY", "0.80"))
B_LEFT, B_RIGHT, B_BOTTOM, B_TOP = 1, 2, 4, 8      # constructive.cpp's codes

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


# ── units: a cluster is one rigid unit ──────────────────────────────────────
def build_units(v):
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
        dims = {i: choose_dims(v, i) for i in mem}
        if len(mem) == 1:
            i = mem[0]
            w, h = dims[i]
            units.append(dict(mem=[i], off=[(0.0, 0.0)], w=w, h=h,
                              pre=_is_pre(v, i)))
            continue
        tot = sum(dims[i][0] * dims[i][1] for i in mem)
        target = math.sqrt(max(tot, EPS))
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
        for row in rows:
            x = 0.0
            for i in row:
                off[i] = (x, y)
                x += dims[i][0]
            uw = max(uw, x)
            y += max(dims[i][1] for i in row)
        units.append(dict(mem=order, off=[off[i] for i in order], w=uw, h=y,
                          pre=any(_is_pre(v, i) for i in mem), dims=dims))
    for u in units:
        u.setdefault("dims", {u["mem"][0]: (u["w"], u["h"])})
    return units


# ── stage A: quadratic global placement ─────────────────────────────────────
def global_place(v, units):
    """Minimise sum w*(du-dv)^2 over unit CENTRES, pins and preplaced pinned.

    Quadratic rather than the true linear HPWL because it is one dense solve at
    n<=120 and its optimum is the right neighbourhood; the exact-linear step
    already exists downstream (the shipped LP).
    """
    U = len(units)
    uof = {}
    for k, u in enumerate(units):
        for i in u["mem"]:
            uof[i] = k
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
    L = np.zeros((U, U))
    bx = np.zeros(U)
    by = np.zeros(U)
    for i, j, w in v["b2b"]:
        if w <= 0 or i >= v["n"] or j >= v["n"]:
            continue
        a, b = uof.get(i), uof.get(j)
        if a is None or b is None or a == b:
            continue
        L[a, a] += w
        L[b, b] += w
        L[a, b] -= w
        L[b, a] -= w
    for p, j, w in v["p2b"]:
        if w <= 0 or j >= v["n"] or p >= len(v["pins"]):
            continue
        a = uof.get(j)
        if a is None:
            continue
        L[a, a] += w
        bx[a] += w * v["pins"][p][0]
        by[a] += w * v["pins"][p][1]
    # anything unconnected would leave a singular row; pull it weakly to the
    # centroid of the pins (or the origin if there are none)
    px = np.mean([p[0] for p in v["pins"]]) if v["pins"] else 0.0
    py = np.mean([p[1] for p in v["pins"]]) if v["pins"] else 0.0
    for k in range(U):
        L[k, k] += 1e-6
        bx[k] += 1e-6 * px
        by[k] += 1e-6 * py
    free = ~fixed
    if not free.any():
        return fx, fy, uof
    for axis, (M, rhs, f) in enumerate(((L, bx, fx), (L, by, fy))):
        pass
    def solve(rhs, fv):
        r = rhs[free] - L[np.ix_(free, fixed)] @ fv[fixed]
        try:
            sol = np.linalg.solve(L[np.ix_(free, free)], r)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(L[np.ix_(free, free)], r, rcond=None)[0]
        out = fv.copy()
        out[free] = sol
        return out
    return solve(bx, fx), solve(by, fy), uof


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


# ── stage C: legalise -- relation selection + longest-path compaction ───────
def legalise(v, units, cx, cy, rounds=40):
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

    axis = {}
    for a in range(U):
        for b in range(a + 1, U):
            ox = (W[a] + W[b]) / 2.0 - abs(cx[b] - cx[a])
            oy = (H[a] + H[b]) / 2.0 - abs(cy[b] - cy[a])
            axis[(a, b)] = 0 if ox <= oy else 1

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
                    worst = max(pred[k], key=lambda p: low[p] + size[p])
                    return None, (k, [worst])
                low[k] = want
            else:
                low[k] = base
        return low, None

    for _ in range(rounds):
        lx, bad = compact(0, px, W)
        if lx is None:
            if bad is None:
                return None, None
            k, ps = bad
            for p in ps:                            # flip the offenders to y
                axis[(min(k, p), max(k, p))] = 1
            continue
        ly, bad = compact(1, py, H)
        if ly is None:
            k, ps = bad
            for p in ps:
                axis[(min(k, p), max(k, p))] = 0
            continue
        return lx, ly
    return None, None


def place(c):
    v = case_view(c)
    units = build_units(v)
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
        nx, ny = legalise(v, units, cx, cy)
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
    for k, u in enumerate(units):
        for t, i in enumerate(u["mem"]):
            ox, oy = u["off"][t]
            w, h = u["dims"][i]
            out[i] = [lx[k] + ox, ly[k] + oy, w, h]
    return out


# ── evaluation ──────────────────────────────────────────────────────────────
def official(c, P):
    m = evaluate_solution({"positions": [list(p) for p in P], "runtime": 1.0},
                          c["base"], c["cons"][:c["n"]], c["b2b"], c["p2b"],
                          c["pins"], c["at"][:c["n"]],
                          target_positions=c["tp"][:c["n"]], median_runtime=1.0)
    return m


def mode_run(a):
    rows, cov, feas = [], 0, 0
    t0 = time.time()
    for c in (CASES[:a.limit] if a.limit else CASES):
        t1 = time.perf_counter()
        P = place(c)
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
                         runtime_seconds=dt, error=None))
    tot = len(CASES[:a.limit] if a.limit else CASES)
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
            for k in ("hpwl_gap", "area_gap", "violations_relative"):
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
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    sys.exit(mode_run(ap.parse_args()))
