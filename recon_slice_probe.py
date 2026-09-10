"""OFFLINE ONLY — M40 reconstruction dual-gate probe (never shipped).

The reconstruction track de-risk. M29 established the renderer (X = B*-tree X-rule
exact + given a good Y-ORDER, gravity reproduces fp_sol) but its from-scratch builder
(tree_build_probe.py, GREEDY most-connected insertion) was DEAD (8.22, 0/100). It
never tried the PROPER way to build a SLICING floorplan from a netlist: RECURSIVE
BISECTION. This probe tests two make-or-break, separable questions on TRAINING
layouts (which carry tree_sol + fp_sol), reusing M29's machinery:

  Gate A  (X-structure recoverable from connectivity?)
    Build a slicing tree by deterministic SPECTRAL recursive bisection (Fiedler-
    median split, region-aspect cut direction). Lay out X with the slicing X-rule
    using the TRUE block dims, then gravity-compact with the ORACLE Y-order (removes
    the Y confound). Compare its weighted QUALITY factor 1+0.5(hgap+agap) to the
    true-tree ceiling (~1.0, trueTree_oracleY) and to our validation quality ~1.274.

  Gate B  (deterministic Y-order can approach oracle-Y?)
    Fix the TRUE tree X (render_bstar). Sweep deterministic Y-orders M29 didn't try
    vs the oracle-Y ceiling.

Both must pass for the reconstruction track to be viable (P1 + P2). Any fail =>
RED-confirmed => converge at 1.3269. Pure offline, zero risk to the shipped binary.

Run:  C:/Users/Nordra/.conda/envs/iccadv/python.exe recon_slice_probe.py
"""
import sys, math, glob
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import (calculate_hpwl_b2b, calculate_hpwl_p2b,
                                calculate_bbox_area, compute_total_score)
from tree_decode_probe import load_layout, render_bstar, compact_down, _cost_of

sys.setrecursionlimit(100000)

ORACLE_TOTAL = 1.1079   # reconstruct_probe.py: feeding fp_sol verbatim (validation)
OUR_TOTAL = 1.3269      # shipped M37 portfolio (validation)
OUR_QUALITY = 1.274     # M28: our validation quality factor 1+0.5(hgap+agap)
OUTLINES = [0.6, 1.0, 1.6]   # root region aspect (w/h) sweep, like tree_build_probe
GATE_A_QUAL_BAR = 1.20  # slice quality factor must beat this (well under OUR_QUALITY)
GATE_B_GAP_BAR = 0.05   # best deterministic Y within 5% of oracle-Y Total


# --------------------------------------------------------------------------- #
# Spectral recursive-bisection slicing tree                                   #
# --------------------------------------------------------------------------- #
def _bisect(members, A, area):
    """Split members into (L, R): balanced Fiedler-median cut; area-balanced
    fallback when the induced subgraph has no internal edges."""
    m = len(members)
    if m <= 1:
        return members, []
    sub = A[np.ix_(members, members)]
    if not np.any(sub):                                   # no edges -> area balance
        L, R, aL, aR = [], [], 0.0, 0.0
        for b in sorted(members, key=lambda b: -area[b]):
            if aL <= aR:
                L.append(b); aL += area[b]
            else:
                R.append(b); aR += area[b]
        return L, R
    d = sub.sum(axis=1)
    lap = np.diag(d) - sub
    _, v = np.linalg.eigh(lap)                            # deterministic
    fied = v[:, 1]                                        # 2nd-smallest eigenvector
    med = float(np.median(fied))
    L = [members[i] for i in range(m) if fied[i] < med]
    R = [members[i] for i in range(m) if fied[i] >= med]
    if not L or not R:                                    # degenerate median -> halve
        L, R = members[:m // 2], members[m // 2:]
    return L, R


def _suma(node, area):
    if node[0] == "leaf":
        return area[node[1]]
    return _suma(node[2], area) + _suma(node[3], area)


def _build_tree(members, A, area, rw, rh):
    """Recursive slicing tree. Cut direction chosen to keep sub-regions near-square
    (V-cut if region wider than tall, else H-cut), split proportional to sub-area."""
    if len(members) == 1:
        return ("leaf", members[0])
    L, R = _bisect(members, A, area)
    aL = sum(area[b] for b in L); aR = sum(area[b] for b in R); tot = aL + aR or 1.0
    if rw >= rh:                                          # vertical cut: split width
        node_d = "V"
        lrw, lrh, rrw, rrh = rw * aL / tot, rh, rw * aR / tot, rh
    else:                                                 # horizontal cut: split height
        node_d = "H"
        lrw, lrh, rrw, rrh = rw, rh * aL / tot, rw, rh * aR / tot
    return ("cut", node_d, _build_tree(L, A, area, lrw, lrh),
                           _build_tree(R, A, area, rrw, rrh))


def _width(node, W):
    if node[0] == "leaf":
        return W[node[1]]
    _, d, L, R = node
    wl, wr = _width(L, W), _width(R, W)
    return wl + wr if d == "V" else max(wl, wr)


def _layout_x(node, x0, W, px):
    """Slicing X-rule with TRUE dims: V-cut places R right of L's packed width;
    H-cut stacks both at the same x (resolved later by Y-compaction)."""
    if node[0] == "leaf":
        px[node[1]] = x0
        return
    _, d, L, R = node
    if d == "V":
        _layout_x(L, x0, W, px)
        _layout_x(R, x0 + _width(L, W), W, px)
    else:
        _layout_x(L, x0, W, px)
        _layout_x(R, x0, W, px)


def _treemap(node, x0, y0, w, h, area, out):
    """Perfect-tiling realization: leaves fill their allocated region (free aspect)."""
    if node[0] == "leaf":
        out[node[1]] = (x0, y0, w, h)
        return
    _, d, L, R = node
    aL = _suma(L, area); tot = (aL + _suma(R, area)) or 1.0
    if d == "V":
        wl = w * aL / tot
        _treemap(L, x0, y0, wl, h, area, out)
        _treemap(R, x0 + wl, y0, w - wl, h, area, out)
    else:
        hl = h * aL / tot
        _treemap(L, x0, y0, w, hl, area, out)
        _treemap(R, x0, y0 + hl, w, h - hl, area, out)


# --------------------------------------------------------------------------- #
# Y-order heuristics (Gate B), all on a fixed X                               #
# --------------------------------------------------------------------------- #
def _conn_drop_order(n, A):
    deg = A.sum(axis=1)
    seed = int(np.argmax(deg))
    placed = [seed]; pset = {seed}
    remaining = set(range(n)) - pset
    while remaining:
        nb = max(remaining, key=lambda i: (sum(A[i, j] for j in pset), deg[i]))
        placed.append(nb); pset.add(nb); remaining.discard(nb)
    return placed


def _spearman(a, b):
    """Rank correlation; nan if degenerate."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- #
def _adj(lay):
    n = lay["n"]
    A = np.zeros((n, n))
    for e in lay["b2b"].tolist():
        i, j, w = int(e[0]), int(e[1]), float(e[2])
        if 0 <= i < n and 0 <= j < n and i != j:
            A[i, j] += w; A[j, i] += w
    return A


def _baseline(lay):
    n = lay["n"]; X, Y, W, H = lay["X"], lay["Y"], lay["W"], lay["H"]
    pos = [(X[b], Y[b], W[b], H[b]) for b in range(n)]
    return {"hpwl_baseline": calculate_hpwl_b2b(pos, lay["b2b"]) +
                             calculate_hpwl_p2b(pos, lay["p2b"], lay["pins"]),
            "area_baseline": calculate_bbox_area(pos)}, pos


def _acc(store, v, m, n):
    store[v]["cost"].append(float(m.cost))
    store[v]["agap"].append(float(m.area_gap))
    store[v]["hgap"].append(float(m.hpwl_gap))
    store[v]["vrel"].append(float(m.violations_relative))
    store[v]["n"].append(n)
    if not m.is_feasible:
        store[v]["infeas"] += 1


def _wq(store, v):
    """weighted quality factor 1+0.5(hgap+agap)."""
    ws = [math.exp(n / 12.0) for n in store[v]["n"]]
    q = [1 + 0.5 * (h + a) for h, a in zip(store[v]["hgap"], store[v]["agap"])]
    return sum(w * x for w, x in zip(ws, q)) / sum(ws)


def _wm(store, v, key):
    ws = [math.exp(n / 12.0) for n in store[v]["n"]]
    return sum(w * x for w, x in zip(ws, store[v][key])) / sum(ws)


def _total(store, v):
    return compute_total_score(store[v]["cost"], store[v]["n"])


def main():
    allf = sorted(glob.glob(str(_DIR / "floorset_lite" / "worker_0" / "layouts_*.th")))
    files = allf[::6]                                # ~15 files = 15 distinct block-counts
    layouts = []
    for f in files:
        d = torch.load(f)
        for L in range(min(6, d[0].shape[0])):       # few layouts/file -> spread n, stay fast
            try:
                layouts.append(load_layout(d, L))
            except AssertionError:
                pass
    # NOTE: 100% of FloorSet cases (train AND validation) carry preplaced blocks
    # (mean ~2.7/case) -> there is NO pure-movable subset. The connectivity slicing
    # tree therefore places preplaced at tree-determined (not their true) positions;
    # this is PESSIMISTIC for Gate A (a real builder pins the known preplaced coords),
    # but it is clean and apples-to-apples vs trueTree which also places every block.
    # The quality factor (dominated by the ~70 movable blocks) is the X signal; the
    # X-order Spearman (movable blocks, pin-free) is the complementary clean measure.
    npp = [sum(int(l["cons"][i][1]) != 0 for i in range(l["n"])) for l in layouts]
    print(f"loaded {len(layouts)} layouts ({len(files)} files); "
          f"preplaced/case mean={sum(npp)/len(npp):.1f} (0 pure-movable by design); "
          f"n in {sorted(set(l['n'] for l in layouts))}")

    variants = ["fp_self", "trueTree_oracleY",          # references / ceilings
                "sliceX_oracleY",                        # Gate A (X via connectivity tree)
                "Y_insOrder", "Y_bstarOrder", "Y_xThenGrav",
                "Y_connDrop", "Y_areaDesc"]              # Gate B (true X + det. Y-order)
    store = {v: {"cost": [], "agap": [], "hgap": [], "vrel": [], "n": [], "infeas": 0}
             for v in variants}
    spear, spw = [], []                                  # movable-block X-order corr (Gate A)

    for lay in layouts:
        n = lay["n"]; W, H, X, Y = lay["W"], lay["H"], lay["X"], lay["Y"]
        base, pos_fp = _baseline(lay)
        A = _adj(lay)
        area = [W[b] * H[b] for b in range(n)]
        tot_area = sum(area)
        oracle_ord = sorted(range(n), key=lambda b: Y[b])
        movable = [b for b in range(n) if int(lay["cons"][b][1]) == 0]

        # --- references ---
        _acc(store, "fp_self", _cost_of(pos_fp, lay, base), n)
        tpx, _ = render_bstar(lay)                      # true tree X (exact)
        _, tpy = compact_down(tpx, W, H, oracle_ord)
        _acc(store, "trueTree_oracleY",
             _cost_of([(tpx[b], tpy[b], W[b], H[b]) for b in range(n)], lay, base), n)

        # --- Gate A: connectivity slicing tree, best outline by quality gap ---
        bestX = None; bestX_px = None
        for asp in OUTLINES:
            rw = math.sqrt(tot_area * asp); rh = tot_area / rw
            tree = _build_tree(list(range(n)), A, area, rw, rh)
            px = [0.0] * n; _layout_x(tree, 0.0, W, px)
            _, py = compact_down(px, W, H, oracle_ord)
            mX = _cost_of([(px[b], py[b], W[b], H[b]) for b in range(n)], lay, base)
            qX = mX.area_gap + mX.hpwl_gap
            if bestX is None or qX < bestX[0]:
                bestX = (qX, mX); bestX_px = px
        _acc(store, "sliceX_oracleY", bestX[1], n)
        if len(movable) >= 3:
            s = _spearman([bestX_px[b] for b in movable], [X[b] for b in movable])
            if not math.isnan(s):
                spear.append(s); spw.append(math.exp(n / 12.0))

        # --- Gate B: fixed TRUE X (tpx), deterministic Y-orders ---
        _, bpy = render_bstar(lay)
        sweeps = {
            "Y_insOrder": [lay["seed"]] + [nb for (_, nb, _) in lay["ins"]],
            "Y_bstarOrder": sorted(range(n), key=lambda b: bpy[b]),
            "Y_xThenGrav": sorted(range(n), key=lambda b: tpx[b]),
            "Y_connDrop": _conn_drop_order(n, A),
            "Y_areaDesc": sorted(range(n), key=lambda b: -area[b]),
        }
        for v, order in sweeps.items():
            _, py = compact_down(tpx, W, H, order)
            _acc(store, v, _cost_of([(tpx[b], py[b], W[b], H[b]) for b in range(n)],
                                    lay, base), n)

    # --------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print(f"references:  per-layout oracle (fp_self) ~ 1.0   |   "
          f"validation: oracle={ORACLE_TOTAL}  ours={OUR_TOTAL} (quality~{OUR_QUALITY})")
    print("=" * 78)
    print(f"{'variant':>16} | {'Total':>7} | {'quality':>7} | {'w.hgap':>7} | "
          f"{'w.agap':>7} | {'w.vrel':>6} | {'infeas':>7}")
    for v in variants:
        print(f"{v:>16} | {_total(store, v):>7.4f} | {_wq(store, v):>7.4f} | "
              f"{_wm(store, v, 'hgap'):>7.4f} | {_wm(store, v, 'agap'):>7.4f} | "
              f"{_wm(store, v, 'vrel'):>6.3f} | {store[v]['infeas']:>3}/{len(layouts)}")
    wspear = sum(w * s for w, s in zip(spw, spear)) / sum(spw) if spw else float("nan")

    # --------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    self_q = _wq(store, "trueTree_oracleY")
    print(f"[self-check] trueTree_oracleY quality = {self_q:.4f} "
          f"(must be ~1.00 — render+oracle-Y reproduces fp_sol; else wiring bug)")

    aq = _wq(store, "sliceX_oracleY"); tq = _wq(store, "trueTree_oracleY")
    gate_a = (aq < GATE_A_QUAL_BAR and aq < OUR_QUALITY) or wspear > 0.6
    print(f"\n[Gate A] slicing-X + oracle-Y quality = {aq:.4f}   "
          f"(true-tree ceiling {tq:.4f} ; our validation {OUR_QUALITY})")
    print(f"         movable X-order Spearman (pin-free) = {wspear:.3f}")
    print(f"         => connectivity recovers X-structure?  "
          f"{'PASS' if gate_a else 'FAIL'} "
          f"(bar: quality < {GATE_A_QUAL_BAR} & < {OUR_QUALITY}, or Spearman > 0.6)")

    y_dets = ["Y_insOrder", "Y_bstarOrder", "Y_xThenGrav", "Y_connDrop", "Y_areaDesc"]
    oracle_t = _total(store, "trueTree_oracleY")
    best_y = min(y_dets, key=lambda v: _total(store, v))
    best_t = _total(store, best_y)
    gap = (best_t - oracle_t) / oracle_t
    gate_b = gap < GATE_B_GAP_BAR
    print(f"\n[Gate B] best deterministic Y-order = {best_y} (Total {best_t:.4f}) "
          f"vs oracle-Y {oracle_t:.4f}  gap {gap*100:+.1f}%")
    print(f"         => deterministic Y approaches oracle?   "
          f"{'PASS' if gate_b else 'FAIL'} (bar: gap < {GATE_B_GAP_BAR*100:.0f}%)")

    go = gate_a and gate_b
    print(f"\n[GO/NO-GO] {'GO -> M41 obstacle-aware builder on validation' if go else 'NO-GO -> RED-confirmed, converge at 1.3269 (update ledger)'}")


if __name__ == "__main__":
    main()
