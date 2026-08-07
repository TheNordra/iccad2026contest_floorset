"""OFFLINE ONLY — index/attribute leak reconstruction probe (never shipped).

M40 (recon_slice_probe.py) RED-confirmed reconstruction on two gates, but it ONLY
tested connectivity->X (spectral recursive bisection) and 5 deterministic Y-orders.
liteLoader.py:30 says tree_sol is the B*-tree the baseline was GENERATED from, and
M40's movable X-order Spearman 0.009 actually shows the NETLIST is generated
INDEPENDENTLY of the geometry (a proximity-based netlist would correlate). So M40
only kills "connectivity -> geometry". It never tested whether the generation
ORDER leaks through the block ENUMERATION INDEX (or area) -- a near-universal
artifact of synthetic benchmark generators (blocks emitted in tree-insertion/DFS
order).

This probe settles that one remaining hypothesis on TRAINING layouts (which carry
tree_sol + fp_sol), reusing the M29/M40 machinery:

  Gate A'  (index/area -> tree / X recoverable?)
    1. STRUCTURAL: is tree_sol itself rebuildable from index alone? seed==0 rate;
       does insertion step k insert block k (nb monotonic in index)?; anchor==nb-1
       rate; dir(0/1) split.  If high, X is FREE at test time (use index order).
    2. PREDICTIVE: Spearman(index, trueX) & Spearman(areaRank, trueX) on movable
       blocks -- the direct analogue of M40's connectivity Spearman 0.009.
    3. QUALITY: build a slicing tree by index-median / area-balance bisection
       (mirror M40's _build_tree but swap the Fiedler cut for index/area), lay X,
       gravity-compact with the ORACLE Y-order (removes the Y confound), score the
       weighted quality factor 1+0.5(hgap+agap) vs the true-tree ceiling (~1.0)
       and our validation quality ~1.274.

  Gate B'  (index/area -> Y-order recoverable?)
    Fix the TRUE tree X (render_bstar); add Y-orders M40 did NOT try (index asc,
    index desc, area asc) vs the oracle-Y ceiling.

Reopen requires Gate A' AND Gate B' to pass (on index OR area). Any double-fail =>
reconstruction is truly RED (information-theoretic, not method-limited) => stop.
Pure offline, zero risk to the shipped binary.

Run:  C:/Users/Nordra/.conda/envs/iccadv/python.exe recon_index_probe.py
"""
import sys, math, glob
from pathlib import Path

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from tree_decode_probe import load_layout, render_bstar, compact_down, _cost_of
from recon_slice_probe import (_width, _layout_x, _baseline, _acc, _wq, _wm,
                               _total, _spearman, ORACLE_TOTAL, OUR_TOTAL,
                               OUR_QUALITY, OUTLINES, GATE_A_QUAL_BAR,
                               GATE_B_GAP_BAR)

sys.setrecursionlimit(100000)


# --------------------------------------------------------------------------- #
# Deterministic slicing tree from a NON-connectivity key (index / area)       #
# --------------------------------------------------------------------------- #
def _bisect_index(members):
    """Split by index median (low half / high half), preserving index order."""
    s = sorted(members)
    m = len(s) // 2
    return s[:m], s[m:]


def _bisect_area(members, area):
    """Area-balanced greedy split (largest-first), like M40's no-edge fallback."""
    L, R, aL, aR = [], [], 0.0, 0.0
    for b in sorted(members, key=lambda b: -area[b]):
        if aL <= aR:
            L.append(b); aL += area[b]
        else:
            R.append(b); aR += area[b]
    return L, R


def _build_tree(members, area, rw, rh, splitter):
    """Recursive slicing tree; cut direction keeps sub-regions near-square,
    split proportional to sub-area (identical region logic to M40)."""
    if len(members) == 1:
        return ("leaf", members[0])
    L, R = splitter(members)
    if not L or not R:                                    # degenerate -> halve
        s = sorted(members); L, R = s[:len(s) // 2], s[len(s) // 2:]
    aL = sum(area[b] for b in L); aR = sum(area[b] for b in R); tot = aL + aR or 1.0
    if rw >= rh:
        d = "V"; lrw, lrh, rrw, rrh = rw * aL / tot, rh, rw * aR / tot, rh
    else:
        d = "H"; lrw, lrh, rrw, rrh = rw, rh * aL / tot, rw, rh * aR / tot
    return ("cut", d, _build_tree(L, area, lrw, lrh, splitter),
                      _build_tree(R, area, rrw, rrh, splitter))


def _x_from_tree(lay, area, splitter):
    """Best-outline X from a key-based slicing tree, gravity-compacted with the
    ORACLE Y-order. Returns (metric, px, quality)."""
    n, W, H, Y = lay["n"], lay["W"], lay["H"], lay["Y"]
    tot_area = sum(area); oracle_ord = sorted(range(n), key=lambda b: Y[b])
    best = None
    for asp in OUTLINES:
        rw = math.sqrt(tot_area * asp); rh = tot_area / rw
        tree = _build_tree(list(range(n)), area, rw, rh, splitter)
        px = [0.0] * n; _layout_x(tree, 0.0, W, px)
        _, py = compact_down(px, W, H, oracle_ord)
        m = _cost_of([(px[b], py[b], W[b], H[b]) for b in range(n)], lay,
                     _baseline(lay)[0])
        q = m.area_gap + m.hpwl_gap
        if best is None or q < best[2]:
            best = (m, px, q)
    return best


# --------------------------------------------------------------------------- #
def main():
    allf = sorted(glob.glob(str(_DIR / "floorset_lite" / "worker_0" / "layouts_*.th")))
    files = allf[::6]                                     # ~15 distinct block-counts
    layouts = []
    for f in files:
        d = torch.load(f)
        for L in range(min(6, d[0].shape[0])):
            try:
                layouts.append(load_layout(d, L))
            except AssertionError:
                pass
    print(f"loaded {len(layouts)} layouts ({len(files)} files); "
          f"n in {sorted(set(l['n'] for l in layouts))}")

    # ----- Gate A'.1 STRUCTURAL: is tree_sol rebuildable from index? --------- #
    seed0 = 0; tot = 0
    step_match = 0; step_tot = 0          # does insertion step k insert block k+1?
    nb_mono = 0                           # is nb strictly increasing along tree_sol?
    anc_prev = 0; anc_lt = 0; ins_tot = 0  # anchor == nb-1 ?  anchor < nb ?
    dir0 = 0; dir1 = 0
    for lay in layouts:
        tot += 1
        if lay["seed"] == 0:
            seed0 += 1
        ins = lay["ins"]
        nbs = [nb for (_, nb, _) in ins]
        nb_mono += int(all(nbs[i] < nbs[i + 1] for i in range(len(nbs) - 1)))
        for k, (a, nb, dr) in enumerate(ins):
            step_tot += 1
            if nb == k + 1:               # block (k+1) inserted at step k (seed=0)
                step_match += 1
            ins_tot += 1
            if a == nb - 1:
                anc_prev += 1
            if a < nb:
                anc_lt += 1
            if dr == 0:
                dir0 += 1
            else:
                dir1 += 1

    # ----- Gate A'.2/3 + references + Gate B' -------------------------------- #
    variants = ["fp_self", "trueTree_oracleY",           # references / ceilings
                "idxX_oracleY", "areaX_oracleY",         # Gate A' (X via index/area tree)
                "Y_indexAsc", "Y_indexDesc", "Y_areaAsc"]  # Gate B' (true X + det. Y)
    store = {v: {"cost": [], "agap": [], "hgap": [], "vrel": [], "n": [], "infeas": 0}
             for v in variants}
    sp_idx, sp_area, spw = [], [], []

    for lay in layouts:
        n = lay["n"]; W, H, X, Y = lay["W"], lay["H"], lay["X"], lay["Y"]
        base, pos_fp = _baseline(lay)
        area = [W[b] * H[b] for b in range(n)]
        oracle_ord = sorted(range(n), key=lambda b: Y[b])
        movable = [b for b in range(n) if int(lay["cons"][b][1]) == 0]

        _acc(store, "fp_self", _cost_of(pos_fp, lay, base), n)
        tpx, _ = render_bstar(lay)                        # true tree X (exact)
        _, tpy = compact_down(tpx, W, H, oracle_ord)
        _acc(store, "trueTree_oracleY",
             _cost_of([(tpx[b], tpy[b], W[b], H[b]) for b in range(n)], lay, base), n)

        mI, pxI, _ = _x_from_tree(lay, area, _bisect_index)
        _acc(store, "idxX_oracleY", mI, n)
        mA, pxA, _ = _x_from_tree(lay, area, lambda mem: _bisect_area(mem, area))
        _acc(store, "areaX_oracleY", mA, n)

        if len(movable) >= 3:
            s_i = _spearman([float(b) for b in movable], [X[b] for b in movable])
            s_a = _spearman([area[b] for b in movable], [X[b] for b in movable])
            if not math.isnan(s_i) and not math.isnan(s_a):
                sp_idx.append(s_i); sp_area.append(s_a)
                spw.append(math.exp(n / 12.0))

        # Gate B': fixed TRUE X, new deterministic Y-orders
        sweeps = {
            "Y_indexAsc": list(range(n)),
            "Y_indexDesc": list(range(n - 1, -1, -1)),
            "Y_areaAsc": sorted(range(n), key=lambda b: area[b]),
        }
        for v, order in sweeps.items():
            _, py = compact_down(tpx, W, H, order)
            _acc(store, v, _cost_of([(tpx[b], py[b], W[b], H[b]) for b in range(n)],
                                    lay, base), n)

    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("Gate A'.1  STRUCTURAL: is tree_sol rebuildable from INDEX alone?")
    print("=" * 78)
    print(f"  seed == block 0           : {seed0}/{tot} ({seed0/tot*100:.0f}%)")
    print(f"  nb strictly increasing    : {nb_mono}/{tot} ({nb_mono/tot*100:.0f}%)")
    print(f"  step k inserts block k+1  : {step_match}/{step_tot} "
          f"({step_match/max(1,step_tot)*100:.0f}%)")
    print(f"  anchor == nb-1            : {anc_prev}/{ins_tot} "
          f"({anc_prev/max(1,ins_tot)*100:.0f}%)")
    print(f"  anchor <  nb              : {anc_lt}/{ins_tot} "
          f"({anc_lt/max(1,ins_tot)*100:.0f}%)")
    print(f"  dir split (right/above)   : {dir0}/{dir1}")

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

    wsi = sum(w * s for w, s in zip(spw, sp_idx)) / sum(spw) if spw else float("nan")
    wsa = sum(w * s for w, s in zip(spw, sp_area)) / sum(spw) if spw else float("nan")

    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    self_q = _wq(store, "trueTree_oracleY")
    print(f"[self-check] trueTree_oracleY quality = {self_q:.4f} "
          f"(must be ~1.00 -- render+oracle-Y reproduces fp_sol; else wiring bug)")

    struct_ok = (seed0 / tot > 0.9 and step_match / max(1, step_tot) > 0.9)
    qi = _wq(store, "idxX_oracleY"); qa = _wq(store, "areaX_oracleY")
    tq = _wq(store, "trueTree_oracleY")
    print(f"\n[Gate A'] index-tree  X+oracle-Y quality = {qi:.4f}   Spearman(index,X) = {wsi:.3f}")
    print(f"          area-tree   X+oracle-Y quality = {qa:.4f}   Spearman(areaR,X) = {wsa:.3f}")
    print(f"          true-tree ceiling {tq:.4f} ; our validation {OUR_QUALITY}")
    gate_a = (struct_ok
              or qi < GATE_A_QUAL_BAR or qa < GATE_A_QUAL_BAR
              or wsi > 0.6 or wsa > 0.6)
    print(f"          => index/area recovers X-structure?  "
          f"{'PASS' if gate_a else 'FAIL'} "
          f"(bar: structural>90%, or quality < {GATE_A_QUAL_BAR}, or Spearman > 0.6)")

    y_dets = ["Y_indexAsc", "Y_indexDesc", "Y_areaAsc"]
    oracle_t = _total(store, "trueTree_oracleY")
    best_y = min(y_dets, key=lambda v: _total(store, v))
    best_t = _total(store, best_y)
    gap = (best_t - oracle_t) / oracle_t
    gate_b = gap < GATE_B_GAP_BAR
    print(f"\n[Gate B'] best new det. Y-order = {best_y} (Total {best_t:.4f}) "
          f"vs oracle-Y {oracle_t:.4f}  gap {gap*100:+.1f}%")
    print(f"          => index/area recovers Y-order?      "
          f"{'PASS' if gate_b else 'FAIL'} (bar: gap < {GATE_B_GAP_BAR*100:.0f}%)")

    go = gate_a and gate_b
    print(f"\n[GO/NO-GO] {'GO -> reconstruction REOPENS (build index/area tree on validation)' if go else 'NO-GO -> reconstruction truly RED (index/area leak null too); converge, stop guessing'}")


if __name__ == "__main__":
    main()
