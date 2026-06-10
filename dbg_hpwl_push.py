"""Prototype: post-placement HPWL slack push on the PORTFOLIO's output positions.

hgap is the dominant cost lever (weighted 0.404 vs agap 0.23 vs vrel 0.04).
Compaction only packs toward frame faces (area) and can spread connected blocks
apart, lifting HPWL. This prototype slides blocks toward their connectivity-
weighted L1-median within available void, keeping them inside the current bbox.

Movable categories (1/2 keep area / bv / gf / mib unchanged, only HPWL drops;
4 can only *decrease* bv):
  1. FREE SINGLE   (boundary==0, cluster==0, not preplaced): slide x and y.
  2. BOUNDARY SINGLE (boundary!=0, cluster==0, not preplaced): slide ONLY the
     free axis (LEFT/RIGHT -> y, TOP/BOTTOM -> x), keeping the constrained axis
     so edge-contact (and thus bv) is preserved. Corner blocks (both LR&TB) skip.
  3. FREE CLUSTER  (>=2 members, no preplaced, no boundary member): RIGID
     translation of the whole group. Internal touching is preserved (gf unchanged
     in theory); no member is a boundary block so bv is unaffected.
     ** REJECTED, off by default (ENABLE_CLUSTER=False).** Two reasons: (i) the
     compound-item packer leaves clusters with no slack -- only 1 free cluster across
     all 100 cases could move at all; (ii) a floating-point rigid translation breaks
     the cluster's *exact* internal abutments at the ULP level, which the harness's
     exact shapely scoring flags as a spurious grouping fragment (the M10 precision
     hazard). Net effect: +0.004% with 1 regression. Not ported to C++.
  4. VIOLATING BOUNDARY SINGLE (M16 candidate): a boundary single whose required
     edge bits are NOT all satisfied already counts 1 bv. Moving it cannot
     increase bv (satisfied blocks pin their edges; this block stays violating
     unless fully repaired). Strategy per pass: (a) if every unsatisfied bit can
     reach its frozen-bbox edge through the void, snap it there -> bv strictly
     drops by 1; (b) otherwise slide the unpinned axes (any axis without a
     satisfied bit) toward the L1-median, accepted only on strict HPWL decrease.
     ** DEAD END, off by default (ENABLE_VIOLATING=False).** Measured on the M15
     portfolio output (2026-06-10, dbg_vio_stats.py): of 202 violating boundary
     blocks across 100 cases, 123 are cluster members (immovable, M10 precision
     wall), 45 preplaced (fixed), and only 34 singles -- ALL 34 of which are
     BLOCKED (snap-to-edge overlaps another block). Zero repairs possible; the
     median-slide fallback yields ~0.00% (only 5th-decimal noise). Not ported.

Why bbox-shrink can't hurt bv: a SATISFIED boundary block lies exactly on its
edge, so it co-defines that extreme -> the bbox cannot shrink past it. Hence no
inward move of other blocks can un-satisfy a satisfied boundary block. (1) and
(2) and (3) only move blocks within the frozen bbox, so the bbox is non-increasing
and bv is non-increasing. We additionally accept a move only if it strictly lowers
the relevant HPWL (guard against rare non-monotone cases).

Measures the TRUE-cost headroom of these levers on the deployed layouts
(portfolio JSON positions, which already carry the C++ M14 free-single push) via
the harness evaluate_solution. The delta is the realizable gain of categories
2+3 on top of M14. If meaningful, port to C++.

Run: python dbg_hpwl_push.py
"""
import json, math, sys
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import (ContestEvaluator, evaluate_solution,
                                calculate_hpwl_b2b, calculate_hpwl_p2b)

TOL = 1e-6
EPS = 1e-6               # evaluator's edge-contact tolerance (must match)
PASSES = 8
B_LEFT, B_RIGHT, B_TOP, B_BOTTOM = 1, 2, 4, 8
ENABLE_BOUNDARY = True   # category 2: boundary-axis slide (shipped in C++ M15)
ENABLE_CLUSTER = False   # category 3: rigid free-cluster slide -- REJECTED (see below),
                         # off by default so a plain run reproduces the shipped M15 result
ENABLE_VIOLATING = False  # category 4: violating-boundary repair -- DEAD END (see above)
ENABLE_SWAP = True       # category 5 (M16): same-size single swap, see below
# 5. SAME-SIZE SINGLE SWAP (M16): swap the positions of two non-preplaced,
#    non-cluster blocks with EXACTLY equal (w, h) and equal boundary code. The
#    geometry multiset is unchanged -> bbox/area identical, overlap-free stays
#    overlap-free, edge-contact count per code unchanged (bv identical), no
#    cluster member moves (gf identical), dims unchanged (mib identical). Only
#    HPWL changes; a swap is accepted only on strict decrease. Soft blocks with
#    equal area and equal boundary code get bit-identical dims from
#    default_soft_dim, so exact-equality grouping finds real swap partners.


def _bbox(pos):
    xmin = min(p[0] for p in pos); ymin = min(p[1] for p in pos)
    xmax = max(p[0] + p[2] for p in pos); ymax = max(p[1] + p[3] for p in pos)
    return xmin, ymin, xmax, ymax


def _adj(b2b, p2b, n):
    """block -> [(neighbor, w)], block -> [(pin, w)]"""
    ba = [[] for _ in range(n)]; pa = [[] for _ in range(n)]
    for e in b2b:
        i, j, w = int(e[0]), int(e[1]), float(e[2])
        if w <= 0 or i < 0 or j < 0 or i >= n or j >= n:
            continue
        ba[i].append((j, w)); ba[j].append((i, w))
    for e in p2b:
        pi, bj, w = int(e[0]), int(e[1]), float(e[2])
        if w <= 0 or bj < 0 or bj >= n:
            continue
        pa[bj].append((pi, w))
    return ba, pa


def _wmedian(targets):
    """Weighted L1 median (minimiser of sum w*|x-t|) over [(t, w)]."""
    if not targets:
        return None
    targets = sorted(targets)
    total = sum(w for _, w in targets)
    acc = 0.0
    for t, w in targets:
        acc += w
        if acc >= total / 2.0:
            return t
    return targets[-1][0]


def push(pos, codes, clus, pre, b2b, p2b, pins, n):
    pos = [list(p) for p in pos]
    ba, pa = _adj(b2b, p2b, n)
    xmin, ymin, xmax, ymax = _bbox(pos)

    free_single = [i for i in range(n)
                   if codes[i] == 0 and clus[i] == 0 and i not in pre]
    bnd_single = [i for i in range(n)
                  if codes[i] != 0 and clus[i] == 0 and i not in pre]
    cl_map = {}
    for i in range(n):
        if clus[i] > 0:
            cl_map.setdefault(clus[i], []).append(i)
    free_clusters = []
    for cid, mem in cl_map.items():
        if len(mem) < 2:
            continue
        if any(i in pre for i in mem):
            continue
        if any(codes[i] != 0 for i in mem):
            continue
        free_clusters.append(mem)

    def hpwl_blk(i, x, y):
        cx, cy = x + pos[i][2] / 2.0, y + pos[i][3] / 2.0
        h = 0.0
        for j, w in ba[i]:
            ncx = pos[j][0] + pos[j][2] / 2.0; ncy = pos[j][1] + pos[j][3] / 2.0
            h += w * (abs(cx - ncx) + abs(cy - ncy))
        for pidx, w in pa[i]:
            h += w * (abs(cx - pins[pidx][0]) + abs(cy - pins[pidx][1]))
        return h

    def void_x(i):
        """[lo, hi] for lower-left x of block i (blocks overlapping in y)."""
        w_, h_ = pos[i][2], pos[i][3]
        lo, hi = xmin, xmax - w_
        for j in range(n):
            if j == i:
                continue
            if pos[j][1] < pos[i][1] + h_ - TOL and pos[i][1] < pos[j][1] + pos[j][3] - TOL:
                if pos[j][0] + pos[j][2] <= pos[i][0] + TOL:
                    lo = max(lo, pos[j][0] + pos[j][2])
                elif pos[j][0] >= pos[i][0] + w_ - TOL:
                    hi = min(hi, pos[j][0] - w_)
        return lo, hi

    def void_y(i):
        """[lo, hi] for lower-left y of block i (blocks overlapping in x)."""
        w_, h_ = pos[i][2], pos[i][3]
        lo, hi = ymin, ymax - h_
        for j in range(n):
            if j == i:
                continue
            if pos[j][0] < pos[i][0] + w_ - TOL and pos[i][0] < pos[j][0] + pos[j][2] - TOL:
                if pos[j][1] + pos[j][3] <= pos[i][1] + TOL:
                    lo = max(lo, pos[j][1] + pos[j][3])
                elif pos[j][1] >= pos[i][1] + h_ - TOL:
                    hi = min(hi, pos[j][1] - h_)
        return lo, hi

    def move_free(i):
        moved = False
        w_, h_ = pos[i][2], pos[i][3]
        lo, hi = void_x(i)
        tx = _wmedian([(pos[j][0] + pos[j][2] / 2.0, w) for j, w in ba[i]]
                      + [(pins[pidx][0], w) for pidx, w in pa[i]])
        if tx is not None and hi >= lo - TOL:
            nx = min(max(tx - w_ / 2.0, lo), hi)
            if abs(nx - pos[i][0]) > TOL and hpwl_blk(i, nx, pos[i][1]) < hpwl_blk(i, pos[i][0], pos[i][1]) - TOL:
                pos[i][0] = nx; moved = True
        lo, hi = void_y(i)
        ty = _wmedian([(pos[j][1] + pos[j][3] / 2.0, w) for j, w in ba[i]]
                      + [(pins[pidx][1], w) for pidx, w in pa[i]])
        if ty is not None and hi >= lo - TOL:
            ny = min(max(ty - h_ / 2.0, lo), hi)
            if abs(ny - pos[i][1]) > TOL and hpwl_blk(i, pos[i][0], ny) < hpwl_blk(i, pos[i][0], pos[i][1]) - TOL:
                pos[i][1] = ny; moved = True
        return moved

    def move_boundary(i):
        code = codes[i]
        has_lr = code & (B_LEFT | B_RIGHT)
        has_tb = code & (B_TOP | B_BOTTOM)
        if has_lr and has_tb:
            return False
        w_, h_ = pos[i][2], pos[i][3]
        if has_lr:                                   # slide y, keep x (on L/R edge)
            lo, hi = void_y(i)
            ty = _wmedian([(pos[j][1] + pos[j][3] / 2.0, w) for j, w in ba[i]]
                          + [(pins[pidx][1], w) for pidx, w in pa[i]])
            if ty is not None and hi >= lo - TOL:
                ny = min(max(ty - h_ / 2.0, lo), hi)
                if abs(ny - pos[i][1]) > TOL and hpwl_blk(i, pos[i][0], ny) < hpwl_blk(i, pos[i][0], pos[i][1]) - TOL:
                    pos[i][1] = ny; return True
        else:                                        # has_tb: slide x, keep y
            lo, hi = void_x(i)
            tx = _wmedian([(pos[j][0] + pos[j][2] / 2.0, w) for j, w in ba[i]]
                          + [(pins[pidx][0], w) for pidx, w in pa[i]])
            if tx is not None and hi >= lo - TOL:
                nx = min(max(tx - w_ / 2.0, lo), hi)
                if abs(nx - pos[i][0]) > TOL and hpwl_blk(i, nx, pos[i][1]) < hpwl_blk(i, pos[i][0], pos[i][1]) - TOL:
                    pos[i][0] = nx; return True
        return False

    def unsat_bits(i):
        """Required-edge bits of block i not currently touching the frozen bbox."""
        x, y, w_, h_ = pos[i]
        code, u = codes[i], 0
        if code & B_LEFT and abs(x - xmin) >= EPS:          u |= B_LEFT
        if code & B_RIGHT and abs(x + w_ - xmax) >= EPS:    u |= B_RIGHT
        if code & B_TOP and abs(y + h_ - ymax) >= EPS:      u |= B_TOP
        if code & B_BOTTOM and abs(y - ymin) >= EPS:        u |= B_BOTTOM
        return u

    def slide_median_x(i):
        lo, hi = void_x(i)
        tx = _wmedian([(pos[j][0] + pos[j][2] / 2.0, w) for j, w in ba[i]]
                      + [(pins[pidx][0], w) for pidx, w in pa[i]])
        if tx is not None and hi >= lo - TOL:
            nx = min(max(tx - pos[i][2] / 2.0, lo), hi)
            if abs(nx - pos[i][0]) > TOL and hpwl_blk(i, nx, pos[i][1]) < hpwl_blk(i, pos[i][0], pos[i][1]) - TOL:
                pos[i][0] = nx; return True
        return False

    def slide_median_y(i):
        lo, hi = void_y(i)
        ty = _wmedian([(pos[j][1] + pos[j][3] / 2.0, w) for j, w in ba[i]]
                      + [(pins[pidx][1], w) for pidx, w in pa[i]])
        if ty is not None and hi >= lo - TOL:
            ny = min(max(ty - pos[i][3] / 2.0, lo), hi)
            if abs(ny - pos[i][1]) > TOL and hpwl_blk(i, pos[i][0], ny) < hpwl_blk(i, pos[i][0], pos[i][1]) - TOL:
                pos[i][1] = ny; return True
        return False

    def move_violating(i):
        """Block i is a violating boundary single (>=1 unsatisfied edge bit).
        (a) Full repair: snap every unsatisfied bit to its frozen-bbox edge if the
            void allows -> bv drops by 1 (accepted unconditionally).
        (b) Else: L1-median slide on any axis without a satisfied bit (cannot
            increase bv -- the block stays violating), strict HPWL decrease only.
        """
        code = codes[i]
        u = unsat_bits(i)
        ux, uy = u & (B_LEFT | B_RIGHT), u & (B_TOP | B_BOTTOM)
        # contradictory codes (needs both L&R, or both T&B edges) cannot be repaired
        can_x = (code & (B_LEFT | B_RIGHT)) != (B_LEFT | B_RIGHT)
        can_y = (code & (B_TOP | B_BOTTOM)) != (B_TOP | B_BOTTOM)
        if can_x and can_y:
            save = list(pos[i])
            ok = True
            if ux:
                lo, hi = void_x(i)
                tgt = xmin if ux & B_LEFT else xmax - pos[i][2]
                if lo - TOL <= tgt <= hi + TOL:
                    pos[i][0] = tgt
                else:
                    ok = False
            if ok and uy:
                lo, hi = void_y(i)              # recomputed after the x snap
                tgt = ymax - pos[i][3] if uy & B_TOP else ymin
                if lo - TOL <= tgt <= hi + TOL:
                    pos[i][1] = tgt
                else:
                    ok = False
            if ok:
                return True                     # fully repaired: bv -= 1
            pos[i][0], pos[i][1] = save[0], save[1]
        # fallback: slide axes that carry no SATISFIED bit (pinned axes stay put)
        moved = False
        x_pinned = (code & (B_LEFT | B_RIGHT)) and not ux
        y_pinned = (code & (B_TOP | B_BOTTOM)) and not uy
        if not x_pinned and slide_median_x(i):
            moved = True
        if not y_pinned and slide_median_y(i):
            moved = True
        return moved

    # same-size swap groups: exact (w, h, boundary_code) over non-pre non-cluster
    swap_groups = {}
    for i in range(n):
        if clus[i] != 0 or i in pre:
            continue
        swap_groups.setdefault((pos[i][2], pos[i][3], codes[i]), []).append(i)
    swap_groups = [m for m in swap_groups.values() if len(m) >= 2]

    def swap_pass():
        moved = False
        for mem in swap_groups:
            for a in range(len(mem)):
                for b in range(a + 1, len(mem)):
                    i, j = mem[a], mem[b]
                    h0 = (hpwl_blk(i, pos[i][0], pos[i][1])
                          + hpwl_blk(j, pos[j][0], pos[j][1]))
                    # swap, then evaluate with both at their new spots (an i-j
                    # edge contributes identically before/after and cancels)
                    pos[i][0], pos[j][0] = pos[j][0], pos[i][0]
                    pos[i][1], pos[j][1] = pos[j][1], pos[i][1]
                    h1 = (hpwl_blk(i, pos[i][0], pos[i][1])
                          + hpwl_blk(j, pos[j][0], pos[j][1]))
                    if h1 < h0 - TOL:
                        moved = True
                    else:                                   # revert
                        pos[i][0], pos[j][0] = pos[j][0], pos[i][0]
                        pos[i][1], pos[j][1] = pos[j][1], pos[i][1]
        return moved

    def shift_cluster(mem, axis):
        """Rigid translation of free cluster `mem` along axis 0=x / 1=y."""
        mset = set(mem)
        if axis == 0:
            dlo = xmin - min(pos[m][0] for m in mem)
            dhi = xmax - max(pos[m][0] + pos[m][2] for m in mem)
        else:
            dlo = ymin - min(pos[m][1] for m in mem)
            dhi = ymax - max(pos[m][1] + pos[m][3] for m in mem)
        for m in mem:
            mx, my, mw, mh = pos[m]
            for j in range(n):
                if j in mset:
                    continue
                jx, jy, jw, jh = pos[j]
                if axis == 0:
                    if jy < my + mh - TOL and my < jy + jh - TOL:       # y-overlap
                        if jx + jw <= mx + TOL:
                            dlo = max(dlo, (jx + jw) - mx)
                        elif jx >= mx + mw - TOL:
                            dhi = min(dhi, jx - (mx + mw))
                else:
                    if jx < mx + mw - TOL and mx < jx + jw - TOL:       # x-overlap
                        if jy + jh <= my + TOL:
                            dlo = max(dlo, (jy + jh) - my)
                        elif jy >= my + mh - TOL:
                            dhi = min(dhi, jy - (my + mh))
        if dhi < dlo - TOL:
            return False
        targets = []
        for m in mem:
            mc = pos[m][0] + pos[m][2] / 2.0 if axis == 0 else pos[m][1] + pos[m][3] / 2.0
            for j, w in ba[m]:
                if j in mset:
                    continue
                jc = pos[j][0] + pos[j][2] / 2.0 if axis == 0 else pos[j][1] + pos[j][3] / 2.0
                targets.append((jc - mc, w))
            for pidx, w in pa[m]:
                pc = pins[pidx][0] if axis == 0 else pins[pidx][1]
                targets.append((pc - mc, w))
        if not targets:
            return False
        d = min(max(_wmedian(targets), dlo), dhi)
        if abs(d) <= TOL:
            return False
        before = sum(w * abs(t) for t, w in targets)
        after = sum(w * abs(d - t) for t, w in targets)
        if after < before - TOL:
            for m in mem:
                pos[m][axis] += d
            return True
        return False

    for _ in range(PASSES):
        moved = False
        if ENABLE_CLUSTER:
            for mem in free_clusters:
                if shift_cluster(mem, 0):
                    moved = True
                if shift_cluster(mem, 1):
                    moved = True
        if ENABLE_BOUNDARY:
            for i in bnd_single:
                if ENABLE_VIOLATING and unsat_bits(i):
                    if move_violating(i):
                        moved = True
                elif move_boundary(i):
                    moved = True
        for i in free_single:
            if move_free(i):
                moved = True
        if ENABLE_SWAP and swap_pass():
            moved = True
        if not moved:
            break
    return [tuple(p) for p in pos]


def main():
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
    port = json.load(open(_DIR / "iccad2026contest" / "optimizer_constructive_results.json"))
    pjson = {t["test_id"]: t for t in port["test_results"]}

    totW = totC0 = totC1 = 0.0
    rows = []
    for idx in range(len(ev.dataset)):
        s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
        at, b2b, p2b, pins, cons = inp
        n = int((at != -1).sum().item())
        base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
        ps = [tuple(p) for p in pjson[idx]["positions"]]
        codes = [int(cons[i, 4]) if cons.dim() > 1 and cons.shape[1] > 4 else 0 for i in range(n)]
        clus = [int(cons[i, 3]) if cons.dim() > 1 and cons.shape[1] > 3 else 0 for i in range(n)]
        pre = {i for i in range(n) if cons.dim() > 1 and cons.shape[1] > 1 and int(cons[i, 1]) != 0}
        pinsl = [(float(pins[k][0]), float(pins[k][1])) for k in range(pins.shape[0])]
        ps2 = push(ps, codes, clus, pre, b2b.tolist(), p2b.tolist(), pinsl, n)

        m0 = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                               b2b, p2b, pins, at[:n], target_positions=tp[:n], median_runtime=1.0)
        m1 = evaluate_solution({'positions': ps2, 'runtime': 1.0}, base, cons[:n],
                               b2b, p2b, pins, at[:n], target_positions=tp[:n], median_runtime=1.0)
        w = math.exp(n / 12.0)
        totW += w; totC0 += w * m0.cost; totC1 += w * m1.cost
        rows.append((idx, n, w, m0.cost, m1.cost, m0.hpwl_gap, m1.hpwl_gap,
                     m0.area_gap, m1.area_gap, m1.is_feasible,
                     m0.boundary_violations, m1.boundary_violations,
                     m0.grouping_violations, m1.grouping_violations))

    nreg = sum(1 for r in rows if r[4] > r[3] + 1e-6)
    nviol = sum(1 for r in rows if r[11] > r[10] or r[13] > r[12])  # bv or gf increased
    print(f"Total Score: orig={totC0/totW:.4f}  pushed={totC1/totW:.4f}  "
          f"delta={100*(totC1-totC0)/totC0:+.2f}%")
    print(f"regressions: {nreg}/100 cases ; cases with bv/gf increase: {nviol}/100")
    print(f"{'case':>4} {'n':>4} {'wt%':>5} {'cost0':>6} {'cost1':>6} "
          f"{'hg0':>6} {'hg1':>6} {'ag0':>6} {'ag1':>6} {'bv':>9} {'gf':>9} {'feas':>5}")
    rows.sort(key=lambda r: -(r[2] * (r[3] - r[4])))
    for (idx, n, w, c0, c1, h0, h1, a0, a1, fe, b0, b1, g0, g1) in rows[:30]:
        print(f"{idx:>4} {n:>4} {100*w/totW:5.2f} {c0:6.3f} {c1:6.3f} "
              f"{h0:6.3f} {h1:6.3f} {a0:6.3f} {a1:6.3f} {b0:>3}->{b1:<4} "
              f"{g0:>3}->{g1:<4} {'' if fe else 'INFEAS'}")


if __name__ == "__main__":
    main()
