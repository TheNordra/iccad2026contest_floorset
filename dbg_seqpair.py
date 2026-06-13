"""Prototype: GLOBAL re-pack via SEQUENCE-PAIR, measure the agap ceiling vs greedy.

The greedy placer (constructive.cpp pack_in_frame) places once and never relocates;
compaction is order-preserving; JUMP freezes the bbox. None re-arrange the whole
layout to a denser global packing, so ~27% void (agap 0.23, the largest open lever)
survives. This measures whether a global packer can beat the deployed greedy cost.

Representation: sequence-pair (Gp, Gm). For modules a,b:
  pa+<pb+ and pa-<pb-  -> a LEFT of b   (x_a + w_a <= x_b)
  pa+>pb+ and pa-<pb-  -> a BELOW b     (y_a + h_a <= y_b)
Coordinates by longest-path DAG (O(n^2), fine for n<=120): compacted, overlap-free.

Two probes (see CLAUDE.md M27 plan):
  L1  deterministic local search seeded from the greedy SP (shippable form)
  L2  fixed-seed annealing from scratch (true ceiling)

Scoring is the TRUE harness evaluate_solution (same import as dbg_hpwl_push.py), so
the prototype cannot fool itself on feasibility/violations.

Usage:
  python dbg_seqpair.py describe [ids...]    # case composition (default: targets)
  python dbg_seqpair.py l1 [ids...]          # L1 local search
  python dbg_seqpair.py l2 [ids...]          # L2 annealing ceiling
"""
import json
import math
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import (ContestEvaluator, evaluate_solution,
                                calculate_hpwl_b2b, calculate_hpwl_p2b)

TOL = 1e-6
EPS = 1e-6
B_LEFT, B_RIGHT, B_TOP, B_BOTTOM = 1, 2, 4, 8

# dense, high-weight, mostly-movable cases where agap headroom should live
TARGETS = [90, 95, 96, 98, 99]
HARD = [62, 85, 89]


def load_cases():
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    port = json.load(open(_DIR / "iccad2026contest" /
                          "optimizer_constructive_results.json"))
    pjson = {t["test_id"]: t for t in port["test_results"]}
    return ev, pjson


def case_data(ev, pjson, idx):
    s = ev.dataset[idx]
    inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    ps = [tuple(float(v) for v in p) for p in pjson[idx]["positions"]]  # (x,y,w,h)
    nc = cons.shape[1] if cons.dim() > 1 else 0
    codes = [int(cons[i, 4]) if nc > 4 else 0 for i in range(n)]
    clus = [int(cons[i, 3]) if nc > 3 else 0 for i in range(n)]
    mib = [int(cons[i, 2]) if nc > 2 else 0 for i in range(n)]
    pre = {i for i in range(n) if nc > 1 and int(cons[i, 1]) != 0}
    fix = {i for i in range(n) if nc > 0 and int(cons[i, 0]) != 0}
    # precompute edge lists ONCE (tolist() per evaluate is the annealing bottleneck)
    b2b_list = [(int(e[0]), int(e[1]), float(e[2])) for e in b2b.tolist()
                if float(e[2]) > 0 and 0 <= int(e[0]) < n and 0 <= int(e[1]) < n]
    pins_list = [(float(p[0]), float(p[1])) for p in pins.tolist()]
    p2b_list = [(int(e[0]), int(e[1]), float(e[2])) for e in p2b.tolist()
                if float(e[2]) > 0 and 0 <= int(e[1]) < n and 0 <= int(e[0]) < len(pins_list)]
    return dict(idx=idx, n=n, at=at, b2b=b2b, p2b=p2b, pins=pins, cons=cons,
                b2b_list=b2b_list, p2b_list=p2b_list, pins_list=pins_list,
                base=base, tp=tp, ps=ps, codes=codes, clus=clus, mib=mib,
                pre=pre, fix=fix, cost=float(pjson[idx]["cost"]),
                hgap=float(pjson[idx]["hpwl_gap"]),
                agap=float(pjson[idx]["area_gap"]),
                vrel=float(pjson[idx]["violations_relative"]))


def _bbox(ps):
    xs = [p[0] for p in ps]; ys = [p[1] for p in ps]
    return (min(xs), min(ys),
            max(p[0] + p[2] for p in ps), max(p[1] + p[3] for p in ps))


def describe(ids, verbose=True):
    ev, pj = load_cases()
    npre0 = ncl0 = nclean = 0
    if verbose:
        print(f"{'case':>4} {'n':>4} {'pre':>4} {'clB(grp)':>9} {'bnd':>4} "
              f"{'free':>4} {'dens':>6} {'agap':>6} {'cost':>7}")
    allids = list(ids) if ids else list(range(len(ev.dataset)))
    for idx in allids:
        c = case_data(ev, pj, idx)
        n = c['n']; ps = c['ps']
        sumA = sum(p[2] * p[3] for p in ps)
        x0, y0, x1, y1 = _bbox(ps)
        bbox = (x1 - x0) * (y1 - y0)
        grps = {c['clus'][i] for i in range(n) if c['clus'][i] > 0}
        nclb = sum(1 for i in range(n) if c['clus'][i] > 0)
        nbnd = sum(1 for i in range(n) if c['codes'][i] != 0)
        nfree = sum(1 for i in range(n)
                    if c['codes'][i] == 0 and c['clus'][i] == 0 and i not in c['pre'])
        if not c['pre']:
            npre0 += 1
        if not grps:
            ncl0 += 1
        if not c['pre'] and not grps:
            nclean += 1
        if verbose:
            tag = " <TARGET" if idx in TARGETS else (" <hard" if idx in HARD else "")
            print(f"{idx:>4} {n:>4} {len(c['pre']):>4} "
                  f"{nclb:>4}({len(grps):>2}) {nbnd:>4} {nfree:>4} "
                  f"{bbox / sumA:>6.3f} {c['agap']:>6.3f} {c['cost']:>7.4f}{tag}")
    print(f"[aggregate over {len(allids)} cases] no-preplaced: {npre0}  "
          f"no-cluster: {ncl0}  no-pre&no-cluster: {nclean}")


# ─── sequence-pair core ───────────────────────────────────────────────────────
# Modules = singles (1 block) + cluster compounds (rigid bbox of the cluster's
# current internal layout). RELAXED ceiling: preplaced are treated as movable too
# (real preplaced are pinned -> the realizable gain is <= this upper bound), and
# cluster members are not translated through the harness (we report a proxy cost,
# not evaluate_solution, so the M10/M15 FP-abutment hazard is irrelevant here).

def build_modules(c, compound=False):
    """Return (modules, blk2mod). Each module: dict(w,h, members=[(bid,offx,offy,w,h)],
    ox,oy) where (ox,oy) is the module bbox origin in the greedy (JSON) layout.

    compound=True compounds each cluster into its rigid bbox -- but a cluster bbox
    OVERLAPS singles nestled in its concavities, which breaks the separating-axis SP
    recovery (the greedy exploits that nesting). compound=False (default) makes every
    block its own module: blocks tile cleanly so recovery is valid. The all-individual
    pack is the OPTIMISTIC area ceiling (it may shatter clusters; the constrained
    packer that keeps them together can only do worse)."""
    n = c['n']; ps = c['ps']; clus = c['clus']
    modules = []
    blk2mod = [-1] * n
    used = set()
    if compound:
        groups = {}
        for i in range(n):
            if clus[i] > 0:
                groups.setdefault(clus[i], []).append(i)
        for g, mem in groups.items():
            ox = min(ps[i][0] for i in mem); oy = min(ps[i][1] for i in mem)
            w = max(ps[i][0] + ps[i][2] for i in mem) - ox
            h = max(ps[i][1] + ps[i][3] for i in mem) - oy
            members = [(i, ps[i][0] - ox, ps[i][1] - oy, ps[i][2], ps[i][3]) for i in mem]
            for i in mem:
                blk2mod[i] = len(modules); used.add(i)
            modules.append(dict(w=w, h=h, members=members, ox=ox, oy=oy))
    for i in range(n):
        if i in used:
            continue
        x, y, w, h = ps[i]
        blk2mod[i] = len(modules)
        modules.append(dict(w=w, h=h, members=[(i, 0.0, 0.0, w, h)], ox=x, oy=y))
    return modules, blk2mod


class _FenMax:
    """1-indexed Fenwick tree for prefix maxima."""
    def __init__(self, n):
        self.n = n; self.t = [0.0] * (n + 1)
    def update(self, i, v):
        i += 1
        while i <= self.n:
            if v > self.t[i]:
                self.t[i] = v
            i += i & (-i)
    def query(self, i):  # max over [0, i)
        r = 0.0
        while i > 0:
            if self.t[i] > r:
                r = self.t[i]
            i -= i & (-i)
        return r


def pack_sp(mw, mh, Gp, Gm):
    """Longest-path SP packing (Fenwick, O(n log n)). Returns X, Y, W, H."""
    n = len(mw)
    posp = [0] * n; posm = [0] * n
    for i, m in enumerate(Gp):
        posp[m] = i
    for i, m in enumerate(Gm):
        posm[m] = i
    X = [0.0] * n; Y = [0.0] * n
    fx = _FenMax(n)
    for m in Gp:                      # x: a left-of m iff posp[a]<posp[m] & posm[a]<posm[m]
        X[m] = fx.query(posm[m])      # max over posm-index < posm[m] already in Gp-prefix
        fx.update(posm[m], X[m] + mw[m])
    fy = _FenMax(n)
    for m in Gm:                      # y: a below m iff posp[a]>posp[m] & posm[a]<posm[m]
        # process in Gm order (posm increasing); among those, need posp[a]>posp[m].
        # index the Fenwick by (n-1-posp) so "posp[a]>posp[m]" -> prefix.
        key = n - 1 - posp[m]
        Y[m] = fy.query(key)
        fy.update(key, Y[m] + mh[m])
    W = max(X[m] + mw[m] for m in range(n))
    H = max(Y[m] + mh[m] for m in range(n))
    return X, Y, W, H


def recover_sp(modules):
    """Recover a VALID sequence pair reproducing the greedy layout. Edges are
    OVERLAP-CONDITIONED (H-left needs y-overlap, V-below/above needs x-overlap): this
    excludes diagonally-separated pairs, which is what makes both constraint graphs
    acyclic (the unconditioned 'left or below' relation cycles on real layouts).
    Gm = topo(H-left u V-below), Gp = topo(H-left u V-above). Diagonal pairs are free
    in the topo and resolved by a lower-left-first tie-break so the packing stays
    close to the greedy. Any resulting permutation pair is a valid SP."""
    nm = len(modules)
    ox = [m['ox'] for m in modules]; oy = [m['oy'] for m in modules]
    w = [m['w'] for m in modules]; h = [m['h'] for m in modules]
    import heapq

    def yov(a, b):
        return oy[a] < oy[b] + h[b] - TOL and oy[b] < oy[a] + h[a] - TOL

    def xov(a, b):
        return ox[a] < ox[b] + w[b] - TOL and ox[b] < ox[a] + w[a] - TOL

    def topo(edge, key):
        adj = [[] for _ in range(nm)]; indeg = [0] * nm
        for a in range(nm):
            for b in range(nm):
                if a != b and edge(a, b):
                    adj[a].append(b); indeg[b] += 1
        hq = [(key(i), i) for i in range(nm) if indeg[i] == 0]
        heapq.heapify(hq); out = []
        while hq:
            _, u = heapq.heappop(hq); out.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(hq, (key(v), v))
        return out if len(out) == nm else None

    hleft = lambda a, b: ox[a] + w[a] <= ox[b] + TOL and yov(a, b)
    vbelow = lambda a, b: oy[a] + h[a] <= oy[b] + TOL and xov(a, b)
    vabove = lambda a, b: oy[b] + h[b] <= oy[a] + TOL and xov(a, b)
    Gm = topo(lambda a, b: hleft(a, b) or vbelow(a, b), lambda i: (ox[i] + oy[i], oy[i]))
    Gp = topo(lambda a, b: hleft(a, b) or vabove(a, b), lambda i: (ox[i] - oy[i], -oy[i]))
    if Gm is None or Gp is None:            # cyclic (shouldn't happen) -> shelf fallback
        Gm = sorted(range(nm), key=lambda m: (oy[m], ox[m]))
        Gp = sorted(range(nm), key=lambda m: (-oy[m], ox[m]))
    return Gp, Gm


def compute_nsoft(c):
    n = c['n']; codes = c['codes']; clus = c['clus']; mib = c['mib']; ps = c['ps']
    nsoft = sum(1 for i in range(n) if codes[i] != 0)
    cl = {}
    for i in range(n):
        if clus[i] > 0:
            cl.setdefault(clus[i], 0)
            cl[clus[i]] += 1
    for s in cl.values():
        nsoft += max(0, s - 1)
    mg = {}
    for i in range(n):
        if mib[i] > 0:
            mg.setdefault(mib[i], set()).add((round(ps[i][2], 4), round(ps[i][3], 4)))
    for sh in mg.values():
        nsoft += max(0, len(sh) - 1)
    return max(nsoft, 1)


def sp_metrics(c, modules, blk2mod, X, Y, W, H):
    """area, hpwl (b2b+p2b), bv (non-cluster boundary singles off-edge) for an SP pack."""
    n = c['n']
    cx = [0.0] * n; cy = [0.0] * n
    for mi, m in enumerate(modules):
        for (bid, offx, offy, bw, bh) in m['members']:
            cx[bid] = X[mi] + offx + bw / 2.0
            cy[bid] = Y[mi] + offy + bh / 2.0
    hpwl = 0.0
    for i, j, w in c['b2b_list']:
        hpwl += w * (abs(cx[i] - cx[j]) + abs(cy[i] - cy[j]))
    pins = c['pins_list']
    for pi, bj, w in c['p2b_list']:
        hpwl += w * (abs(cx[bj] - pins[pi][0]) + abs(cy[bj] - pins[pi][1]))
    bv = 0
    codes = c['codes']
    for mi, m in enumerate(modules):
        if len(m['members']) != 1:        # compound: skip (boundary check is messy)
            continue
        bid = m['members'][0][0]
        cd = codes[bid]
        if cd == 0:
            continue
        bx, by = X[mi], Y[mi]
        bw, bh = m['w'], m['h']
        ok = True
        if cd & B_LEFT:   ok = ok and abs(bx - 0.0) < 1e-4
        if cd & B_RIGHT:  ok = ok and abs(bx + bw - W) < 1e-4
        if cd & B_TOP:    ok = ok and abs(by + bh - H) < 1e-4
        if cd & B_BOTTOM: ok = ok and abs(by - 0.0) < 1e-4
        if not ok:
            bv += 1
    return W * H, hpwl, bv


def est_cost(area, hpwl, vrel, gref):
    """Map an SP pack to estimated contest cost using the greedy's gaps as scale.
    gref = (greedy_area, greedy_hpwl, agap_g, hgap_g). Clamp gaps at 0 (no credit
    past baseline). At (greedy_area, greedy_hpwl, greedy_vrel) this returns the
    greedy cost, so deltas are directly comparable."""
    ga, gh, ag, hg = gref
    new_ag = max(0.0, area / ga * (1.0 + ag) - 1.0)
    new_hg = max(0.0, hpwl / gh * (1.0 + hg) - 1.0)
    return (1.0 + 0.5 * (new_ag + new_hg)) * math.exp(2.0 * vrel)


def run_case(c, mode, iters=4000, seed=12345, verbose=True, seed_mode="greedy"):
    import random
    rng = random.Random(seed)
    n = c['n']
    modules, blk2mod = build_modules(c, compound=False)
    nm = len(modules)
    mw = [m['w'] for m in modules]; mh = [m['h'] for m in modules]
    nsoft = compute_nsoft(c)
    # greedy reference (exact from JSON positions)
    gx0, gy0, gx1, gy1 = _bbox(c['ps'])
    garea = (gx1 - gx0) * (gy1 - gy0)
    _, ghpwl, _ = sp_metrics(c, modules, blk2mod,
                             [m['ox'] - gx0 for m in modules],
                             [m['oy'] - gy0 for m in modules],
                             gx1 - gx0, gy1 - gy0)
    gref = (garea, ghpwl, c['agap'], c['hgap'])
    hw = 0.12 if n >= 116 else 0.06

    # Hold vrel at the greedy's level: this isolates the AREA+HPWL ceiling (the
    # question is whether global re-packing the SAME rectangles is denser). Raw SP
    # LB-packing drops boundary blocks off their edges (bv high) -- a separate
    # concern the constrained packer must handle; reported separately, not charged
    # into the est. So est here = optimistic area+hpwl upper bound.
    def evaluate(Gp, Gm):
        X, Y, W, H = pack_sp(mw, mh, Gp, Gm)
        area, hpwl, bv = sp_metrics(c, modules, blk2mod, X, Y, W, H)
        return est_cost(area, hpwl, c['vrel'], gref), area, hpwl, bv

    import os
    Gp, Gm = recover_sp(modules)
    if seed_mode == "shelf":          # sanity: anneal from a deliberately bad seed
        Gm = sorted(range(nm), key=lambda m: (modules[m]['oy'], modules[m]['ox']))
        Gp = sorted(range(nm), key=lambda m: (-modules[m]['oy'], modules[m]['ox']))
    c0, a0, h0, bv0 = evaluate(Gp, Gm)            # seed (greedy SP -> ~greedy)
    best = (c0, Gp[:], Gm[:]); bestm = (a0, h0, bv0)
    cur_c = c0; cur = (Gp[:], Gm[:])
    dbg = os.environ.get("SP_DEBUG")
    if mode in ("l2", "l1"):
        T0, T1 = max(0.02 * c0, 0.005), 0.0002
        win = max(2, nm // 8)         # local-move window
        for it in range(iters):
            frac = it / max(1, iters - 1)
            T = T0 * (T1 / T0) ** frac if mode == "l2" else 0.0   # l1 = hill-climb
            Gp2, Gm2 = cur[0][:], cur[1][:]
            mt = rng.random()
            a = rng.randrange(nm)
            if rng.random() < 0.7:                 # local move (adjacent-ish)
                b = min(max(a + rng.randint(-win, win), 0), nm - 1)
            else:
                b = rng.randrange(nm)              # global move
            if a == b:
                continue
            if mt < 0.4:                           # swap in Gp only
                Gp2[a], Gp2[b] = Gp2[b], Gp2[a]
            elif mt < 0.8:                         # swap in Gm only
                Gm2[a], Gm2[b] = Gm2[b], Gm2[a]
            else:                                  # swap in both
                Gp2[a], Gp2[b] = Gp2[b], Gp2[a]
                Gm2[a], Gm2[b] = Gm2[b], Gm2[a]
            cc, aa, hh, bb = evaluate(Gp2, Gm2)
            if cc < cur_c or (T > 0 and rng.random() < math.exp((cur_c - cc) / T)):
                cur_c = cc; cur = (Gp2, Gm2)
                if cc < best[0]:
                    best = (cc, Gp2[:], Gm2[:]); bestm = (aa, hh, bb)
            if dbg and (it + 1) % max(1, iters // 5) == 0:
                print(f"    [{c['idx']} it{it+1}] best={best[0]:.4f} "
                      f"area={bestm[0]:.0f} cur={cur_c:.4f} T={T:.4f}")
    gcost = c['cost']                 # greedy true cost from JSON
    if verbose:
        print(f"case {c['idx']:>3} n={n:>3} mods={nm:>3}  "
              f"greedy cost={gcost:.4f} area={garea:.0f} hpwl={ghpwl:.0f}  | "
              f"SP-of-greedy est={c0:.4f}(bv{bv0}) | "
              f"best est={best[0]:.4f} area={bestm[0]:.0f}"
              f"({bestm[0]/garea:.3f}x) hpwl={bestm[1]:.0f}"
              f"({bestm[1]/ghpwl:.3f}x) bv={bestm[2]}")
    return dict(idx=c['idx'], n=n, gcost=gcost, sp_greedy=c0,
                best=best[0], area_ratio=bestm[0] / garea,
                hpwl_ratio=bestm[1] / ghpwl, bv=bestm[2])


def run(ids, mode):
    import os
    ev, pj = load_cases()
    ids = ids or TARGETS
    iters = int(os.environ.get("SP_ITERS", "4000"))
    seed_mode = os.environ.get("SP_SEEDMODE", "greedy")
    rows = []
    for idx in ids:
        c = case_data(ev, pj, idx)
        rows.append(run_case(c, mode, iters=iters, seed_mode=seed_mode))
    print()
    nbeat = sum(1 for r in rows if r['best'] < r['gcost'] - 1e-4)
    print(f"[{mode}] {nbeat}/{len(rows)} cases: best-est < greedy-cost  "
          f"(RELAXED upper bound: preplaced unpinned, vrel optimistic)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "describe"
    ids = [int(a) for a in sys.argv[2:]] if len(sys.argv) > 2 else None
    if mode == "describe":
        # default: targets + hard, plus an all-100 aggregate line
        if ids is None:
            describe(TARGETS + HARD)
            print()
            describe(None, verbose=False)
        else:
            describe(ids)
    elif mode in ("sp", "l2", "l1"):
        run(ids, mode)
    else:
        print(f"mode '{mode}' not implemented yet")


if __name__ == "__main__":
    main()
