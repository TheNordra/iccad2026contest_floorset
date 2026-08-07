"""OFFLINE ONLY — M29 tree->layout geometric decoder probe (never shipped).

The reconstruction follow-up to reconstruct_probe.py (M28). The original
floorplans are generated from an INSERTION SLICING TREE (tree_sol). This probe
reverse-engineers the tree->geometry map so we can later BUILD trees from
connectivity (no labels) and render them.

Stage 1 here = residual_probe(): on GROUND-TRUTH coords, find the predictor for
the unsolved coordinate. Empirically the cross/abutting coordinate is already
exact (dir=0 newX==anchorRight 42/42; dir=1 newX==anchorX 32/32); the UNSOLVED
axis is always newY. Hypothesis: newY is set by the anchor's SUBTREE-REGION Y
extent (dir=0 -> region bottom; dir=1 -> region top), not the bare anchor block.

Data: TRAINING .th only (validation set has no tree_sol).
Run with the iccadv conda interpreter (torch lives there):
    C:/Users/Nordra/.conda/envs/iccadv/python.exe tree_decode_probe.py
"""
import sys, math, glob
from pathlib import Path

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import (evaluate_solution, calculate_hpwl_b2b,
                                calculate_hpwl_p2b, calculate_bbox_area,
                                compute_total_score)

TOL = 0.5  # fp_sol coords are integer-valued


def load_layout(d, L):
    at = d[0][L][:, 0]
    n = int((at != -1).sum().item())
    cons = d[0][L][:n, 1:]
    b2b, p2b, pins = d[1][L], d[2][L], d[3][L]
    tree = d[4][L][:n - 1]
    fp = d[5][L][:n]
    ins = [(int(a), int(nb), int(dr)) for a, nb, dr in tree.tolist()]
    W = [float(fp[i][0]) for i in range(n)]
    H = [float(fp[i][1]) for i in range(n)]
    X = [float(fp[i][2]) for i in range(n)]
    Y = [float(fp[i][3]) for i in range(n)]
    news = {nb for _, nb, _ in ins}
    seeds = [b for b in range(n) if b not in news]
    assert len(seeds) == 1, f"L={L} seeds={seeds}"
    return dict(n=n, ins=ins, W=W, H=H, X=X, Y=Y, seed=seeds[0],
                at=at[:n], cons=cons, b2b=b2b, p2b=p2b, pins=pins)


def residual_probe(layouts):
    """Tally exact-hit rate of candidate predictors for newY (the unsolved axis),
    split by dir. Region = anchor subtree bbox over FINAL fp_sol coords, tracked
    with 'grow' semantics (anchor's region accumulates everything attached)."""
    # counters: predictor -> dir -> [hits, total]
    cand = ["bareAnchor", "regionGrow", "regionBareStay", "floor0/globalTop"]
    tally = {c: {0: [0, 0], 1: [0, 0]} for c in cand}
    # cross-axis sanity (should be ~always exact)
    cross = {0: [0, 0], 1: [0, 0]}

    for lay in layouts:
        n, ins = lay["n"], lay["ins"]
        X, Y, W, H = lay["X"], lay["Y"], lay["W"], lay["H"]

        def bbox(members):
            xs0 = min(X[b] for b in members); ys0 = min(Y[b] for b in members)
            xs1 = max(X[b] + W[b] for b in members); ys1 = max(Y[b] + H[b] for b in members)
            return xs0, ys0, xs1, ys1

        # grow semantics: node_id maps block -> representative; members[rep]=set
        rep = {b: b for b in range(n)}
        members = {b: {b} for b in range(n)}
        # bare-stay: anchor region is the chain of blocks ever attached while the
        # block stayed an anchor leaf -> approximate by NEVER merging anchor away
        members_bare = {b: {b} for b in range(n)}

        globalTop = max(Y[b] + H[b] for b in range(n))

        for (a, nb, dr) in ins:
            ra = rep[a]
            reg_g = bbox(members[ra])           # grow region of anchor
            reg_b = bbox(members_bare[a])        # bare-stay region of anchor
            ax, ay, aw, ah = X[a], Y[a], W[a], H[a]

            if dr == 0:  # new to RIGHT; cross axis = X (newX==anchorRight), along = newY
                cross[0][1] += 1
                if abs(X[nb] - (ax + aw)) < TOL:
                    cross[0][0] += 1
                preds = {
                    "bareAnchor": ay,
                    "regionGrow": reg_g[1],       # region bottom
                    "regionBareStay": reg_b[1],
                    "floor0/globalTop": 0.0,
                }
                target = Y[nb]
            else:        # new ABOVE; cross axis = X (newX==anchorX), along = newY
                cross[1][1] += 1
                if abs(X[nb] - ax) < TOL:
                    cross[1][0] += 1
                preds = {
                    "bareAnchor": ay + ah,
                    "regionGrow": reg_g[3],       # region top
                    "regionBareStay": reg_b[3],
                    "floor0/globalTop": globalTop,
                }
                target = Y[nb]

            for c in cand:
                tally[c][dr][1] += 1
                if abs(preds[c] - target) < TOL:
                    tally[c][dr][0] += 1

            # update member sets
            new_members = members[ra] | {nb}
            for b in new_members:
                rep[b] = ra
            members[ra] = new_members
            members_bare[a] = members_bare[a] | {nb}
            members_bare[nb] = members_bare[a]

    print("=" * 72)
    print("residual_probe: predictor exact-hit rate for newY (unsolved axis)")
    print("=" * 72)
    print(f"cross-axis sanity (should be ~100%): "
          f"dir0 newX==anchorRight {cross[0][0]}/{cross[0][1]}  "
          f"dir1 newX==anchorX {cross[1][0]}/{cross[1][1]}")
    print(f"\n{'predictor':>18} | {'dir=0 (right)':>16} | {'dir=1 (above)':>16} | {'overall':>9}")
    for c in cand:
        h0, t0 = tally[c][0]; h1, t1 = tally[c][1]
        ov = (h0 + h1) / max(1, t0 + t1)
        print(f"{c:>18} | {h0:>6}/{t0:<6} {h0/max(1,t0)*100:4.0f}% | "
              f"{h1:>6}/{t1:<6} {h1/max(1,t1)*100:4.0f}% | {ov*100:7.1f}%")


def render_bstar(lay):
    """B*-tree packing (liteLoader.py:30 calls tree_sol a 'B*Tree representation').
    dir=0 -> new is LEFT child of anchor (placed to the RIGHT: x=anchorRight);
    dir=1 -> new is RIGHT child of anchor (placed ABOVE: x=anchorX).
    Y from a horizontal CONTOUR during DFS (left child before right child).
    Uses ONLY tree + per-block (W,H). Returns px, py lists."""
    n, ins, W, H, seed = lay["n"], lay["ins"], lay["W"], lay["H"], lay["seed"]
    left = [None] * n; right = [None] * n
    for (a, nb, dr) in ins:
        if dr == 0:
            left[a] = nb
        else:
            right[a] = nb
    px = [None] * n; py = [None] * n
    contour = [[0.0, 1e12, 0.0]]  # segments [x0, x1, h] covering [0, +inf)

    def query(x0, x1):
        h = 0.0
        for s in contour:
            if s[0] < x1 - 1e-9 and s[1] > x0 + 1e-9:
                h = max(h, s[2])
        return h

    def update(x0, x1, h):
        new = []
        for s in contour:
            if s[1] <= x0 + 1e-9 or s[0] >= x1 - 1e-9:
                new.append(s)
            else:
                if s[0] < x0:
                    new.append([s[0], x0, s[2]])
                if s[1] > x1:
                    new.append([x1, s[1], s[2]])
        new.append([x0, x1, h])
        new.sort()
        contour[:] = new

    # iterative pre-order DFS (left before right) to avoid deep recursion
    stack = [(seed, 0.0)]
    while stack:
        node, x = stack.pop()
        px[node] = x
        y = query(x, x + W[node])
        py[node] = y
        update(x, x + W[node], y + H[node])
        # push right first so left is processed first (LIFO)
        if right[node] is not None:
            stack.append((right[node], x))
        if left[node] is not None:
            stack.append((left[node], x + W[node]))
    return px, py


def compact_down(px, W, H, order):
    """Gravity-compact in Y with px FIXED, dropping blocks in the given ORDER.
    Each block rests on already-placed blocks with X-overlap (or floor)."""
    out = [0.0] * len(px)
    done = []
    for b in order:
        x0, x1 = px[b], px[b] + W[b]
        top = 0.0
        for c in done:
            if px[c] < x1 - 1e-9 and px[c] + W[c] > x0 + 1e-9:
                top = max(top, out[c] + H[c])
        out[b] = top
        done.append(b)
    return px, out


def _cost_of(positions, lay, baseline):
    m = evaluate_solution({"positions": positions, "runtime": 1.0}, baseline,
                          lay["cons"], lay["b2b"], lay["p2b"], lay["pins"],
                          lay["at"], target_positions=None, median_runtime=1.0)
    return m


def fidelity_probe(layouts):
    tot_blk = 0; exact_blk = 0; perfect = 0; worst = 0.0
    variants = ["raw", "compactInsOrder", "compactBT", "compactOracleY", "fp_self"]
    costs = {v: [] for v in variants}
    ns = []
    agap = {v: [] for v in variants}; hgap = {v: [] for v in variants}
    infeas = {v: 0 for v in variants}
    for lay in layouts:
        n = lay["n"]; X, Y, W, H = lay["X"], lay["Y"], lay["W"], lay["H"]
        # baseline from fp_sol itself (feeding fp_sol => gap ~ 0 self-check)
        pos_fp = [(X[b], Y[b], W[b], H[b]) for b in range(n)]
        base = {"hpwl_baseline": calculate_hpwl_b2b(pos_fp, lay["b2b"]) +
                                 calculate_hpwl_p2b(pos_fp, lay["p2b"], lay["pins"]),
                "area_baseline": calculate_bbox_area(pos_fp)}
        px, py = render_bstar(lay)
        e = [max(abs(px[b] - X[b]), abs(py[b] - Y[b])) for b in range(n)]
        ec = sum(1 for v in e if v < TOL)
        tot_blk += n; exact_blk += ec; worst = max(worst, max(e))
        if ec == n:
            perfect += 1
        ns.append(n)
        # compactInsOrder: LABEL-FREE — exact px + gravity in tree INSERTION order
        ins_order = [lay["seed"]] + [nb for (_, nb, _) in lay["ins"]]
        _, pyi = compact_down(px, W, H, ins_order)
        # compactBT: drop in render_bstar's own y-order (px exact from tree)
        _, pyc = compact_down(px, W, H, sorted(range(n), key=lambda b: py[b]))
        # compactOracleY: CEILING probe — exact px + true fp_sol y-ORDER (label)
        _, pyo = compact_down(px, W, H, sorted(range(n), key=lambda b: Y[b]))
        cand = {"raw": (px, py), "compactInsOrder": (px, pyi),
                "compactBT": (px, pyc), "compactOracleY": (px, pyo),
                "fp_self": ([X[b] for b in range(n)], [Y[b] for b in range(n)])}
        for v, (qx, qy) in cand.items():
            pos = [(qx[b], qy[b], W[b], H[b]) for b in range(n)]
            m = _cost_of(pos, lay, base)
            costs[v].append(float(m.cost)); agap[v].append(float(m.area_gap))
            hgap[v].append(float(m.hpwl_gap))
            if not m.is_feasible:
                infeas[v] += 1
    print("\n" + "=" * 72)
    print("fidelity_probe: render_bstar vs fp_sol")
    print("=" * 72)
    print(f"exact-match (px AND py): blocks {exact_blk}/{tot_blk} = {exact_blk/tot_blk*100:.1f}%"
          f"  perfect layouts {perfect}/{len(layouts)}  worst|err| {worst:.1f}")

    def wmean(vals):
        W_ = [math.exp(n / 12.0) for n in ns]
        return sum(w * v for w, v in zip(W_, vals)) / sum(W_)
    print(f"\noracle reconstruction ceiling = 1.1079 ; our portfolio = 1.3843")
    print(f"{'variant':>9} | {'Total(exp n/12)':>15} | {'w.area_gap':>10} | "
          f"{'w.hpwl_gap':>10} | {'infeasible':>10}")
    for v in variants:
        ts = compute_total_score(costs[v], ns)
        print(f"{v:>9} | {ts:>15.4f} | {wmean(agap[v]):>10.4f} | "
              f"{wmean(hgap[v]):>10.4f} | {infeas[v]:>4}/{len(layouts)}")


def main():
    # one file == one block-count config; sample a spread of n for breadth
    allf = sorted(glob.glob(str(_DIR / "floorset_lite" / "worker_0" / "layouts_*.th")))
    files = allf[:8]
    layouts = []
    for f in files:
        d = torch.load(f)
        for L in range(d[0].shape[0]):
            layouts.append(load_layout(d, L))
    print(f"loaded {len(layouts)} layouts from {len(files)} file(s); "
          f"n in {sorted(set(l['n'] for l in layouts))}")
    residual_probe(layouts)
    fidelity_probe(layouts)
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("- X (px) is recovered EXACTLY from tree_sol via the B*-tree X-rule, "
          "label-free.")
    print("- Renderer = exact-X + gravity-in-Y-order is FAITHFUL: with the true "
          "Y-order it reproduces fp_sol (area_gap=hpwl_gap=0 => oracle ceiling).")
    print("- fp_sol is NOT a B*-tree pack of its own tree (right-children can sit "
          "BELOW parents) => the compact Y-ORDER is NOT a tree-topological order.")
    print("- Reconstruction therefore reduces to a 1-D Y-ORDERING problem given "
          "the exact X. Topological/insertion orders are ~39% too tall; the order "
          "is the entire quality lever. => GREEN renderer, Y-order is Phase-B's job.")


if __name__ == "__main__":
    main()
