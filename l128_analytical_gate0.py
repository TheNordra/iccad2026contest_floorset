"""L128 Gate 0 — is the analytical floorplanner's PREMISE true?

OFFLINE PROBE — never shipped. Uses fp_sol as an ORACLE INPUT only (same class as
M26 oracle-perm, M68 ML-seed, M79 shape oracle); nothing is trained on it.

WHY THIS BEFORE A SOLVER. Three perfect-information bounds say the bottleneck is
not what we feed the packer -- perfect ORDER +0.005% (M26), perfect SEED +0.001%
(M68), perfect SHAPE +0.099% (M79) -- while the gap to the label floor is 10.4%
(1.2368 -> 1.1079). By elimination the deficit is the TOPOLOGY, which is exactly
what an analytical (continuous, gradient-based) floorplanner would attack. That is
a good reason to want one. It is not evidence that one would work, and the
pipeline it needs has a step nobody has measured:

    analytical solve  ->  continuous, overlapping positions
                      ->  extract a topology
                      ->  LEGALISE into non-overlapping rectangles   <- unmeasured
                      ->  our post-processing

The teammate measured the analogous step through THEIR packer and it was a cliff
(oracle_pack_ceiling: a full-marks answer legalised to 3.7518; zero slack put 42
cases at 1.017 each but left 58% unplaceable). Nobody has measured it through OUR
constraint-graph LP, which is a different legaliser and a much better one for this
job. This file measures it, and it needs NO new solver:

  * `build_and_solve` DERIVES its topology from whatever positions it is handed
    (:2140 picks the max-gap relation per pair and freezes it);
  * its separation rows are exact non-overlap in the NEW coordinates
    (`x_i + d_i + w_i + dw_i <= x_j + d_j`), so an OVERLAPPING input is legalised
    rather than preserved -- the rhs is the current gap, negative when they
    overlap, and the row forces it open;
  * its HPWL objective is EXACT (aux column per (edge,axis) with two rows giving
    a true absolute value), not a linearisation. Only the AREA constraint is
    linearised, which is what rho=0.06 bounds.

ARMS
  calib    label positions verbatim -> official cost. The floor, ~1.1079, and a
           check that this harness agrees with m79_shape_oracle_probe.py calib.
  ours     the shipped anchor json -> official cost. Must reproduce
           1.2367916697725434 exactly, or the harness is wrong.
  topolab  label topology + LABEL shapes -> LP -> official cost. Pipeline sanity:
           the LP must not destroy a layout that is already at the floor.
  topo     THE GATE. label arrangement + OUR shapes -> LP legalises -> official
           cost. This is the analytical pipeline with a perfect analytical stage.

READING THE GATE. `topo` near 1.11-1.15 => the topology transfers through our
legaliser, the deficit really is topological, and an analytical solver is worth
building. `topo` near 1.2368 or worse => a perfect topology does not survive
legalisation into our reachable set, and the route is dead for the same reason
M27 closed the packer rewrite -- which would save the remaining days.

⚠️ The LP objective here is the SHIPPED baseline-free one (own hpwl, structural
floor for area), not the label-derived one -- see `lp-baseline-is-label-derived`.
A ceiling probe is allowed label inputs, but using a label-derived OBJECTIVE would
inflate the ceiling with something the deployed path cannot have.

Run:
  <python> -u l128_analytical_gate0.py calib
  <python> -u l128_analytical_gate0.py topo --iters 4
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

# NOT `import m53_l3_probe`: the wrapper's `l3` is an internal shim class
# (optimizer_constructive.py:1806), so the LP reads oc.l3.CASES -- a separate
# module import would populate a store nothing ever looks at.
l3 = oc.l3

ANCHOR = _DIR / "results_L114_48c_lp_anchor.json"
ANCHOR_TOTAL = 1.2367916697725434
LABEL_FLOOR = 1.1079            # fp_sol verbatim, recorded in CLAUDE.md

print("[l128] loading dataset ...", flush=True)
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
TOTW = sum(c["w"] for c in CASES)


def label_pos(c):
    """fp_sol verbatim as (x, y, w, h) per block."""
    return [tuple(float(v) for v in c["tp"][i]) for i in range(c["n"])]


def our_pos(c):
    j = json.load(open(ANCHOR))
    raise RuntimeError("use OURS, loaded once")   # pragma: no cover


_OURS = {r["test_id"]: [tuple(float(v) for v in p) for p in r["positions"]]
         for r in json.load(open(ANCHOR))["test_results"]}


def cost_of(c, P):
    m = evaluate_solution({"positions": [list(p) for p in P], "runtime": 1.0},
                          c["base"], c["cons"][:c["n"]], c["b2b"], c["p2b"],
                          c["pins"], c["at"][:c["n"]],
                          target_positions=c["tp"][:c["n"]], median_runtime=1.0)
    return float(m.cost), bool(m.is_feasible)


def weighted(per):
    return (sum(CASES[i]["w"] * v for i, v in per.items())
            / sum(CASES[i]["w"] for i in per))


# Exactly the wrapper's own split (optimizer_constructive._hard_masks):
#   cn[1] != 0            -> PREPLACED: position frozen
#   cn[0] != 0 or cn[1]   -> FIXED:     shape frozen (a fixed block may still move)
# Conflating the two is a real trap -- it reads as "pinned block moved" on layouts
# the official evaluator calls feasible.
def _pre(c, i):
    return int(c["cons"][i][1]) != 0


def _shape_frozen(c, i):
    cn = c["cons"][i]
    return int(cn[0]) != 0 or int(cn[1]) != 0


def seed_layout(c, scale, blend=1.0, free_only=True):
    """The label's ARRANGEMENT carrying OUR shapes.

    Pinned blocks keep their exact required position and size (moving them is an
    automatic hard-constraint failure, and in the label they are already there).
    Movable blocks take our width/height at the label's centre, optionally scaled
    about the label bbox centre -- our layouts run ~17% larger in bbox area
    (utilisation 82.2% vs the label's 96.6%, L95), so at scale 1.0 our blocks
    overlap heavily and the LP has to open every gap at once.
    """
    L, O = label_pos(c), _OURS[c["idx"]]
    xs = [p[0] for p in L]
    ys = [p[1] for p in L]
    x1 = max(p[0] + p[2] for p in L)
    y1 = max(p[1] + p[3] for p in L)
    cx, cy = (min(xs) + x1) / 2.0, (min(ys) + y1) / 2.0
    if scale == "auto":
        la = (x1 - min(xs)) * (y1 - min(ys))
        oa = ((max(p[0] + p[2] for p in O) - min(p[0] for p in O))
              * (max(p[1] + p[3] for p in O) - min(p[1] for p in O)))
        s = math.sqrt(max(oa, 1e-12) / max(la, 1e-12))
    else:
        s = float(scale)
    out = []
    for i in range(c["n"]):
        if _pre(c, i):
            out.append(L[i])              # required position, cannot move
            continue
        # Blend the ASPECT from the label's to ours at constant area (areas are
        # identical to 0 ulp on every movable soft block -- M79). t=0 is the
        # label's own shape, t=1 is ours; the curve between them is how much
        # shape change the label's topology can absorb.
        lw, lh, ow, oh = L[i][2], L[i][3], O[i][2], O[i][3]
        # 🚨 A CLUSTERED block may NOT take a different shape here. decompose()
        # makes each cluster ONE RIGID UNIT whose member offsets come from this
        # seed, and separation rows skip intra-unit pairs (`if ui == uj:
        # continue`). Resizing a member without re-packing the cluster interior
        # therefore bakes an overlap in that nothing can ever fix -- it reads as
        # "the topology cannot absorb shape change" when it is really this
        # harness. Re-packing a cluster interior is `make_group_item`'s job, in
        # C++. So the shape question is asked only of blocks that are their own
        # unit, and the answer is scoped to them.
        if free_only and int(c["cons"][i][3]) != 0:
            w, h = lw, lh
            out.append((L[i][0], L[i][1], w, h))
            continue
        if blend >= 1.0 or _shape_frozen(c, i) or lw <= 0 or lh <= 0 \
                or ow <= 0 or oh <= 0:
            w, h = ow, oh
        elif blend <= 0.0:
            w, h = lw, lh
        else:
            A = ow * oh
            r = math.exp((1 - blend) * math.log(lw / lh)
                         + blend * math.log(ow / oh))
            w = math.sqrt(A * r)
            h = A / w
        lcx = L[i][0] + L[i][2] / 2.0
        lcy = L[i][1] + L[i][3] / 2.0
        ncx, ncy = cx + (lcx - cx) * s, cy + (lcy - cy) * s
        out.append((ncx - w / 2.0, ncy - h / 2.0, w, h))
    return out


def legalise(c, P0, iters, rho):
    """Hand P0 to the shipped LP and let it derive a topology and legalise.

    Baseline-free objective, exactly as `_shape_lp` builds it.
    """
    key = "l128"
    sumA = sum(max(0.0, float(c["at"][i])) for i in range(c["n"]))
    hp = oc._proxy_metrics(P0, c["at"], c["b2b"], c["p2b"], c["pins"],
                           c["cons"], c["n"])["hpwl"]
    base = {"hpwl_baseline": max(float(hp), 1e-6),
            "area_baseline": max(sumA / oc._LP_UTIL, 1e-6)}
    l3.CASES[key] = oc._lp_build_case(c["n"], c["at"], c["b2b"], c["p2b"],
                                      c["pins"], c["cons"], base)
    saved, oc.PRUNE_B = oc.PRUNE_B, None      # exact: no HPWL pruning in a probe
    P, status = P0, "start"
    try:
        for it in range(iters):
            newP, tele, _B = oc.lp_pass(key, P, rho, sep_trim=False)
            status = tele["status"]
            if newP is None:
                break
            P = newP
    finally:
        oc.PRUNE_B = saved
        l3.CASES.pop(key, None)
        oc._HARD_MASKS.pop(key, None)
    return P, status


def _legal(c, P):
    """Overlap / area / pinned check, independent of the evaluator."""
    A = np.asarray(P, dtype=float)
    n = c["n"]
    if np.any(A[:, 2] <= 0) or np.any(A[:, 3] <= 0):
        return False, "nonpositive dims"
    L = np.asarray(label_pos(c), dtype=float)
    for i in range(n):
        if _pre(c, i) and (abs(A[i, 0] - L[i, 0]) > 1e-9
                           or abs(A[i, 1] - L[i, 1]) > 1e-9):
            return False, f"preplaced block {i} moved"
    at = np.asarray([float(c["at"][i]) for i in range(n)])
    soft = np.array([not _shape_frozen(c, i) for i in range(n)])
    if np.any(soft & (np.abs(A[:, 2] * A[:, 3] - at) > 0.01 * at)):
        return False, "area band"
    x2, y2 = A[:, 0] + A[:, 2], A[:, 1] + A[:, 3]
    ox = np.minimum(x2[:, None], x2[None, :]) - np.maximum(A[:, 0][:, None], A[:, 0][None, :])
    oy = np.minimum(y2[:, None], y2[None, :]) - np.maximum(A[:, 1][:, None], A[:, 1][None, :])
    bad = (ox > oc.EPS_OVL) & (oy > oc.EPS_OVL)
    np.fill_diagonal(bad, False)
    return (not bool(bad.any())), ("overlap" if bad.any() else "ok")


def run(a):
    cases = [c for c in CASES if c["n"] >= a.nmin]
    if a.limit:
        cases = cases[:a.limit]
    per, feas, fails = {}, 0, {}
    t0 = time.time()
    for k, c in enumerate(cases):
        if a.mode == "calib":
            P = label_pos(c)
        elif a.mode == "ours":
            P = _OURS[c["idx"]]
        elif a.mode == "topolab":
            P, st = legalise(c, label_pos(c), a.iters, a.rho)
            fails[c["idx"]] = st
        else:                                              # topo -- the gate
            P0 = seed_layout(c, a.scale, a.blend, not a.all_blocks)
            P, st = legalise(c, P0, a.iters, a.rho)
            fails[c["idx"]] = st
        ok, why = _legal(c, P)
        cost, f = cost_of(c, P)
        if not f:
            cost = 10.0
        else:
            feas += 1
        per[c["idx"]] = cost
        if a.verbose and (not ok or cost > 2.0):
            print(f"    case {c['idx']:>3} n={c['n']:>3} cost {cost:.4f} "
                  f"legal={ok}({why}) lp={fails.get(c['idx'], '-')}", flush=True)
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(cases)} ({time.time() - t0:.0f}s)", flush=True)
    tot = weighted(per)
    print(f"\n=== {a.mode}  scale={a.scale} iters={a.iters} rho={a.rho} ===")
    print(f"  cases           {len(cases)}   feasible {feas}/{len(cases)}")
    print(f"  weighted total  {tot:.9f}")
    print(f"  vs ours  ({ANCHOR_TOTAL:.6f})   {100 * (1 - tot / ANCHOR_TOTAL):+.4f}%")
    print(f"  vs floor ({LABEL_FLOOR:.4f})       "
          f"{100 * (tot / LABEL_FLOOR - 1):+.4f}% above")
    if fails:
        from collections import Counter
        print(f"  LP status: {dict(Counter(fails.values()))}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "ours", "topolab", "topo"])
    ap.add_argument("--scale", default="auto")
    ap.add_argument("--all-blocks", action="store_true",
                    help="also reshape CLUSTERED blocks (creates intra-unit "
                         "overlap the LP cannot fix -- see seed_layout)")
    ap.add_argument("--blend", type=float, default=1.0,
                    help="topo: 0 = label shapes, 1 = our shapes")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.06)
    ap.add_argument("--nmin", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
