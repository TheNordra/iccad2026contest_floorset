"""M59 pilot probe (OFFLINE, never shipped): REFINE rejected states -> L3 LP seeds.

M57_PLAN.md section 4. REFINE's per-pass c2 layouts (constructive.cpp refine loop)
are distinct separation topologies that get discarded after serving as guide.
The L2-seeds result proved "worse pre-LP, better post-LP" states exist (+0.19%);
this pilot asks whether REFINE's own rejected states are such seeds.

Pipeline (cases {62,65,85,88,89,91,97}, the hard-case list from the plan):
  1. winner host ksel per case looked up READ-ONLY from m53_l3_cache.pkl by
     mirroring mode_portfull's top-32 pre-LP proxy + post-LP proxy re-selection;
     asserted bit-exact against the anchor json cost.
  2. byte-gate: constructive_m59.exe (dump OFF) positions == cache ("run",ci,ksel)
     bit-exact -> the m46 copy is unperturbed.
  3. dump run: ICCAD_REFINE_DUMP=<file> -> per-frame pre-refine c1 (r=-1) and
     per-pass c2 states; stdout must stay bit-identical to the gate run.
  4. dedupe by L3 pair-relation signature (per-pair argmax separating axis,
     m53_l3_probe rule), drop states matching the anchor/pre-LP-run signatures,
     keep the min-layout_score representative per signature, cap 8 by sc.
  5. each seed -> 2-pass LP (--area semantics), official strict eval keep-guard
     (feasible AND strictly improving), best-over-seeds vs the anchor value.
Pilot does NOT re-run compaction/push on seeds (plan: direct LP only).

Anchor: results_L3_port_top32_area.json (offline L3 anchor, 1.3003478581).
Kill gate: weighted gain < 0.05% AND no significant single case -> RED
(stage 2 dual-guided relation flips is then dropped too).

LP code below is COPIED from m53_l3_probe.py (global rule: no import+patch of
existing probes). Cache: m59_cache.pkl (own file; m53 caches are read-only).
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

os.environ["ICCAD_L1_POOL"] = "1"        # BEFORE oc import (84-profile L1 pool)
os.environ["ICCAD_ADAPTIVE_POOL"] = "0"

from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from shapely.geometry import box as _sbox  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
import optimizer_constructive as oc  # noqa: E402
from optimizer_claude import _serialize_input, _parse_output  # noqa: E402
from proxy_analysis import build_opt_target_pos  # noqa: E402

EPS_BND = 1e-6
RH = 1.4
TARGET_CASES = (62, 65, 85, 88, 89, 91, 97)
CAP = 8
LP_ITERS = 2       # seed LP passes (plan section 4: --area, 2 passes)
L3_LP_ITERS = 8    # m53_l3_cache lp-entry key: portfull ran with --iters 8
ANCHOR_JSON = _DIR / "results_L3_port_top32_area.json"
L3_CACHE = _DIR / "m53_l3_cache.pkl"          # READ-ONLY
M59_CACHE = _DIR / "m59_cache.pkl"
EXE59 = str(_DIR / "constructive_m59.exe")
EXE_SHIPPED = str(_DIR / "constructive.exe")
SCRATCH = Path(os.environ.get(
    "M59_SCRATCH",
    r"C:\Users\Nordra\AppData\Local\Temp\claude"
    r"\C--Users-Nordra-Downloads-ICCAD2026-FloorSet-FloorSet"
    r"\46ab355a-60c3-435d-85c7-9ebf23e734d6\scratchpad"))

PROFILES = list(oc._PROFILES)
assert len(PROFILES) == 84, f"L1 pool expected 84 profiles, got {len(PROFILES)}"

# ── dataset (copied from m53_l3_probe.py) ────────────────────────────────────
print("[load] dataset ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
CASES, W = {}, {}
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _inp, _lab = _s["input"], _s["label"]
    _at, _b2b, _p2b, _pins, _cons = _inp
    _n = int((_at != -1).sum().item())
    W[_idx] = math.exp(_n / 12.0)
    _base, _tp = _ev._extract_baseline(_idx, _lab, _b2b, _p2b, _pins, _n)
    CASES[_idx] = dict(
        idx=_idx, n=_n, base=_base, tp=_tp, at=_at, b2b=_b2b, p2b=_p2b,
        pins=_pins, cons=_cons,
        b2l=[(int(e[0]), int(e[1]), float(e[2]))
             for e in _b2b.tolist() if int(e[0]) != -1],
        p2l=[(int(e[0]), int(e[1]), float(e[2]))
             for e in _p2b.tolist() if int(e[0]) != -1],
        pin=[(float(p[0]), float(p[1])) for p in _pins.tolist()],
        cn=[[int(v) for v in _cons[i].tolist()] for i in range(_n)],
    )
TOTW = sum(W.values())
print("[load] 100 cases ready", flush=True)


def cost_eval(ci, ps):
    """Official strict scoring (target_positions passed -> hard checks on)."""
    c = CASES[ci]
    return evaluate_solution(
        {"positions": ps, "runtime": 1.0}, c["base"], c["cons"][: c["n"]],
        c["b2b"], c["p2b"], c["pins"], c["at"][: c["n"]],
        target_positions=c["tp"][: c["n"]], median_runtime=1.0)


def comp_split(P, mem):
    """Cluster members -> connected components exactly as the evaluator sees
    them (unary_union geoms; zero tolerance)."""
    boxes = {i: _sbox(P[i][0], P[i][1], P[i][0] + P[i][2], P[i][1] + P[i][3])
             for i in mem}
    u = unary_union(list(boxes.values()))
    geoms = [u] if u.geom_type == "Polygon" else list(u.geoms)
    comps = [[] for _ in geoms]
    for i in mem:
        b = boxes[i]
        k = max(range(len(geoms)), key=lambda t: geoms[t].intersection(b).area)
        comps[k].append(i)
    return comps


# ── LP build + solve (one pass) — copied from m53_l3_probe.py ────────────────
def build_and_solve(ci, P, freeze_units, area_obj=False, force_bnd=()):
    c = CASES[ci]
    n, cn = c["n"], c["cn"]

    # units: rigid cluster components / free singles; preplaced (and any
    # component containing one) frozen
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

    # HPWL objective (aux linearization); intra-unit / all-frozen edges const.
    h_base = max(float(c["base"].get("hpwl_baseline", 1.0)), 1e-6)
    hw_scale = 0.5 / h_base
    cx = [P[i][0] + P[i][2] / 2.0 for i in range(n)]
    cy = [P[i][1] + P[i][3] / 2.0 for i in range(n)]
    const_h = 0.0
    obj0 = 0.0

    def edge_axis(t, ui, uj, off, dC):
        # t >= dC + d_i - d_j   and   t >= -(dC + d_i - d_j)
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

    # pairwise separation: keep the currently separating axis
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
            gap, ul, ur, off = max(cands, key=lambda t: t[0])
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

    for i in force_bnd:
        ui = unit_of[i]
        if ui is None:
            return None  # frozen: structurally unrepairable
        code = cn[i][4]
        x, y, w, h = P[i]
        if code & 1:
            add_eq([(ui, 1.0), (XMIN, -1.0)], -x)
        if code & 2:
            add_eq([(ui, 1.0), (XMAX, -1.0)], -(x + w))
        if code & 4:
            add_eq([(U + ui, 1.0), (YMAX, -1.0)], -(y + h))
        if code & 8:
            add_eq([(U + ui, 1.0), (YMIN, -1.0)], -y)

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
    # bbox never grows
    add_ub([(XMAX, 1.0), (XMIN, -1.0)], W0)
    add_ub([(YMAX, 1.0), (YMIN, -1.0)], H0)

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


def hpwl_of(ci, P):
    c = CASES[ci]
    tot = 0.0
    for i, j, w in c["b2l"]:
        tot += w * (abs((P[i][0] + P[i][2] / 2) - (P[j][0] + P[j][2] / 2))
                    + abs((P[i][1] + P[i][3] / 2) - (P[j][1] + P[j][3] / 2)))
    for p, i, w in c["p2l"]:
        px, py = c["pin"][p]
        tot += w * (abs((P[i][0] + P[i][2] / 2) - px)
                    + abs((P[i][1] + P[i][3] / 2) - py))
    return tot


def apply_deltas(P, units, dx, dy):
    newP = [tuple(p) for p in P]
    for u, mem in enumerate(units):
        ddx = 0.0 if abs(dx[u]) < 1e-12 else float(dx[u])
        ddy = 0.0 if abs(dy[u]) < 1e-12 else float(dy[u])
        if ddx == 0.0 and ddy == 0.0:
            continue
        for i in mem:
            x, y, w, h = P[i]
            newP[i] = (x + ddx, y + ddy, w, h)
    return newP


def lp_pass(ci, P, area_obj=False, force_bnd=()):
    """One LP pass with the cluster-precision re-solve ladder."""
    freeze = set()
    for attempt in range(3):
        B = build_and_solve(ci, P, freeze, area_obj=area_obj,
                            force_bnd=force_bnd)
        if B is None:
            return None, dict(status="force_frozen")
        if B["res"].status != 0:
            return None, dict(status=f"lp_status_{B['res'].status}")
        U = B["U"]
        x = B["res"].x
        newP = apply_deltas(P, B["units"], x[:U], x[U:2 * U])
        broken = [g for g, c0 in B["group_comp0"].items()
                  if len(comp_split(newP, [i for i in range(CASES[ci]["n"])
                                           if CASES[ci]["cn"][i][3] == g])) > c0]
        if not broken:
            return newP, dict(status="ok", U=U, bnd_skip=B["bnd_skip"],
                              pred_h=hpwl_of(ci, newP),
                              obj0=B["obj0"], attempts=attempt + 1,
                              frozen=len(freeze))
        for g in broken:
            freeze |= B["group_units"][g]
    return None, dict(status="cluster_break")


# ── M59-specific: signatures, winner lookup, dump run ────────────────────────
def pair_signature(P, n):
    """Per-pair argmax separating axis (mirrors the LP topology rule at
    m53_l3_probe.py:221-225, incl. first-max tie behaviour) -> bytes."""
    A = np.asarray(P, dtype=np.float64)[:n]
    X1, Y1 = A[:, 0], A[:, 1]
    X2, Y2 = A[:, 0] + A[:, 2], A[:, 1] + A[:, 3]
    G = np.stack((X1[None, :] - X2[:, None],    # 0: i left of j
                  X1[:, None] - X2[None, :],    # 1: j left of i
                  Y1[None, :] - Y2[:, None],    # 2: i below j
                  Y1[:, None] - Y2[None, :]))   # 3: j below i
    arg = np.argmax(G, axis=0).astype(np.uint8)
    iu = np.triu_indices(n, k=1)
    return arg[iu].tobytes()


def winner_host(ci, db, anch_cost):
    """Mirror mode_portfull's selection: top-32 by pre-LP proxy, post-LP proxy
    re-selection -> ksel. Asserted bit-exact against the anchor json."""
    c = CASES[ci]
    A_hat = 1.035 * max(sum(max(0.0, float(c["at"][i]))
                            for i in range(c["n"])), 1e-9)
    pm = {k: db[("pm", ci, k)] for k in range(84)}
    hmin = min(v[1] for v in pm.values()) or 1.0
    prox = {k: (pm[k][0] / A_hat + RH * pm[k][1] / hmin)
            * math.exp(2 * pm[k][2]) for k in pm}
    top = sorted(pm, key=lambda k: prox[k])[:32]
    res = {k: db[("lp", ci, k, L3_LP_ITERS, True)] for k in top}
    h2 = {k: res[k][1] for k in top}
    hmin2 = min(h2.values()) or 1.0
    ab = float(c["base"].get("area_baseline", 1.0))
    prox2 = {k: ((1 + res[k][2]) * ab / A_hat + RH * h2[k] / hmin2)
             * math.exp(2 * res[k][3]) for k in top}
    ksel = min(top, key=lambda k: prox2[k])
    assert res[ksel][0] == anch_cost, (
        f"case {ci}: cache ksel={ksel} lp cost {res[ksel][0]!r} != anchor "
        f"{anch_cost!r} (data drift)")
    return ksel


def run_exe(ci, k, dump_path=None):
    c = CASES[ci]
    otp = build_opt_target_pos(c["tp"], c["cons"], c["n"])
    txt = _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"], c["pins"],
                           c["cons"], otp, gnn_hint=None)
    env = dict(os.environ)
    env.update(PROFILES[k])
    if dump_path is not None:
        env["ICCAD_REFINE_DUMP"] = str(dump_path)
    r = subprocess.run([EXE59], input=txt, capture_output=True, text=True,
                       env=env, timeout=600)
    return _parse_output(r.stdout, c["n"])


def parse_dump(path, n):
    """-> list of dict(f, r, sc, imp, P) in file order."""
    states = []
    with open(path) as fh:
        line = fh.readline()
        while line:
            assert line.startswith("STATE "), f"bad dump line: {line!r}"
            kv = dict(tok.split("=") for tok in line.split()[1:])
            assert int(kv["n"]) == n, f"dump n={kv['n']} != case n={n}"
            P = []
            for _ in range(n):
                x, y, w, h = fh.readline().split()
                P.append((float(x), float(y), float(w), float(h)))
            states.append(dict(f=int(kv["f"]), r=int(kv["r"]),
                               sc=float(kv["sc"]), imp=int(kv["imp"]), P=P))
            line = fh.readline()
    return states


def lp_seed(ci, P):
    """Mirror mode_portfull's inner LP loop on one seed state."""
    m0 = cost_eval(ci, P)
    bestP, bestc, bestm = P, m0.cost, m0
    for _ in range(LP_ITERS):
        newP, tele = lp_pass(ci, P, area_obj=True)
        if newP is None:
            break
        m = cost_eval(ci, newP)
        if m.is_feasible and m.cost < bestc - 1e-12:
            bestP, bestc, bestm = newP, m.cost, m
            P = newP
        else:
            break
    return m0, bestP, bestc, bestm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None,
                    help="comma list override (default 62,65,85,88,89,91,97)")
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args()
    cases = ([int(x) for x in args.cases.split(",")] if args.cases
             else list(TARGET_CASES))

    aj = json.load(open(ANCHOR_JSON))
    anchor_total = aj["total_score"]
    ANCH = {t["test_id"]: t for t in aj["test_results"]}
    print(f"[anchor] {ANCHOR_JSON.name} total={anchor_total:.10f}", flush=True)

    l3 = pickle.load(open(L3_CACHE, "rb"))          # READ-ONLY
    sig_expect = repr((repr(PROFILES),
                       hashlib.md5(open(EXE_SHIPPED, "rb").read()).hexdigest()))
    assert l3.get("sig") == sig_expect, \
        "m53_l3_cache signature != current pool/exe -> data drift, abort"
    db = l3["db"]

    # own cache (resume): sig = m59 exe + profiles
    msig = repr((repr(PROFILES),
                 hashlib.md5(open(EXE59, "rb").read()).hexdigest()))
    m59 = {}
    if M59_CACHE.exists():
        try:
            _c = pickle.load(open(M59_CACHE, "rb"))
            if _c.get("sig") == msig:
                m59 = _c["db"]
            else:
                print("[cache] m59 signature mismatch -> reset", flush=True)
        except Exception:
            print("[cache] m59 unreadable -> reset", flush=True)

    def save():
        tmp = M59_CACHE.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"sig": msig, "db": m59}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, M59_CACHE)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for ci in cases:
        c = CASES[ci]
        n = c["n"]
        anch_cost = ANCH[ci]["cost"]
        anch_P = [tuple(p) for p in ANCH[ci]["positions"]]

        # LP-module sanity (gate0 semantics): strict eval reproduces the anchor
        m_a = cost_eval(ci, anch_P)
        assert m_a.cost == anch_cost, \
            f"case {ci}: strict eval {m_a.cost!r} != anchor {anch_cost!r}"

        ksel = winner_host(ci, db, anch_cost)
        runP = [tuple(p) for p in db[("run", ci, ksel)]]

        # byte-gate: dump-OFF run must reproduce the cached shipped positions
        gk = ("gate", ci, ksel)
        if gk not in m59:
            p_off = [tuple(p) for p in run_exe(ci, ksel)]
            assert p_off == runP, f"case {ci}: m59 exe (dump off) != cache run"
            m59[gk] = True
            save()
        # dump run: stdout must stay bit-identical
        dump_path = SCRATCH / f"m59_dump_{ci}.txt"
        p_on = [tuple(p) for p in run_exe(ci, ksel, dump_path=dump_path)]
        assert p_on == runP, f"case {ci}: dump run perturbed stdout"
        states = parse_dump(dump_path, n)

        # dedupe by pair-relation signature; drop anchor/pre-LP topologies
        excl = {pair_signature(anch_P, n), pair_signature(runP, n)}
        best_by_sig = {}
        for st in states:
            sg = pair_signature(st["P"], n)
            if sg in excl:
                continue
            if sg not in best_by_sig or st["sc"] < best_by_sig[sg]["sc"]:
                best_by_sig[sg] = st
        seeds = sorted(best_by_sig.items(), key=lambda kv: kv[1]["sc"])
        seeds = seeds[:args.cap]
        print(f"case {ci:3d} n={n:3d} host k={ksel} states={len(states)} "
              f"distinct={len(best_by_sig)} seeds={len(seeds)} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

        # LP each seed
        best_cost, best_P, best_m, best_tag = None, None, None, None
        pre_best = None
        for sg, st in seeds:
            kk = ("seed", ci, ksel,
                  hashlib.md5(sg).hexdigest(), LP_ITERS, True)
            if kk in m59:
                pre_c, post_c, feas, bP = m59[kk]
            else:
                m0, bP, post_c, bm = lp_seed(ci, [tuple(p) for p in st["P"]])
                pre_c, feas = m0.cost, bool(bm.is_feasible)
                m59[kk] = (pre_c, post_c, feas, bP)
                save()
            tag = f"f{st['f']}r{st['r']}"
            print(f"    seed {tag:8s} sc={st['sc']:.6g} pre {pre_c:.6f} "
                  f"-> post {post_c:.6f} feas={int(feas)}"
                  f"{'  *BEATS-ANCHOR*' if feas and post_c < anch_cost - 1e-12 else ''}",
                  flush=True)
            if pre_best is None or pre_c < pre_best:
                pre_best = pre_c
            if feas and (best_cost is None or post_c < best_cost):
                best_cost, best_P, best_tag = post_c, bP, tag
        d = (anch_cost - best_cost) if (best_cost is not None
                                        and best_cost < anch_cost - 1e-12) else 0.0
        wc = W[ci] * d / TOTW / anchor_total * 100
        rows.append(dict(ci=ci, n=n, ksel=ksel, states=len(states),
                         distinct=len(best_by_sig), seeds=len(seeds),
                         pre_best=pre_best, best=best_cost, best_tag=best_tag,
                         anchor=anch_cost, d=d, wc=wc,
                         P=best_P if d > 0 else None))
        print(f"case {ci:3d} anchor {anch_cost:.6f} bestLP "
              f"{best_cost if best_cost is not None else float('nan'):.6f} "
              f"d={d:+.6f} wContr={wc:+.4f}%", flush=True)

    gain = sum(r["wc"] for r in rows)
    nimp = sum(1 for r in rows if r["d"] > 0)
    print(f"\n== M59 pilot: {nimp}/{len(rows)} cases improved, weighted gain "
          f"{gain:+.4f}% of anchor {anchor_total:.10f} "
          f"({time.perf_counter() - t0:.0f}s) ==", flush=True)

    # results json: anchor copy, improved cases overwritten
    out = json.loads(json.dumps(aj))
    out["submission_name"] = "M59_refine_seed"
    out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    bytid = {t["test_id"]: t for t in out["test_results"]}
    for r in rows:
        if r["d"] > 0 and r["P"] is not None:
            m = cost_eval(r["ci"], [tuple(p) for p in r["P"]])
            bytid[r["ci"]].update(
                cost=m.cost, hpwl_gap=m.hpwl_gap, area_gap=m.area_gap,
                violations_relative=m.violations_relative,
                is_feasible=bool(m.is_feasible),
                positions=[list(p) for p in r["P"]])
    tot = sum(W[t["test_id"]] * t["cost"]
              for t in out["test_results"]) / TOTW
    out["total_score"] = tot
    out["summary"]["num_feasible"] = sum(
        t["is_feasible"] for t in out["test_results"])
    out["summary"]["avg_cost"] = sum(
        t["cost"] for t in out["test_results"]) / 100
    dump = _DIR / "results_M59_refine_seed.json"
    json.dump(out, open(dump, "w"))
    print(f"[dump] {dump.name}  total_score={tot:.10f}", flush=True)

    # machine-readable row table for the report
    json.dump([{k: v for k, v in r.items() if k != "P"} for r in rows],
              open(_DIR / "m59_rows.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
