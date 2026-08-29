"""L282 (OFFLINE, never shipped): relocate a unit OFF the critical chain.

L281 closed every topology edit that LENGTHENS the critical chain and left
exactly one thing open (L281 S10.1 item 2): the DUAL move -- take a unit off the
chain so the chain gets SHORTER, letting the LP shrink the bbox and pay down
area_gap.  It is feasible by construction where L281's moves were not: a shorter
chain fits in the same box.

Gate 0 (`l282_chain_gate.py`, no LP, all 100 graded cases) says the redundancy
is real but the bracket is wide:

    removing the best single unit from the binding chain
      optimistic  (other axis absorbs it for free)      +0.6282 %
      pessimistic (it lands on the other critical path) -3.1957 %
      chain shortening as % of the row: p50 0.82 %, p90 5.64 %, max 14.57 %
      binding floor is the CHAIN in 93/100 cases (frozen span in only 7)

So the sign is decided by WHERE the unit goes, which is what this measures.

DIFFERENCE FROM L281 -- same machinery, two changes:
  * candidate units are the ones ON the binding critical chain, not the ones
    with the largest wire prize;
  * targets are ranked by the PREDICTED NEW BBOX AREA, computed exactly from
    the new constraint graphs, not by wire.
Everything else -- force_rel semantics, the certificate, the LP, the official
strict scorer, and a control given the identical LP and polish budget -- is
L281's and is imported rather than reimplemented.

modes:
  probe    run cases
  report   aggregate
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import m53_l3_probe as m53                                        # noqa: E402
import m64_flip_probe as m64                                      # noqa: E402
import l281_reloc_probe as L                                      # noqa: E402

CASES, W = L.CASES, L.W
cost_eval = L.cost_eval
MIRROR = L.MIRROR
PROBE_VERSION = "l282-v1"

ANCH, SIG, DB = None, None, None
_dirty = 0


# -- lifted verbatim from l282_chain_gate.py (that file runs a 100-case
#    analysis at module level, so it must not be imported) ---------------
def chain_excl(n, edges, wt, excl):
    """Longest node-weighted path with the nodes in `excl` deleted."""
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a in excl or b in excl:
            continue
        adj[a].append(b)
        indeg[b] += 1
    q = [i for i in range(n) if indeg[i] == 0 and i not in excl]
    dist = [0.0 if i in excl else wt[i] for i in range(n)]
    head, seen = 0, 0
    while head < len(q):
        a = q[head]
        head += 1
        seen += 1
        da = dist[a]
        for b in adj[a]:
            if da + wt[b] > dist[b]:
                dist[b] = da + wt[b]
            indeg[b] -= 1
            if indeg[b] == 0:
                q.append(b)
    if seen != n - len(excl):
        return float("inf")
    return max((dist[i] for i in range(n) if i not in excl), default=0.0)


def critical_nodes(n, edges, wt):
    """Nodes lying on SOME longest path (zero-slack nodes)."""
    adj, radj, indeg = [[] for _ in range(n)], [[] for _ in range(n)], [0] * n
    for a, b in edges:
        adj[a].append(b)
        radj[b].append(a)
        indeg[b] += 1
    q = [i for i in range(n) if indeg[i] == 0]
    up = [wt[i] for i in range(n)]
    order, head = [], 0
    while head < len(q):
        a = q[head]
        head += 1
        order.append(a)
        for b in adj[a]:
            if up[a] + wt[b] > up[b]:
                up[b] = up[a] + wt[b]
            indeg[b] -= 1
            if indeg[b] == 0:
                q.append(b)
    if len(order) != n:
        return None, None
    down = [0.0] * n
    for a in reversed(order):
        best = 0.0
        for b in adj[a]:
            if down[b] + wt[b] > best:
                best = down[b] + wt[b]
        down[a] = best
    Lm = max(up[i] + down[i] for i in range(n))
    return Lm, [i for i in range(n) if up[i] + down[i] > Lm - 1e-9]


def save_db():
    global _dirty
    tmp = _DIR / "l282_cache.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dict(sig=SIG, db=DB), f)
    os.replace(tmp, _DIR / "l282_cache.pkl")
    _dirty = 0


def put(k, v):
    global _dirty
    DB[k] = v
    _dirty += 1
    if _dirty >= 20:
        save_db()


def fast_unit_edges(n, unit_of, ukey, ku, mem, fr):
    """L281.unit_edges restricted to pairs that involve ku -- O(|u| * n)
    instead of O(n^2), which is what makes ranking hundreds of targets cheap."""
    EH, EV = [], []
    us = set(mem[ku])
    for i in us:
        ki = ukey[i]
        for j in range(n):
            if j in us or unit_of[i] == unit_of[j]:
                continue
            kj = ukey[j]
            pk = (ki, kj) if ki <= kj else (kj, ki)
            kc = fr.get(pk)
            if kc is None:
                continue
            k = kc if ki <= kj else MIRROR[kc]
            a, b = (i, j) if k in (0, 2) else (j, i)
            (EH if k < 2 else EV).append((a, b))
    return EH, EV


def gen_chain_targets(ku, box, keys, bb, nslot):
    """Diverse abutment slots: for every other unit, the four positions that
    put u immediately right / left / above / below it, with the free coordinate
    aligned to that unit.  Not wire-driven -- this move is about geometry."""
    xmin0, xmax0, ymin0, ymax0 = bb
    ux0, ux1, uy0, uy1 = box[ku]
    ex, ey = ux1 - ux0, uy1 - uy0
    if ex > xmax0 - xmin0 + 1e-9 or ey > ymax0 - ymin0 + 1e-9:
        return []

    def cx(x):
        return min(max(x, xmin0), xmax0 - ex)

    def cy(y):
        return min(max(y, ymin0), ymax0 - ey)

    out = []
    for kv in keys:
        if kv == ku:
            continue
        vx0, vx1, vy0, vy1 = box[kv]
        out.append((cx(vx1), cy(vy0)))
        out.append((cx(vx0 - ex), cy(vy0)))
        out.append((cx(vx0), cy(vy1)))
        out.append((cx(vx0), cy(vy0 - ey)))
    seen, ded = set(), []
    for x, y in out:
        k = (round(x, 9), round(y, 9))
        if k in seen:
            continue
        seen.add(k)
        ded.append((x, y))
    return ded[:nslot] if nslot and len(ded) > nslot else ded


def probe_case(ci, nunits, ncand, nslot, polish_iters):
    ent = ANCH[ci]
    P0 = [tuple(p) for p in ent["positions"]]
    c0 = ent["cost"]
    n = CASES[ci]["n"]
    if ("gate", ci) not in DB:
        m = cost_eval(ci, P0)
        if m.cost != c0:
            raise SystemExit(f"anchor gate FAIL case {ci}")
        put(("gate", ci), True)

    ck = ("ctrlp", ci, polish_iters)
    if ck not in DB:
        r = L.reloc_solve(ci, P0, None)
        if r["status"] == "ok" and r["feas"]:
            bc, _bP = L.polish(ci, r["newP"], r["cost"], polish_iters)
            put(ck, dict(cost=bc, feas=True))
        else:
            put(ck, dict(cost=float("inf"), feas=False))
        save_db()
    cbase = min([c0] + ([DB[ck]["cost"]] if DB[ck]["feas"] else []))

    units, unit_of, ukey, box, mem = L.unit_geo(ci, P0)
    pin, bb = L.pinned_keys(ci, P0, units, unit_of)
    keys = sorted(box)
    EH, EV = L.base_graph(ci, P0, unit_of, ukey, None)
    wW = [P0[i][2] for i in range(n)]
    wH = [P0[i][3] for i in range(n)]
    W0, H0 = bb[1] - bb[0], bb[3] - bb[2]
    lH, cH = critical_nodes(n, EH, wW)
    lV, cV = critical_nodes(n, EV, wH)
    if lH is None:
        print(f"case {ci}: anchor topology cyclic?! skipped", flush=True)
        return []
    froX = [(P0[i][0], P0[i][0] + P0[i][2]) for i in range(n)
            if unit_of[i] is None]
    froY = [(P0[i][1], P0[i][1] + P0[i][3]) for i in range(n)
            if unit_of[i] is None]
    spanX = (max(b for _a, b in froX) - min(a for a, _b in froX)) if froX else 0.0
    spanY = (max(b for _a, b in froY) - min(a for a, _b in froY)) if froY else 0.0
    binding = "H" if (1.0 - lH / W0) <= (1.0 - lV / H0) else "V"
    crit = cH if binding == "H" else cV
    cand = [k for k in {ukey[i] for i in crit}
            if k[0] == "U" and k not in pin]
    # rank candidates by how much removing them shortens the binding chain
    ranked = []
    for k in cand:
        after = chain_excl(n, EH if binding == "H" else EV,
                           wW if binding == "H" else wH, set(mem[k]))
        ranked.append((after, k))
    ranked.sort()
    ranked = ranked[:nunits]

    t0 = time.perf_counter()
    area0 = W0 * H0
    nsolve, movers = 0, []
    for _after, ku in ranked:
        base_EH = [(a, b) for a, b in EH
                   if ukey[a] != ku and ukey[b] != ku]
        base_EV = [(a, b) for a, b in EV
                   if ukey[a] != ku and ukey[b] != ku]
        cur_fr, _g = L.induced_rel(ku, box[ku], box, keys)
        ex = box[ku][1] - box[ku][0]
        ey = box[ku][3] - box[ku][2]
        scored = []
        for x, y in gen_chain_targets(ku, box, keys, bb, nslot):
            fr, _gm = L.induced_rel(ku, (x, x + ex, y, y + ey), box, keys)
            if fr == cur_fr:
                continue
            if L.n_binding(ku, box[ku], fr, box, keys) <= 0:
                continue                     # vacuous: moves nothing (L281 S6)
            eh, ev = fast_unit_edges(n, unit_of, ukey, ku, mem, fr)
            okH, nlH = L.longest_chain(n, base_EH + eh, wW)
            if not okH:
                continue
            okV, nlV = L.longest_chain(n, base_EV + ev, wH)
            if not okV:
                continue
            rH, rV = max(nlH, spanX), max(nlV, spanY)
            if rH > W0 + 1e-9 or rV > H0 + 1e-9:
                continue                     # would not fit the current box
            scored.append((rH * rV, x, y, fr, rH, rV))
        scored.sort(key=lambda t: t[0])
        picks = [s for s in scored if s[0] < area0 - 1e-9][:ncand]
        for ic, (pa, x, y, fr, rH, rV) in enumerate(picks):
            key = ("rel", ci, ku, ic)
            if key not in DB:
                r = L.reloc_solve(ci, P0, fr)
                r["tgt"], r["pred_area"] = (x, y), pa
                r["pred_shrink"] = 1.0 - pa / area0
                if r["status"] == "ok" and r["feas"]:
                    bc, bP = L.polish(ci, r["newP"], r["cost"], polish_iters)
                    r["polished"] = bc
                    if bc < cbase - 1e-6:
                        r["positions"] = [list(p) for p in bP]
                r.pop("newP", None)
                put(key, r)
                nsolve += 1
            r = DB[key]
            if r["status"] == "ok":
                bst = min(r["cost"], r.get("polished", float("inf")))
                print(f"  c{ci} {ku} #{ic} pred-shrink "
                      f"{100 * r['pred_shrink']:5.2f}%  cost {r['cost']:.6f} "
                      f"pol {bst:.6f} d={cbase - bst:+.2e} "
                      f"feas={int(r['feas'])} ({r['dt']:.1f}s)", flush=True)
                if r["feas"] and cbase - bst > 1e-6:
                    movers.append((ku, ic, cbase - bst, bst))
            else:
                print(f"  c{ci} {ku} #{ic} pred-shrink "
                      f"{100 * r['pred_shrink']:5.2f}%  {r['status']} "
                      f"({r['dt']:.1f}s)", flush=True)
    save_db()
    print(f"case {ci:3d} n={n:3d} [{binding}] chain units {len(cand)}, "
          f"{nsolve} LP solves, {len(movers)} movers, base {cbase:.6f} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    for ku, ic, d, pc in movers:
        print(f"    MOVER {ku} #{ic} d={d:+.6f} -> {pc:.6f}", flush=True)
    return movers


def mode_report():
    hist, best, cbase = {}, {}, {}
    for ci in sorted({k[1] for k in DB if k[0] == "rel"}):
        cands = [ANCH[ci]["cost"]]
        for k, v in DB.items():
            if k[0] == "ctrlp" and k[1] == ci and v.get("feas"):
                cands.append(v["cost"])
        cbase[ci] = min(cands)
    preds, reals = [], []
    for k, v in DB.items():
        if k[0] != "rel":
            continue
        ci = k[1]
        st = v["status"]
        if st == "ok":
            cb = min(v["cost"], v.get("polished", float("inf")))
            d = cbase[ci] - cb
            st = ("mover" if v["feas"] and d > 1e-6 else
                  "worse" if v["feas"] else "ok_infeasible")
            if v["feas"]:
                preds.append(v["pred_shrink"])
                reals.append(d / cbase[ci])
            if v["feas"] and cb < best.get(ci, float("inf")):
                best[ci] = cb
        hist[st] = hist.get(st, 0) + 1
    tot = sum(hist.values()) or 1
    print("== L282 chain-shortening: status histogram ==")
    for s, c in sorted(hist.items(), key=lambda t: -t[1]):
        print(f"  {s:16s} {c:5d}   {100.0 * c / tot:5.1f} %")
    inf = sum(c for s, c in hist.items() if s == "ok_infeasible"
              or s.startswith("lp_status") or s in ("ladder_kill",
                                                    "cluster_break"))
    print(f"  LP-infeasible / infeasible-solution: {inf}/{tot} = "
          f"{100.0 * inf / tot:.1f} %")
    if preds:
        pr = sorted(preds)
        print(f"\n== predicted bbox shrink of the candidates that were solved "
              f"(n={len(pr)}) ==")
        print(f"   p25 {100 * pr[len(pr) // 4]:.3f} %   "
              f"p50 {100 * pr[len(pr) // 2]:.3f} %   "
              f"p75 {100 * pr[3 * len(pr) // 4]:.3f} %   "
              f"max {100 * pr[-1]:.3f} %")
        rs = sorted(reals)
        print(f"   realised cost delta: p25 {100 * rs[len(rs) // 4]:+.4f} %   "
              f"p50 {100 * rs[len(rs) // 2]:+.4f} %   "
              f"p75 {100 * rs[3 * len(rs) // 4]:+.4f} %   "
              f"best {100 * rs[-1]:+.4f} %")
    cases = sorted(cbase)
    wsum = sum(W[ci] for ci in cases) or 1e-9
    bt = sum(W[ci] * cbase[ci] for ci in cases) / wsum
    g = sum(W[ci] * max(0.0, cbase[ci] - best.get(ci, cbase[ci]))
            for ci in cases) / wsum
    print(f"\n== union-oracle over {len(cases)} cases (base {bt:.6f}) ==")
    print(f"   vs the polished control : {100.0 * g / bt:+.4f} %")
    print(f"   cases with any gain     : "
          f"{sum(1 for ci in cases if cbase[ci] - best.get(ci, cbase[ci]) > 1e-9)}"
          f"/{len(cases)}")
    for ci in cases:
        b = best.get(ci, cbase[ci])
        print(f"   case {ci:3d}: base {cbase[ci]:.6f} best {b:.6f}  "
              f"{100.0 * (cbase[ci] - b) / cbase[ci]:+.4f} %")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "report"])
    ap.add_argument("--anchor", default=str(_DIR / "results_L274_base_48c.json"))
    ap.add_argument("--cases", default="85,88,91")
    ap.add_argument("--units", type=int, default=6)
    ap.add_argument("--cand", type=int, default=6)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--polish-iters", type=int, default=3)
    args = ap.parse_args()

    raw = open(args.anchor, "rb").read()
    SIG = (hashlib.md5(raw).hexdigest(), PROBE_VERSION)
    _aj = json.loads(raw)
    ANCH = {t["test_id"]: t for t in _aj["test_results"]}
    L.ANCH = ANCH
    print(f"[anchor] {Path(args.anchor).name} "
          f"total={_aj['total_score']:.10f}", flush=True)

    DB = {}
    _cp = _DIR / "l282_cache.pkl"
    if _cp.exists():
        try:
            _d = pickle.load(open(_cp, "rb"))
        except Exception:
            _d = None
        if _d and _d.get("sig") == SIG:
            DB = _d["db"]
            print(f"[cache] resume {len(DB)} entries", flush=True)

    if args.mode == "report":
        mode_report()
    else:
        allm = []
        for ci in [int(x) for x in args.cases.split(",")]:
            allm += probe_case(ci, args.units, args.cand, args.slots,
                               args.polish_iters)
        save_db()
        print(f"\n== probe done: {len(allm)} movers ==", flush=True)
