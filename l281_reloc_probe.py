"""L281 relocation probe (OFFLINE, never shipped): move ONE unit to a different
place in the topology -- which re-derives EVERY pair relation involving that
unit at once -- then re-solve the constraint-graph LP and score it.

WHY THIS MOVE (HANDOFF_2026-08-27_RELOCATION S1.2 / L280 S5)
  M64 flipped one UNIT PAIR's separation relation and left the other ~3000-4900
  pairs on their anchor disjunct: 459/529 = 86.8 % LP-infeasible, 0 movers.  Its
  own HONEST-SCOPE note says the semantics may be self-inflicting.  A relocation
  is by construction a realisable topology, so the thesis is that a large part
  of that 86.8 % was the move, not the instance.

MOVE SEMANTICS (the design decision, stated so it can be attacked)
  A relocation is specified by a TARGET POSITION for unit u's bounding box, not
  by an ordinal in a 1-D ordering.  Reason: the anchor's relation set is the
  per-pair max-gap disjunct (m53_l3_probe.py:213-231), and that set is NOT in
  general a sequence pair -- the tournament "u left of v OR v below u" can carry
  3-cycles for a perfectly legal placement (A left B, B left C, A below C when
  the y-gap of (A,C) exceeds its x-gap).  So there is no ordinal to move in.  A
  target position induces a relation for every pair (u,v) directly, from ONE
  consistent geometric configuration -- which is what "coherent" has to mean.

  Unit-level relations are read off unit BOUNDING BOXES, the exact granularity
  force_rel applies at ("slide all of A past all of B"), so a witness at unit
  level satisfies every constituent block-pair row too.

THE CERTIFICATE (what makes this measurable rather than anecdotal)
  The forced LP is feasible only if, at BLOCK level, the horizontal constraint
  graph and the vertical constraint graph are both acyclic and their longest
  node-weighted chains fit inside the anchor bbox:
      chain i1 -> i2 -> ... -> ik horizontally  =>  sum(w) <= XMAX-XMIN <= W0
  Both are exact and cheap, so every candidate move is classified BEFORE any LP
  runs, and an LP infeasibility can be attributed:
      CYCLIC / OVERSIZED   the move is not a realisable topology  (self-inflicted)
      COHERENT but LP-infeasible   the instance really is that tight
  M64 could not make this distinction.  `census` mode runs the same certificate
  over M64's own move (single unit-pair flip) on the same anchor, so the two
  move semantics are compared on identical geometry.

CONTROL (the thing M64 did not need and this probe does)
  M64's anchor was a fixpoint of this same LP, so "anchor cost" was a fair
  baseline.  This probe re-anchors on the SHIPPED in-set-100 positions
  (results_L274_base_48c.json = the graded shape, L275's rule), and the shipped
  path's LP is a DIFFERENT LP.  So every case also gets a no-force LP pass, and
  relocation deltas are reported against that control as well as the anchor.
  Without it a gain is indistinguishable from "the research LP is better".

modes:
  gate     anchor reproduces json cost + forcing a unit's CURRENT topology over
           every one of its pairs is a bit-exact no-op
  census   NO LP: coherence certificate for relocation vs M64's single-pair flip
  probe    LP + official strict scorer on the coherent relocations
  report   histograms -> the infeasibility fork, then the cost distribution
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
import m53_l3_probe as m53          # noqa: E402  (loads the dataset once)
import m64_flip_probe as m64        # noqa: E402  (build_units / LP with force_rel)

CASES, W, TOTW = m53.CASES, m53.W, m53.TOTW
cost_eval, EPS_BND = m53.cost_eval, m53.EPS_BND
MIRROR = m64.MIRROR                  # (1,0,3,2)
PROBE_VERSION = "l281-v2"

ANCH, SIG, DB = None, None, None
_dirty = 0


# -- geometry ---------------------------------------------------------------
def unit_geo(ci, P):
    """Unit bounding boxes keyed by the SAME canonical key force_rel uses."""
    units, unit_of, group_units, group_comp0, frozen_blk, ukey = \
        m64.build_units(ci, P)
    n = CASES[ci]["n"]
    box, mem = {}, {}
    for i in range(n):
        k = ukey[i]
        x, y, w, h = P[i]
        b = box.get(k)
        if b is None:
            box[k] = [x, x + w, y, y + h]
        else:
            b[0], b[1] = min(b[0], x), max(b[1], x + w)
            b[2], b[3] = min(b[2], y), max(b[3], y + h)
        mem.setdefault(k, []).append(i)
    return units, unit_of, ukey, box, mem


def pinned_keys(ci, P, units, unit_of):
    """Unit keys carrying a boundary/extreme equality row in the LP build
    (mirror of build_and_solve_flip's `sides` loop)."""
    c = CASES[ci]
    n, cn = c["n"], c["cn"]
    xmin0 = min(P[i][0] for i in range(n))
    xmax0 = max(P[i][0] + P[i][2] for i in range(n))
    ymin0 = min(P[i][1] for i in range(n))
    ymax0 = max(P[i][1] + P[i][3] for i in range(n))

    def touch_ok(i, code):
        x, y, w, h = P[i]
        return ((not code & 1 or abs(x - xmin0) < EPS_BND)
                and (not code & 2 or abs(x + w - xmax0) < EPS_BND)
                and (not code & 4 or abs(y + h - ymax0) < EPS_BND)
                and (not code & 8 or abs(y - ymin0) < EPS_BND))

    sat = [(i, code) for i, code in ((i, cn[i][4]) for i in range(n)
                                     if cn[i][4] != 0) if touch_ok(i, code)]
    sides = ((1, min(range(n), key=lambda i: P[i][0])),
             (2, max(range(n), key=lambda i: P[i][0] + P[i][2])),
             (4, max(range(n), key=lambda i: P[i][1] + P[i][3])),
             (8, min(range(n), key=lambda i: P[i][1])))
    out = set()
    for bit, mdef in sides:
        tied = {unit_of[i] for i, code in sat if code & bit}
        if not tied:
            continue
        for u in tied | {unit_of[mdef]}:
            if u is not None:
                out.add(("U", min(units[u])))
    return out, (xmin0, xmax0, ymin0, ymax0)


def wire_terms(ci, P, unit_of, ukey, ku):
    """Live edges incident to unit key ku as (value, weight) per axis: the
    objective for a rigid displacement d of u alone is sum w*|d - value|.
    Also returns the per-neighbour-unit weight, for candidate generation."""
    c, n = CASES[ci], CASES[ci]["n"]
    cx = [P[i][0] + P[i][2] / 2.0 for i in range(n)]
    cy = [P[i][1] + P[i][3] / 2.0 for i in range(n)]
    vx, vy, nb = [], [], {}
    for i, j, w in c["b2l"]:
        ui, uj = unit_of[i], unit_of[j]
        if w <= 0.0 or ui == uj:
            continue
        if ukey[i] == ku:
            vx.append((cx[j] - cx[i], w))
            vy.append((cy[j] - cy[i], w))
            nb[ukey[j]] = nb.get(ukey[j], 0.0) + w
        elif ukey[j] == ku:
            vx.append((cx[i] - cx[j], w))
            vy.append((cy[i] - cy[j], w))
            nb[ukey[i]] = nb.get(ukey[i], 0.0) + w
    for p, i, w in c["p2l"]:
        if w <= 0.0 or unit_of[i] is None or ukey[i] != ku:
            continue
        px, py = c["pin"][p]
        vx.append((px - cx[i], w))
        vy.append((py - cy[i], w))
    return vx, vy, nb


def wmedian(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    tot = sum(w for _v, w in s)
    acc = 0.0
    for v, w in s:
        acc += w
        if acc >= tot / 2.0:
            return v
    return s[-1][0]


def wcost(vals, d):
    return sum(w * abs(d - v) for v, w in vals)


# -- the move: target position -> force_rel over every pair involving u ------
def induced_rel(ku, ub, box, keys):
    """Relation of u (at bbox ub) against every other unit, canonicalised the
    way build_and_solve_flip reads force_rel.  Returns (force_rel, min gap);
    min gap < 0 means the target overlaps -- legal as a MOVE (the LP is allowed
    to push the others aside), it just is not a zero-displacement witness."""
    fr, gmin = {}, float("inf")
    ux0, ux1, uy0, uy1 = ub
    for kv in keys:
        if kv == ku:
            continue
        vx0, vx1, vy0, vy1 = box[kv]
        g = (vx0 - ux1, ux0 - vx1, vy0 - uy1, uy0 - vy1)
        k = 0                                # first-wins argmax == LP's max()
        for t in (1, 2, 3):
            if g[t] > g[k]:
                k = t
        if g[k] < gmin:
            gmin = g[k]
        if ku <= kv:
            fr[(ku, kv)] = k
        else:
            fr[(kv, ku)] = MIRROR[k]
    return fr, gmin


def n_binding(ku, ub_cur, fr, box, keys):
    """How many of the forced relations are VIOLATED at u's current position.

    🔑 Not every changed relation is a move.  For a diagonal pair both the
    horizontal and the vertical separation already hold, so switching between
    them rewrites the LP row without excluding the current placement: the LP
    returns the same solution and the 'relocation' relocated nothing.  Only the
    pairs whose forced direction has a NEGATIVE gap at the current position
    force u to actually go somewhere.  `nflip` counts rewrites; this counts
    moves, and the difference is large."""
    ux0, ux1, uy0, uy1 = ub_cur
    nb = 0
    for kv in keys:
        if kv == ku:
            continue
        pk = (ku, kv) if ku <= kv else (kv, ku)
        kc = fr.get(pk)
        if kc is None:
            continue
        k = kc if ku <= kv else MIRROR[kc]
        vx0, vx1, vy0, vy1 = box[kv]
        g = (vx0 - ux1, ux0 - vx1, vy0 - uy1, uy0 - vy1)[k]
        if g < -1e-9:
            nb += 1
    return nb


def block_rel(ci, P, unit_of, ukey):
    """Per canonical unit pair, the SET of block-level argmax canonical
    directions the unforced LP actually uses (m64.rank_pairs' `amset`)."""
    n = CASES[ci]["n"]
    am = {}
    for i in range(n):
        xi, yi, wi, hi = P[i]
        for j in range(i + 1, n):
            if unit_of[i] == unit_of[j]:
                continue
            xj, yj, wj, hj = P[j]
            g = (xj - (xi + wi), xi - (xj + wj),
                 yj - (yi + hi), yi - (yj + hj))
            k = 0
            for t in (1, 2, 3):
                if g[t] > g[k]:
                    k = t
            ki, kj = ukey[i], ukey[j]
            if ki <= kj:
                pk, kc = (ki, kj), k
            else:
                pk, kc = (kj, ki), MIRROR[k]
            am.setdefault(pk, set()).add(kc)
    return am


# -- the coherence certificate ----------------------------------------------
def base_graph(ci, P, unit_of, ukey, skip_key):
    """Block-level H / V constraint edges for every pair NOT involving
    `skip_key`, using the same direction rule the unforced LP uses.
    Edge (a,b) on axis H means x_a + w_a <= x_b."""
    n = CASES[ci]["n"]
    EH, EV = [], []
    for i in range(n):
        xi, yi, wi, hi = P[i]
        ki = ukey[i]
        for j in range(i + 1, n):
            if unit_of[i] == unit_of[j]:
                continue
            kj = ukey[j]
            if ki == skip_key or kj == skip_key:
                continue
            xj, yj, wj, hj = P[j]
            g = (xj - (xi + wi), xi - (xj + wj),
                 yj - (yi + hi), yi - (yj + hj))
            k = 0
            for t in (1, 2, 3):
                if g[t] > g[k]:
                    k = t
            if k == 0:
                EH.append((i, j))
            elif k == 1:
                EH.append((j, i))
            elif k == 2:
                EV.append((i, j))
            else:
                EV.append((j, i))
    return EH, EV


def unit_edges(ci, P, unit_of, ukey, ku, fr):
    """The H / V edges contributed by every block pair that spans unit ku,
    under the forced relation dict `fr` (canonical unit-pair -> direction)."""
    n = CASES[ci]["n"]
    EH, EV = [], []
    for i in range(n):
        ki = ukey[i]
        for j in range(i + 1, n):
            kj = ukey[j]
            if unit_of[i] == unit_of[j] or (ki != ku and kj != ku):
                continue
            pk = (ki, kj) if ki <= kj else (kj, ki)
            kc = fr.get(pk)
            if kc is None:
                continue
            k = kc if ki <= kj else MIRROR[kc]
            if k == 0:
                EH.append((i, j))
            elif k == 1:
                EH.append((j, i))
            elif k == 2:
                EV.append((i, j))
            else:
                EV.append((j, i))
    return EH, EV


def longest_chain(n, edges, wt):
    """Kahn topological pass: (acyclic?, longest node-weighted path length)."""
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    q = [i for i in range(n) if indeg[i] == 0]
    dist = [wt[i] for i in range(n)]
    seen = 0
    head = 0
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
    if seen != n:
        return False, float("inf")
    return True, max(dist) if n else 0.0


def certificate(ci, P, EHb, EVb, EHu, EVu, bb):
    """Necessary conditions for the forced LP to be feasible.  Sound: a FAIL
    proves infeasibility, a PASS proves nothing."""
    n = CASES[ci]["n"]
    wt_w = [P[i][2] for i in range(n)]
    wt_h = [P[i][3] for i in range(n)]
    okH, lH = longest_chain(n, EHb + EHu, wt_w)
    okV, lV = longest_chain(n, EVb + EVu, wt_h)
    W0, H0 = bb[1] - bb[0], bb[3] - bb[2]
    if not okH or not okV:
        return dict(ok=False, why="cyclic", cycH=not okH, cycV=not okV)
    if lH > W0 + 1e-9 or lV > H0 + 1e-9:
        return dict(ok=False, why="oversized", lH=lH, lV=lV, W0=W0, H0=H0,
                    exH=lH - W0, exV=lV - H0)
    return dict(ok=True, why="coherent", lH=lH, lV=lV, W0=W0, H0=H0)


# -- candidate generation ---------------------------------------------------
def gen_targets(ku, box, keys, bb, nb, vx, vy, nnb):
    """Wire-driven relocation targets: abut u against each of its heaviest
    neighbour units on all four sides, plus the unconstrained wire optimum."""
    ux0, ux1, uy0, uy1 = box[ku]
    ex, ey = ux1 - ux0, uy1 - uy0
    xmin0, xmax0, ymin0, ymax0 = bb
    if ex > xmax0 - xmin0 + 1e-9 or ey > ymax0 - ymin0 + 1e-9:
        return []
    xstar, ystar = ux0 + wmedian(vx), uy0 + wmedian(vy)

    def clipx(x):
        return min(max(x, xmin0), xmax0 - ex)

    def clipy(y):
        return min(max(y, ymin0), ymax0 - ey)

    out = [(clipx(xstar), clipy(ystar))]
    for kv, _w in sorted(nb.items(), key=lambda t: -t[1])[:nnb]:
        if kv == ku or kv not in box:
            continue
        vx0, vx1, vy0, vy1 = box[kv]
        for x in (vx1, vx0 - ex):                     # right of v / left of v
            for y in (clipy(ystar), clipy(vy0), clipy(vy1 - ey)):
                out.append((clipx(x), y))
        for y in (vy1, vy0 - ey):                     # above v / below v
            for x in (clipx(xstar), clipx(vx0), clipx(vx1 - ex)):
                out.append((x, clipy(y)))
    seen, ded = set(), []
    for x, y in out:
        k = (round(x, 9), round(y, 9))
        if k in seen:
            continue
        seen.add(k)
        ded.append((x, y))
    return ded


def rank_units(ci, P):
    """Units ranked by the exact wire prize of moving them ALONE to their
    weighted-L1-median: an upper bound on what relocating that unit can buy."""
    c = CASES[ci]
    units, unit_of, ukey, box, mem = unit_geo(ci, P)
    pin, bb = pinned_keys(ci, P, units, unit_of)
    hw = 0.5 / max(float(c["base"].get("hpwl_baseline", 1.0)), 1e-6)
    keys = sorted(box)
    out = []
    for ku in keys:
        if ku[0] != "U":                     # frozen pseudo-units cannot move
            continue
        vx, vy, nb = wire_terms(ci, P, unit_of, ukey, ku)
        if not vx:
            continue
        dx, dy = wmedian(vx), wmedian(vy)
        prize = (wcost(vx, 0.0) - wcost(vx, dx)
                 + wcost(vy, 0.0) - wcost(vy, dy)) * hw
        out.append(dict(ku=ku, prize=prize, dx=dx, dy=dy,
                        pinned=ku in pin, nedge=len(vx), nmem=len(mem[ku])))
    out.sort(key=lambda r: -r["prize"])
    return out, dict(box=box, keys=keys, bb=bb, pin=pin, hw=hw,
                     unit_of=unit_of, ukey=ukey, units=units, mem=mem)


# -- cache ------------------------------------------------------------------
def save_db():
    global _dirty
    tmp = _DIR / "l281_cache.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dict(sig=SIG, db=DB), f)
    os.replace(tmp, _DIR / "l281_cache.pkl")
    _dirty = 0


def put(key, val):
    global _dirty
    DB[key] = val
    _dirty += 1
    if _dirty >= 20:
        save_db()


# -- census: certificate only, no LP ----------------------------------------
def census_case(ci, nunits, nnb):
    """Coherence rates for BOTH move semantics on identical geometry."""
    P0 = [tuple(p) for p in ANCH[ci]["positions"]]
    ranked, G = rank_units(ci, P0)
    box, keys, bb = G["box"], G["keys"], G["bb"]
    unit_of, ukey = G["unit_of"], G["ukey"]
    t0 = time.perf_counter()

    # ---- move A: RELOCATION (this probe) ----
    relo = dict(coherent=0, cyclic=0, oversized=0, total=0, free_slot=0,
                exH=[], exV=[], nflip=[], demand=0.0, supply=0.0)
    hw = G["hw"]
    picks = []
    for rec in ranked:
        if rec["pinned"] or len(picks) >= nunits:
            continue
        ku = rec["ku"]
        vx, vy, nb = wire_terms(ci, P0, unit_of, ukey, ku)
        tg = gen_targets(ku, box, keys, bb, nb, vx, vy, nnb)
        if not tg:
            continue
        EHb, EVb = base_graph(ci, P0, unit_of, ukey, ku)
        cur_fr, _g = induced_rel(ku, box[ku], box, keys)
        seen, kept = set(), []
        for x, y in tg:
            ex = box[ku][1] - box[ku][0]
            ey = box[ku][3] - box[ku][2]
            fr, gmin = induced_rel(ku, (x, x + ex, y, y + ey), box, keys)
            sig = tuple(sorted(fr.items()))
            if sig in seen or fr == cur_fr:
                continue
            seen.add(sig)
            nflip = sum(1 for pk, k in fr.items() if cur_fr.get(pk) != k)
            EHu, EVu = unit_edges(ci, P0, unit_of, ukey, ku, fr)
            cert = certificate(ci, P0, EHb, EVb, EHu, EVu, bb)
            relo["total"] += 1
            relo["nflip"].append(nflip)
            if gmin >= -1e-9:
                relo["free_slot"] += 1
            if cert["ok"]:
                relo["coherent"] += 1
                kept.append((x, y, fr, nflip, wcost(vx, x - box[ku][0])
                             + wcost(vy, y - box[ku][2])))
            elif cert["why"] == "cyclic":
                relo["cyclic"] += 1
            else:
                relo["oversized"] += 1
                relo["exH"].append(cert["exH"])
                relo["exV"].append(cert["exV"])
        # demand = the unconstrained first-order wire prize for this unit;
        # supply = the best of it that survives the coherence certificate.
        w0 = wcost(vx, 0.0) + wcost(vy, 0.0)
        relo["demand"] = max(relo["demand"], hw * (w0 - wcost(vx, rec["dx"])
                                                   - wcost(vy, rec["dy"])))
        if kept:
            relo["supply"] = max(relo["supply"],
                                 hw * (w0 - min(k[4] for k in kept)))
        picks.append((ku, kept))

    # ---- move B: M64's single unit-pair flip, same anchor, same certificate --
    R64 = m64.rank_pairs(ci, P0, 24)
    m64s = dict(coherent=0, cyclic=0, oversized=0, total=0)
    EHall, EVall = base_graph(ci, P0, unit_of, ukey, None)
    for score, pk, dirs, cur, het in R64["ranked"]:
        EHb, EVb = [], []
        ka, kb = pk
        # rebuild base excluding this pair only
        drop = set()
        n = CASES[ci]["n"]
        for i in range(n):
            for j in range(i + 1, n):
                if unit_of[i] == unit_of[j]:
                    continue
                ki, kj = ukey[i], ukey[j]
                p = (ki, kj) if ki <= kj else (kj, ki)
                if p == pk:
                    drop.add((i, j))
        EHb = [(a, b) for a, b in EHall
               if (min(a, b), max(a, b)) not in drop]
        EVb = [(a, b) for a, b in EVall
               if (min(a, b), max(a, b)) not in drop]
        for k in dirs:
            EHu, EVu = unit_edges(ci, P0, unit_of, ukey, ka, {pk: k})
            EHu2, EVu2 = unit_edges(ci, P0, unit_of, ukey, kb, {pk: k})
            eh = list({*EHu, *EHu2})
            ev = list({*EVu, *EVu2})
            cert = certificate(ci, P0, EHb, EVb, eh, ev, bb)
            m64s["total"] += 1
            if cert["ok"]:
                m64s["coherent"] += 1
            elif cert["why"] == "cyclic":
                m64s["cyclic"] += 1
            else:
                m64s["oversized"] += 1

    put(("census", ci, nunits, nnb), dict(relo=relo, m64=m64s, picks=picks,
                             npk=R64["npk"], nunits=R64["nunits"]))
    save_db()
    r, m = relo, m64s
    print(f"case {ci:3d} n={CASES[ci]['n']:3d} units={R64['nunits']:3d} "
          f"pairs={R64['npk']:5d}  ({time.perf_counter() - t0:.0f}s)",
          flush=True)
    print(f"    RELOCATION  {r['total']:4d} moves: coherent {r['coherent']:4d} "
          f"({100.0 * r['coherent'] / max(r['total'], 1):5.1f} %)  "
          f"cyclic {r['cyclic']:4d}  oversized {r['oversized']:4d}  |  "
          f"lands in free space {r['free_slot']:3d}", flush=True)
    print(f"    M64 1-PAIR  {m['total']:4d} moves: coherent {m['coherent']:4d} "
          f"({100.0 * m['coherent'] / max(m['total'], 1):5.1f} %)  "
          f"cyclic {m['cyclic']:4d}  oversized {m['oversized']:4d}", flush=True)
    return relo, m64s


# -- one relocation ---------------------------------------------------------
def reloc_solve(ci, P0, fr, relax=1.0):
    t0 = time.perf_counter()
    newP, tele = m64.lp_pass_flip(ci, P0, area_obj=True, force_rel=fr,
                                  bbox_relax=relax)
    dt = time.perf_counter() - t0
    if newP is None:
        return dict(status=tele["status"], attempt=tele.get("attempt"), dt=dt)
    m = cost_eval(ci, newP)
    return dict(status="ok", dt=dt, cost=m.cost, feas=bool(m.is_feasible),
                hgap=m.hpwl_gap, agap=m.area_gap,
                vrel=m.violations_relative, attempt=tele["attempts"],
                newP=newP)


def polish(ci, P, c, iters):
    """m53 fixpoint polish -- the SAME budget must be given to the control,
    otherwise the relocation arm is being compared against a shorter search
    and the 'gain' is just the extra passes (L156's no-op-control rule)."""
    bP, bc = P, c
    for _ in range(iters):
        nP, _t = m53.lp_pass(ci, bP, area_obj=True)
        if nP is None:
            break
        mm = cost_eval(ci, nP)
        if mm.is_feasible and mm.cost < bc - 1e-12:
            bc, bP = mm.cost, nP
        else:
            break
    return bc, bP


def probe_case(ci, nunits, ncand, nnb, polish_iters):
    ent = ANCH[ci]
    P0 = [tuple(p) for p in ent["positions"]]
    c0 = ent["cost"]
    gk = ("gate", ci)
    if gk not in DB:
        m = cost_eval(ci, P0)
        if m.cost != c0:
            raise SystemExit(f"anchor gate FAIL case {ci}: {m.cost!r} != {c0!r}")
        put(gk, True)

    ck = ("ctrl", ci)
    if ck not in DB:
        r = reloc_solve(ci, P0, None)
        r.pop("newP", None)
        put(ck, r)
        save_db()
    ctrl = DB[ck]
    # the control gets the SAME polish budget as the relocation arm
    cpk = ("ctrlp", ci, polish_iters)
    if cpk not in DB:
        r = reloc_solve(ci, P0, None)
        if r["status"] == "ok" and r["feas"]:
            bc, _bP = polish(ci, r["newP"], r["cost"], polish_iters)
            put(cpk, dict(cost=bc, feas=True))
        else:
            put(cpk, dict(cost=float("inf"), feas=False))
        save_db()
    ctrlp = DB[cpk]
    cbase = min([c0] + ([ctrl["cost"]] if ctrl.get("feas") else [])
                + ([ctrlp["cost"]] if ctrlp.get("feas") else []))

    cs = ("census", ci, nunits, nnb)
    if cs not in DB:
        census_case(ci, nunits, nnb)
    picks = DB[cs]["picks"]

    t_case = time.perf_counter()
    nsolve, movers = 0, []
    for ku, kept in picks:
        kept = sorted(kept, key=lambda t: t[4])[:ncand]
        for ic, (x, y, fr, nflip, wcst) in enumerate(kept):
            key = ("rel2", ci, ku, ic)
            if key not in DB:
                r = reloc_solve(ci, P0, fr)
                r["nflip"], r["tgt"] = nflip, (x, y)
                if r["status"] == "ok":
                    r["delta"] = cbase - r["cost"]
                    if r["feas"]:
                        # polish EVERY feasible solution, not only the ones that
                        # already win -- otherwise the arm and the control are
                        # not being given the same search budget
                        bc, bP = polish(ci, r["newP"], r["cost"], polish_iters)
                        r["polished"] = bc
                        if bc < cbase - 1e-6:
                            r["positions"] = [list(p) for p in bP]
                r.pop("newP", None)
                put(key, r)
                nsolve += 1
            r = DB[key]
            if r["status"] == "ok":
                bst = min(r["cost"], r.get("polished", float("inf")))
                print(f"  c{ci} {ku} #{ic} nflip={r['nflip']:3d} "
                      f"cost {r['cost']:.6f} pol {bst:.6f} "
                      f"d={cbase - bst:+.2e} feas={int(r['feas'])} "
                      f"({r['dt']:.1f}s)", flush=True)
                if r["feas"] and cbase - bst > 1e-6:
                    movers.append((ku, ic, cbase - bst, bst))
            else:
                print(f"  c{ci} {ku} #{ic} nflip={r['nflip']:3d} {r['status']} "
                      f"att={r.get('attempt')} ({r['dt']:.1f}s)", flush=True)
    save_db()
    cc = ctrl.get("cost", float("nan"))
    print(f"case {ci:3d} n={CASES[ci]['n']:3d}: {nsolve} new LP solves, "
          f"{len(movers)} movers, ctrl {cc:.6f} vs anchor {c0:.6f} "
          f"({time.perf_counter() - t_case:.0f}s)", flush=True)
    for ku, ic, d, pc in movers:
        print(f"    MOVER {ku} #{ic} d={d:+.6f} polished {pc:.6f}", flush=True)
    return movers


# -- report -----------------------------------------------------------------
def mode_report():
    cen = [(k[1], v) for k, v in DB.items() if k[0] == "census"]
    if cen:
        tr = dict(coherent=0, cyclic=0, oversized=0, total=0, free_slot=0)
        tm = dict(coherent=0, cyclic=0, oversized=0, total=0)
        nfl = []
        for _ci, v in cen:
            for a in tr:
                tr[a] += v["relo"][a]
            for a in tm:
                tm[a] += v["m64"][a]
            nfl += v["relo"]["nflip"]
        print("== COHERENCE CENSUS (certificate only, no LP) ==")
        print(f"  {'move':<14}{'n':>6}{'coherent':>12}{'cyclic':>10}"
              f"{'oversized':>11}")
        for lbl, d in (("RELOCATION", tr), ("M64 1-pair flip", tm)):
            print(f"  {lbl:<14}{d['total']:>6}"
                  f"{d['coherent']:>7} {100.0 * d['coherent'] / max(d['total'], 1):5.1f}%"
                  f"{d['cyclic']:>10}{d['oversized']:>11}")
        if nfl:
            nfl.sort()
            print(f"  relocation pair-flips per move: min {nfl[0]} "
                  f"p50 {nfl[len(nfl) // 2]} max {nfl[-1]}")
        print(f"  relocations landing in existing free space: "
              f"{tr['free_slot']}/{tr['total']} "
              f"({100.0 * tr['free_slot'] / max(tr['total'], 1):.1f} %)")
        # demand vs supply, recomputed from the cached census so it covers every
        # census entry.  SUPPLY counts only BINDING coherent targets: a vacuous
        # target leaves u where it is, so crediting it with the wire cost of the
        # target position would be counting a move that never happens.
        dem = []
        for ci, v in cen:
            P = [tuple(p) for p in ANCH[ci]["positions"]]
            ranked, G = rank_units(ci, P)
            box, keys, hw = G["box"], G["keys"], G["hw"]
            uo, uk = G["unit_of"], G["ukey"]
            bd = bs = 0.0
            for ku, kept in v["picks"]:
                vx, vy, _nb = wire_terms(ci, P, uo, uk, ku)
                w0 = wcost(vx, 0.0) + wcost(vy, 0.0)
                bd = max(bd, hw * (w0 - wcost(vx, wmedian(vx))
                                   - wcost(vy, wmedian(vy))))
                for t in kept:
                    x, y, fr, _nf, wcst = t[0], t[1], t[2], t[3], t[4]
                    if n_binding(ku, box[ku], fr, box, keys) <= 0:
                        continue
                    bs = max(bs, hw * (w0 - wcst))
            dem.append((ci, bd, bs))
        if dem:
            wsum = sum(W[ci] for ci, _d, _s in dem)
            bt = sum(W[ci] * ANCH[ci]["cost"] for ci, _d, _s in dem) / wsum
            gd = sum(W[ci] * d for ci, d, _s in dem) / wsum
            gs = sum(W[ci] * s for ci, _d, s in dem) / wsum
            print(f"\n  == wire DEMAND vs coherent SUPPLY ({len(dem)} cases, "
                  f"weighted base {bt:.6f}) ==")
            print(f"     demand (best unit -> its wire optimum, "
                  f"no constraints) : {100.0 * gd / bt:+.4f} %")
            print(f"     supply (best target that passes the certificate)  "
                  f"    : {100.0 * gs / bt:+.4f} %   "
                  f"= {100.0 * gs / max(gd, 1e-12):.1f} % of demand")

    # recompute `nbind` for every cached LP row from its stored target, so the
    # binding/vacuous split does not require re-running any LP
    geo = {}
    for ci in sorted({k[1] for k in DB if k[0] == "rel2"}):
        P = [tuple(p) for p in ANCH[ci]["positions"]]
        _u, _uo, _uk, box, _m = unit_geo(ci, P)
        geo[ci] = (box, sorted(box))
    for key, r in DB.items():
        if key[0] != "rel2" or "tgt" not in r:
            continue
        ci, ku = key[1], key[2]
        box, keys = geo[ci]
        ex = box[ku][1] - box[ku][0]
        ey = box[ku][3] - box[ku][2]
        x, y = r["tgt"]
        fr, _g = induced_rel(ku, (x, x + ex, y, y + ey), box, keys)
        r["nbind"] = n_binding(ku, box[ku], fr, box, keys)

    # base per case = the best the SAME machinery reaches with no relocation:
    # the anchor, the no-force LP, and that LP given the same polish budget.
    cbase = {}
    for ci in sorted({k[1] for k in DB if k[0] in ("rel2", "ctrl")}):
        cands = [ANCH[ci]["cost"]]
        ct = DB.get(("ctrl", ci), {})
        if ct.get("feas"):
            cands.append(ct["cost"])
        for k, v in DB.items():
            if k[0] == "ctrlp" and k[1] == ci and v.get("feas"):
                cands.append(v["cost"])
        cbase[ci] = min(cands)

    hist, rows, best = {}, [], {}
    for key, r in DB.items():
        if key[0] != "rel2":
            continue
        _t, ci, ku, ic = key
        st = r["status"]
        if st == "ok":
            cb = min(r["cost"], r.get("polished", float("inf")))
            d = cbase[ci] - cb
            st = ("mover" if r["feas"] and d > 1e-6 else
                  "tiny_win" if r["feas"] and d > 1e-12 else
                  "worse" if r["feas"] else "ok_infeasible")
            rows.append((ci, ku, ic, r, d))
            if r["feas"] and cb < best.get(ci, float("inf")):
                best[ci] = cb
        elif str(st).startswith("lp_status") and r.get("attempt", 1) > 1:
            st = "ladder_kill"
        hist[st] = hist.get(st, 0) + 1
    tot = sum(hist.values())
    if not tot:
        return
    print("\n== L281 relocation LP: status histogram (per (unit, target)) ==")
    for st, c in sorted(hist.items(), key=lambda t: -t[1]):
        print(f"  {st:18s} {c:5d}   {100.0 * c / max(tot, 1):5.1f} %")
    inf = sum(c for st, c in hist.items()
              if st.startswith("lp_status") or st in ("ladder_kill",
                                                      "cluster_break"))
    print(f"\nTHE FORK -- LP-infeasible rate on CERTIFIED-COHERENT moves: "
          f"{inf}/{tot} = {100.0 * inf / max(tot, 1):.1f} %   (M64 = 86.8 %)")

    # the same histogram restricted to moves that actually displace the unit
    hb, nb_tot = {}, 0
    for key, r in DB.items():
        if key[0] != "rel2" or r.get("nbind", 0) <= 0:
            continue
        ci = key[1]
        st = r["status"]
        if st == "ok":
            cb = min(r["cost"], r.get("polished", float("inf")))
            d = cbase[ci] - cb
            st = ("mover" if r["feas"] and d > 1e-6 else
                  "tiny_win" if r["feas"] and d > 1e-12 else
                  "worse" if r["feas"] else "ok_infeasible")
        elif str(st).startswith("lp_status") and r.get("attempt", 1) > 1:
            st = "ladder_kill"
        hb[st] = hb.get(st, 0) + 1
        nb_tot += 1
    vac = tot - nb_tot
    print(f"\n== restricted to BINDING moves (the forced relation is violated "
          f"at the current position, so u must actually move) ==")
    print(f"  vacuous rewrites excluded: {vac}/{tot} = "
          f"{100.0 * vac / max(tot, 1):.1f} %   "
          f"(same LP row set, same optimum -- not a relocation)")
    for st, c in sorted(hb.items(), key=lambda t: -t[1]):
        print(f"  {st:18s} {c:5d}   {100.0 * c / max(nb_tot, 1):5.1f} %")
    infb = sum(c for st, c in hb.items()
               if st.startswith("lp_status") or st in ("ladder_kill",
                                                       "cluster_break"))
    print(f"  LP-infeasible among binding: {infb}/{nb_tot} = "
          f"{100.0 * infb / max(nb_tot, 1):.1f} %")

    ds = sorted(d for _c, _k, _i, r, d in rows if r["feas"])
    if ds:
        n = len(ds)
        print(f"\n== feasible-solution delta vs the polished control (n={n}, "
              f"positive = better) ==")
        for q, lbl in ((n - 1, "best"), (3 * n // 4, "p75"), (n // 2, "p50"),
                       (n // 4, "p25"), (0, "worst")):
            print(f"  {lbl:5s} {ds[min(q, n - 1)]:+.6f}")
    for ci in sorted({c for c, _k, _i, _r, _d in rows}):
        e = ANCH[ci]
        sub = [r for c, _k, _i, r, _d in rows if c == ci and r["feas"]]
        if not sub:
            continue
        mh = sum(r["hgap"] for r in sub) / len(sub) - e["hpwl_gap"]
        ma = sum(r["agap"] for r in sub) / len(sub) - e["area_gap"]
        mv = (sum(r["vrel"] for r in sub) / len(sub)
              - e["violations_relative"])
        print(f"  case {ci:3d}: mean d_hgap {mh:+.5f} d_agap {ma:+.5f} "
              f"d_vrel {mv:+.5f}  ({len(sub)} feasible)")

    cases = sorted({k[1] for k in DB if k[0] == "rel2"})
    print("\n== control (same LP + same polish budget, no relocation) ==")
    gA = gC = 0.0
    for ci in cases:
        c0 = ANCH[ci]["cost"]
        ct = DB.get(("ctrl", ci), {})
        cc = ct.get("cost", float("nan")) if ct.get("feas") else float("nan")
        cp = min([v["cost"] for k, v in DB.items()
                  if k[0] == "ctrlp" and k[1] == ci and v.get("feas")],
                 default=float("nan"))
        bf = best.get(ci, float("inf"))
        gA += W[ci] * max(0.0, c0 - bf)
        gC += W[ci] * max(0.0, cbase[ci] - bf)
        print(f"  case {ci:3d}: anchor {c0:.6f}  ctrl {cc:.6f}  "
              f"ctrl+polish {cp:.6f}  best_reloc {bf:.6f}")
    wsum = sum(W[ci] for ci in cases) or 1e-9
    bt = sum(W[ci] * ANCH[ci]["cost"] for ci in cases) / wsum
    bc = sum(W[ci] * cbase[ci] for ci in cases) / wsum
    print(f"\n== union-oracle over {len(cases)} probed cases ==")
    print(f"   vs anchor          (base {bt:.6f}) : "
          f"{100.0 * gA / wsum / bt:+.4f} %")
    print(f"   vs polished control(base {bc:.6f}) : "
          f"{100.0 * gC / wsum / bc:+.4f} %   <- the honest one")


# -- gate: forcing the CURRENT topology of a unit must be a no-op ------------
def mode_gate(ci):
    ent = ANCH[ci]
    P0 = [tuple(p) for p in ent["positions"]]
    m = cost_eval(ci, P0)
    assert m.cost == ent["cost"], f"anchor gate FAIL: {m.cost!r}"
    print(f"[gate] case {ci} anchor reproduces json cost exactly")
    ranked, G = rank_units(ci, P0)
    box, keys = G["box"], G["keys"]
    am = block_rel(ci, P0, G["unit_of"], G["ukey"])
    pick, ncand = None, 0
    for rec in ranked:
        if rec["pinned"]:
            continue
        fr, _g = induced_rel(rec["ku"], box[rec["ku"]], box, keys)
        ncand += 1
        if all(am.get(pk) == {k} for pk, k in fr.items()):
            pick = (rec["ku"], fr, _g)
            break
    if pick is None:
        raise SystemExit(f"GATE FAIL: no unit whose bbox topology restates the "
                         f"anchor rows exactly, among {ncand} candidates")
    ku, fr, gmin = pick
    print(f"[gate] unit {ku}: current-topology force over {len(fr)} pairs, "
          f"all restate the anchor rows exactly, min gap {gmin:.3e}")
    a, ta = m64.lp_pass_flip(ci, P0, area_obj=True)
    b, tb = m64.lp_pass_flip(ci, P0, area_obj=True, force_rel=fr)
    assert a is not None and b is not None, (ta, tb)
    same = all(ax == bx for pa, pb in zip(a, b) for ax, bx in zip(pa, pb))
    print(f"[gate] forced-current == unforced, bit-identical: {same}")
    if not same:
        nd = sum(1 for pa, pb in zip(a, b)
                 for ax, bx in zip(pa, pb) if ax != bx)
        raise SystemExit(f"GATE FAIL: {nd} coordinates differ")
    # certificate must certify the anchor's own topology as coherent
    EHb, EVb = base_graph(ci, P0, G["unit_of"], G["ukey"], ku)
    EHu, EVu = unit_edges(ci, P0, G["unit_of"], G["ukey"], ku, fr)
    cert = certificate(ci, P0, EHb, EVb, EHu, EVu, G["bb"])
    print(f"[gate] certificate on the anchor's own topology: {cert}")
    if not cert["ok"]:
        raise SystemExit("GATE FAIL: certificate rejects the anchor itself")
    print("[gate] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "census", "probe", "report"])
    ap.add_argument("--anchor", default=str(_DIR / "results_L274_base_48c.json"))
    ap.add_argument("--cases", default="85,88,91")
    ap.add_argument("--units", type=int, default=8)
    ap.add_argument("--cand", type=int, default=3)
    ap.add_argument("--nnb", type=int, default=4)
    ap.add_argument("--polish-iters", type=int, default=3)
    args = ap.parse_args()

    raw = open(args.anchor, "rb").read()
    SIG = (hashlib.md5(raw).hexdigest(), PROBE_VERSION)
    _aj = json.loads(raw)
    ANCH = {t["test_id"]: t for t in _aj["test_results"]}
    print(f"[anchor] {Path(args.anchor).name} "
          f"total={_aj['total_score']:.10f}", flush=True)

    DB = {}
    _cp = _DIR / "l281_cache.pkl"
    if _cp.exists():
        try:
            _d = pickle.load(open(_cp, "rb"))
        except Exception:
            _d = None
        if _d and _d.get("sig") == SIG:
            DB = _d["db"]
            print(f"[cache] resume {len(DB)} entries", flush=True)
        else:
            print("[cache] sig mismatch -> reset", flush=True)

    ids = [int(x) for x in args.cases.split(",")] if args.cases else []
    if args.mode == "gate":
        mode_gate(ids[0])
    elif args.mode == "report":
        mode_report()
    elif args.mode == "census":
        for ci in ids:
            census_case(ci, args.units, args.nnb)
        save_db()
    else:
        allm = []
        for ci in ids:
            allm += probe_case(ci, args.units, args.cand, args.nnb,
                               args.polish_iters)
        save_db()
        print(f"\n== probe done: {len(allm)} movers ==", flush=True)
