"""L282 Gate 0 (NO LP): can the critical chain be shortened at all?

L281 closed every topology edit that LENGTHENS the critical chain.  The dual
move is the one thing left: take a unit OFF the chain so the chain gets shorter,
which lets the LP shrink the bbox and pay down area_gap.  Unlike L281's moves
this one is feasible by construction -- a shorter chain fits in the same box.

Whether it can buy anything at all is decided by two quantities, both exact and
both computable without an LP:

  (1) CHAIN REDUNDANCY.  Deleting unit u from the binding axis's constraint
      graph entirely is the best case for relocating it (it is then constrained
      only on the other axis).  So `lH - lH(without u)`, maximised over movable
      non-pinned units on the critical chain, is an UPPER BOUND on how much one
      relocation can shorten the row.  If the graph carries two node-disjoint
      near-critical paths this is ~0 and the axis is dead.

  (2) FROZEN SPAN.  `build_and_solve_flip` bounds XMIN <= min frozen x and
      XMAX >= max frozen (x+w), so the bbox can never shrink below the
      preplaced blocks' own extent, whatever the chain does.

The prize is bounded by whichever binds first.  Cost model: shrinking one row by
factor f scales area by f, and d(0.5*area_gap) = 0.5*(f-1)*(1+area_gap), priced
against the case's own cost bracket.  This ignores the wire cost of squeezing,
so it is an upper bound on the AREA side only.
"""
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import l281_reloc_probe as L                                       # noqa: E402

CASES, W = L.CASES, L.W


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


anchor = sys.argv[1] if len(sys.argv) > 1 else str(
    _DIR / "results_L274_base_48c.json")
aj = json.loads(open(anchor, "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
print(f"[anchor] {Path(anchor).name} total={aj['total_score']:.10f}",
      flush=True)

rows = []
for ci in sorted(ANCH):
    e = ANCH[ci]
    P = [tuple(p) for p in e["positions"]]
    n = CASES[ci]["n"]
    units, unit_of, ukey, box, mem = L.unit_geo(ci, P)
    pin, bb = L.pinned_keys(ci, P, units, unit_of)
    EH, EV = L.base_graph(ci, P, unit_of, ukey, None)
    wW = [P[i][2] for i in range(n)]
    wH = [P[i][3] for i in range(n)]
    W0, H0 = bb[1] - bb[0], bb[3] - bb[2]
    lH, cH = critical_nodes(n, EH, wW)
    lV, cV = critical_nodes(n, EV, wH)
    if lH is None or lV is None:
        continue
    # the binding axis = the one with less slack
    sH, sV = 1.0 - lH / W0, 1.0 - lV / H0
    if sH <= sV:
        axis, edges, wt, row, ln, crit = "H", EH, wW, W0, lH, cH
        fro_lo = [P[i][0] for i in range(n) if unit_of[i] is None]
        fro_hi = [P[i][0] + P[i][2] for i in range(n) if unit_of[i] is None]
    else:
        axis, edges, wt, row, ln, crit = "V", EV, wH, H0, lV, cV
        fro_lo = [P[i][1] for i in range(n) if unit_of[i] is None]
        fro_hi = [P[i][1] + P[i][3] for i in range(n) if unit_of[i] is None]
    frozen_span = (max(fro_hi) - min(fro_lo)) if fro_lo else 0.0

    # candidate units: movable, not boundary/extreme-pinned, on the chain
    cand = set()
    for i in crit:
        k = ukey[i]
        if k[0] == "U" and k not in pin:
            cand.add(k)
    # the OTHER axis has to absorb u: relocating it off this chain puts it on
    # that one.  optimistic = the other row does not grow; pessimistic = u lands
    # on the other axis's critical path and adds its whole extent.
    if axis == "H":
        oth_row, oth_len = H0, lV
        ext = {k: box[k][3] - box[k][2] for k in cand}
    else:
        oth_row, oth_len = W0, lH
        ext = {k: box[k][1] - box[k][0] for k in cand}
    best_after, best_u, best_oth = ln, None, oth_row
    for k in cand:
        after = chain_excl(n, edges, wt, set(mem[k]))
        if after < best_after:
            best_after, best_u = after, k
            best_oth = max(oth_row, oth_len + ext[k])
    # what the row could become, and what that is worth
    floor = max(best_after, frozen_span)
    f = max(floor, 1e-9) / row
    agap = e["area_gap"]
    brk = 1.0 + 0.5 * (e["hpwl_gap"] + agap)
    prize = -0.5 * (f - 1.0) * (1.0 + agap) / brk       # positive = better
    fp = f * (best_oth / oth_row)                       # other axis pays too
    prize_p = -0.5 * (fp - 1.0) * (1.0 + agap) / brk
    rows.append((ci, n, axis, row, ln, best_after, frozen_span, floor,
                 len(cand), len(crit), prize, prize_p,
                 1.0 - oth_len / oth_row))
    print(f"case {ci:3d} n={n:3d} [{axis}] row {row:9.4f} chain {ln:9.4f} "
          f"-> best-after {best_after:9.4f}  frozen span {frozen_span:9.4f}  "
          f"floor {floor:9.4f}  cand {len(cand):3d}/{len(crit):3d}  "
          f"prize {100 * prize:+7.4f} %", flush=True)

wsum = sum(W[r[0]] for r in rows)
bt = sum(W[r[0]] * ANCH[r[0]]["cost"] for r in rows) / wsum
g = sum(W[r[0]] * ANCH[r[0]]["cost"] * r[10] for r in rows) / wsum
nz = sum(1 for r in rows if r[10] > 1e-9)
chain_binds = sum(1 for r in rows if r[5] >= r[6] - 1e-9)
froz_binds = len(rows) - chain_binds
noshrink = sum(1 for r in rows if r[7] >= r[3] - 1e-9)
print(f"\n== {len(rows)} cases, weighted exp(n/12), base {bt:.6f} ==")
gp = sum(W[r[0]] * ANCH[r[0]]["cost"] * r[11] for r in rows) / wsum
oslack = sorted(100.0 * r[12] for r in rows)
print(f"  OPTIMISTIC  (other axis absorbs u for free)  : {100 * g / bt:+.4f} %")
print(f"  PESSIMISTIC (u lands on the other critical path): "
      f"{100 * gp / bt:+.4f} %")
print(f"  slack on the OTHER axis: p25 {oslack[len(oslack) // 4]:.4f} %  "
      f"p50 {oslack[len(oslack) // 2]:.4f} %  "
      f"p75 {oslack[3 * len(oslack) // 4]:.4f} %")
print(f"  cases where the other axis has room for the moved unit: "
      f"{sum(1 for r in rows if r[11] > 1e-9)}/{len(rows)}")
print(f"  cases where any shortening is possible at all : {nz}/{len(rows)}")
print(f"  cases where the row cannot shrink at all      : {noshrink}/{len(rows)}")
print(f"  binding floor is the CHAIN  : {chain_binds}/{len(rows)}")
print(f"  binding floor is FROZEN span: {froz_binds}/{len(rows)}")
drops = sorted(100.0 * (r[4] - r[5]) / r[3] for r in rows)
print(f"  chain shortening as % of the row: p50 {drops[len(drops) // 2]:.4f} %"
      f"   p90 {drops[int(0.9 * len(drops))]:.4f} %   max {drops[-1]:.4f} %")
