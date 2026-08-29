"""L279 -- are the 23 preplaced boundary violations structurally unsatisfiable?

L277 found 59 in-set boundary violations, of which **23 sit on `preplaced`
blocks**. A preplaced block's position is given by the problem, so its boundary
requirement is satisfied only if the layout's bounding box edge lands exactly
where that block's edge already is. That makes it a FRAME question, not a
placement question -- and L136 found and fixed one whole family of exactly this
shape (`MARGIN` = 1e-4 against the scorer's `TOL` = 1e-6 left the frame 1e-4 wider
than the preplaced extent, handing those blocks a violation nothing could satisfy;
worth +0.5972%).

The audit's `kind == "MARGIN"` test says that exact signature is extinct. This asks
whether a DIFFERENT structural impossibility is hiding in the remaining 23, using
two independent tests:

  HARD   does another PREPLACED block extend beyond this one on the side it must
         touch? A preplaced block's POSITION is given, so the bbox edge can never
         retreat past it -- unsatisfiable for anyone, us or the label.
         🚨 `fixed` does NOT belong in this test. `is_fixed` pins only the SHAPE;
         the position stays free (constructive.cpp:1745-1747 sets `placed[i]=1`
         for is_preplaced only). Including it made 10 of 23 rows come out both
         HARD and label-SATISFIED, which is impossible -- the label shares our
         preplaced positions. That self-contradiction is what caught the bug.

  LABEL  does the GROUND TRUTH layout satisfy this same constraint? The label is a
         real, legal layout, so if it satisfies the constraint the requirement is
         reachable and we are leaving something on the table; if the label fails
         it too, no achievable layout in evidence satisfies it.

The two together partition the 23 into "nobody can", "the reference does not
either", and "the reference does, so we are losing it".

⚠️ The label test is an ORACLE -- it reads ground truth, so it can diagnose but can
never be part of a shipped mechanism. Same status as M26/M68/M79's oracle probes.

READ-ONLY.

  <python> l279_preplaced.py results_L274_base_48c.json
"""
import json
import math
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import ContestEvaluator                # noqa: E402

EPS = 1e-6
SIDES = [(1, "L"), (2, "R"), (4, "T"), (8, "B")]


def _bbox(P):
    return (min(p[0] for p in P), min(p[1] for p in P),
            max(p[0] + p[2] for p in P), max(p[1] + p[3] for p in P))


def _miss(P, i, code):
    """Which required sides block i fails to touch, and by how much."""
    x0, y0, x1, y1 = _bbox(P)
    bx, by, bw, bh = P[i]
    out = {}
    if code & 1 and abs(bx - x0) >= EPS:
        out["L"] = bx - x0
    if code & 2 and abs(bx + bw - x1) >= EPS:
        out["R"] = x1 - (bx + bw)
    if code & 4 and abs(by + bh - y1) >= EPS:
        out["T"] = y1 - (by + bh)
    if code & 8 and abs(by - y0) >= EPS:
        out["B"] = by - y0
    return out


def _label_pos(polygons, n):
    P = []
    for i in range(n):
        blk = polygons[i]
        v = blk[blk[:, 0] != -1]
        if len(v) == 0:
            P.append([0.0, 0.0, 1.0, 1.0])
            continue
        mn = v.min(dim=0).values
        mx = v.max(dim=0).values
        P.append([float(mn[0]), float(mn[1]),
                  float(mx[0] - mn[0]), float(mx[1] - mn[1])])
    return P


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results_L274_base_48c.json"
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    res = {r["test_id"]: r for r in json.load(open(src))["test_results"]}

    rows = []
    for idx in sorted(res):
        P = res[idx].get("positions")
        if not P:
            continue
        s = ev.dataset[idx]
        at, _b2b, _p2b, _pins, cons = s["input"]
        polygons, _metrics = s["label"]
        n = int((at != -1).sum().item())
        P = [list(map(float, p)) for p in P[:n]]
        L = _label_pos(polygons, n)
        fixed = [float(cons[i, 0]) != 0 for i in range(n)]
        pre = [float(cons[i, 1]) != 0 for i in range(n)]
        bnd = [int(cons[i, 4].item()) for i in range(n)]

        for i in range(n):
            if not bnd[i] or not pre[i]:
                continue
            miss = _miss(P, i, bnd[i])
            if not miss:
                continue
            side, dist = min(miss.items(), key=lambda kv: abs(kv[1]))
            # HARD: does an immovable block extend beyond us on that side?
            bx, by, bw, bh = P[i]
            hard = False
            for t in range(n):
                if t == i or not pre[t]:   # NOT fixed[t] -- see the docstring
                    continue
                tx, ty, tw, th = P[t]
                if side == "L" and tx < bx - EPS:
                    hard = True
                if side == "R" and tx + tw > bx + bw + EPS:
                    hard = True
                if side == "B" and ty < by - EPS:
                    hard = True
                if side == "T" and ty + th > by + bh + EPS:
                    hard = True
                if hard:
                    break
            lab_miss = _miss(L, i, bnd[i])
            rows.append(dict(case=idx, blk=i, side=side, dist=abs(dist),
                             hard=hard, label_ok=(side not in lab_miss),
                             n=int(res[idx]["block_count"])))

    print("preplaced boundary violations, in-set 100, current shipped code: {}"
          .format(len(rows)))
    print()
    nh = sum(1 for r in rows if r["hard"])
    nl = sum(1 for r in rows if r["label_ok"])
    both = sum(1 for r in rows if r["hard"] and r["label_ok"])
    print("  HARD  an immovable (preplaced/fixed) block extends beyond it   {:3d}"
          .format(nh))
    print("  LABEL the ground truth SATISFIES the same constraint           {:3d}"
          .format(nl))
    print("        ... of which also HARD (contradiction -> check the test)  {:3d}"
          .format(both))
    print()
    print("  partition:")
    print("    nobody can satisfy it (HARD, label fails too)      {:3d}".format(
        sum(1 for r in rows if r["hard"] and not r["label_ok"])))
    print("    label fails it too, not provably hard              {:3d}".format(
        sum(1 for r in rows if not r["hard"] and not r["label_ok"])))
    print("    LABEL SATISFIES IT -> reachable, we are losing it  {:3d}".format(
        sum(1 for r in rows if r["label_ok"])))
    print()
    print("  {:>5s} {:>5s} {:>4s} {:>6s} {:>10s} {:>6s} {:>9s}".format(
        "case", "n", "blk", "side", "dist", "HARD", "label_ok"))
    for r in sorted(rows, key=lambda r: (-r["n"], r["case"])):
        print("  {:5d} {:5d} {:4d} {:>6s} {:10.4f} {:>6s} {:>9s}".format(
            r["case"], r["n"], r["blk"], r["side"], r["dist"],
            "yes" if r["hard"] else "-", "SAT" if r["label_ok"] else "fails"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
