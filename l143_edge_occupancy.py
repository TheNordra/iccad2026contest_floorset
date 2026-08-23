"""L143 - WHO is sitting on the edge the violator needed?

L141 established that 99.6% of held-out boundary misses are on sides with spare
total capacity, so the miss is a packing decision. This asks the next question,
which decides what the fix has to be:

  * if the largest CONTIGUOUS free interval on that edge is smaller than the
    violating block, the edge is FRAGMENTED -- the space exists but not in one
    piece, so the fix is sequencing/reservation (place boundary blocks first,
    or keep a run of edge free for them);
  * if a big enough gap exists, the placer simply preferred another position for
    that block, and the fix is in the candidate scoring for that block;
  * and either way, whether the blocks holding the edge are themselves
    boundary-constrained ("entitled") or not ("squatters") says whether the edge
    is contested at all.

READ-ONLY.

  <python> -u l143_edge_occupancy.py l140_oos_s1_c48.json --sample s1
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

import l141_edge_capacity as L141                                   # noqa: E402

EPS = 1e-6
BITS = {1: "L", 2: "R", 4: "T", 8: "B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--sample", default="")
    ap.add_argument("--inset", action="store_true")
    a = ap.parse_args()

    frag = fits = 0
    gaps, squat, entitled, contested = [], 0, 0, 0
    for idx, P, cons, n in L141._cases(a):
        P = [list(map(float, p)) for p in P[:n]]
        bnd = cons[:n, 4]
        if int((bnd != 0).sum().item()) == 0:
            continue
        x0 = min(p[0] for p in P)
        y0 = min(p[1] for p in P)
        x1 = max(p[0] + p[2] for p in P)
        y1 = max(p[1] + p[3] for p in P)
        for bit, side in BITS.items():
            req = [i for i in range(n) if int(bnd[i].item()) & bit]
            if not req:
                continue
            touch = {1: lambda p: abs(p[0] - x0) < EPS,
                     2: lambda p: abs(p[0] + p[2] - x1) < EPS,
                     4: lambda p: abs(p[1] + p[3] - y1) < EPS,
                     8: lambda p: abs(p[1] - y0) < EPS}[bit]
            miss = [i for i in req if not touch(P[i])]
            if not miss:
                continue
            # who is on the edge, and where
            lo, hi = (y0, y1) if side in "LR" else (x0, x1)
            occ = []
            for t in range(n):
                if not touch(P[t]):
                    continue
                s = P[t][1] if side in "LR" else P[t][0]
                e = s + (P[t][3] if side in "LR" else P[t][2])
                occ.append((s, e, int(bnd[t].item()) & bit != 0))
            occ.sort()
            for _s, _e, ent in occ:
                if ent:
                    entitled += 1
                else:
                    squat += 1
            # Largest contiguous free run, measured on the STRIP the violator
            # would actually occupy: putting block i on the left edge claims
            # [x0, x0+w_i] x [y, y+h_i], so ANY block overlapping that strip
            # blocks it -- not just the ones already touching the edge. The
            # first version of this probe only counted edge-touching blocks and
            # therefore over-reported available space.
            for i in miss:
                ext = P[i][3] if side in "LR" else P[i][2]
                depth = P[i][2] if side in "LR" else P[i][3]
                if side == "L":
                    d0, d1 = x0, x0 + depth
                elif side == "R":
                    d0, d1 = x1 - depth, x1
                elif side == "B":
                    d0, d1 = y0, y0 + depth
                else:
                    d0, d1 = y1 - depth, y1
                occ2 = []
                for t in range(n):
                    if t == i:
                        continue
                    ds, de = ((P[t][0], P[t][0] + P[t][2]) if side in "LR"
                              else (P[t][1], P[t][1] + P[t][3]))
                    if min(de, d1) - max(ds, d0) <= EPS:      # outside the strip
                        continue
                    s = P[t][1] if side in "LR" else P[t][0]
                    occ2.append((s, s + (P[t][3] if side in "LR" else P[t][2])))
                occ2.sort()
                free, cur = [], lo
                for s, e in occ2:
                    if s > cur:
                        free.append(s - cur)
                    cur = max(cur, e)
                if hi > cur:
                    free.append(hi - cur)
                biggest = max(free) if free else 0.0
                gaps.append(biggest / max(ext, 1e-12))
                if biggest + EPS < ext:
                    frag += 1
                else:
                    fits += 1
                    if any(not ent for _s, _e, ent in occ):
                        contested += 1

    tot = frag + fits
    print(f"\n=== L143 edge occupancy: {Path(a.results).name} ===\n")
    print(f"boundary side-misses examined: {tot}")
    print(f"  edge FRAGMENTED (no gap big enough)  {frag:>5}  "
          f"({100 * frag / max(tot, 1):.1f}%)")
    print(f"  a big enough gap EXISTED             {fits:>5}  "
          f"({100 * fits / max(tot, 1):.1f}%)")
    print(f"    of those, non-boundary blocks hold part of the edge: "
          f"{contested}")
    if gaps:
        print(f"\nlargest free run / violator extent:  median {st.median(gaps):.3f}"
              f"   p10 {sorted(gaps)[int(0.1 * len(gaps))]:.3f}"
              f"   p90 {sorted(gaps)[int(0.9 * len(gaps))]:.3f}")
    print(f"\nblocks holding constrained edges: {entitled} entitled "
          f"(boundary-constrained) vs {squat} squatters "
          f"({100 * squat / max(entitled + squat, 1):.1f}% squatters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
