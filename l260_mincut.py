"""L260 -- how big must the ruin set be? The minimum displacement to open a slot.

L259: at the jam, the largest unplaced block has ZERO legal positions, while
smaller ones have thousands. So the obstruction is contiguity, and the only route
left is coordinated re-placement (M27). This sizes it.

For the blocking block (w,h), over every candidate anchor, count how many ALREADY
PLACED blocks its footprint would overlap. The minimum over anchors is the
**minimum number of blocks that must be displaced** to open a slot for it -- a
lower bound on the ruin set, and therefore the minimum scale of any M27-style
coordinated move.

Anchors are the cross product of
    {0} u {right edges} u {left edges - w}   x   {0} u {top edges} u {bottom - h}
so the block can be pushed against an obstacle from either side. Exact rectangle
arithmetic, no raster.

Reported against L255's target: re-place ~10% of the design -> s ~ 1.06 -> +3.23%.

  <python> l260_mincut.py --cases 8
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l259.exe"
TOL = 1e-9
_ARGV = list(sys.argv)


def min_displacement(placed, w, h, fw, fh):
    """-> (min_count, area_of_that_set, n_anchors). placed: list of (x,y,w,h)."""
    if not placed:
        return 0, 0.0, 0
    P = np.array([[q[0], q[1], q[0] + q[2], q[1] + q[3]] for q in placed])
    A = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])
    xs = {0.0}
    ys = {0.0}
    for (a, b, c, d) in placed:
        if a + c <= fw - w + TOL:
            xs.add(a + c)
        if a - w >= -TOL:
            xs.add(a - w)
        if b + d <= fh - h + TOL:
            ys.add(b + d)
        if b - h >= -TOL:
            ys.add(b - h)
    xs = np.array([v for v in sorted(xs) if v >= -TOL and v + w <= fw + TOL])
    ys = np.array([v for v in sorted(ys) if v >= -TOL and v + h <= fh + TOL])
    if xs.size == 0 or ys.size == 0:
        return -1, 0.0, 0
    best_c, best_a = 10 ** 9, 0.0
    # loop x, vectorise over y and rectangles (keeps memory bounded)
    for x in xs:
        oxs = (x + w > P[:, 0] + TOL) & (P[:, 2] > x + TOL)          # (R,)
        if not oxs.any():
            # a whole free column: any y with no vertical overlap wins outright
            pass
        oy = ((ys[:, None] + h > P[None, :, 1] + TOL) &
              (P[None, :, 3] > ys[:, None] + TOL))                    # (Y,R)
        hit = oy & oxs[None, :]
        cnt = hit.sum(axis=1)
        j = int(np.argmin(cnt))
        if cnt[j] < best_c:
            best_c = int(cnt[j])
            best_a = float(A[hit[j]].sum())
        elif cnt[j] == best_c:
            best_a = min(best_a, float(A[hit[j]].sum()))
    return best_c, best_a, int(xs.size * ys.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args(_ARGV[1:])

    sys.argv = ["x"]
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
    import optimizer_constructive as oc
    sys.path.insert(0, str(DIR))
    from l259_feasible import parse

    C = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs("s1")}
    keys = sorted([k for k in C if k[0] == "s1"], key=lambda k: -C[k]["n"])[:a.cases]
    LADDER = ",".join("{:.4f}".format(1.00 + i * 0.01) for i in range(26))
    RH = oc._RH
    loaded = {}
    rows = []
    for key in keys:
        ck = key[1]
        e = C[key]
        fk, L, n = spec_of[ck]
        if fk not in loaded:
            loaded.clear()
            loaded[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(loaded[fk], L)
        otp = m67.build_opt_target_pos(lay["tp"], lay["cons"], n)
        hint = None
        if bool(oc._l137_env()) or bool(oc._l137_active(n)):
            try:
                hint = oc._gordian_hint(n, lay["at"], lay["b2b"], lay["p2b"],
                                        lay["pins"], lay["cons"], otp)
            except Exception:
                hint = None
        inp = oc._serialize_input(n, lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                                  lay["cons"], otp, gnn_hint=hint)
        idxs = sorted(e["recs"])
        met = [e["recs"][i] for i in idxs]
        A_hat = 1.035 * max(e["sumA"], 1e-9)
        hmin = min(m["hpwl"] for m in met) or 1.0
        prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
                for m in met]
        widx = idxs[min(range(len(idxs)), key=lambda t: prox[t])]
        prof = dict(oc._PROFILES[widx])
        ov = oc._profile_env(widx, n)
        if ov:
            prof.update(ov)
        env = dict(os.environ)
        env.update(prof)
        env["ICCAD_FRAME_SCALES"] = LADDER
        env["ICCAD_L252"] = "1"
        env["ICCAD_L259"] = "1"
        r = subprocess.run([str(PROBE)], input=inp, capture_output=True, text=True,
                           env=env, timeout=300)
        jams, tries, frames, tot = parse(r.stderr)
        oks = [i for i, o in tries.items() if o]
        below = [i for i in jams if oks and i < min(oks)]
        if not below:
            continue
        J = jams[max(below)]
        sumA = tot or e["sumA"]
        # the blocking block = the largest unplaced one
        bi, bw, bh = max(J["left"], key=lambda t: t[1] * t[2])
        cnt, area, na = min_displacement(J["placed"], bw, bh, J["fw"], J["fh"])
        rows.append(dict(n=n, nleft=len(J["left"]), bw=bw, bh=bh, cnt=cnt,
                         area=area, sumA=sumA, na=na,
                         frac=area / max(sumA, 1e-9)))
        print("   n={:3d}  {:2d} left  blocker {:.1f}x{:.1f}  -> displace >= {} blocks"
              "  ({:.1f} area = {:.2f}% of design)  [{} anchors]".format(
                  n, len(J["left"]), bw, bh, cnt, area,
                  100.0 * area / max(sumA, 1e-9), na))

    if not rows:
        print("nothing measured")
        return 1
    pickle.dump(rows, open(DIR / "l260_mincut.pkl", "wb"))
    cs = sorted(r["cnt"] for r in rows)
    fs = sorted(r["frac"] for r in rows)
    print()
    print("=" * 70)
    print("L260 -- minimum ruin set to unblock the jam, {} cases".format(len(rows)))
    print("=" * 70)
    print("  blocks that must be displaced   min {}  p50 {}  max {}".format(
        cs[0], cs[len(cs) // 2], cs[-1]))
    print("  their area, as % of the design  min {:.2f}%  p50 {:.2f}%  max {:.2f}%".format(
        100 * fs[0], 100 * fs[len(fs) // 2], 100 * fs[-1]))
    print()
    print("  L255's target for +3.23% was 're-place ~10% of the design'.")
    print("  This is a LOWER bound: the displaced blocks then need homes too, and")
    print("  only the single largest blocker is counted -- the other unplaced")
    print("  blocks may need their own slots.")
    p50 = fs[len(fs) // 2]
    print()
    if p50 * 100 < 2.0:
        print("  => the minimum coordinated move is SMALL ({:.2f}% of the design at"
              " the median).".format(100 * p50))
        print("     A bounded local search over a few blocks is the right shape,")
        print("     not a from-scratch global placer.")
    else:
        print("  => the minimum coordinated move is already {:.2f}% of the design,"
              " before".format(100 * p50))
        print("     any cascade. That is re-packer scale, i.e. L129 territory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
