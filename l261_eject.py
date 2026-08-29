"""L261 -- does the ejection chain terminate? This is the M27 sizing number.

L260: displacing exactly ONE placed block opens a slot for the blocker, in 8/8
cases, at a median of 0.75% of the design. But that is a lower bound -- the
displaced block then needs a home of its own, which may displace another. The
depth at which that cascade terminates IS the scale of the mechanism:

    depth 1-3   a bounded ejection chain. Cheap, local, implementable inside the
                existing packer as a repair at the jam.
    deep/never  a re-packer. L129 territory, days of engineering.

Greedy chain: for the block that needs a home, take the anchor that displaces the
fewest already-placed blocks (0 if one exists -> that branch terminates); evict
those, place, and push the evicted onto the queue. Bounded in depth and in
per-block evictions so a cycle cannot spin.

Every unplaced block from the jam is queued, not just the blocker, so a SUCCESS
here means the whole layout completes at that frame.

  <python> l261_eject.py --cases 10 --maxdepth 40
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


def best_anchor(P, A, w, h, fw, fh):
    """-> (count, x, y, victim_idx array) minimising displaced-block count."""
    if P.shape[0] == 0:
        return 0, 0.0, 0.0, np.zeros(0, dtype=int)
    xs = {0.0}
    ys = {0.0}
    for k in range(P.shape[0]):
        a, b, c2, d2 = P[k]
        if c2 <= fw - w + TOL:
            xs.add(float(c2))
        if a - w >= -TOL:
            xs.add(float(a - w))
        if d2 <= fh - h + TOL:
            ys.add(float(d2))
        if b - h >= -TOL:
            ys.add(float(b - h))
    xs = np.array([v for v in sorted(xs) if -TOL <= v and v + w <= fw + TOL])
    ys = np.array([v for v in sorted(ys) if -TOL <= v and v + h <= fh + TOL])
    if xs.size == 0 or ys.size == 0:
        return -1, 0.0, 0.0, np.zeros(0, dtype=int)
    best = (10 ** 9, 0.0, 0.0, None, 1e18)
    for x in xs:
        oxs = (x + w > P[:, 0] + TOL) & (P[:, 2] > x + TOL)
        oy = ((ys[:, None] + h > P[None, :, 1] + TOL) &
              (P[None, :, 3] > ys[:, None] + TOL))
        hit = oy & oxs[None, :]
        cnt = hit.sum(axis=1)
        j = int(np.argmin(cnt))
        c = int(cnt[j])
        ar = float(A[hit[j]].sum())
        if c < best[0] or (c == best[0] and ar < best[4]):
            best = (c, float(x), float(ys[j]), np.where(hit[j])[0], ar)
        if best[0] == 0:
            break
    return best[0], best[1], best[2], (best[3] if best[3] is not None
                                       else np.zeros(0, dtype=int))


def chain(placed, left, fw, fh, maxdepth, maxevict):
    """-> (ok, depth, total_evictions, max_queue)"""
    P = np.array([[q[0], q[1], q[0] + q[2], q[1] + q[3]] for q in placed],
                 dtype=float) if placed else np.zeros((0, 4))
    A = ((P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])) if P.shape[0] else np.zeros(0)
    queue = [(w, h) for _i, w, h in sorted(left, key=lambda t: -(t[1] * t[2]))]
    evicted = 0
    depth = 0
    maxq = len(queue)
    while queue:
        depth += 1
        if depth > maxdepth or evicted > maxevict:
            return False, depth, evicted, maxq
        w, h = queue.pop(0)
        c, x, y, vic = best_anchor(P, A, w, h, fw, fh)
        if c < 0:
            return False, depth, evicted, maxq
        if c > 0:
            for k in sorted(vic.tolist(), reverse=True):
                queue.append((float(P[k, 2] - P[k, 0]), float(P[k, 3] - P[k, 1])))
                P = np.delete(P, k, axis=0)
                A = np.delete(A, k, axis=0)
            evicted += len(vic)
        P = np.vstack([P, [[x, y, x + w, y + h]]])
        A = np.append(A, w * h)
        maxq = max(maxq, len(queue))
    return True, depth, evicted, maxq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--maxdepth", type=int, default=40)
    ap.add_argument("--maxevict", type=int, default=60)
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
        ok, d, ev, mq = chain(J["placed"], J["left"], J["fw"], J["fh"],
                              a.maxdepth, a.maxevict)
        rows.append(dict(n=n, nleft=len(J["left"]), ok=ok, depth=d, evict=ev,
                         maxq=mq))
        print("   n={:3d}  {:2d} left  -> {:7s}  depth {:3d}  evictions {:3d}"
              "  peak queue {:2d}".format(
                  n, len(J["left"]), "SOLVED" if ok else "no", d, ev, mq))

    if not rows:
        print("nothing run")
        return 1
    pickle.dump(rows, open(DIR / "l261_eject.pkl", "wb"))
    ok = [r for r in rows if r["ok"]]
    print()
    print("=" * 66)
    print("L261 ejection chain -- {} jams".format(len(rows)))
    print("=" * 66)
    print("  completed the layout at the tighter frame:  {}/{}".format(
        len(ok), len(rows)))
    if ok:
        ds = sorted(r["evict"] for r in ok)
        print("  evictions needed   min {}  p50 {}  max {}".format(
            ds[0], ds[len(ds) // 2], ds[-1]))
        qs = sorted(r["maxq"] for r in ok)
        print("  peak queue depth   min {}  p50 {}  max {}".format(
            qs[0], qs[len(qs) // 2], qs[-1]))
    print()
    print("  This is a GREEDY chain with no backtracking: SOLVED is a constructive")
    print("  proof that the frame is packable; 'no' is not a proof that it is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
