"""L259 -- at the moment the greedy jams, is the remainder placeable AT ALL?

The recreate bound. L254 says the jam has >=10x the needed area free and a median
of 3 blocks left; L256 showed the shipped greedy cannot use that space. This asks
whether ANY method could, by solving the sub-problem exactly:

    given the blocks already placed and the frame, is there a legal placement of
    every remaining block?

Bounded backtracking over corner points (the standard candidate set for rectangle
packing: for each placed rectangle, its right-bottom and left-top corners, plus
the origin). Optimistic on purpose -- soft constraints (boundary/group/MIB) are
ignored and dims are nominal, so a YES is a geometric upper bound on what a better
recreate could do, and a NO is strong.

  YES on most cases -> a stronger recreate exists; L255 prices the target at
                       +3.23% (re-place ~10% -> s ~ 1.06)
  NO               -> the jam is unrepairable GIVEN THAT PREFIX, so the fix must
                       change the prefix = full re-placement = M27 = L129

  <python> l259_feasible.py --limit 20
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l259.exe"
CACHE = DIR / "l252_cache.pkl"
TOL = 1e-9
_ARGV = list(sys.argv)


def parse(stderr):
    """-> ({frame_idx: (fw,fh,ndone,N,placed,left)}, tries, frames, tot)"""
    jams, cur = {}, None
    tries, frames, tot = {}, [], None
    for line in stderr.splitlines():
        if line.startswith("L259JAM "):
            _, fi, fw, fh, nd, N = line.split()
            cur = dict(fi=int(fi), fw=float(fw), fh=float(fh),
                       nd=int(nd), N=int(N), placed=[], left=[])
            jams[int(fi)] = cur
        elif line.startswith("L259P ") and cur is not None:
            _, i, x, y, w, h = line.split()
            cur["placed"].append((float(x), float(y), float(w), float(h)))
        elif line.startswith("L259U ") and cur is not None:
            _, i, w, h = line.split()
            cur["left"].append((int(i), float(w), float(h)))
        elif line.startswith("L252TRY "):
            _, i, ok, sc = line.split()
            tries[int(i)] = int(ok)
        elif line.startswith("L252FRM "):
            _, i, w, h = line.split()
            frames.append((int(i), float(w), float(h)))
        elif line.startswith("L252TOT "):
            tot = float(line.split()[1])
    return jams, tries, frames, tot


def feasible(placed, left, fw, fh, budget=300000):
    """Bounded backtracking over the FULL bottom-left candidate set.

    The candidate set must be the CROSS PRODUCT {0} u {right edges} x {0} u {top
    edges}: a block can take its x from one rectangle and its y from another, so
    per-rectangle corners alone are not complete. The first version used only
    per-rect corners and returned NO after 1 node on every case -- a solver bug
    that looked exactly like a finding.

    -> (True/False/None, nodes). None = node budget exhausted.
    """
    import numpy as np
    left = sorted(left, key=lambda t: -(t[1] * t[2]))
    dims = [(w, h) for _i, w, h in left]
    nodes = [0]
    exhausted = [False]

    def cands(occ):
        xs = {0.0}
        ys = {0.0}
        for (a, b, c, d) in occ:
            if a + c < fw - TOL:
                xs.add(a + c)
            if b + d < fh - TOL:
                ys.add(b + d)
        xs = np.array(sorted(xs))
        ys = np.array(sorted(ys))
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return X.ravel(), Y.ravel()

    def free_spots(occ, w, h):
        X, Y = cands(occ)
        keep = (X + w <= fw + TOL) & (Y + h <= fh + TOL)
        X, Y = X[keep], Y[keep]
        if X.size == 0:
            return X, Y
        A = np.array([[o[0], o[1], o[0] + o[2], o[1] + o[3]] for o in occ])             if occ else np.zeros((0, 4))
        if A.shape[0]:
            ov = ((X[:, None] + w > A[None, :, 0] + TOL) &
                  (A[None, :, 2] > X[:, None] + TOL) &
                  (Y[:, None] + h > A[None, :, 1] + TOL) &
                  (A[None, :, 3] > Y[:, None] + TOL))
            ok = ~ov.any(axis=1)
            X, Y = X[ok], Y[ok]
        order = np.lexsort((X, Y))          # bottom-left first
        return X[order], Y[order]

    def rec(k, occ):
        if k == len(dims):
            return True
        w, h = dims[k]
        X, Y = free_spots(occ, w, h)
        for j in range(X.size):
            nodes[0] += 1
            if nodes[0] > budget:
                exhausted[0] = True
                return False
            if rec(k + 1, occ + [(float(X[j]), float(Y[j]), w, h)]):
                return True
        return False

    ok = rec(0, list(placed))
    if not ok and exhausted[0]:
        return None, nodes[0]
    return ok, nodes[0]


def sanity(placed, left, fw, fh):
    """Is the dumped state even self-consistent? Returns a short string."""
    bad = 0
    for i in range(len(placed)):
        a = placed[i]
        if a[0] < -TOL or a[1] < -TOL or a[0] + a[2] > fw + TOL or a[1] + a[3] > fh + TOL:
            bad += 1
    ov = 0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = placed[i], placed[j]
            if (a[0] + a[2] > b[0] + TOL and b[0] + b[2] > a[0] + TOL and
                    a[1] + a[3] > b[1] + TOL and b[1] + b[3] > a[1] + TOL):
                ov += 1
    freea = fw * fh - sum(q[2] * q[3] for q in placed)
    needa = sum(w * h for _i, w, h in left)
    # what is the biggest empty axis-aligned rectangle anchored at a candidate?
    import numpy as np
    xs = sorted({0.0} | {q[0] + q[2] for q in placed if q[0] + q[2] < fw - TOL})
    ys = sorted({0.0} | {q[1] + q[3] for q in placed if q[1] + q[3] < fh - TOL})
    bw = bh = 0.0
    barea = 0.0
    for x in xs:
        for y in ys:
            # grow right then up from (x,y) without hitting an obstacle
            w = fw - x
            for (a, b, c, d) in placed:
                if b + d > y + TOL and b < fh and a + c > x + TOL and a >= x - TOL:
                    if b <= y + TOL < b + d - TOL or (b > y and b < fh):
                        pass
            # simple exact scan: max w such that no obstacle intersects [x,x+w)x[y,y+eps)
            wlim = fw - x
            for (a, b, c, d) in placed:
                if b < y + 1e-6 + TOL and b + d > y + TOL and a + c > x + TOL:
                    wlim = min(wlim, max(0.0, a - x))
            hlim = fh - y
            for (a, b, c, d) in placed:
                if a < x + 1e-6 + TOL and a + c > x + TOL and b + d > y + TOL:
                    hlim = min(hlim, max(0.0, b - y))
            if wlim * hlim > barea:
                barea, bw, bh = wlim * hlim, wlim, hlim
    big = max(left, key=lambda t: t[1] * t[2]) if left else (0, 0, 0)
    return ("oob={} ovl={} free/need={:.1f}x  biggest-left {:.1f}x{:.1f}"
            "  best-slot ~{:.1f}x{:.1f}".format(
                bad, ov, freea / max(needa, 1e-9), big[1], big[2], bw, bh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--maxleft", type=int, default=12,
                    help="skip jams with more unplaced blocks than this")
    ap.add_argument("--budget", type=int, default=300000)
    a = ap.parse_args(_ARGV[1:])

    sys.argv = ["x"]
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
    import optimizer_constructive as oc

    if len(list(oc._pool_indices(120))) != 51:
        print("!! not the shipped pool")
        return 1
    RH = oc._RH
    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)
    LADDER = ",".join("{:.4f}".format(1.00 + i * 0.01) for i in range(26))

    C = pickle.load(open(CACHE, "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs(a.sample)}
    keys = sorted([k for k in C if k[0] == a.sample], key=lambda k: -C[k]["n"])[:a.limit]
    loaded = {}
    rows = []
    for kn, key in enumerate(keys):
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
                           env=env, timeout=TO)
        jams, tries, frames, tot = parse(r.stderr)
        oks = [i for i, o in tries.items() if o]
        if not oks or not jams:
            continue
        first_ok = min(oks)
        below = [i for i in jams if i < first_ok]
        if not below:
            continue
        edge = max(below)
        J = jams[edge]
        s_of = {i: math.sqrt(max(w * h, 1e-18) / max(tot or e["sumA"], 1e-18))
                for i, w, h in frames}
        nleft = len(J["left"])
        if nleft > a.maxleft:
            rows.append(dict(n=n, nleft=nleft, verdict="skip", nodes=0,
                             s=s_of.get(edge, float("nan"))))
            print("   n={:3d}  {:2d} left  -- skipped (over --maxleft)".format(n, nleft))
            continue
        sn = sanity(J["placed"], J["left"], J["fw"], J["fh"])
        ok, nodes = feasible(J["placed"], J["left"], J["fw"], J["fh"], a.budget)
        v = "YES" if ok else ("NO" if ok is False else "budget")
        rows.append(dict(n=n, nleft=nleft, verdict=v, nodes=nodes,
                         s=s_of.get(edge, float("nan"))))
        print("   n={:3d}  s={:.4f}  {:2d} left  -> {:6s}  ({} nodes)  [{}]".format(
            n, s_of.get(edge, float("nan")), nleft, v, nodes, sn))

    if not rows:
        print("nothing solved")
        return 1
    pickle.dump(rows, open(DIR / "l259_feasible.pkl", "wb"))
    dec = [r for r in rows if r["verdict"] in ("YES", "NO")]
    print()
    print("=" * 62)
    print("L259 recreate bound -- {} jams, {} decided".format(len(rows), len(dec)))
    print("=" * 62)
    for v in ("YES", "NO", "budget", "skip"):
        k = sum(1 for r in rows if r["verdict"] == v)
        if k:
            print("  {:7s} {:3d}".format(v, k))
    if dec:
        y = sum(1 for r in dec if r["verdict"] == "YES")
        print()
        print("  => the remainder IS placeable in {}/{} decided jams ({:.0f}%)".format(
            y, len(dec), 100.0 * y / len(dec)))
        print("     YES means a stronger recreate exists for that case;")
        print("     NO means the prefix itself has to change (M27 territory).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
