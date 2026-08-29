"""L259 sanity -- brute-force ONE jam on a raster, to arbitrate solver vs diagnostic.

Two bugs in a row in the backtracking solver (sparse candidate set, then 0 spots
on the fixed one) and a slot diagnostic that over-estimates (max free width and
max free height are computed independently, so an L-shaped region reads as a
rectangle). Neither is trustworthy. This rasterises the jam and asks the question
with no cleverness at all.
"""
import os
import pickle
import subprocess
import sys
import math
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l259.exe"
RES = int(os.environ.get("RES", "600"))

sys.argv = ["x"]
import torch                                                   # noqa: E402
import m67_oos_probe as m67                                    # noqa: E402
import m77_oos_probe as m77                                    # noqa: E402
os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
import optimizer_constructive as oc                            # noqa: E402
sys.path.insert(0, str(DIR))
from l259_feasible import parse                                # noqa: E402

C = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs("s1")}
keys = sorted([k for k in C if k[0] == "s1"], key=lambda k: -C[k]["n"])
LADDER = ",".join("{:.4f}".format(1.00 + i * 0.01) for i in range(26))
RH = oc._RH
loaded = {}

for key in keys[:6]:
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
    fw, fh, placed, left = J["fw"], J["fh"], J["placed"], J["left"]

    # rasterise: cell (i,j) occupied if its CENTRE lies inside any placed rect
    gx = np.linspace(0, fw, RES, endpoint=False) + fw / RES / 2
    gy = np.linspace(0, fh, RES, endpoint=False) + fh / RES / 2
    occ = np.zeros((RES, RES), dtype=np.int32)
    for (a, b, c, d) in placed:
        i0 = np.searchsorted(gx, a)
        i1 = np.searchsorted(gx, a + c)
        j0 = np.searchsorted(gy, b)
        j1 = np.searchsorted(gy, b + d)
        occ[i0:i1, j0:j1] = 1
    S = occ.cumsum(0).cumsum(1)
    S = np.pad(S, ((1, 0), (1, 0)))

    print("=== n={} fw={:.2f} fh={:.2f}  {} placed  {} left  (raster {}) ===".format(
        n, fw, fh, len(placed), len(left), RES))
    print("    raster occupancy {:.1f}%  (area occupancy {:.1f}%)".format(
        100.0 * occ.mean(),
        100.0 * sum(q[2] * q[3] for q in placed) / (fw * fh)))
    for (bi, w, h) in sorted(left, key=lambda t: -(t[1] * t[2]))[:3]:
        cw = max(1, int(math.ceil(w / (fw / RES))))
        ch = max(1, int(math.ceil(h / (fh / RES))))
        if cw > RES or ch > RES:
            print("    block {:3d} {:.1f}x{:.1f}: bigger than the frame".format(bi, w, h))
            continue
        tot_win = (S[cw:, ch:] - S[:-cw, ch:] - S[cw:, :-ch] + S[:-cw, :-ch])
        free = (tot_win == 0)
        nfree = int(free.sum())
        print("    block {:3d} {:6.1f}x{:<6.1f} -> {} free raster positions{}".format(
            bi, w, h, nfree, "  <== FITS" if nfree else "  (no fit)"))
    print()
