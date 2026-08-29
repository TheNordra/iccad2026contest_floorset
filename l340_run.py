"""L340 driver: feed a case to the C++ B*-tree SA and score the result officially.

THE WEIGHT. Our score's quality factor is 1 + 0.5*(hpwl/hpwl_L + area/area_L), so the
objective whose gradient matches it is  area + (area_L/hpwl_L) * hpwl.  That makes
HW* = area_L / hpwl_L the RIGHT weight -- and both terms are label values.

area_L is available label-free: L320 measured label utilisation at p50 0.9693 and
w*h = A exactly, so area_L ~= sum(A)/0.971 to about +-1%. hpwl_L is NOT available.
So this probe sweeps HW as a multiple of the oracle HW* to map the frontier first;
whether a label-free proxy lands near the optimum is the next question, not this one.

SCOPE: preplaced blocks are packed like any other block (a B*-tree cannot express a
fixed coordinate), so these layouts are not submittable. This measures what the
manifold can reach on the objective we are scored on.
"""
import glob
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, "iccad2026contest")
from iccad2026_evaluate import (calculate_bbox_area, calculate_hpwl_b2b,  # noqa: E402
                                calculate_hpwl_p2b)

LAB = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth")}
DAT = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litedata_1.pth")}
EXE = os.path.abspath("l340_btree.exe")


def divpairs(A, lo=1 / 3, hi=3.0):
    A = int(round(A))
    out = []
    w = 1
    while w * w <= A:
        if A % w == 0:
            for a, b in ((w, A // w), (A // w, w)):
                if lo - 1e-12 <= a / b <= hi + 1e-12:
                    out.append((a, b))
        w += 1
    return sorted(set(out)) or [(A, 1)]


def load(n):
    d = torch.load(DAT[n], weights_only=False)[0]
    meta, b2b, p2b, pins = d[0], d[1], d[2], d[3]
    m8 = torch.load(LAB[n], weights_only=False)[0][0]
    nb = int((meta[:, 0] > 0).sum())
    return meta[:nb], b2b, p2b, pins, float(m8[0]), float(m8[-2]) + float(m8[-1]), nb


def run(n, hw, iters, seed=1):
    meta, b2b, p2b, pins, arL, hpL, nb = load(n)
    shapes = [divpairs(float(meta[k, 0])) for k in range(nb)]
    P = pins[pins[:, 0] >= 0].float()
    e = b2b[(b2b[:, 0] >= 0) & (b2b[:, 1] >= 0) & (b2b[:, 0] < nb) & (b2b[:, 1] < nb)]
    e = e[e[:, 0] != e[:, 1]]
    pe = p2b[(p2b[:, 0] >= 0) & (p2b[:, 1] >= 0) & (p2b[:, 1] < nb) & (p2b[:, 0] < len(P))]
    lines = ["%d %.12g %d %d" % (nb, hw, iters, seed)]
    for k in range(nb):
        lines.append("%d %d %s" % (k, len(shapes[k]),
                                   " ".join("%d %d" % s for s in shapes[k])))
    lines.append(str(len(e)))
    for r in e:
        lines.append("%d %d %.12g" % (int(r[0]), int(r[1]), float(r[2])))
    lines.append(str(len(P)))
    for p in P:
        lines.append("%.12g %.12g" % (float(p[0]), float(p[1])))
    lines.append(str(len(pe)))
    for r in pe:
        lines.append("%d %d %.12g" % (int(r[0]), int(r[1]), float(r[2])))
    t0 = time.time()
    out = subprocess.run([EXE], input="\n".join(lines), capture_output=True,
                         text=True, timeout=3600)
    dt = time.time() - t0
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:400])
    rows = out.stdout.strip().split("\n")
    W, H = (int(x) for x in rows[0].split()[:2])
    pos = []
    for r in rows[1:1 + nb]:
        x, y, w, h = (int(v) for v in r.split())
        pos.append((x, y, w, h))
    hp = calculate_hpwl_b2b(pos, b2b) + calculate_hpwl_p2b(pos, p2b, pins)
    ar = calculate_bbox_area(pos)
    sumA = sum(int(round(float(meta[k, 0]))) for k in range(nb))
    return dict(n=n, util=sumA / (W * H), hg=max(0.0, (hp - hpL) / hpL),
                ag=max(0.0, (ar - arL) / arL), dt=dt, hw_star=arL / hpL,
                pos=pos, W=W, H=H)


if __name__ == "__main__":
    NS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "40,80,120").split(",")]
    IT = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    MULTS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3
                                else "0,0.25,1,4").split(",")]
    print("== L340 C++ B*-tree SA, area + HW*wirelength, %d iterations ==" % IT)
    print("   HW* = area_L / hpwl_L is the weight matching our score's gradient")
    print()
    print("   %-5s %8s %8s %9s %9s %9s %8s"
          % ("n", "HW/HW*", "util", "hpwl_gap", "area_gap", "quality", "time"))
    for n in NS:
        _m, _b, _p, _pi, arL, hpL, _nb = load(n)
        star = arL / hpL
        for mu in MULTS:
            r = run(n, star * mu, IT)
            q = 1 + 0.5 * (r["hg"] + r["ag"])
            print("   %-5d %8.2f %8.4f %9.4f %9.4f %9.4f %7.1fs"
                  % (n, mu, r["util"], r["hg"], r["ag"], q, r["dt"]))
        print()
    print("   our shipped mix arm: util 0.877  hpwl_gap 0.2402  area_gap 0.1176"
          "  quality 1.1789")
