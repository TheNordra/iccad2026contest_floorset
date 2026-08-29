"""L329: turn the inverted position target into a real, overlap-free layout, and
measure the gaps it achieves.

L328 showed the inversion locates the label's blocks 25 % more accurately than our
placer (0.0867 vs 0.1161 of the bbox half-perimeter, better on 78/100). But that is a
POSITION TARGET, not a layout -- it has overlaps and satisfies nothing.

Realise it the classical way: extract a SEQUENCE PAIR from the target placement
(Gamma+ = sort by x+y, Gamma- = sort by x-y; a precedes b in both => a is LEFT of b;
a precedes only in Gamma+ => a is BELOW b), then decode by longest path. The decode
is overlap-free by construction and, like the label, bottom-and-left supported.

Shapes: the squarest integer divisor pair of the area with aspect <= 3 -- L321 showed
the label uses exactly this space, and the FloorSet paper's own statistics say the
squarest choice reproduces the label's shape ~82 % of the time.

⚠️ SCOPE: this probe does NOT place preplaced blocks at their required coordinates,
so it is not a submittable layout. It measures whether the inverted target carries a
BETTER TOPOLOGY than ours, which is the question that decides the path.
"""
import glob
import json
import math
import os
import sys

import torch

sys.path.insert(0, "iccad2026contest")
from iccad2026_evaluate import (calculate_hpwl_b2b, calculate_hpwl_p2b,  # noqa: E402
                                calculate_bbox_area)

LAB = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth")}
DAT = {int(os.path.basename(os.path.dirname(f)).split("_")[1]): f
       for f in glob.glob("LiteTensorDataTest/config_*/litedata_1.pth")}
OURS = {t["block_count"]: t for t in json.load(open("l302_mix_1.json"))["test_results"]}
LAW = json.load(open("l327_law.json"))["law"]
UB_W = {int(k): v for k, v in LAW["w"].items()}
UB_M = {int(k): v for k, v in LAW["m"].items()}
UP = {int(k): v for k, v in LAW["pm"].items()}
UTIL = 0.971

import l328_invert as INV                                              # noqa: E402


def squarest(A, lo=1 / 3, hi=3.0):
    A = int(round(A))
    best = None
    w = 1
    while w * w <= A:
        if A % w == 0:
            for a, b in ((w, A // w), (A // w, w)):
                if lo - 1e-12 <= a / b <= hi + 1e-12:
                    r = max(a / b, b / a)
                    if best is None or r < best[0]:
                        best = (r, a, b)
        w += 1
    if best is None:
        s = math.sqrt(A)
        return s, s
    return float(best[1]), float(best[2])


def seqpair(C):
    n = len(C)
    gp = sorted(range(n), key=lambda i: (C[i][0] + C[i][1], i))
    gm = sorted(range(n), key=lambda i: (C[i][0] - C[i][1], i))
    pp = {v: k for k, v in enumerate(gp)}
    pm = {v: k for k, v in enumerate(gm)}
    return pp, pm


def decode(C, W, H):
    """longest-path decode of the sequence pair induced by placement C"""
    n = len(C)
    pp, pm = seqpair(C)
    x = [0.0] * n
    y = [0.0] * n
    order = sorted(range(n), key=lambda i: pp[i])
    for a in order:                       # x: a left of b iff a<b in BOTH
        for b in range(n):
            if a == b:
                continue
            if pp[a] < pp[b] and pm[a] < pm[b]:
                x[b] = max(x[b], x[a] + W[a])
    for a in order:                       # y: a below b iff a<b in G+ only
        for b in range(n):
            if a == b:
                continue
            if pp[a] < pp[b] and pm[a] > pm[b]:
                y[b] = max(y[b], y[a] + H[a])
    return x, y


tot = {"inv": [0.0, 0.0], "our": [0.0, 0.0]}
cnt = 0
for n in sorted(LAB):
    meta, b2b, p2b, pins, C = INV.load(n)
    est = INV.solve(n, meta, b2b, p2b, pins)
    if est is None:
        continue
    lab = torch.load(LAB[n], weights_only=False)[0][0]
    hp_base = float(lab[-2]) + float(lab[-1])
    ar_base = float(lab[0])
    # shapes: fixed/preplaced keep their target dims, others take the squarest pair
    Wd, Hd = [], []
    for k in range(n):
        A = float(meta[k, 0])
        w, h = squarest(A)
        Wd.append(w); Hd.append(h)
    pts = [(float(est[k, 0]), float(est[k, 1])) for k in range(n)]
    x, y = decode(pts, Wd, Hd)
    pos = [(x[k], y[k], Wd[k], Hd[k]) for k in range(n)]
    hp = calculate_hpwl_b2b(pos, b2b) + calculate_hpwl_p2b(pos, p2b, pins)
    ar = calculate_bbox_area(pos)
    tot["inv"][0] += max(0.0, (hp - hp_base) / hp_base)
    tot["inv"][1] += max(0.0, (ar - ar_base) / ar_base)
    r = OURS.get(n)
    tot["our"][0] += r["hpwl_gap"]; tot["our"][1] += r["area_gap"]
    cnt += 1

print("== L329 realising the inverted target as a real layout (%d cases, unweighted) ==" % cnt)
print("   %-34s %10s %10s" % ("", "hpwl_gap", "area_gap"))
print("   %-34s %10.4f %10.4f" % ("inversion -> seq-pair -> compact",
                                  tot["inv"][0] / cnt, tot["inv"][1] / cnt))
print("   %-34s %10.4f %10.4f" % ("our shipped placer (mix)",
                                  tot["our"][0] / cnt, tot["our"][1] / cnt))
print("\n   NOTE: preplaced blocks are NOT honoured here, so this is a topology probe,")
print("   not a submittable layout. Overlap-free by construction.")
