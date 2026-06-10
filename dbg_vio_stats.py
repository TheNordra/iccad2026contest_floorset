"""One-off: classify ALL violating boundary blocks in the PORTFOLIO output
(optimizer_constructive_results.json) across 100 cases.

For each violating block report: single / cluster-member / preplaced, and for
singles whether snapping every unsatisfied bit to its bbox edge is overlap-free
(i.e. whether the M16 constrained-axis repair could fix it at all).
"""
import json, sys
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator

L, R, T, B = 1, 2, 4, 8
EPS = 1e-6

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
port = json.load(open(_DIR / "iccad2026contest" / "optimizer_constructive_results.json"))
pjson = {t["test_id"]: t for t in port["test_results"]}

def overlaps(ps, self_i, x, y, w, h):
    for j, (px, py, pw, ph) in enumerate(ps):
        if j == self_i:
            continue
        ox = min(x + w, px + pw) - max(x, px)
        oy = min(y + h, py + ph) - max(y, py)
        if ox > 1e-6 and oy > 1e-6:
            return j
    return -1

tot = dict(single=0, cluster=0, preplaced=0, single_free=0, single_blocked=0)
per_case = []
for idx in range(len(ev.dataset)):
    s = ev.dataset[idx]; inp = s["input"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    ps = [tuple(p) for p in pjson[idx]["positions"]]
    xmin = min(p[0] for p in ps); ymin = min(p[1] for p in ps)
    xmax = max(p[0] + p[2] for p in ps); ymax = max(p[1] + p[3] for p in ps)
    cs = ck = cp = cf = cb = 0
    for i in range(n):
        code = int(cons[i, 4]) if cons.shape[1] > 4 else 0
        if code == 0:
            continue
        x, y, w, h = ps[i]
        clu = int(cons[i, 3]) if cons.shape[1] > 3 else 0
        pre = (cons[i, 1] != 0) if cons.shape[1] > 1 else False
        tch = []
        if code & L: tch.append(abs(x - xmin) < EPS)
        if code & R: tch.append(abs(x + w - xmax) < EPS)
        if code & T: tch.append(abs(y + h - ymax) < EPS)
        if code & B: tch.append(abs(y - ymin) < EPS)
        if all(tch):
            continue
        if pre:
            cp += 1
        elif clu > 0:
            ck += 1
        else:
            cs += 1
            nx, ny = x, y
            if code & L: nx = xmin
            if code & R: nx = xmax - w
            if code & B: ny = ymin
            if code & T: ny = ymax - h
            if overlaps(ps, i, nx, ny, w, h) >= 0:
                cb += 1
            else:
                cf += 1
    tot["single"] += cs; tot["cluster"] += ck; tot["preplaced"] += cp
    tot["single_free"] += cf; tot["single_blocked"] += cb
    if cs + ck + cp:
        per_case.append((idx, n, cs, cf, cb, ck, cp))

print(f"TOTAL violating boundary blocks across 100 cases:")
print(f"  singles   = {tot['single']}  (snap-to-edge overlap-free: {tot['single_free']}, blocked: {tot['single_blocked']})")
print(f"  cluster   = {tot['cluster']}")
print(f"  preplaced = {tot['preplaced']}")
print(f"\nper-case (cases with any violation): case n | single(free/blocked) cluster preplaced")
for idx, n, cs, cf, cb, ck, cp in per_case:
    print(f"  {idx:>3} {n:>4} | single={cs}({cf}/{cb}) cluster={ck} preplaced={cp}")
