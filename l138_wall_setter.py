"""L138 — who sets the wall, and is that profile ever worth its cost?

L137's clean timing showed that on every heavy case measured the binding term of
M67-E's wall model is `max_k dt_k`, not `sum_k dt_k / cores`:

    case 90  max 0.925  sum/48 0.582
    case 96  max 1.402  sum/48 0.870
    case 99  max 1.542  sum/48 0.883

So the per-case wall is ONE profile's runtime. Cutting cost therefore means
either making the max-setter faster or not running it -- and M41/M42 already did
exactly that once, dropping the six ORDER_SWAP/ORDER_MOVE profiles that "set the
18-20s wall yet the proxy never selects them on big cases". This asks the same
question of the pool as it stands today, and it asks it with the REAL selector
rather than a reimplementation:

    proxy = (area/A_hat + 1.4 * hpwl/hmin) * exp(2 * vrel)      (oc:2122-2125)

with `hmin` the minimum HPWL over the profiles actually in the pool -- which is
why "just drop it" cannot be assumed: removing a profile can re-scale the hpwl
term for every survivor and move the winner somewhere else. So each candidate
drop is re-selected, not subtracted.

Reports per case: the max-setter, whether it wins, what the wall becomes without
it, and -- the number that matters -- whether the SELECTED layout changes.

  <python> -u l138_wall_setter.py --minn 100
  <python> -u l138_wall_setter.py --minn 80 --repeat 2
"""
import argparse
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, "iccad2026contest")

import optimizer_constructive as oc
from optimizer_claude import _serialize_input, _parse_output
from iccad2026_evaluate import ContestEvaluator
from proxy_analysis import build_opt_target_pos

_A_HAT = 1.035


def proxy_of(metrics, hmin, A_hat):
    return [(m["area"] / A_hat + oc._RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
            for m in metrics]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minn", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--maxk", type=int, default=10)
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    oc._ensure_compiled()
    ev = ContestEvaluator(data_path=".", verbose=False)
    ev._load_dataset()
    benv = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}

    rows = []
    for idx in range(100):
        s = ev.dataset[idx]
        at, b2b, p2b, pins, cons = s["input"]
        n = int((at != -1).sum().item())
        if n < a.minn:
            continue
        base, tp = ev._extract_baseline(idx, s["label"], b2b, p2b, pins, n)
        tt = tp if torch.is_tensor(tp) else torch.tensor(
            [[float(v) for v in q] for q in tp])
        otp = build_opt_target_pos(tt[:n], cons[:n], n)
        txt = _serialize_input(n, at[:n], b2b, p2b, pins, cons[:n], otp,
                               gnn_hint=None)
        pool = oc._pool_indices(n)

        dts, mets = [], []
        for pi in pool:
            e = dict(benv)
            e.update(oc._profile_env(pi, n))
            e.update(oc._PROFILES[pi])
            best = None
            for _ in range(a.repeat):
                t0 = time.perf_counter()
                r = subprocess.run([str(oc._BIN)], input=txt, capture_output=True,
                                   text=True, env=e)
                dt = time.perf_counter() - t0
                if best is None or dt < best[0]:
                    best = (dt, r.stdout)
            dts.append(best[0])
            P = _parse_output(best[1], n)
            mets.append(oc._proxy_metrics([tuple(map(float, q)) for q in P],
                                          at[:n], b2b, p2b, pins, cons[:n], n))

        A_hat = _A_HAT * float(sum(max(0.0, float(at[i])) for i in range(n)))
        hmin = min(m["hpwl"] for m in mets) or 1.0
        px = proxy_of(mets, hmin, A_hat)
        win = min(range(len(pool)), key=lambda k: px[k])
        slow = max(range(len(pool)), key=lambda k: dts[k])

        # DROP-K sweep. Each K is a fresh selection over the survivors, never a
        # subtraction: hmin is the pool minimum, so removing profiles re-scales
        # the hpwl term for everyone left and can move the winner.
        order = sorted(range(len(pool)), key=lambda k: -dts[k])   # slowest first
        wall = max(max(dts), sum(dts) / a.cores)
        curve = []
        for K in range(0, a.maxk + 1):
            keep = [k for k in range(len(pool)) if k not in set(order[:K])]
            if len(keep) < 2:
                break
            hm = min(mets[k]["hpwl"] for k in keep) or 1.0
            pk = proxy_of([mets[k] for k in keep], hm, A_hat)
            wk = keep[min(range(len(keep)), key=lambda t: pk[t])]
            wl = max(max(dts[k] for k in keep), sum(dts[k] for k in keep) / a.cores)
            curve.append((K, wl, wk == win))
        rows.append((idx, n, pool[slow], dts[slow], pool[win], slow == win,
                     wall, curve))
        surv = max((K for K, _w, ok in curve if ok), default=0)
        print(f"  case {idx:>3} n={n:>3}  wall {wall:.3f}s by prof {pool[slow]:>3}"
              f"  winner {pool[win]:>3}{' <-- SAME' if slow == win else ''}"
              f"   winner survives drop-K up to K={surv}"
              f"   wall@K={surv}: {dict((K,round(w,3)) for K,w,_o in curve)[surv]:.3f}s",
              flush=True)

    if not rows:
        return 0
    W = lambda r: math.exp(r[1] / 12.0)
    ws = sum(W(r) for r in rows)
    w1 = sum(W(r) * r[6] for r in rows) / ws
    w2 = sum(W(r) * r[7] for r in rows) / ws
    same = sum(1 for r in rows if r[8])
    selfwin = sum(1 for r in rows if r[5])
    print(f"\n=== L138: {len(rows)} cases, n>={a.minn}, {a.cores}c pool ===")
    print(f"  max-setter IS the winner on      {selfwin}/{len(rows)} cases")
    print(f"  winner UNCHANGED after dropping  {same}/{len(rows)} cases")
    print(f"  weighted wall  {w1:.4f}s -> {w2:.4f}s   {100*(w2-w1)/w1:+.2f}%")
    print("\n  (dropping is only free where the winner is unchanged; where the")
    print("   max-setter IS the winner, its cost is buying the selected layout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
