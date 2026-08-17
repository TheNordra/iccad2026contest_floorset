"""L139 — turn L138's per-case upper bound into ONE deployable fixed drop set.

L138 measured that dropping the 10 slowest profiles PER CASE cuts the weighted
wall 12.95% with the winner preserved on 12/13 heavy cases. That is an upper
bound and not deployable: the drop set was chosen from each case's own timings,
which a real run cannot do. A shippable version needs ONE fixed set of profile
indices that are slow AND never selected ACROSS cases -- the same question
M41/M42 answered for the ORDER_SWAP/ORDER_MOVE profiles.

Two modes, because the timing is the expensive part and the analysis is not:

  survey   run every pool profile on every case in the band, record dt + the
           proxy metrics, dump to json. The WINNER is deterministic (it depends
           on positions, not on timing), so repeat=1 is exact for selection and
           only the dt ranking carries noise.
  analyse  try candidate drop sets against that json -- no re-timing. Each
           candidate is RE-SELECTED, never subtracted: `hmin` is the pool
           minimum, so removing profiles re-scales the hpwl term for every
           survivor and can move the winner (oc:2122).

  <python> -u l139_fixed_dropset.py survey  --minn 80 --out l139_survey.json
  <python> -u l139_fixed_dropset.py analyse --in  l139_survey.json
"""
import argparse
import json
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


def _proxy(mets, keep, A_hat):
    """The shipped selector over a SUBSET, re-scaled. Returns the winning index
    into `keep`."""
    hmin = min(mets[k]["hpwl"] for k in keep) or 1.0
    best, bi = None, None
    for t, k in enumerate(keep):
        m = mets[k]
        p = (m["area"] / A_hat + oc._RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
        if best is None or p < best:
            best, bi = p, k
    return bi


def _cases(a):
    """(idx, at, b2b, p2b, pins, cons, tp) for the in-set 100 or the OOS sample.

    The OOS path is the whole point of this file's second half: a fixed drop set
    is only safe if none of the dropped profiles WINS out of sample, and that is
    measurable directly, without touching the shipped pool logic."""
    if not a.oos:
        ev = ContestEvaluator(data_path=".", verbose=False)
        ev._load_dataset()
        for idx in range(100):
            s = ev.dataset[idx]
            at, b2b, p2b, pins, cons = s["input"]
            n = int((at != -1).sum().item())
            if n < a.minn:
                continue
            _base, tp = ev._extract_baseline(idx, s["label"], b2b, p2b, pins, n)
            yield idx, n, at, b2b, p2b, pins, cons, tp
        return
    import m77_oos_probe as M
    import m67_oos_probe as m67
    from collections import defaultdict
    byf = defaultdict(list)
    for ck, fk, lay, n in M._specs(a.oos):
        if n >= a.minn:
            byf[fk].append((ck, lay, n))
    seen = 0
    for fk, items in byf.items():
        d = torch.load(m67._path_of(fk))
        for ck, lay, n in items:
            if a.limit and seen >= a.limit:
                return
            L = m67._load_case(d, lay)
            seen += 1
            yield ck, n, L["at"], L["b2b"], L["p2b"], L["pins"], L["cons"], L["tp"]


def mode_survey(a):
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    oc._ensure_compiled()
    benv = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}

    out = []
    t_all = time.time()
    for idx, n, at, b2b, p2b, pins, cons, tp in _cases(a):
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
            m = oc._proxy_metrics([tuple(map(float, q)) for q in P], at[:n],
                                  b2b, p2b, pins, cons[:n], n)
            mets.append({k: float(m[k]) for k in ("area", "hpwl", "vrel")})
        A_hat = _A_HAT * float(sum(max(0.0, float(at[i])) for i in range(n)))
        out.append(dict(idx=str(idx), n=n, pool=list(pool), dts=dts, mets=mets,
                        A_hat=A_hat))
        print(f"  case {str(idx)[-18:]:>18} n={n:>3}  {len(pool)} prof  "
              f"max {max(dts):.3f}s  ({time.time() - t_all:.0f}s elapsed)",
              flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"\nwrote {a.out}  ({len(out)} cases)")
    return 0


def mode_analyse(a):
    data = json.load(open(a.inp))
    W = lambda c: math.exp(c["n"] / 12.0)
    ws = sum(W(c) for c in data)

    def evaluate(drop):
        """(weighted wall, cases whose winner moved) for a FIXED index set."""
        wall = 0.0
        moved = []
        for c in data:
            pool, dts, mets = c["pool"], c["dts"], c["mets"]
            base_win = _proxy(mets, list(range(len(pool))), c["A_hat"])
            keep = [k for k in range(len(pool)) if pool[k] not in drop]
            if len(keep) < 2:
                return None, None
            w = _proxy(mets, keep, c["A_hat"])
            if w != base_win:
                moved.append(c["idx"])
            wall += W(c) * max(max(dts[k] for k in keep),
                               sum(dts[k] for k in keep) / a.cores)
        return wall / ws, moved

    base_wall, _ = evaluate(set())
    print(f"=== L139 analyse: {len(data)} cases, baseline weighted wall "
          f"{base_wall:.4f}s ===\n")

    # per-profile stats ACROSS cases: how often selected, and where its runtime
    # sits in its own case's pool (rank 0 = slowest)
    stats = {}
    for c in data:
        pool, dts, mets = c["pool"], c["dts"], c["mets"]
        win = pool[_proxy(mets, list(range(len(pool))), c["A_hat"])]
        order = sorted(range(len(pool)), key=lambda k: -dts[k])
        rank = {pool[k]: r for r, k in enumerate(order)}
        for k, pi in enumerate(pool):
            st = stats.setdefault(pi, dict(seen=0, wins=0, ranks=[], share=[]))
            st["seen"] += 1
            st["wins"] += 1 if pi == win else 0
            st["ranks"].append(rank[pi])
            st["share"].append(dts[k] / max(dts))
    for pi, st in stats.items():
        st["mean_rank"] = sum(st["ranks"]) / len(st["ranks"])
        st["mean_share"] = sum(st["share"]) / len(st["share"])

    never = [pi for pi, st in stats.items() if st["wins"] == 0]
    never.sort(key=lambda pi: stats[pi]["mean_rank"])
    print(f"profiles that NEVER win in this band: {len(never)}/{len(stats)}")
    print(f"{'prof':>5}{'seen':>6}{'wins':>6}{'mean rank':>11}{'mean dt/max':>13}")
    for pi in never[:14]:
        st = stats[pi]
        print(f"{pi:>5}{st['seen']:>6}{st['wins']:>6}{st['mean_rank']:>11.1f}"
              f"{st['mean_share']:>13.3f}")

    print(f"\ngreedy fixed drop set (slowest-first among never-winners):")
    print(f"{'size':>5}{'weighted wall':>15}{'vs base':>9}{'winners moved':>15}  drop set")
    drop = set()
    for pi in never:
        cand = drop | {pi}
        wall, moved = evaluate(cand)
        if wall is None:
            break
        tag = "OK" if not moved else f"MOVED {moved[:4]}"
        print(f"{len(cand):>5}{wall:>15.4f}{100*(wall-base_wall)/base_wall:>+8.2f}%"
              f"{tag:>15}  {sorted(cand)}")
        if moved:
            print("  ^ stop: a fixed set is only free while no winner moves")
            break
        drop = cand
    print(f"\nFIXED DROP SET = {sorted(drop)}")
    wall, moved = evaluate(drop)
    print(f"  weighted wall {base_wall:.4f}s -> {wall:.4f}s "
          f"({100*(wall-base_wall)/base_wall:+.2f}%), winners moved: {moved or 'none'}")
    print("\n  In-set only. Pool pruning has a bad prior here: M67-D measured the")
    print("  adaptive cuts' OOS quality tax at +2.825% against an in-set +0.106%.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["survey", "analyse"])
    ap.add_argument("--minn", type=int, default=80)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--out", default="l139_survey.json")
    ap.add_argument("--in", dest="inp", default="l139_survey.json")
    ap.add_argument("--oos", default="", help="sample name (s1/s2); empty = in-set")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drop", default="", help="analyse: test THIS fixed set")
    a = ap.parse_args()
    raise SystemExit(mode_survey(a) if a.mode == "survey" else mode_analyse(a))
