"""L146 - REFINE band-cut (M49/M50) arm runner, HEAVY BAND ONLY.

Same solve/evaluate path as l140_oos_soft_audit.py `run`, with two differences
that the task requires:

  * case selection is by BLOCK COUNT, never by --limit.  m77_oos_probe._specs()
    returns cases sorted by n ASCENDING, so --limit selects the LIGHTEST cases;
    this arm has to be measured on n>100, which carries 89.6% of the OOS
    sample's weight (81.2% of the official beta weight).
  * it also records the route-A frame-queue WORK total (sum of subprocess wall
    over every frame task) per case, so the CPU-work ratio -- the sum-bound
    limit, which is machine-width independent -- can be reported next to the
    wall ratio measured on this 32-logical box.

ICCAD_* knobs are captured BEFORE importing m77_oos_probe (which strips them at
import time) and restored afterwards, exactly as l137/l140 do.

  <python> -u l146_refine_band_run.py --sample s1 --cores 48 --nmin 101 \
           --take-per-n 1 --out l146_off_r1.json --tag off
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

_KNOBS = {k: v for k, v in os.environ.items() if k.startswith("ICCAD_")}

import torch                                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--nmin", type=int, default=101, help="keep n >= nmin")
    ap.add_argument("--nmax", type=int, default=10**9)
    ap.add_argument("--take-per-n", type=int, default=0,
                    help="0 = all; k = first k cases of each distinct n")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import m77_oos_probe as M
    specs = M._specs(a.sample)
    os.environ.update(_KNOBS)
    if a.cores:
        os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)

    sel, seen = [], defaultdict(int)
    for s in specs:                       # (case_key, file_key, layout, n)
        n = s[3]
        if not (a.nmin <= n <= a.nmax):
            continue
        if a.take_per_n and seen[n] >= a.take_per_n:
            continue
        seen[n] += 1
        sel.append(s)

    import m67_oos_probe as m67
    import optimizer_constructive as oc
    from proxy_analysis import build_opt_target_pos
    from iccad2026_evaluate import evaluate_solution

    knobs = {k: v for k, v in os.environ.items()
             if k.startswith("ICCAD_") and k != "ICCAD_ADAPTIVE_CORES"}
    print(f"[l146run] tag={a.tag} {len(sel)} cases n in "
          f"[{min(s[3] for s in sel)},{max(s[3] for s in sel)}], sample {a.sample}, "
          f"ADAPTIVE_CORES={os.environ.get('ICCAD_ADAPTIVE_CORES')}, "
          f"binary={oc._BIN.name if oc._BIN else '?'}\n"
          f"[l146run] knobs={knobs}\n"
          f"[l146run] band_env(105)={oc._band_env(105)}  "
          f"pool@118={len(oc._pool_indices(118))}  "
          f"route_a_default={oc._route_a_default()} queue={oc._route_a_cores()}",
          flush=True)

    opt = oc.MyOptimizer(verbose=False)
    by_file = defaultdict(list)
    order = {}
    for i, (ck, fk, lay_id, n) in enumerate(sel):
        by_file[fk].append((ck, lay_id, n))
        order[ck] = i

    rows = []
    t0 = time.time()
    for fk, items in by_file.items():
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in items:
            lay = m67._load_case(d, lay_id)
            assert lay["n"] == n
            lay["base"], _dev = m67._baseline_official(lay)
            tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            w0, k0 = oc._ROUTE_A_WORK, oc._ROUTE_A_TASKS
            t1 = time.perf_counter()
            P = opt.solve(n, lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                          lay["cons"], otp)
            dt = time.perf_counter() - t1
            work = oc._ROUTE_A_WORK - w0
            tasks = oc._ROUTE_A_TASKS - k0
            m = evaluate_solution({"positions": [list(p) for p in P],
                                   "runtime": 1.0},
                                  lay["base"], lay["cons"], lay["b2b"],
                                  lay["p2b"], lay["pins"], lay["at"],
                                  target_positions=tt[:n], median_runtime=1.0)
            rows.append(dict(
                test_id=order[ck], key=ck, n=n,
                positions=[list(map(float, p)) for p in P],
                runtime_seconds=dt, ra_work=work, ra_tasks=tasks,
                feasible=bool(m.is_feasible), cost=float(m.cost),
                hpwl_gap=float(m.hpwl_gap), area_gap=float(m.area_gap),
                vrel=float(m.violations_relative),
                v_bnd=int(m.boundary_violations),
                v_grp=int(m.grouping_violations),
                v_mib=int(m.mib_violations),
                nsoft=int(m.max_possible_violations)))
            print(f"  {ck:<34} n={n:>3} {dt:7.2f}s work={work:8.1f}s "
                  f"tasks={tasks:>5} cost={m.cost:.5f}", flush=True)

    rows.sort(key=lambda r: r["test_id"])
    ws = sum(math.exp(r["n"] / 12.0) for r in rows)
    wc = sum(math.exp(r["n"] / 12.0) * r["cost"] for r in rows) / ws
    print(f"\n[l146run] tag={a.tag} cases={len(rows)} weighted cost {wc:.6f} "
          f"wall {time.time() - t0:.1f}s  solve-sum "
          f"{sum(r['runtime_seconds'] for r in rows):.1f}s  work-sum "
          f"{sum(r['ra_work'] for r in rows):.1f}s", flush=True)
    json.dump(dict(submission_name=f"L146-{a.tag}", sample=a.sample,
                   cores=a.cores, tag=a.tag, knobs=knobs,
                   test_results=rows), open(a.out, "w"))
    print(f"[l146run] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
