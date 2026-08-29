"""PORTFOLIO A/B -- the gate neither the colleague's screen nor my solo runs cleared.

Every A/B in this line (theirs and mine) runs ONE profile solo.  The shipped
optimizer runs the whole pool per case and keeps the best by the baseline-free
proxy in optimizer_constructive._solve_impl:

    proxy = (area/A_hat + _RH*hpwl/hmin) * exp(2*vrel),  A_hat = 1.035*sum(at)

A min over ~13-51 profiles is already near its floor, so a per-profile gain
generally SHRINKS under the portfolio.  This replicates that selection exactly
for both arms and reports the official weighted cost of the selected layouts.

  <python> -u l144_verify0_portfolio.py --skip 224 --cases 16 --weight 200

Read-only: runs existing binaries, modifies nothing.
"""
import argparse
import collections
import math
import os
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

import torch                                                        # noqa: E402
import m67_oos_probe as m67                                         # noqa: E402
import m77_oos_probe as m77                                         # noqa: E402
import optimizer_constructive as oc                                 # noqa: E402
from optimizer_claude import _serialize_input, _parse_output        # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402
from iccad2026_evaluate import evaluate_solution                    # noqa: E402


def pick(cands, margs, area_targets, n):
    """Exact copy of the shipped proxy selection."""
    metrics = [oc._proxy_metrics(p, *margs) for p in cands]
    sumA = sum(max(0.0, float(area_targets[i])) for i in range(n))
    A_hat = 1.035 * max(sumA, 1e-9)
    hmin = min(m["hpwl"] for m in metrics) or 1.0
    best, bp = cands[0], float("inf")
    for p, m in zip(cands, metrics):
        pr = (m["area"] / A_hat + oc._RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
        if pr < bp:
            bp, best = pr, p
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--skip", type=int, default=224)
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--exe", default="constructive_l144v1.exe")
    ap.add_argument("--var", default="ICCAD_BND_ABUT")
    ap.add_argument("--weight", default="200")
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()
    exe = _DIR / a.exe

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = m77._specs(a.sample)[a.skip:a.skip + a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    rows = []
    secs = {"off": 0.0, "on": 0.0}
    bnd = {"off": 0, "on": 0}
    moved = 0
    runs = 0
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in byf[fk]:
            lay = m67._load_case(d, lay_id)
            lay["base"], _dev = m67._baseline_official(lay)
            tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            inp = _serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp)
            margs = (lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                     lay["cons"], n)
            sel = {}
            for tag in ("off", "on"):
                cands = []
                for i in oc._pool_indices(n):
                    env = dict(os.environ)
                    env.update(oc._PROFILES[i])
                    env.update(oc._profile_env(i, n))
                    if tag == "on":
                        env[a.var] = a.weight
                    t0 = time.time()
                    r = subprocess.run([str(exe)], input=inp,
                                       capture_output=True, text=True,
                                       timeout=1200, env=env)
                    secs[tag] += time.time() - t0
                    runs += 1
                    try:
                        P = _parse_output(r.stdout, n)
                    except Exception:
                        continue
                    if P:
                        cands.append([list(p) for p in P])
                sel[tag] = pick(cands, margs, lay["at"], n)
                m = evaluate_solution({"positions": sel[tag], "runtime": 1.0},
                                      lay["base"], lay["cons"], lay["b2b"],
                                      lay["p2b"], lay["pins"], lay["at"],
                                      target_positions=tt[:n],
                                      median_runtime=1.0)
                bnd[tag] += int(m.boundary_violations)
                sel[tag + "_cost"] = float(m.cost)
            if sel["off"] != sel["on"]:
                moved += 1
            rows.append((ck, n, math.exp(n / 12.0),
                         sel["off_cost"], sel["on_cost"]))
            print(f"  {ck:>26} n={n:>3} "
                  f"{sel['off_cost']:.5f} -> {sel['on_cost']:.5f} "
                  f"{sel['on_cost'] - sel['off_cost']:+.5f}", flush=True)

    W = sum(r[2] for r in rows)
    c0 = sum(r[2] * r[3] for r in rows) / W
    cw = sum(r[2] * r[4] for r in rows) / W
    print(f"\n=== PORTFOLIO A/B ({a.var}={a.weight}): {a.sample} "
          f"specs[{a.skip}:{a.skip + a.cases}], full pool, {runs} solver runs ===")
    print(f"weighted cost  {c0:.6f} -> {cw:.6f}   {100 * (c0 - cw) / c0:+.3f}%")
    print(f"bnd {bnd['off']} -> {bnd['on']}   selected layout changed "
          f"{moved}/{len(rows)} cases")
    print(f"wall OFF {secs['off']:.1f}s  ON {secs['on']:.1f}s  "
          f"ON/OFF {secs['on'] / secs['off']:.4f}x")
    js = []
    for i in range(len(rows)):
        sub = rows[:i] + rows[i + 1:]
        Ws = sum(r[2] for r in sub)
        b = sum(r[2] * r[3] for r in sub) / Ws
        js.append(100 * (b - sum(r[2] * r[4] for r in sub) / Ws) / b)
    js.sort()
    print(f"jackknife min {js[0]:+.3f}%  max {js[-1]:+.3f}%  "
          f"<=0: {sum(1 for v in js if v <= 0)}/{len(js)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
