"""Concentration + jackknife on one slice of the OOS s1 set.

The A/B harness prints a weighted mean but keys per-case rows by (case, lay_id),
so you cannot see the exp(n/12) weight a row carries.  This prints, per case:
n, the case's share of the slice weight, dcost, and the weighted contribution --
then a leave-one-out jackknife of the slice-level percentage.

If dropping a single case flips the sign, the slice result is one case, not a
mechanism.  Also reports wall time per arm, since the real bar is NET of runtime.

  <python> -u l144_verify0_conc.py --skip 192 --cases 48 --weight 200
Read-only: creates nothing, modifies nothing.
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

EXE = _DIR / "constructive_l144v1.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--skip", type=int, default=192)
    ap.add_argument("--cases", type=int, default=48)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--weight", default="200")
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = m77._specs(a.sample)[a.skip:a.skip + a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    rows = []
    secs = {"0": 0.0, a.weight: 0.0}
    bnd = {"0": 0, a.weight: 0}
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in byf[fk]:
            lay = m67._load_case(d, lay_id)
            lay["base"], _dev = m67._baseline_official(lay)
            tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            inp = _serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp)
            env0 = dict(os.environ)
            env0.update(oc._PROFILES[a.profile])
            env0.update(oc._profile_env(a.profile, n))
            got = {}
            for w in ("0", a.weight):
                env = dict(env0)
                if w != "0":
                    env["ICCAD_BND_ABUT"] = w
                t0 = time.time()
                r = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=1200, env=env)
                secs[w] += time.time() - t0
                P = _parse_output(r.stdout, n)
                m = evaluate_solution({"positions": [list(p) for p in P],
                                       "runtime": 1.0},
                                      lay["base"], lay["cons"], lay["b2b"],
                                      lay["p2b"], lay["pins"], lay["at"],
                                      target_positions=tt[:n],
                                      median_runtime=1.0)
                got[w] = (float(m.cost), int(m.boundary_violations))
                bnd[w] += int(m.boundary_violations)
            rows.append((ck, n, math.exp(n / 12.0),
                         got["0"][0], got[a.weight][0],
                         got["0"][1], got[a.weight][1]))

    W = sum(r[2] for r in rows)
    c0 = sum(r[2] * r[3] for r in rows) / W
    cw = sum(r[2] * r[4] for r in rows) / W
    print(f"\n=== concentration: {a.sample} specs[{a.skip}:{a.skip + a.cases}], "
          f"profile {a.profile}, ICCAD_BND_ABUT={a.weight} ===")
    print(f"weighted cost  {c0:.6f} -> {cw:.6f}   "
          f"{100 * (c0 - cw) / c0:+.3f}%   bnd {bnd['0']} -> {bnd[a.weight]}")
    print(f"wall time      OFF {secs['0']:.1f}s   ON {secs[a.weight]:.1f}s   "
          f"ON/OFF {secs[a.weight] / secs['0']:.4f}x")

    moved = [r for r in rows if abs(r[4] - r[3]) > 1e-12]
    print(f"\ncases moved {len(moved)}/{len(rows)}")
    print(f"  {'case':>26} {'n':>4} {'wshare':>7} {'dcost':>9} "
          f"{'w*dcost':>11} {'bnd':>7}")
    for r in sorted(moved, key=lambda r: r[2] * (r[4] - r[3])):
        print(f"  {r[0]:>26} {r[1]:>4} {100 * r[2] / W:6.2f}% "
              f"{r[4] - r[3]:>+9.5f} {r[2] * (r[4] - r[3]):>+11.1f} "
              f"{r[5]:>3} ->{r[6]:>3}")

    # jackknife: drop one case, recompute the slice percentage
    print("\nleave-one-out jackknife of the slice percentage:")
    js = []
    for i in range(len(rows)):
        sub = rows[:i] + rows[i + 1:]
        Ws = sum(r[2] for r in sub)
        a0 = sum(r[2] * r[3] for r in sub) / Ws
        aw = sum(r[2] * r[4] for r in sub) / Ws
        js.append((100 * (a0 - aw) / a0, rows[i][0], rows[i][1]))
    js.sort()
    print(f"  min {js[0][0]:+.3f}% (dropping {js[0][1]} n={js[0][2]})")
    print(f"  max {js[-1][0]:+.3f}% (dropping {js[-1][1]} n={js[-1][2]})")
    neg = sum(1 for v, _, _ in js if v <= 0)
    print(f"  leave-one-out values <= 0: {neg}/{len(js)}  -> "
          + ("SIGN IS NOT ROBUST to dropping one case"
             if neg else "sign survives dropping any single case"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
