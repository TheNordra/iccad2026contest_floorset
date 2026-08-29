"""PLACEBO CONTROL for the L144-V1 ABUT result.

ABUT at w=200 measures +0.379% (profile 0) / +0.503% (profile 1) on the full
240-case OOS s1 set.  Before that can be called a mechanism, it has to beat the
null hypothesis "ANY perturbation of the boundary-item candidate ranking is worth
about +0.4% on 240 cases, because the shipped tie-break (1e-3*y + 1e-4*x) is
arbitrary".

The control is free: ICCAD_BND_EDGE_RUN in constructive_l144.exe is a DIFFERENT
penalty on the SAME cls==0 branch with the SAME lexicographic guard, and it was
already declared RED.  Run it over the same 240 cases.

  ABUT +0.4%, EDGE_RUN ~0 or negative  -> the gain is specific to "abut or corner"
  both ~ +0.4%                         -> the gain is generic perturbation

Read-only: runs existing binaries, modifies nothing.

  <python> -u l144_verify0_placebo.py --exe constructive_l144.exe \
      --var ICCAD_BND_EDGE_RUN --weight 3000 --cases 240
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--cases", type=int, default=240)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--exe", default="constructive_l144.exe")
    ap.add_argument("--var", default="ICCAD_BND_EDGE_RUN")
    ap.add_argument("--weight", default="3000")
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
            pos = {}
            for tag in ("off", "on"):
                env = dict(env0)
                if tag == "on":
                    env[a.var] = a.weight
                t0 = time.time()
                r = subprocess.run([str(exe)], input=inp, capture_output=True,
                                   text=True, timeout=1200, env=env)
                secs[tag] += time.time() - t0
                P = _parse_output(r.stdout, n)
                pos[tag] = [list(p) for p in P]
                m = evaluate_solution({"positions": pos[tag], "runtime": 1.0},
                                      lay["base"], lay["cons"], lay["b2b"],
                                      lay["p2b"], lay["pins"], lay["at"],
                                      target_positions=tt[:n],
                                      median_runtime=1.0)
                got[tag] = float(m.cost)
                bnd[tag] += int(m.boundary_violations)
            if pos["off"] != pos["on"]:
                moved += 1
            rows.append((ck, n, math.exp(n / 12.0), got["off"], got["on"]))

    W = sum(r[2] for r in rows)
    c0 = sum(r[2] * r[3] for r in rows) / W
    cw = sum(r[2] * r[4] for r in rows) / W
    print(f"\n=== PLACEBO/CONTROL: {a.exe}  {a.var}={a.weight}  "
          f"{a.sample} specs[{a.skip}:{a.skip + a.cases}]  profile {a.profile} ===")
    print(f"weighted cost  {c0:.6f} -> {cw:.6f}   {100 * (c0 - cw) / c0:+.3f}%")
    print(f"bnd {bnd['off']} -> {bnd['on']}     cases moved {moved}/{len(rows)}")
    print(f"wall OFF {secs['off']:.1f}s  ON {secs['on']:.1f}s  "
          f"ON/OFF {secs['on'] / secs['off']:.4f}x")
    js = []
    for i in range(len(rows)):
        sub = rows[:i] + rows[i + 1:]
        Ws = sum(r[2] for r in sub)
        js.append(100 * (sum(r[2] * r[3] for r in sub) / Ws
                         - sum(r[2] * r[4] for r in sub) / Ws)
                  / (sum(r[2] * r[3] for r in sub) / Ws))
    js.sort()
    print(f"jackknife min {js[0]:+.3f}%  max {js[-1]:+.3f}%  "
          f"<=0: {sum(1 for v in js if v <= 0)}/{len(js)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
