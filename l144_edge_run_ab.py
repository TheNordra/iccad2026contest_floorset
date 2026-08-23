"""L144 - single-profile A/B of ICCAD_BND_EDGE_RUN before paying for the screen.

The twin screen (l124_r3_scale) costs ~40 min per sample. This is the cheap
precondition: on ONE profile, does the contiguity term actually remove boundary
violations, and what does it cost in hpwl/area? A mechanism that cannot move the
count on its own profile has nothing for a twin to arbitrate.

Runs `constructive_l144.exe` directly (off-path gate: 612/612 byte-identical
with the flag off). The shipping exe and every shipped artefact are untouched.

  <python> -u l144_edge_run_ab.py --sample s1 --cases 16 --weights 0,3000,10000,30000
"""
import argparse
import collections
import math
import os
import subprocess
import sys
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

EXE = _DIR / "constructive_l144.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--weights", default="0,3000,10000,30000")
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    ws = [w.strip() for w in a.weights.split(",")]
    specs = m77._specs(a.sample)[:a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    acc = {w: collections.Counter() for w in ws}
    tot = {w: collections.defaultdict(float) for w in ws}
    wsum = 0.0
    changed = collections.Counter()
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
            wt = math.exp(n / 12.0)
            wsum += wt
            ref = None
            for w in ws:
                env = dict(env0)
                if w != "0":
                    env["ICCAD_BND_EDGE_RUN"] = w
                r = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                P = _parse_output(r.stdout, n)
                if ref is None:
                    ref = P
                elif list(P) != list(ref):
                    changed[w] += 1
                m = evaluate_solution({"positions": [list(p) for p in P],
                                       "runtime": 1.0},
                                      lay["base"], lay["cons"], lay["b2b"],
                                      lay["p2b"], lay["pins"], lay["at"],
                                      target_positions=tt[:n],
                                      median_runtime=1.0)
                acc[w]["bnd"] += int(m.boundary_violations)
                acc[w]["grp"] += int(m.grouping_violations)
                acc[w]["mib"] += int(m.mib_violations)
                acc[w]["infeas"] += (0 if m.is_feasible else 1)
                for k in ("cost", "hpwl_gap", "area_gap", "violations_relative"):
                    tot[w][k] += wt * float(getattr(m, k))

    print(f"\n=== L144 EDGE_RUN A/B: {a.sample}, {len(specs)} cases, "
          f"profile {a.profile} (solo, not portfolio) ===\n")
    print(f"{'weight':>8} {'cost':>10} {'hpwl':>9} {'area':>9} {'vrel':>9} "
          f"{'bnd':>5} {'grp':>5} {'mib':>5} {'moved':>6} {'infeas':>6}")
    base = tot[ws[0]]["cost"] / wsum
    for w in ws:
        t = tot[w]
        print(f"{w:>8} {t['cost'] / wsum:>10.6f} {t['hpwl_gap'] / wsum:>9.4f} "
              f"{t['area_gap'] / wsum:>9.4f} "
              f"{t['violations_relative'] / wsum:>9.5f} "
              f"{acc[w]['bnd']:>5} {acc[w]['grp']:>5} {acc[w]['mib']:>5} "
              f"{changed[w]:>6} {acc[w]['infeas']:>6}"
              + (f"   {100 * (base - t['cost'] / wsum) / base:+.3f}%"
                 if w != ws[0] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
