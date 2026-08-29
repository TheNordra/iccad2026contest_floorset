"""L144v2 -- single-profile A/B of ICCAD_BND_DEMAND_ORDER.

Copy of l144_edge_run_ab.py retargeted at constructive_l144v2.exe and the
demand-order flag. Same question, cheapest form: on ONE profile, does ordering
each bscore class by along-edge demand actually remove boundary violations, and
what does it cost in hpwl/area? A mechanism that cannot move the count on its
own profile has nothing for a twin screen to arbitrate.

Off-path gate for this binary: l144v2_gate.py (408/408 byte-identical with the
flag off, 230/408 pairs move with it on). The shipping exe is untouched.

  <python> -u l144v2_ab.py --sample s1 --cases 16 --profile 0 --values 0,1
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

EXE = _DIR / "constructive_l144v2.exe"
FLAG = "ICCAD_BND_DEMAND_ORDER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--values", default="0,1")
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--percase", action="store_true",
                    help="print the per-case boundary count for every value")
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    vs = [v.strip() for v in a.values.split(",")]
    specs = m77._specs(a.sample)[:a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    acc = {v: collections.Counter() for v in vs}
    tot = {v: collections.defaultdict(float) for v in vs}
    percase = {v: {} for v in vs}
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
            for v in vs:
                env = dict(env0)
                if v != "0":
                    env[FLAG] = v
                r = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                P = _parse_output(r.stdout, n)
                if ref is None:
                    ref = P
                elif list(P) != list(ref):
                    changed[v] += 1
                m = evaluate_solution({"positions": [list(p) for p in P],
                                       "runtime": 1.0},
                                      lay["base"], lay["cons"], lay["b2b"],
                                      lay["p2b"], lay["pins"], lay["at"],
                                      target_positions=tt[:n],
                                      median_runtime=1.0)
                acc[v]["bnd"] += int(m.boundary_violations)
                acc[v]["grp"] += int(m.grouping_violations)
                acc[v]["mib"] += int(m.mib_violations)
                acc[v]["infeas"] += (0 if m.is_feasible else 1)
                percase[v][ck] = (int(m.boundary_violations), float(m.cost))
                for k in ("cost", "hpwl_gap", "area_gap", "violations_relative"):
                    tot[v][k] += wt * float(getattr(m, k))

    print(f"\n=== L144v2 {FLAG} A/B: {a.sample}, {len(specs)} cases, "
          f"profile {a.profile} (solo, not portfolio) ===\n")
    print(f"{'value':>8} {'cost':>10} {'hpwl':>9} {'area':>9} {'vrel':>9} "
          f"{'bnd':>5} {'grp':>5} {'mib':>5} {'moved':>6} {'infeas':>6}")
    base = tot[vs[0]]["cost"] / wsum
    for v in vs:
        t = tot[v]
        print(f"{v:>8} {t['cost'] / wsum:>10.6f} {t['hpwl_gap'] / wsum:>9.4f} "
              f"{t['area_gap'] / wsum:>9.4f} "
              f"{t['violations_relative'] / wsum:>9.5f} "
              f"{acc[v]['bnd']:>5} {acc[v]['grp']:>5} {acc[v]['mib']:>5} "
              f"{changed[v]:>6} {acc[v]['infeas']:>6}"
              + (f"   {100 * (base - t['cost'] / wsum) / base:+.3f}%"
                 if v != vs[0] else ""))

    if len(vs) > 1:
        v0, v1 = vs[0], vs[-1]
        better = [c for c in percase[v0]
                  if percase[v1][c][0] < percase[v0][c][0]]
        worse = [c for c in percase[v0]
                 if percase[v1][c][0] > percase[v0][c][0]]
        print(f"\nboundary count, {v1} vs {v0}: "
              f"better on {len(better)} cases {sorted(better)}, "
              f"worse on {len(worse)} cases {sorted(worse)}")
    if a.percase:
        print(f"\n{'case':>6} " + " ".join(f"{('bnd@' + v):>8}" for v in vs)
              + "   " + " ".join(f"{('cost@' + v):>10}" for v in vs))
        for c in sorted(percase[vs[0]]):
            print(f"{c:>6} " + " ".join(f"{percase[v][c][0]:>8}" for v in vs)
                  + "   " + " ".join(f"{percase[v][c][1]:>10.5f}" for v in vs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
