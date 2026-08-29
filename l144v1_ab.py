"""L144-V1 - single-profile A/B of ICCAD_BND_ABUT ("abut or corner").

Same harness as l144_edge_run_ab.py, pointed at the V1 binary and the V1 flag.
EDGE_RUN's gradient was RED (count 26 -> 26/27/28/30 at every weight 30..30000
while cost fell up to 6%): a gradient taxes the FIRST item of each edge, which
the wire term should own. V1 instead charges a DISCRETE w only when a compliant
candidate touches neither a frame corner of its edge nor a block already sitting
on that same side, so it can only forbid leaving unusable holes.

Runs `constructive_l144v1.exe` directly (off-path gate l144v1_gate.py: 408/408
byte-identical on stdout AND stderr with the flag off). The shipping exe and
every shipped artefact are untouched.

`moved` is the liveness count: 0 means silent no-op. `flip` is the in-solver
count of boundary-item placements whose winner the penalty actually changed.

  <python> -u l144v1_ab.py --sample s1 --cases 16 --weights 0,1000,5000,20000
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

EXE = _DIR / "constructive_l144v1.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--skip", type=int, default=0,
                    help="take specs[skip:skip+cases] -- lets a second run use a "
                         "DISJOINT slice, so a weight tuned on the first slice "
                         "gets a genuine held-out confirmation")
    ap.add_argument("--percase", action="store_true",
                    help="dump per-case cost/bnd deltas, to show whether a gain "
                         "is spread or is one lucky case")
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--weights", default="0,1000,5000,20000")
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    ws = [w.strip() for w in a.weights.split(",")]
    specs = m77._specs(a.sample)[a.skip:a.skip + a.cases]
    percase = collections.defaultdict(dict)
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
                    env["ICCAD_BND_ABUT"] = w
                r = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                for line in r.stderr.splitlines():
                    if line.startswith("ABUTLIVE"):
                        for tok in line.split()[1:]:
                            k, v = tok.split("=")
                            if k != "w":
                                acc[w]["lv_" + k] += int(v)
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
                percase[(ck, lay_id)][w] = (float(m.cost),
                                            int(m.boundary_violations))
                for k in ("cost", "hpwl_gap", "area_gap", "violations_relative"):
                    tot[w][k] += wt * float(getattr(m, k))

    print(f"\n=== L144-V1 ABUT A/B: {a.sample}, {len(specs)} cases, "
          f"profile {a.profile} (solo, not portfolio) ===\n")
    print(f"{'weight':>8} {'cost':>10} {'hpwl':>9} {'area':>9} {'vrel':>9} "
          f"{'bnd':>5} {'grp':>5} {'mib':>5} {'moved':>6} {'infeas':>6} "
          f"{'flip':>8} {'pen':>8} {'place':>8}")
    base = tot[ws[0]]["cost"] / wsum
    for w in ws:
        t = tot[w]
        print(f"{w:>8} {t['cost'] / wsum:>10.6f} {t['hpwl_gap'] / wsum:>9.4f} "
              f"{t['area_gap'] / wsum:>9.4f} "
              f"{t['violations_relative'] / wsum:>9.5f} "
              f"{acc[w]['bnd']:>5} {acc[w]['grp']:>5} {acc[w]['mib']:>5} "
              f"{changed[w]:>6} {acc[w]['infeas']:>6} "
              f"{acc[w]['lv_flip']:>8} {acc[w]['lv_pen']:>8} "
              f"{acc[w]['lv_place']:>8}"
              + (f"   {100 * (base - t['cost'] / wsum) / base:+.3f}%"
                 if w != ws[0] else ""))
    if all(changed[w] == 0 for w in ws[1:]):
        print("\n*** SILENT NO-OP: not one case moved at any weight ***")
    if a.percase:
        w0 = ws[0]
        for w in ws[1:]:
            rows = [(k, v[w0], v[w]) for k, v in percase.items()
                    if v[w] != v[w0]]
            print(f"\nper-case changes at weight {w} "
                  f"({len(rows)}/{len(percase)} cases differ):")
            print(f"  {'case':>18} {'cost0':>10} {'costW':>10} "
                  f"{'dcost':>10} {'bnd0':>5} {'bndW':>5}")
            for k, b, t in sorted(rows, key=lambda r: r[2][0] - r[1][0]):
                print(f"  {str(k):>18} {b[0]:>10.5f} {t[0]:>10.5f} "
                      f"{t[0] - b[0]:>+10.5f} {b[1]:>5} {t[1]:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
