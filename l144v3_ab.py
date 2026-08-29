"""L144 V3 - single-profile A/B of ICCAD_BND_COMPACT_SAFE.

Same protocol as l144_edge_run_ab.py (one profile, N held-out cases, exact
official metrics), but for the compaction guard. Modes:

  0  off (shipped behaviour)
  1  FILTER    : never consider a compaction candidate whose boundary count
                 exceeds the pre-compaction layout's
  2  VETO      : shipped selection, then discard the whole compaction if its
                 boundary count went up
  3  FILTER+GF : mode 1 and also never increase group fragments

With --trace it additionally runs the probe with ICCAD_BND_CS_TRACE=1 and
reports the liveness counters straight out of compact_layout (how many solves
saw compaction raise bv, how many candidates the filter rejected, how many
vetoes fired).

  <python> -u l144v3_ab.py --sample s1 --cases 16 --modes 0,1 --trace
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

EXE = _DIR / "constructive_l144v3.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first K specs; m77._specs is sorted by n, so "
                         "--offset 158 lands on the n>=100 band that owns the "
                         "exp(n/12) weight")
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--modes", default="0,1")
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--percase", action="store_true")
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    ms = [m.strip() for m in a.modes.split(",")]
    specs = m77._specs(a.sample)[a.offset:a.offset + a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    acc = {m: collections.Counter() for m in ms}
    tot = {m: collections.defaultdict(float) for m in ms}
    tr = {m: collections.Counter() for m in ms}
    percase = collections.defaultdict(dict)
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
            for m in ms:
                env = dict(env0)
                if m != "0":
                    env["ICCAD_BND_COMPACT_SAFE"] = m
                if a.trace:
                    env["ICCAD_BND_CS_TRACE"] = "1"
                r = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                P = _parse_output(r.stdout, n)
                if a.trace:
                    for line in r.stderr.splitlines():
                        if not line.startswith("CSAFE "):
                            continue
                        t = line.split()
                        bv0, bv1 = int(t[3]), int(t[4])
                        gf0, gf1 = int(t[6]), int(t[7])
                        tr[m]["calls"] += 1
                        tr[m]["bv_pre"] += bv0
                        tr[m]["bv_post"] += bv1
                        tr[m]["gf_pre"] += gf0
                        tr[m]["gf_post"] += gf1
                        if bv1 > bv0:
                            tr[m]["calls_bv_up"] += 1
                        if bv1 < bv0:
                            tr[m]["calls_bv_down"] += 1
                        if gf1 > gf0:
                            tr[m]["calls_gf_up"] += 1
                        tr[m]["rej"] += int(t[8].split("=")[1])
                        tr[m]["veto"] += int(t[9].split("=")[1])
                if ref is None:
                    ref = P
                elif list(P) != list(ref):
                    changed[m] += 1
                mt = evaluate_solution({"positions": [list(p) for p in P],
                                        "runtime": 1.0},
                                       lay["base"], lay["cons"], lay["b2b"],
                                       lay["p2b"], lay["pins"], lay["at"],
                                       target_positions=tt[:n],
                                       median_runtime=1.0)
                acc[m]["bnd"] += int(mt.boundary_violations)
                acc[m]["grp"] += int(mt.grouping_violations)
                acc[m]["mib"] += int(mt.mib_violations)
                acc[m]["infeas"] += (0 if mt.is_feasible else 1)
                for k in ("cost", "hpwl_gap", "area_gap",
                          "violations_relative"):
                    tot[m][k] += wt * float(getattr(mt, k))
                percase[ck][m] = (float(mt.cost), int(mt.boundary_violations),
                                  float(mt.area_gap), float(mt.hpwl_gap), n)
            print(f"  case {ck} n={n} "
                  + " ".join(f"m{m}:cost={percase[ck][m][0]:.4f},"
                             f"bnd={percase[ck][m][1]}" for m in ms),
                  flush=True)

    print(f"\n=== L144 V3 BND_COMPACT_SAFE A/B: {a.sample}, {len(specs)} cases, "
          f"profile {a.profile} (solo, not portfolio) ===\n")
    print(f"{'mode':>6} {'cost':>10} {'hpwl':>9} {'area':>9} {'vrel':>9} "
          f"{'bnd':>5} {'grp':>5} {'mib':>5} {'moved':>6} {'infeas':>6}")
    base = tot[ms[0]]["cost"] / wsum
    for m in ms:
        t = tot[m]
        print(f"{m:>6} {t['cost'] / wsum:>10.6f} {t['hpwl_gap'] / wsum:>9.4f} "
              f"{t['area_gap'] / wsum:>9.4f} "
              f"{t['violations_relative'] / wsum:>9.5f} "
              f"{acc[m]['bnd']:>5} {acc[m]['grp']:>5} {acc[m]['mib']:>5} "
              f"{changed[m]:>6} {acc[m]['infeas']:>6}"
              + (f"   {100 * (base - t['cost'] / wsum) / base:+.3f}%"
                 if m != ms[0] else ""))
    if a.trace:
        print("\nliveness (compact_layout calls; REFRAME profiles call it twice):")
        print(f"{'mode':>6} {'calls':>7} {'bv_up':>7} {'bv_dn':>7} {'gf_up':>7} "
              f"{'bv_pre':>7} {'bv_post':>8} {'gf_pre':>7} {'gf_post':>8} "
              f"{'rej':>6} {'veto':>5}")
        for m in ms:
            c = tr[m]
            print(f"{m:>6} {c['calls']:>7} {c['calls_bv_up']:>7} "
                  f"{c['calls_bv_down']:>7} {c['calls_gf_up']:>7} "
                  f"{c['bv_pre']:>7} {c['bv_post']:>8} {c['gf_pre']:>7} "
                  f"{c['gf_post']:>8} {c['rej']:>6} {c['veto']:>5}")
    if a.percase:
        print("\nper case (cost / bnd / area_gap / hpwl_gap):")
        for ck in sorted(percase):
            row = percase[ck]
            n = row[ms[0]][4]
            print(f"  {ck:>6} n={n:>4} " + " | ".join(
                f"m{m} {row[m][0]:.4f} b{row[m][1]} a{row[m][2]:.4f} "
                f"h{row[m][3]:.4f}" for m in ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
