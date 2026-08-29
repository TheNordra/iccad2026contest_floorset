"""L144v2 -- off-path gate for constructive_l144v2.exe (ICCAD_BND_DEMAND_ORDER).

Two things must both hold before any A/B number is worth reading:

  1. OFF-PATH: with the flag ABSENT, the probe binary must be byte-identical to
     the shipping constructive.exe on every (case, profile) pair. If it is not,
     every later delta is contaminated by an unrelated code change.
  2. LIVENESS: with ICCAD_BND_DEMAND_ORDER=1 the stdout must actually CHANGE on
     some pairs. A flag that parses but reaches no decision prints a perfectly
     clean A/B table of zeros -- the silent no-op failure mode. The gate reports
     the changed count so "0 boundary reduction" can never be confused with
     "the mechanism never ran".

The shipping exe is only ever READ (subprocess), never rebuilt or touched.

  <python> -u l144v2_gate.py --sample s1 --cases 8 --profiles pool
"""
import argparse
import collections
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
from optimizer_claude import _serialize_input                       # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402

SHIP = _DIR / "constructive.exe"
EXE = _DIR / "constructive_l144v2.exe"
FLAG = "ICCAD_BND_DEMAND_ORDER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--profiles", default="pool",
                    help="comma-separated pool indices, or 'pool' for all")
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = m77._specs(a.sample)[:a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    same = diff = 0
    on_same = on_diff = 0
    bad = []
    live_cases = set()
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in byf[fk]:
            lay = m67._load_case(d, lay_id)
            tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            inp = _serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp)
            idxs = (list(oc._pool_indices(n)) if a.profiles == "pool"
                    else [int(s) for s in a.profiles.split(",")])
            for i in idxs:
                env = dict(os.environ)
                env.update(oc._PROFILES[i])
                env.update(oc._profile_env(i, n))
                assert FLAG not in env, "flag leaked into the OFF environment"
                x = subprocess.run([str(SHIP)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                y = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                if x.stdout == y.stdout:
                    same += 1
                else:
                    diff += 1
                    bad.append((ck, i))
                envn = dict(env)
                envn[FLAG] = "1"
                z = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=envn)
                if z.stdout == y.stdout:
                    on_same += 1
                else:
                    on_diff += 1
                    live_cases.add(ck)

    runs = same + diff
    print(f"\n=== L144v2 gate: {a.sample}, {len(specs)} cases, "
          f"profiles={a.profiles}, {runs} (case,profile) pairs ===\n")
    print(f"1. OFF-PATH  flag absent, l144v2 vs constructive.exe")
    print(f"   identical {same}/{runs}, different {diff}   -> "
          + ("PASS" if not diff else "*** FAIL ***"))
    if bad:
        print(f"   first offenders: {bad[:10]}")
    print(f"\n2. LIVENESS  {FLAG}=1 vs the same binary OFF")
    print(f"   changed   {on_diff}/{runs}   unchanged {on_same}"
          f"   ({len(live_cases)} distinct cases moved)   -> "
          + ("LIVE" if on_diff else "*** SILENT NO-OP ***"))
    return 0 if (not diff and on_diff) else 1


if __name__ == "__main__":
    raise SystemExit(main())
