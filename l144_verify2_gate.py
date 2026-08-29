"""L144 VERIFY2 - independent off-path gate + liveness for constructive_l144v3.exe.

Three runs per (case, profile):
  A = constructive.exe                       (shipped)
  B = constructive_l144v3.exe, flag unset    (must be byte-identical to A)
  C = constructive_l144v3.exe, SAFE=<mode>   (liveness: how often does it differ)

A gate that only proves A==B says nothing about whether the mechanism can move
anything; this reports both in one pass so the "silent no-op" mode is impossible
to hide.

  <python> -u l144_verify2_gate.py --sample s1 --cases 8 --mode 1
"""
import argparse
import collections
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
from optimizer_claude import _serialize_input                       # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402

SHIP = _DIR / "constructive.exe"
PROBE = _DIR / "constructive_l144v3.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--mode", default="1")
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = m77._specs(a.sample)[a.offset:a.offset + a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    T = collections.Counter()
    bad = []
    live_by_case = collections.Counter()
    pool_by_case = collections.Counter()
    t0 = time.time()
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in byf[fk]:
            lay = m67._load_case(d, lay_id)
            tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            inp = _serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp)
            for i in oc._pool_indices(n):
                env = dict(os.environ)
                env.update(oc._PROFILES[i])
                env.update(oc._profile_env(i, n))
                A = subprocess.run([str(SHIP)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env).stdout
                B = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env).stdout
                envc = dict(env)
                envc["ICCAD_BND_COMPACT_SAFE"] = a.mode
                C = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=envc).stdout
                T["runs"] += 1
                pool_by_case[ck] += 1
                if A == B:
                    T["offpath_same"] += 1
                else:
                    T["offpath_DIFF"] += 1
                    bad.append((ck, n, i))
                if C != B:
                    T["live_diff"] += 1
                    live_by_case[ck] += 1
            print(f"  case {ck} n={n} pool={pool_by_case[ck]} "
                  f"cum same={T['offpath_same']} DIFF={T['offpath_DIFF']} "
                  f"live={T['live_diff']}/{T['runs']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n=== VERIFY2 GATE: {a.sample} off={a.offset} {len(specs)} cases, "
          f"{T['runs']} (profile x case) triples, mode={a.mode} ===")
    print(f"OFF-PATH  identical {T['offpath_same']}/{T['runs']}  DIFF "
          f"{T['offpath_DIFF']}  -> "
          + ("PASS" if not T["offpath_DIFF"] else "*** FAIL ***"))
    print(f"LIVENESS  flag-on differs from flag-off in {T['live_diff']}/"
          f"{T['runs']} runs "
          f"({100.0 * T['live_diff'] / max(1, T['runs']):.1f}%)  -> "
          + ("LIVE" if T["live_diff"] else "*** SILENT NO-OP ***"))
    for ck in sorted(pool_by_case):
        print(f"   case {ck:>6} pool={pool_by_case[ck]:>3} "
              f"live={live_by_case[ck]:>3}")
    for b in bad[:20]:
        print(f"   OFFPATH DIFF case={b[0]} n={b[1]} profile={b[2]}")
    return 0 if not T["offpath_DIFF"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
