"""L144 V3 - off-path gate for constructive_l144v3.exe.

The V3 probe adds ICCAD_BND_COMPACT_SAFE (boundary-aware compaction guard) and
ICCAD_BND_CS_TRACE (stderr counters only). With both unset the binary must be
byte-identical to the shipped constructive.exe on stdout, over every profile the
adaptive pool would actually run for each case.

  <python> -u l144v3_gate.py --sample s1 --cases 8

Nothing in the shipping tree is written or read for effect; both binaries are run
as children on the same serialized input.
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
PROBE = _DIR / "constructive_l144v3.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first K specs (m77._specs is sorted by n)")
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = m77._specs(a.sample)[a.offset:a.offset + a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    T = collections.Counter()
    runs = 0
    bad = []
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
                x = subprocess.run([str(SHIP)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                y = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                runs += 1
                if x.stdout == y.stdout:
                    T["same"] += 1
                else:
                    T["DIFF"] += 1
                    bad.append((ck, n, i))
            print(f"  case {ck} n={n} pool={len(oc._pool_indices(n))} "
                  f"cum same={T['same']} DIFF={T['DIFF']}", flush=True)

    print(f"\n=== L144 V3 OFF-PATH GATE: {a.sample}, {len(specs)} cases, "
          f"{runs} (profile x case) comparisons ===")
    print(f"identical {T['same']}/{runs}, different {T['DIFF']}  -> "
          + ("PASS" if not T["DIFF"] else "*** FAIL ***"))
    for b in bad[:20]:
        print(f"   DIFF case={b[0]} n={b[1]} profile={b[2]}")
    return 0 if not T["DIFF"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
