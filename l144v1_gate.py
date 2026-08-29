"""L144-V1 off-path gate: constructive_l144v1.exe with ICCAD_BND_ABUT unset must
be byte-identical to the shipping constructive.exe.

V1 adds one flag (ICCAD_BND_ABUT, the "abut or corner" discrete gap penalty) plus
stderr-only liveness counters. Every new statement is behind `BND_ABUT_W>0`, so
with the flag absent the probe binary has to reproduce the shipping stdout
exactly, character for character, on every (case, profile) pair. This proves it
before any A/B number is believed.

Modelled on l144_bnd_trace.py --gate, but always runs the FULL pool for each
case (oc._pool_indices(n)) instead of a single profile.

  <python> -u l144v1_gate.py --sample s1 --cases 8
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
PROBE = _DIR / "constructive_l144v1.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--live", default="",
                    help="if set, ALSO run the probe with ICCAD_BND_ABUT=<live> "
                         "and report the liveness counters (not part of the gate)")
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = m77._specs(a.sample)[:a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    same = diff = 0
    bad = []
    live = collections.Counter()
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
                if x.stdout == y.stdout:
                    same += 1
                else:
                    diff += 1
                    bad.append((ck, lay_id, n, i))
                # constructive.exe already writes a METRICS line to stderr; the
                # gate is that the probe adds NOTHING to it with the flag off.
                if y.stderr != x.stderr:
                    diff += 1
                    same -= 1
                    bad.append((ck, lay_id, n, i, "stderr"))
                if a.live:
                    env["ICCAD_BND_ABUT"] = a.live
                    z = subprocess.run([str(PROBE)], input=inp,
                                       capture_output=True, text=True,
                                       timeout=600, env=env)
                    live["runs"] += 1
                    if z.stdout != x.stdout:
                        live["moved_runs"] += 1
                    for line in z.stderr.splitlines():
                        if line.startswith("ABUTLIVE"):
                            for tok in line.split()[1:]:
                                k, v = tok.split("=")
                                if k != "w":
                                    live[k] += int(v)

    tot = same + diff
    print(f"\n=== L144-V1 off-path gate: {a.sample}, {len(specs)} cases, "
          f"full pool, {tot} (case,profile) runs ===")
    print(f"identical {same}/{tot}   different {diff}   -> "
          + ("PASS" if not diff else "*** FAIL ***"))
    for b in bad[:20]:
        print(f"  DIFF {b}")
    if a.live:
        print(f"\nliveness at ICCAD_BND_ABUT={a.live} over {live['runs']} runs:")
        print(f"  boundary-item placements with a compliant candidate "
              f"{live['place']}")
        print(f"  winner PAID the abut penalty                      "
              f"{live['pen']}")
        print(f"  penalty CHANGED the winner                        "
              f"{live['flip']}")
        print(f"  solver runs whose stdout differs from shipped     "
              f"{live['moved_runs']}/{live['runs']}")
    return 0 if not diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
