"""Independent re-gate of the L144-V1 ABUT claim.

Differences from l144v1_gate.py (deliberate, so this is not just a re-run of
their script with their binary):

  * gates MY OWN build (l144_verify0_v1.exe, compiled here from their
    constructive_l144v1.cpp) against the shipping constructive.exe, so a stale
    or hand-doctored .exe cannot pass;
  * ALSO runs their constructive_l144v1.exe with the flag ON and my build with
    the flag ON on the same input, and requires the two to agree -- this is the
    check that their .exe is genuinely the compile of their .cpp;
  * counts, with the flag ON, how many runs moved and how many did NOT, so a
    silent no-op cannot hide behind an aggregate.

Read-only: touches no existing file.
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
MINE = _DIR / "l144_verify0_v1.exe"
THEIRS = _DIR / "constructive_l144v1.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--live", default="5000")
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = m77._specs(a.sample)[a.skip:a.skip + a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    off_same = off_diff = 0
    exe_same = exe_diff = 0
    moved = notmoved = 0
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
                s = subprocess.run([str(SHIP)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                m = subprocess.run([str(MINE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                # (1) off-path gate on MY build: stdout AND stderr
                if s.stdout == m.stdout and s.stderr == m.stderr:
                    off_same += 1
                else:
                    off_diff += 1
                    bad.append((ck, lay_id, n, i,
                                "stdout" if s.stdout != m.stdout else "stderr"))
                # (2) flag ON: my build vs their shipped probe binary
                envl = dict(env)
                envl["ICCAD_BND_ABUT"] = a.live
                mo = subprocess.run([str(MINE)], input=inp, capture_output=True,
                                    text=True, timeout=600, env=envl)
                to = subprocess.run([str(THEIRS)], input=inp,
                                    capture_output=True, text=True,
                                    timeout=600, env=envl)
                if mo.stdout == to.stdout:
                    exe_same += 1
                else:
                    exe_diff += 1
                # (3) liveness of the ON path
                if mo.stdout != s.stdout:
                    moved += 1
                else:
                    notmoved += 1
                for line in mo.stderr.splitlines():
                    if line.startswith("ABUTLIVE"):
                        for tok in line.split()[1:]:
                            k, v = tok.split("=")
                            if k != "w":
                                live[k] += int(v)

    tot = off_same + off_diff
    print(f"\n=== L144-VERIFY0 gate: {a.sample}, specs[{a.skip}:{a.skip + a.cases}], "
          f"full pool, {tot} (case,profile) runs ===")
    print(f"[1] MY build, flag OFF, vs constructive.exe (stdout+stderr): "
          f"identical {off_same}/{tot}  -> "
          + ("PASS" if not off_diff else "*** FAIL ***"))
    for b in bad[:20]:
        print(f"    DIFF {b}")
    print(f"[2] MY build vs THEIR .exe, flag ON at {a.live}: "
          f"identical {exe_same}/{tot}  -> "
          + ("their .exe matches their .cpp"
             if not exe_diff else f"*** MISMATCH {exe_diff} ***"))
    print(f"[3] liveness at ICCAD_BND_ABUT={a.live}: "
          f"runs moved {moved}/{tot}, unchanged {notmoved}")
    print(f"    place={live['place']} pen={live['pen']} flip={live['flip']}"
          + (f"   pen/place={100 * live['pen'] / live['place']:.1f}%"
             f"  flip/place={100 * live['flip'] / live['place']:.1f}%"
             if live['place'] else ""))
    return 0 if not off_diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
