"""L144 verify1 -- does constructive_l144v2.exe actually come from
constructive_l144v2.cpp?

g++ output is not byte-reproducible, so md5(exe) proves nothing on its own. The
test that matters: a freshly compiled binary from the claimed source must agree
with the shipped-in probe binary on every (case, profile) pair, BOTH with the
flag off and with it on. If it does not, the .exe under test is stale or was
built from something else and every number in the screen is unanchored.

Read-only w.r.t. the repo; the rebuilt exe lives in the scratchpad.
"""
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

import torch                                                         # noqa: E402
import m67_oos_probe as m67                                          # noqa: E402
import m77_oos_probe as m77                                          # noqa: E402
import optimizer_constructive as oc                                  # noqa: E402
from optimizer_claude import _serialize_input                        # noqa: E402
from proxy_analysis import build_opt_target_pos                      # noqa: E402

CLAIMED = _DIR / "constructive_l144v2.exe"
REBUILT = Path(os.environ.get(
    "V1_REBUILT",
    r"C:\Users\0150B8~1\AppData\Local\Temp\claude\C--ICCAD-ml"
    r"\574c551c-a4ae-4d5d-8ef2-d147eceabc4b\scratchpad\v1_rebuild.exe"))
FLAG = "ICCAD_BND_DEMAND_ORDER"

os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
specs = m77._specs("s1")[:6]
byf = collections.defaultdict(list)
for ck, fk, lay_id, n in specs:
    byf[fk].append((ck, lay_id, n))

same_off = diff_off = same_on = diff_on = 0
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
            for on in (False, True):
                e = dict(env)
                if on:
                    e[FLAG] = "1"
                a = subprocess.run([str(CLAIMED)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=e).stdout
                b = subprocess.run([str(REBUILT)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=e).stdout
                if on:
                    if a == b:
                        same_on += 1
                    else:
                        diff_on += 1
                else:
                    if a == b:
                        same_off += 1
                    else:
                        diff_off += 1

print(f"\n=== rebuilt-from-source vs claimed .exe, {len(specs)} cases x full pool ===")
print(f"  flag OFF: identical {same_off}/{same_off+diff_off}")
print(f"  flag ON : identical {same_on}/{same_on+diff_on}")
print("  -> " + ("PASS: the .exe matches the claimed .cpp"
                 if not (diff_off or diff_on) else "*** FAIL: exe != source ***"))
