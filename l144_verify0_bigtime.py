"""Time one of the HEAVY OOS cases (n=120, 2.2% of the whole 240-case weighted
score each) so the heavy-end A/B can be budgeted.  Read-only."""
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

os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
specs = m77._specs("s1")
ck, fk, lay_id, n = specs[239]
d = torch.load(m67._path_of(fk))
lay = m67._load_case(d, lay_id)
tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
otp = build_opt_target_pos(tt[:n], lay["cons"], n)
inp = _serialize_input(n, lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                       lay["cons"], otp)
env = dict(os.environ)
env.update(oc._PROFILES[0])
env.update(oc._profile_env(0, n))
for tag, w in [("off", None), ("w=10", "10")]:
    e = dict(env)
    if w:
        e["ICCAD_BND_ABUT"] = w
    t0 = time.time()
    r = subprocess.run([str(_DIR / "constructive_l144v1.exe")], input=inp,
                       capture_output=True, text=True, timeout=1200, env=e)
    print(f"{ck} n={n} {tag:5s} {time.time() - t0:7.2f}s rc={r.returncode}")
