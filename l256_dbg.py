"""L256 liveness: run ONE heavy case on the proxy-winning profile with the debug
emitters on, and print exactly where the shrink loop stops."""
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l256.exe"
CACHE = DIR / "l252_cache.pkl"

# SNAPSHOT BEFORE THE IMPORT. m67_oos_probe.py:61-63 deletes every ICCAD_* at
# import time, and sys.argv is clobbered below -- reading either afterwards gives
# the shipped defaults while the flags look like they were honoured.
_ARGV = list(sys.argv)
_ENV0 = dict(os.environ)

sys.argv = ["x"]
import torch                                                    # noqa: E402
import m67_oos_probe as m67                                     # noqa: E402
import m77_oos_probe as m77                                     # noqa: E402
os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
import optimizer_constructive as oc                             # noqa: E402

NCASE = int(_ARGV[1]) if len(_ARGV) > 1 else 3
RUIN = _ENV0.get("RUIN", "0.12")
STEP = _ENV0.get("STEP", "0.99")
ITERS = _ENV0.get("ITERS", "40")
MODE = _ENV0.get("MODE", "1")

C = pickle.load(open(CACHE, "rb"))
spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs("s1")}
keys = sorted([k for k in C if k[0] == "s1"], key=lambda k: -C[k]["n"])[:NCASE]
RH = oc._RH
loaded = {}
for key in keys:
    ck = key[1]
    e = C[key]
    fk, L, n = spec_of[ck]
    if fk not in loaded:
        loaded.clear()
        loaded[fk] = torch.load(m67._path_of(fk))
    lay = m67._load_case(loaded[fk], L)
    otp = m67.build_opt_target_pos(lay["tp"], lay["cons"], n)
    hint = None
    if bool(oc._l137_env()) or bool(oc._l137_active(n)):
        try:
            hint = oc._gordian_hint(n, lay["at"], lay["b2b"], lay["p2b"],
                                    lay["pins"], lay["cons"], otp)
        except Exception:
            hint = None
    inp = oc._serialize_input(n, lay["at"], lay["b2b"], lay["p2b"], lay["cons"] and lay["pins"],
                              lay["cons"], otp, gnn_hint=hint) if False else \
        oc._serialize_input(n, lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                            lay["cons"], otp, gnn_hint=hint)
    idxs = sorted(e["recs"])
    met = [e["recs"][i] for i in idxs]
    A_hat = 1.035 * max(e["sumA"], 1e-9)
    hmin = min(m["hpwl"] for m in met) or 1.0
    prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
            for m in met]
    widx = idxs[min(range(len(idxs)), key=lambda t: prox[t])]
    prof = dict(oc._PROFILES[widx])
    ov = oc._profile_env(widx, n)
    if ov:
        prof.update(ov)
    env = dict(os.environ)
    env.update(prof)
    env["ICCAD_L256"] = "1"
    env["ICCAD_L256_DBG"] = "1"
    env["ICCAD_L256_RUIN"] = RUIN
    env["ICCAD_L256_STEP"] = STEP
    env["ICCAD_L256_ITERS"] = ITERS
    env["ICCAD_L256_MODE"] = MODE
    r = subprocess.run([str(PROBE)], input=inp, capture_output=True, text=True,
                       env=env, timeout=300)
    print("=== n={} profile={}  RUIN={} STEP={} ITERS={} ===".format(n, widx, RUIN, STEP, ITERS))
    for line in r.stderr.splitlines():
        if line.startswith("L256DBG"):
            print("   " + line)
    print()
