"""L307c -- how much of M is the L124 twins, which the BETA package did not have.

The reconstruction runs 43 profiles: 35 base + the 8 `_M124_IDX` twins, which are
cores-gated >= 40 and so switch on when ADAPTIVE_CORES=48 is forced.  L124 landed
2026-08-13; the beta package was uploaded 2026-07-30.  So the graded run was 35
profiles and `M = max over 43` is an upper bound on the quantity the grader
actually paid.  M appears in the NUMERATOR of f, so this biases f UP -- the
anti-conservative direction for a GREEN verdict.

Twins are identifiable in the captured env: `_profile_env` sets ICCAD_MIB_BUCKET
on exactly those indices.
"""
import os, pickle, subprocess, sys, time
from pathlib import Path

DIR = Path(__file__).parent
CAP = pickle.load(open(DIR / "l306_capture.pkl", "rb"))
BIN = str(DIR / "constructive.exe")
nmin = int(sys.argv[1]) if len(sys.argv) > 1 else 108
out = {}
for c in CAP:
    if c["n"] < nmin:
        continue
    rows = []
    for env_over, inp in c["profiles"]:
        env = dict(os.environ); env.update(env_over)
        t0 = time.perf_counter()
        subprocess.run([BIN], input=inp, capture_output=True, text=True, env=env)
        rows.append((time.perf_counter() - t0, "ICCAD_MIB_BUCKET" in env_over))
    m43 = max(d for d, _ in rows)
    m35 = max(d for d, t in rows if not t)
    out[c["n"]] = (m43, m35, sum(1 for _, t in rows if t))
    print("  n=%3d  twins %d   max over 43 %6.3f   max over 35 %6.3f   ratio %.3f"
          % (c["n"], out[c["n"]][2], m43, m35, m43 / m35), flush=True)
pickle.dump(out, open(DIR / "l307c_twins.pkl", "wb"))
r = [v[0] / v[1] for v in out.values()]
print("mean inflation of M from the twins: %.3f  (n=%d cases)" % (sum(r) / len(r), len(r)))
