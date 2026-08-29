"""L307d -- is the CURRENT binary a fair stand-in for the one the grader ran?

f's numerator is the uncontended slowest profile, measured with today's
`constructive.exe`.  The graded package was M73 (`7f38893`), before L114's shape
LP, L131's abutment fix and L136's FRAME_EPS -- and L136 explicitly made the
frame smaller, which changes how long a pack takes.  If today's binary is faster,
the numerator is too small and f is understated.

Times the SAME captured stdin through both binaries on the heavy band.
"""
import os, pickle, subprocess, sys, time
from pathlib import Path

DIR = Path(__file__).parent
CAP = pickle.load(open(DIR / "l306_capture.pkl", "rb"))
NEW, OLD = str(DIR / "constructive.exe"), str(DIR / "constructive_m73.exe")
nmin = int(sys.argv[1]) if len(sys.argv) > 1 else 101
out = {}
for c in CAP:
    if c["n"] < nmin:
        continue
    a = b = 0.0
    ok = 0
    for env_over, inp in c["profiles"]:
        env = dict(os.environ); env.update(env_over)
        t0 = time.perf_counter()
        subprocess.run([NEW], input=inp, capture_output=True, text=True, env=env)
        a = max(a, time.perf_counter() - t0)
        t0 = time.perf_counter()
        r = subprocess.run([OLD], input=inp, capture_output=True, text=True, env=env)
        b = max(b, time.perf_counter() - t0)
        ok += r.returncode == 0 and bool(r.stdout.strip())
    out[c["n"]] = (a, b, ok, len(c["profiles"]))
    print("  n=%3d  M(current) %6.3f   M(M73) %6.3f   ratio %.3f   M73 ok %d/%d"
          % (c["n"], a, b, b / a, ok, len(c["profiles"])), flush=True)
pickle.dump(out, open(DIR / "l307d_m73.pkl", "wb"))
r = [v[1] / v[0] for v in out.values()]
print("M73 / current, max-profile: mean %.3f  min %.3f  max %.3f"
      % (sum(r) / len(r), min(r), max(r)))
