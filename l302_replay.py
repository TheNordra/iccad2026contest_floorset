"""L302 phase 2 -- replay every captured profile SERIALLY and time it.

The parallel wall on this box is contended (16 physical cores, 32 logical, 43
profiles), so a profile's wall inside a real run is not its compute time.  The
grader has 48 cores and c* is at most 22.5, so THERE the case wall is the single
slowest profile -- an uncontended, single-threaded quantity.  To compare like
with like we need the uncontended time here, which means running one at a time.

  <python> l302_replay.py [--cases 21,120] [--reps 1] [--out l302_serial.pkl]
"""
import argparse, os, pickle, subprocess, sys, time
from pathlib import Path

DIR = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--nmin", type=int, default=21)
ap.add_argument("--nmax", type=int, default=120)
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--out", default="l302_serial.pkl")
a = ap.parse_args()

CAP = pickle.load(open(DIR / "l302_capture.pkl", "rb"))
BIN = str(DIR / "constructive.exe")
res = {}
t_start = time.time()
for c in CAP:
    if not (a.nmin <= c["n"] <= a.nmax):
        continue
    dts = []
    for env_over, inp in c["profiles"]:
        env = dict(os.environ)
        env.update(env_over)
        best = 1e18
        for _ in range(a.reps):
            t0 = time.perf_counter()
            r = subprocess.run([BIN], input=inp, capture_output=True, text=True, env=env)
            best = min(best, time.perf_counter() - t0)
            ok = r.returncode == 0 and bool(r.stdout.strip())
        dts.append((best, ok))
    res[c["n"]] = dict(case=c["case"], n=c["n"], wall_parallel=c["wall_parallel"],
                       dt=[d for d, _ in dts], ok=sum(1 for _, o in dts if o))
    s = sorted(d for d, _ in dts)
    print("  n=%3d  profiles %2d  max %6.3f  sum %8.3f  c*=%5.2f  parallel %6.3f  ratio par/max %.2f"
          % (c["n"], len(dts), s[-1], sum(s), sum(s) / s[-1], c["wall_parallel"],
             c["wall_parallel"] / s[-1]), flush=True)
pickle.dump(res, open(DIR / a.out, "wb"))
print("done %d cases in %.1f s" % (len(res), time.time() - t_start))
