"""L307b -- re-measure a block-count range and keep the ELEMENTWISE MINIMUM.

The first pass ran 12:22-13:06 and another session started a 43-way parallel
evaluation at 12:52, so the last ~30 % of it (which is the heavy band, because the
replay walks n in order) was contended.  Contention can only inflate a time, so a
second pass merged with min() is the fix, and it is the project's own min-of-N
rule applied to the thing that matters most.
"""
import os, pickle, subprocess, sys, time
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import optimizer_constructive as oc                                # noqa: E402

nmin = int(sys.argv[1]) if len(sys.argv) > 1 else 95
CAP = pickle.load(open(DIR / "l306_capture.pkl", "rb"))
OLD = pickle.load(open(DIR / "l307_serial.pkl", "rb"))
BIN = str(DIR / "constructive.exe")
t0all = time.time()
for c in CAP:
    n = c["n"]
    if n < nmin:
        continue
    dts, pos0 = [], None
    for env_over, inp in c["profiles"]:
        env = dict(os.environ); env.update(env_over)
        t0 = time.perf_counter()
        r = subprocess.run([BIN], input=inp, capture_output=True, text=True, env=env)
        dts.append(time.perf_counter() - t0)
        if pos0 is None and r.returncode == 0 and r.stdout.strip():
            pos0 = oc._parse_output(r.stdout, n)
    tp = []
    if pos0 is not None and "margs" in c:
        for _ in range(3):
            t1 = time.perf_counter()
            oc._proxy_metrics(pos0, *c["margs"])
            tp.append(time.perf_counter() - t1)
    M2, C2 = max(dts), len(dts) * (min(tp) if tp else 0.0)
    o = OLD[n]
    print("  n=%3d  M %6.3f -> %6.3f (%+5.1f%%)   C %6.3f -> %6.3f"
          % (n, o["M"], M2, 100 * (M2 / o["M"] - 1), o["C"], C2), flush=True)
    o["M"] = min(o["M"], M2)
    o["C"] = min(o["C"], C2)
    o["SUM"] = min(o["SUM"], sum(dts))
pickle.dump(OLD, open(DIR / "l307_serial.pkl", "wb"))
print("merged (min) in %.1f s" % (time.time() - t0all))
