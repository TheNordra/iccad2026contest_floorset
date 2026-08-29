"""L307 -- replay every captured profile SERIALLY (uncontended), then re-time the
proxy uncontended.  Nothing else may run on the box while this does.

    M(n)  = max over the 43 profiles of the uncontended subprocess wall
    SUM   = sum of all 43, only used to show c*
    C(n)  = 43 x one uncontended _proxy_metrics   (M47: exactly one runs at a time)
    S(n)  = _serialize_input, timed in situ before the pool started
"""
import os, pickle, statistics, subprocess, sys, time
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import optimizer_constructive as oc                                # noqa: E402

CAP = pickle.load(open(DIR / "l306_capture.pkl", "rb"))
BIN = str(DIR / "constructive.exe")
out = {}
t0all = time.time()
for c in CAP:
    n = c["n"]
    dts = []
    pos0 = None
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
    proxy1 = min(tp) if tp else 0.0
    out[n] = dict(n=n, npro=len(dts), M=max(dts), SUM=sum(dts), proxy1=proxy1,
                  C=len(dts) * proxy1, S=c.get("t_serialize", 0.0), wall=c["wall"])
    print("  n=%3d  M %6.3f  SUM %8.3f  c* %5.1f  proxy1 %.4f  C %6.3f  S %.4f  parallel %6.3f"
          % (n, out[n]["M"], out[n]["SUM"], out[n]["SUM"] / out[n]["M"],
             proxy1, out[n]["C"], out[n]["S"], c["wall"]), flush=True)
pickle.dump(out, open(DIR / "l307_serial.pkl", "wb"))
print("done in %.1f s ; sum M %.2f  sum C %.2f  sum S %.2f"
      % (time.time() - t0all, sum(v["M"] for v in out.values()),
         sum(v["C"] for v in out.values()), sum(v["S"] for v in out.values())))
