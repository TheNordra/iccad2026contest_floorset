"""L304 -- the serial-Python halves of a case, measured UNCONTENDED.

`_solve_impl` has exactly three cost centres and the code says which is which:

  M  the slowest profile subprocess.  43 profiles <= 48 grader cores => ONE wave,
     so on the grader the pool's subprocess component IS this single number.
     Measured uncontended by l302_replay.py.
  C  `_proxy_metrics`, run **on the main thread, one at a time** (the M47 comment
     at :2726 is explicit -- concurrent proxies were 4x worse under the GIL), so
     43 x one proxy, strictly serialised, overlapping the still-running profiles.
  S  everything outside the pool: `_serialize_input` and the final argmin.

Both C and S are single-threaded Python/shapely, i.e. the same KIND of work as
the shape LP.  Timing them here uncontended is what lets the grader's published
per-case wall be decomposed instead of divided by a whole-case ratio.
"""
import os, pickle, statistics, subprocess, sys, time
from pathlib import Path

DIR = Path(__file__).parent
os.environ.update(dict(ICCAD_ADAPTIVE_CORES="48", ICCAD_SHAPE_LP="0",
                       ICCAD_ROUTE_A="0", ICCAD_M80_TIER="0", ICCAD_HINT_MODE="0",
                       ICCAD_L223_REFINE_HEAVY="4", ICCAD_L231_REFINE_MID="8"))
sys.path.insert(0, str(DIR)); sys.path.insert(0, r"C:/ICCAD_ml")
import torch                                                       # noqa: E402
from lite_dataset_test import FloorplanDatasetLiteTest             # noqa: E402
import optimizer_constructive as oc                                # noqa: E402

assert oc._effective_cores_hi() >= 40
CAP = {c["n"]: c for c in pickle.load(open(DIR / "l302_capture.pkl", "rb"))}
ds = FloorplanDatasetLiteTest(r"C:/ICCAD_ml/")
BIN = str(DIR / "constructive.exe")
out = {}
for i in range(100):
    s = ds[i]
    at, b2b, p2b, pins, cons = s["input"]
    n = len(cons)
    c = CAP[n]
    npro = len(c["profiles"])
    # S1: input serialisation, min of 3
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        oc._serialize_input(n, at, b2b, p2b, pins, cons, None, None)
        ts.append(time.perf_counter() - t0)
    s_build = min(ts)
    # one profile run -> positions for the proxy
    env = dict(os.environ); env.update(c["profiles"][0][0])
    r = subprocess.run([BIN], input=c["profiles"][0][1], capture_output=True,
                       text=True, env=env)
    pos = oc._parse_output(r.stdout, n)
    margs = (at, b2b, p2b, pins, cons, n)
    tp = []
    for _ in range(3):
        t0 = time.perf_counter()
        oc._proxy_metrics(pos, *margs)
        tp.append(time.perf_counter() - t0)
    c1 = min(tp)
    out[n] = dict(n=n, npro=npro, s_build=s_build, proxy1=c1, C=npro * c1)
    if n % 10 == 0 or n > 115:
        print("  n=%3d  profiles %2d  serialize %.4f s  proxy1 %.4f s  C=%.3f s"
              % (n, npro, s_build, c1, npro * c1), flush=True)
pickle.dump(out, open(DIR / "l304_pyparts.pkl", "wb"))
print("sum S_build %.2f s   sum C %.2f s" % (sum(v["s_build"] for v in out.values()),
                                             sum(v["C"] for v in out.values())))
