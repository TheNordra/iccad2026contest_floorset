"""L302 phase 1 -- capture the EXACT (env, stdin) every profile subprocess gets.

Why a spy and not a reconstruction: `_l137_env()` is non-empty at >=40 cores and
the hint block rides in the shared stdin, so building the input by hand builds a
DIFFERENT case than the deployment path (research-handoff trap #1). The only safe
way to replay a profile is to replay the bytes the wrapper actually sent.

Config = the BETA package, reconstructed with the shipped kill switches (L285):
that is the only configuration the grader has ever measured per case, and its
per-case wall is in beta_evaluation_results.json.

  <python> l302_capture.py            -> l302_capture.pkl
"""
import os, pickle, sys, time
from pathlib import Path

DIR = Path(__file__).parent
BETA_ENV = dict(ICCAD_ADAPTIVE_CORES="48", ICCAD_SHAPE_LP="0", ICCAD_ROUTE_A="0",
                ICCAD_M80_TIER="0", ICCAD_HINT_MODE="0",
                ICCAD_L223_REFINE_HEAVY="4", ICCAD_L231_REFINE_MID="8")
os.environ.update(BETA_ENV)
sys.path.insert(0, str(DIR))
sys.path.insert(0, r"C:/ICCAD_ml")
sys.path.insert(0, str(DIR / "iccad2026contest"))

import torch                                                       # noqa: E402
from lite_dataset_test import FloorplanDatasetLiteTest             # noqa: E402
import optimizer_constructive as oc                                # noqa: E402

# hard gate: the flags must actually be live in THIS process
assert oc._effective_cores_hi() >= 40, "cores gate not armed"
assert os.environ.get("ICCAD_SHAPE_LP") == "0"

CAP = []
_orig = oc._run_profile


def spy(env_over, inp, n):
    CAP[-1]["profiles"].append((dict(env_over), inp))
    return _orig(env_over, inp, n)


oc._run_profile = spy
ds = FloorplanDatasetLiteTest(r"C:/ICCAD_ml/")
opt = oc.MyOptimizer(verbose=False)
out = []
for i in range(100):
    s = ds[i]
    at, b2b, p2b, pins, cons = s["input"]
    n = int((at > 0).sum()) if at.dim() == 1 else at.shape[0]
    n = len(cons)
    CAP.append({"case": i, "n": n, "profiles": []})
    t0 = time.time()
    opt.solve(n, at, b2b, p2b, pins, cons)
    CAP[-1]["wall_parallel"] = time.time() - t0
    print("  case %3d n=%3d  profiles %2d  parallel wall %6.3f s"
          % (i, n, len(CAP[-1]["profiles"]), CAP[-1]["wall_parallel"]), flush=True)
pickle.dump(CAP, open(DIR / "l302_capture.pkl", "wb"))
print("captured %d cases, %d profile calls"
      % (len(CAP), sum(len(c["profiles"]) for c in CAP)))
