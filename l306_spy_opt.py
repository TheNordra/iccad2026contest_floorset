"""L306 -- an optimizer wrapper that records what a case actually costs.

Loaded by the OFFICIAL evaluator (`--evaluate l306_spy_opt.py`) so the inputs are
the deployment inputs by construction.  My first attempt drove `solve()` from my
own loop with `target_positions=None`; that silently removes every preplaced
block's position, makes the pack far easier, and the whole run came out 1.7x
faster.  The research handoff's trap #1 in a new costume -- so the driver is now
the evaluator itself, and the gate is that the run reproduces L285's beta-config
total 1.259897682.

Records per case:  the (env, stdin) of every profile subprocess, the wall of
_serialize_input (runs before the pool, uncontended), and one (positions, margs)
pair so _proxy_metrics can be re-timed uncontended afterwards.
"""
import atexit, pickle, sys, time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import optimizer_constructive as _oc                               # noqa: E402
from iccad2026_evaluate import FloorplanOptimizer                  # noqa: E402

CAP = []
_rp, _si, _pm = _oc._run_profile, _oc._serialize_input, _oc._proxy_metrics


def _spy_rp(env_over, inp, n):
    if CAP:
        CAP[-1]["profiles"].append((dict(env_over), inp))
    return _rp(env_over, inp, n)


def _spy_si(*a, **k):
    t0 = time.perf_counter()
    r = _si(*a, **k)
    if CAP:
        CAP[-1]["t_serialize"] = time.perf_counter() - t0
    return r


def _spy_pm(positions, *margs):
    if CAP and "margs" not in CAP[-1]:
        CAP[-1]["margs"] = margs
        CAP[-1]["pos"] = positions
    return _pm(positions, *margs)


_oc._run_profile, _oc._serialize_input, _oc._proxy_metrics = _spy_rp, _spy_si, _spy_pm


class MyOptimizer(_oc.MyOptimizer):
    def solve(self, block_count, *a, **k):
        CAP.append({"n": int(block_count), "profiles": []})
        t0 = time.time()
        try:
            return super().solve(block_count, *a, **k)
        finally:
            CAP[-1]["wall"] = time.time() - t0


atexit.register(lambda: pickle.dump(CAP, open(_DIR / "l306_capture.pkl", "wb")))
