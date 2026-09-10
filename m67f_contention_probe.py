"""M67-F Phase 2 — E-core memory-contention probe (OFFLINE-ONLY, never shipped).

WHY
---
M67-F Phase 1 measured theta_pool = 0.7636 (GREEN): restoring the M42/M45 pool cuts
recovers 76% of the +2.8% OOS quality tax. Shipping the cores-gated tier-5 (restore
M42 at >=32 cores) is blocked on Phase 2: proving that at 48 cores -- where the wall
is the max-setter, not sum/cores -- restoring ~21 profiles (heavy band 13 -> 34) does
NOT inflate the wall.

That whole argument rests on the M67-E wall model
    W(pool, cores) = max(max_i dt_i, sum_i dt_i / cores, sum PT)
and its UNMEASURED assumption: a single profile's dt_i is INVARIANT under parallelism.
audit_cache dt was only ever measured at ~11-way on the 12-core box (the model is
already off by 9% at 41-way oversubscription). At 48 cores both the shipped (~13) and
restore (~34) heavy-band pools fit within 48 physical cores (no oversubscription), so
the ONLY thing that can make restore cost more is memory-subsystem contention slowing
the max-setter itself -- exactly what the wall model assumes away.

We cannot run 48 cores locally, but we can measure the CONTENTION SLOPE: pin the
constructive.exe max-setter to homogeneous E-cores and watch dt(k)/dt(1) as k=1..8
identical copies co-run (1 process per E-core, no oversubscription).

WHY E-CORES = A CONSERVATIVE UPPER BOUND
----------------------------------------
Gracemont E-cores have the LEAST cache/bandwidth per core on this box (a 4-core cluster
shares one 2 MB L2 ~= 512 KB/core, no private LLC slice). Eight of them hammering a
single mobile memory controller is the harshest per-core squeeze this machine can make.
If dt(k)/dt(1) stays flat even here, the workload (<=120-block greedy packing, small
working set, mostly L2/LLC-resident) is NOT memory-subsystem-bound -> adding co-runners
on a far better-provisioned 48c ICELAKE server inflates the max-setter even less.
So: LOCAL GREEN => server GREEN (one-directional safe inference); LOCAL RED => kill
conservatively (do not bet the server's slack rescues it). The dt(k)/dt(1) RATIO cancels
the E-cores' absolute slowness -- we measure the contention slope, not raw speed.

DECISION (pre-registered, mirrors M64/M65 pilot discipline). Gate on the WORST combo's
dt(8)/dt(1):
    GREEN  <= 1.03  and no rising slope (extrap k=16 < 1.05)
            -> wall model holds under the conservative bound; Phase 2's only remaining
               item is reconciling vs Beta's actual per-case runtime_seconds. M67-F alive.
    AMBER  1.03 .. 1.10
            -> locally inconclusive; defer to real 48c wall data, do NOT ship tier-5 on
               the model alone. (Feed the measured factor f into m67e_rf48.py: restore
               band walls x f, re-project for the exact net.)
    RED    >= 1.10  or clear monotone rise (extrap k=16 > 1.08)
            -> "restore is free at 48c" is falsified even under the local upper bound
               -> write M67-F death into the ledger; M42/M45 stay cut; do NOT ship tier-5.

Submission form is untouched by this file (pure measurement).

Run:  python -u m67f_contention_probe.py > m67f_contention_stdout.txt 2>&1
"""
import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# Strip ICCAD_* so imports and every child process start from shipped-default env
# (a stray knob in the parent shell would silently change what we time).
for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

import pickle

from iccad2026_evaluate import ContestEvaluator          # noqa: E402
from optimizer_claude import _serialize_input, _parse_output  # noqa: E402
from proxy_analysis import build_opt_target_pos          # noqa: E402
import optimizer_constructive as oc                       # noqa: E402

EXE = str(_DIR / "constructive.exe")
NLOG = os.cpu_count() or 16                                # 16 logical on i7-1260P

# Pre-registered gate constants (see module docstring).
GREEN_MAX = 1.03
RED_MAX = 1.10
GREEN_EXTRAP16 = 1.05
RED_EXTRAP16 = 1.08

# ── ctypes affinity (no psutil on this box) ─────────────────────────────────
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_k32.SetProcessAffinityMask.restype = ctypes.c_int
_k32.GetCurrentProcess.restype = ctypes.c_void_p


def _set_affinity(handle: int, mask: int) -> None:
    if not _k32.SetProcessAffinityMask(ctypes.c_void_p(handle), ctypes.c_size_t(mask)):
        raise ctypes.WinError(ctypes.get_last_error())


def _pin_self(mask: int) -> None:
    _set_affinity(_k32.GetCurrentProcess(), mask)


# ── dataset prep (mirrors profile_audit.py:69-82; no import of that module —
#    importing it runs a whole audit). Built lazily per case. ────────────────
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
_txt_cache = {}


def case_txt(idx: int):
    if idx not in _txt_cache:
        s = _ev.dataset[idx]
        inp, lab = s["input"], s["label"]
        at, b2b, p2b, pins, cons = inp
        n = int((at != -1).sum().item())
        _base, tp = _ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
        otp = build_opt_target_pos(tp, cons, n)
        txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
        _txt_cache[idx] = (txt, n)
    return _txt_cache[idx]


# ── heavy restore-pool max-setters ──────────────────────────────────────────
# Fallback list = audit_cache restore-pool argmax dt per heavy case (verified
# 2026-07-23): (case, profile_idx). Recomputed from audit_cache when present.
_FALLBACK_COMBOS = [(99, 6), (98, 40), (95, 18), (89, 22), (85, 23)]


def _restore_pool(n: int):
    os.environ["ICCAD_ADAPTIVE_POOL"] = "1"
    os.environ["ICCAD_M67F_RESTORE"] = "1"
    pool = set(oc._pool_indices(n))
    del os.environ["ICCAD_M67F_RESTORE"]
    del os.environ["ICCAD_ADAPTIVE_POOL"]
    return pool


def pick_combos(want: int):
    """Top-`want` heavy cases by restore-pool max-setter dt (from audit_cache)."""
    cache = _DIR / "audit_cache.pkl"
    if not cache.exists():
        print("[combos] audit_cache.pkl missing -> fallback list", flush=True)
        return _FALLBACK_COMBOS[:want]
    try:
        data = pickle.load(open(cache, "rb"))["data"]
    except Exception as e:
        print(f"[combos] audit_cache load failed ({e}) -> fallback", flush=True)
        return _FALLBACK_COMBOS[:want]
    rows = []
    for ci in range(100):
        ns = [len(data[(ci, k)][0]) for k in range(len(oc._PROFILES))
              if (ci, k) in data and data[(ci, k)][0]]
        if not ns:
            continue
        n = max(ns)
        pool = _restore_pool(n)
        cand = [(data[(ci, k)][1], k) for k in pool if (ci, k) in data]
        if not cand:
            continue
        dtmax, kmax = max(cand)
        rows.append((dtmax, ci, kmax))
    rows.sort(reverse=True)
    combos = [(ci, k) for _dt, ci, k in rows[:want]]
    return combos or _FALLBACK_COMBOS[:want]


# ── timed runs ──────────────────────────────────────────────────────────────
def _spawn(env_over):
    env = dict(os.environ)
    env.update(env_over)
    return subprocess.Popen([EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)


def run_pinned(txt, env_over, core, expect_n):
    """Single process pinned to one logical core, timed end-to-end."""
    p = _spawn(env_over)
    _set_affinity(int(p._handle), 1 << core)   # pin BEFORE feeding stdin (compute
    t0 = time.perf_counter()                   # hasn't started; proc blocks on read)
    out, _err = p.communicate(input=txt)
    dt = time.perf_counter() - t0
    ok = p.returncode == 0 and len(_parse_output(out, expect_n)) == expect_n
    return dt, ok


def run_batch(txt, env_over, cores, expect_n):
    """k identical copies, each pinned to a distinct core, released together by a
    barrier so they contend for the whole compute. Returns per-copy wall times."""
    k = len(cores)
    procs = []
    for c in cores:
        p = _spawn(env_over)
        _set_affinity(int(p._handle), 1 << c)  # all pinned & blocked on stdin
        procs.append(p)
    barrier = threading.Barrier(k)
    results = [None] * k

    def worker(i):
        p = procs[i]
        barrier.wait()                         # all k start writing+computing together
        t0 = time.perf_counter()
        out, _err = p.communicate(input=txt)   # communicate drains stdout -> no pipe deadlock
        dt = time.perf_counter() - t0
        results[i] = (dt, p.returncode == 0 and len(_parse_output(out, expect_n)) == expect_n)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r[0] for r in results], all(r[1] for r in results)


# ── E-core calibration ──────────────────────────────────────────────────────
def calibrate(reps: int):
    """Rank the 16 logical CPUs by single-run wall (sequential, no co-runners).
    The 8 slowest = E-cores. Uses the first heavy combo at REFINE=4 (fast; ranking
    is scale-invariant)."""
    ci, pk = pick_combos(1)[0]
    txt, n = case_txt(ci)
    env = dict(oc._PROFILES[pk], ICCAD_REFINE_ITERS="4")
    run_pinned(txt, env, 0, n)                 # warm exe/case cache
    med = {}
    for core in range(NLOG):
        ts = []
        for _ in range(reps):
            dt, ok = run_pinned(txt, env, core, n)
            if not ok:
                raise RuntimeError(f"calibration run failed on core {core}")
            ts.append(dt)
        med[core] = statistics.median(ts)
        print(f"  core {core:>2}: {med[core]:.3f}s", flush=True)
    order = sorted(range(NLOG), key=lambda c: med[c])   # fast -> slow
    ecores = sorted(order[-8:])
    return med, ecores


# ── contention sweep ────────────────────────────────────────────────────────
def sweep(ci, pk, ecores, ks, reps, refine):
    txt, n = case_txt(ci)
    env = dict(oc._PROFILES[pk])
    if refine:
        env["ICCAD_REFINE_ITERS"] = str(refine)
    run_pinned(txt, env, ecores[0], n)          # warm-up (discarded)
    curve = {}
    for k in ks:
        cores = ecores[:k]
        means, maxes = [], []
        for _ in range(reps):
            dts, ok = run_batch(txt, env, cores, n)
            if not ok:
                raise RuntimeError(f"batch failed: case {ci} prof {pk} k={k}")
            means.append(sum(dts) / k)
            maxes.append(max(dts))
        curve[k] = {"mean": statistics.median(means),
                    "maxc": statistics.median(maxes),
                    "rep_means": means}
        print(f"    k={k}: dt_mean={curve[k]['mean']:.3f}s "
              f"dt_max={curve[k]['maxc']:.3f}s", flush=True)
    return n, curve


def _linfit(xs, ys):
    m = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = m * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, ys[0]
    b = (m * sxy - sx * sy) / denom
    a = (sy - b * sx) / m
    return b, a          # slope per +1 co-runner, intercept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", type=int, default=5)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--calib-reps", type=int, default=3)
    ap.add_argument("--refine", type=int, default=4,
                    help="REFINE_ITERS for the primary sweep (4 = shipped Beta for "
                         "n>100). 0 = profile default (K=12).")
    ap.add_argument("--k12-check", action="store_true", default=True,
                    help="also run the heaviest combo at K=12 (k in {1,4,8}) as a "
                         "conservative cross-check.")
    ap.add_argument("--no-k12-check", dest="k12_check", action="store_false")
    ap.add_argument("--ecores", type=str, default="",
                    help="comma list to override E-core detection, e.g. 8,9,..,15")
    ap.add_argument("--out", type=str, default=str(_DIR / "results_M67F_contention.json"))
    args = ap.parse_args()

    ks = list(range(1, 9))
    print(f"=== M67-F Phase 2 contention probe ===", flush=True)
    print(f"box: i7-1260P (4P+8E / {NLOG} logical); exe={EXE}", flush=True)
    print(f"params: combos={args.combos} reps={args.reps} refine={args.refine or 'K12'} "
          f"k12_check={args.k12_check}", flush=True)

    # Pin the orchestrator to P-cores (logical 0-7) so its I/O threads never steal
    # E-core cycles from the measured processes.
    _pin_self(0x00FF)

    print("\n[1] E-core calibration (sequential single runs, slowest 8 = E-cores)",
          flush=True)
    calib, ecores = calibrate(args.calib_reps)
    if args.ecores:
        ecores = sorted(int(x) for x in args.ecores.split(","))
        print(f"  E-cores OVERRIDDEN -> {ecores}", flush=True)
    expected = list(range(8, 16))
    warn = None
    if ecores != expected:
        warn = f"detected E-cores {ecores} != expected {expected}"
        print(f"  WARNING: {warn} (using detected set)", flush=True)
    else:
        print(f"  E-cores = {ecores} (matches Alder Lake logical 8-15)", flush=True)

    combos = pick_combos(args.combos)
    print(f"\n[2] heavy restore-pool max-setters: {combos}", flush=True)

    results = {"params": vars(args), "ecores": ecores, "calib": calib,
               "expected_ecores": expected, "warn": warn, "combos": {}, "k12": {}}

    print(f"\n[3] contention sweep (REFINE={args.refine or 'K12'})", flush=True)
    for ci, pk in combos:
        print(f"  case {ci} prof #{pk}:", flush=True)
        n, curve = sweep(ci, pk, ecores, ks, args.reps, args.refine)
        infl = {k: curve[k]["mean"] / curve[1]["mean"] for k in ks}
        results["combos"][f"{ci}:{pk}"] = {"n": n, "curve": curve, "infl": infl}

    # K=12 conservative cross-check on the heaviest combo.
    if args.k12_check:
        ci, pk = combos[0]
        print(f"\n[4] K=12 conservative cross-check: case {ci} prof #{pk} k in {{1,4,8}}",
              flush=True)
        n, curve = sweep(ci, pk, ecores, [1, 4, 8], max(3, args.reps - 1), 0)
        infl = {k: curve[k]["mean"] / curve[1]["mean"] for k in (1, 4, 8)}
        results["k12"] = {"case": ci, "prof": pk, "n": n, "curve": curve, "infl": infl}

    # ── aggregate + pre-registered gate ─────────────────────────────────────
    infl8 = {key: c["infl"][8] for key, c in results["combos"].items()}
    infl8_vals = list(infl8.values())
    infl8_max = max(infl8_vals)
    infl8_med = statistics.median(infl8_vals)
    worst_key = max(infl8, key=infl8.get)
    # aggregate curve = median inflation across combos per k; slope + extrapolation
    agg = {k: statistics.median(c["infl"][k] for c in results["combos"].values())
           for k in ks}
    slope, intercept = _linfit(ks, [agg[k] for k in ks])
    extrap16 = intercept + slope * 16
    extrap34 = intercept + slope * 34

    if infl8_max <= GREEN_MAX and extrap16 < GREEN_EXTRAP16:
        verdict = "GREEN"
    elif infl8_max >= RED_MAX or extrap16 > RED_EXTRAP16:
        verdict = "RED"
    else:
        verdict = "AMBER"

    results["summary"] = {
        "infl8_per_combo": infl8, "infl8_max": infl8_max, "infl8_median": infl8_med,
        "worst_combo": worst_key, "agg_curve": agg, "slope_per_corunner": slope,
        "extrap_k16": extrap16, "extrap_k34": extrap34, "verdict": verdict,
        "gate": {"GREEN_MAX": GREEN_MAX, "RED_MAX": RED_MAX,
                 "GREEN_EXTRAP16": GREEN_EXTRAP16, "RED_EXTRAP16": RED_EXTRAP16},
    }
    if results["k12"]:
        results["summary"]["k12_infl8"] = results["k12"]["infl"][8]

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'combo':>10} {'n':>4} " + " ".join(f"k{k:>5}" for k in ks), flush=True)
    for key, c in results["combos"].items():
        print(f"{key:>10} {c['n']:>4} " +
              " ".join(f"{c['infl'][k]:>6.3f}" for k in ks), flush=True)
    print(f"\nagg infl:  " + " ".join(f"k{k}={agg[k]:.3f}" for k in ks), flush=True)
    print(f"dt(8)/dt(1): max={infl8_max:.3f} ({worst_key})  median={infl8_med:.3f}",
          flush=True)
    print(f"slope/co-runner={slope:+.4f}  extrap k16={extrap16:.3f}  k34={extrap34:.3f}",
          flush=True)
    if results["k12"]:
        print(f"K=12 cross-check dt(8)/dt(1) = {results['k12']['infl'][8]:.3f} "
              f"(case {results['k12']['case']} #{results['k12']['prof']})", flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print("  (conservative upper bound: E-cores are the box's most cache/bandwidth-"
          "starved cores; flat here => flat on the 48c server)", flush=True)
    if verdict == "GREEN":
        print("  => wall model holds; M67-F Phase 2 only needs Beta per-case "
              "runtime_seconds to reconcile.", flush=True)
    elif verdict == "RED":
        print("  => 'restore is free at 48c' falsified under the local upper bound; "
              "write M67-F death into ledger, do NOT ship tier-5.", flush=True)
    else:
        print("  => locally inconclusive; defer to real 48c wall data. Feed the "
              "measured factor into m67e_rf48.py for the exact net.", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ndumped {args.out}", flush=True)


if __name__ == "__main__":
    main()
