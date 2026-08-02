"""M83 contention slope on THIS box (OFFLINE, never shipped, produces wall-clock).

WHY (2026-08-02): R1 measured the grader POOL (35 profiles, forced via
ICCAD_ADAPTIVE_CORES=48) against the local pool (13) and found every one of the
20 big-band cases got SLOWER -- median 1.32x, worst 1.69x.  The wall model
    W = max(max_i dt_i, sum_i dt_i / cores, sum_i pt_i)
predicted median 1.000x, worst 1.142x.  The model's assumption that dt_i is
INVARIANT under concurrency is therefore measurably false, and that assumption
underpins:
  * M67-E's "the 22 restored profiles are free at 48c (dW=+0.00%)"
  * G1's tier-5 verdict
  * G2's route-A (frame parallelism) upper bound

But the R1 number CANNOT be read as a grader result: this box is 16 PHYSICAL /
32 logical, so 35 processes is ~2.2x oversubscribed on physical cores, while the
grader's 48 cores host 35 processes with room to spare.  Different regime.

So measure the SLOPE instead, which is transferable:
    dt(k)/dt(1) for k = 1 .. 35
with two structural breaks to look for:
    k > 16  -> SMT siblings start sharing a physical core
    k > 32  -> genuine oversubscription (this box only)
If the curve is FLAT up to ~32 then 35 processes on a 48-core grader is also
flat, and the model's assumption is fine THERE even though it fails here.  If it
rises steeply before 16, the grader is at risk and the teammate needs to know.

No pinning: the shipped wrapper does not pin either, so letting the OS place the
processes is the production-representative measurement (and on this homogeneous
AMD part there are no fast/slow core classes to control for, unlike the Intel
hybrid box M67-F had to pin around).

Modes: combos | sweep | report
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

import m67f_contention_probe as cp                # noqa: E402 (loads dataset)
import optimizer_constructive as oc               # noqa: E402

OUT = _DIR / "results_M83_contention_slope.json"
KS = (1, 4, 8, 13, 16, 20, 24, 28, 32, 35)
PHYS, LOGI = 16, 32                               # this box
AUDIT = _DIR / "audit_cache_m71_kband.pkl"


def run_batch_free(txt, env_over, k, expect_n):
    """k identical copies released together by a barrier, NOT pinned.

    m67f_contention_probe.run_batch pins copy i to logical core i, which on this
    box would be wrong: Windows/AMD enumerate SMT siblings adjacently, so pinning
    to 0..15 lands on only ~8 PHYSICAL cores and manufactures contention that the
    real workload never sees.  The shipped wrapper does not pin, and the OS knows
    the topology, so letting it place the processes is both correct and
    production-representative.
    """
    import threading
    procs = [cp._spawn(env_over) for _ in range(k)]
    barrier = threading.Barrier(k)
    res = [None] * k

    def worker(i):
        p = procs[i]
        barrier.wait()
        t0 = time.perf_counter()
        out, _err = p.communicate(input=txt)
        res[i] = (time.perf_counter() - t0,
                  p.returncode == 0
                  and len(cp._parse_output(out, expect_n)) == expect_n)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return [r[0] for r in res], all(r[1] for r in res)


def shipped_env(k_prof, n):
    """Per-profile env exactly as the shipped wrapper builds it at 48 cores."""
    env = dict(oc._PROFILES[k_prof])
    env.update(getattr(oc, "_M71_ENV", {}))       # M71 rides every profile
    if n > 100:
        env["ICCAD_REFINE_ITERS"] = "4"           # M49 band overlay
    elif n > 60:
        env["ICCAD_REFINE_ITERS"] = "8"           # M50 band overlay
    return env


def pick(want=3):
    """Heaviest cases by their 48c-pool max-setter dt, from the SHIPPED-config
    audit (M71 + K overlay).  These are the profiles that actually set the wall."""
    import pickle
    data = pickle.load(open(AUDIT, "rb"))["data"]
    import os
    rows = []
    for ci in range(100):
        n = max(len(data[(ci, k)][0]) for k in range(41) if (ci, k) in data)
        if n <= 100:
            continue
        os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
        pool = oc._pool_indices(n)
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
        cand = [(data[(ci, k)][1], k) for k in pool if (ci, k) in data]
        if cand:
            dt, k = max(cand)
            rows.append((dt, ci, k, n))
    rows.sort(reverse=True)
    return [(ci, k, n, dt) for dt, ci, k, n in rows[:want]]


def mode_combos():
    for ci, k, n, dt in pick(6):
        print(f"case {ci:>3} n={n:>3}  max-setter profile #{k:<3} audit dt={dt:.3f}s")


def mode_sweep(want, reps):
    combos = pick(want)
    print(f"box: {PHYS} physical / {LOGI} logical.  no pinning (production-shaped)")
    print(f"k sweep {KS}   reps {reps}   breaks: k>{PHYS} SMT, k>{LOGI} oversubscribed\n")
    out = {}
    for ci, kprof, n, adt in combos:
        txt, expect_n = cp.case_txt(ci)
        env = shipped_env(kprof, n)
        base = None
        row = {}
        print(f"== case {ci} n={n} profile #{kprof} (audit dt {adt:.3f}s) ==")
        print(f"{'k':>4} {'mean':>8} {'max':>8} {'ratio(mean)':>12} {'ratio(max)':>11}  note")
        for k in KS:
            meds = []
            for _r in range(reps):
                dts, ok = run_batch_free(txt, env, k, expect_n)
                if not ok:
                    print(f"{k:>4}   FAILED (bad output)")
                    break
                meds.append(dts)
            if not meds:
                continue
            mean_k = st.median([st.mean(d) for d in meds])
            max_k = st.median([max(d) for d in meds])
            if base is None:
                base = mean_k
            note = ("" if k <= PHYS else
                    ("SMT" if k <= LOGI else "OVERSUBSCRIBED"))
            print(f"{k:>4} {mean_k:>8.3f} {max_k:>8.3f} {mean_k/base:>12.3f} "
                  f"{max_k/base:>11.3f}  {note}")
            row[k] = {"mean": mean_k, "max": max_k, "ratio_mean": mean_k / base,
                      "ratio_max": max_k / base}
        out[f"{ci}_{kprof}"] = row
        print()
    # verdict
    print("=" * 74)
    r16 = [v[16]["ratio_mean"] for v in out.values() if 16 in v]
    r32 = [v[32]["ratio_mean"] for v in out.values() if 32 in v]
    r35 = [v[35]["ratio_mean"] for v in out.values() if 35 in v]
    if r16:
        print(f"dt(16)/dt(1)  median {st.median(r16):.3f}  worst {max(r16):.3f}"
              f"   <- all on distinct PHYSICAL cores")
    if r32:
        print(f"dt(32)/dt(1)  median {st.median(r32):.3f}  worst {max(r32):.3f}"
              f"   <- SMT siblings shared, still not oversubscribed")
    if r35:
        print(f"dt(35)/dt(1)  median {st.median(r35):.3f}  worst {max(r35):.3f}"
              f"   <- OVERSUBSCRIBED (this box only; grader is not)")
    print("\nREAD: the grader hosts 35 processes on 48 cores = the k<=32 regime")
    print("      here, NOT the k=35 column.  If dt(32)/dt(1) is flat, the wall")
    print("      model's dt-invariance is fine on the grader and R1's 1.32x is a")
    print("      local oversubscription artefact.  If it is already steep at")
    print("      k<=16, the model is wrong everywhere and tier-5 needs re-pricing.")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["combos", "sweep"])
    ap.add_argument("--combos", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.mode == "combos":
        mode_combos()
    else:
        mode_sweep(a.combos, a.reps)
    print(f"[{a.mode} {time.perf_counter() - t0:.0f}s]")
