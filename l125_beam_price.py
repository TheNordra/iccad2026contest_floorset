"""L125 — what does a beam twin cost in RuntimeFactor at 48 cores?

OFFLINE (never shipped). Pure cache analysis: reads `audit_cache_ship.pkl` and
runs no solver.

This is L125-B0 written down. B0 was done by hand on 2026-08-12 and only its
conclusions survived (in `l125-beam-affordable-only-at-w2`); this file reproduces
them and fixes the one assumption B0 had to make.

THE MODEL. At 48 cores the wall is the MAX-SETTER on 100/100 cases (M67-E,
measured), so a beam twin costs whatever it adds to the wall — not what it adds to
itself. Weighted dRF over the in-set 100:

    dRF = sum_i w_i * [ (max(wall_i, m * dt_i,p) / wall_i)^0.3 - 1 ] / sum_i w_i

with w_i = exp(n_i/12) and gamma = 0.3. This IGNORES the RF floor of 0.7, which
49/100 cases already sit on, and where the extra time is entirely free => every
number here is an UPPER BOUND on the cost.

WHAT B0 ASSUMED AND THIS FIXES. B0 had no beam, so it priced m = W exactly. The
measured multiplier is not W: `l125_beam_probe.py ab` reads p50 2.30 / p90 2.93 at
W=2 (branch levels re-expand W times, nodes are copied, and — the part that is not
overhead — a beam completes frames the greedy abandons, so more frames reach the
post-processing). `--mult` prices the measured number instead of the assumed one.

TWO APPROXIMATIONS, both inherited from B0 and both stated rather than hidden:
  * the pool here is the audit cache's 41 shipped profiles + OM16, so the M80 and
    L124 tiers (which only exist at >=40 cores) are missing from the max. That
    understates today's wall, so it overstates the twin's cost.
  * dt is the audit cache's measured serial time on THIS machine.

Run:
  <python> l125_beam_price.py rank --mult 2.30
  <python> l125_beam_price.py set --mult 2.30 --src 33 11 9
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in sorted(k for k in os.environ if k.startswith("ICCAD_")):
    del os.environ[_k]

import optimizer_constructive as oc                                  # noqa: E402

GAMMA = 0.3
CACHE = _DIR / "audit_cache_ship.pkl"


def load():
    if not CACHE.exists():
        sys.exit("audit_cache_ship.pkl missing (profile_audit.py ship)")
    blob = pickle.load(open(CACHE, "rb"))
    data = blob["data"]
    ns = {}
    for (ci, _k), (pos, _dt) in data.items():
        ns.setdefault(ci, len(pos))
    return data, ns


def pools(data, ns, cores):
    """{case: [profile index]} exactly as the wrapper would build it."""
    kmax = 1 + max(k for _ci, k in data)
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(cores)
    out = {}
    for ci, n in ns.items():
        idx = [k for k in oc._pool_indices(n) if k < kmax and (ci, k) in data]
        out[ci] = idx
    del os.environ["ICCAD_ADAPTIVE_CORES"]
    return out


def drf(data, ns, pool, srcs, mult):
    """Weighted dRF of appending beam twins of `srcs` (upper bound: no 0.7 floor)."""
    num = den = 0.0
    raised = 0
    for ci, idx in pool.items():
        if not idx:
            continue
        w = math.exp(ns[ci] / 12.0)
        old = max(data[(ci, k)][1] for k in idx)
        add = max((mult * data[(ci, k)][1] for k in srcs if (ci, k) in data),
                  default=0.0)
        new = max(old, add)
        if new > old:
            raised += 1
        num += w * ((new / old) ** GAMMA - 1.0)
        den += w
    return num / den, raised


# ── measured dt for the WHOLE 48-core pool ──────────────────────────────────
# audit_cache_ship.pkl stops at the shipped prefix, so the M80 (86-93) and L124
# (94-101) tiers -- which only exist at >=40 cores, and which is where the beam
# screen puts its value -- have no dt anywhere. Measure them, SERIALLY (dt is a
# measurement, profile_audit.py's rule) and in one consistent pass rather than
# splicing today's numbers onto the audit cache's.
DTC = _DIR / "l125_dt_cache.pkl"


def _cases(nmin):
    from iccad2026_evaluate import ContestEvaluator
    from proxy_analysis import build_opt_target_pos
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    out = []
    for idx in range(100):
        s = ev.dataset[idx]
        at, b2b, p2b, pins, cons = s["input"]
        n = int((at != -1).sum().item())
        if n < nmin:
            continue
        _b, tp = ev._extract_baseline(idx, s["label"], b2b, p2b, pins, n)
        out.append(dict(idx=idx, n=n, at=at, b2b=b2b, p2b=p2b, pins=pins,
                        cons=cons, otp=build_opt_target_pos(tp, cons, n)))
    return out


def mode_measure(a):
    from optimizer_claude import _serialize_input
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    cases = _cases(a.nmin)
    pool = sorted(set(oc._pool_indices(max(c["n"] for c in cases))))
    del os.environ["ICCAD_ADAPTIVE_CORES"]
    base = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    C = pickle.load(open(DTC, "rb")) if DTC.exists() else {}
    arms = [("off", str(_DIR / "constructive.exe"), {})]
    if a.beam:
        arms.append(("on", str(_DIR / "constructive_l125.exe"),
                     {"ICCAD_BEAM": "1"}))
    todo = [(c, k, arm) for c in cases for k in pool for arm in arms
            if (c["idx"], k, arm[0], a.reps) not in C]
    print(f"[cfg] {len(cases)} cases (n>={a.nmin}) x {len(pool)} profiles x "
          f"{len(arms)} arms, reps={a.reps}: {len(todo)} to measure", flush=True)
    t0 = time.time()
    for j, (c, k, (an, exe, extra)) in enumerate(todo):
        env = dict(base)
        env.update(oc._PROFILES[k])
        env.update(oc._band_env(c["n"]))
        env.update(oc._m71_env())
        env.update(extra)
        txt = _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"], c["pins"],
                               c["cons"], c["otp"], gnn_hint=None)
        best = float("inf")
        for _r in range(a.reps):
            t1 = time.perf_counter()
            subprocess.run([exe], input=txt, capture_output=True, text=True,
                           env=env)
            best = min(best, time.perf_counter() - t1)
        C[(c["idx"], k, an, a.reps)] = best
        if (j + 1) % 40 == 0:
            el = time.time() - t0
            print(f"  {j + 1}/{len(todo)} ({el:.0f}s, eta "
                  f"{el / (j + 1) * (len(todo) - j - 1):.0f}s)", flush=True)
            pickle.dump(C, open(DTC, "wb"))
    pickle.dump(C, open(DTC, "wb"))
    print(f"[measure] cache has {len(C)} entries")
    return 0


def mode_afford(a):
    """Which pool profiles can carry a beam twin, using the measured dt."""
    if not DTC.exists():
        sys.exit("l125_dt_cache.pkl missing (run `measure` first)")
    C = pickle.load(open(DTC, "rb"))
    cases = sorted({ci for ci, _k, _a, _r in C})
    pool = sorted({k for _ci, k, _a, _r in C})
    ns = {c["idx"]: c["n"] for c in _cases(0)}
    have_on = any(an == "on" for _ci, _k, an, _r in C) and not a.use_mult
    rows = []
    for k in pool:
        num = den = 0.0
        raised = 0
        for ci in cases:
            dts = [C[(ci, q, "off", a.reps)] for q in pool
                   if (ci, q, "off", a.reps) in C]
            if not dts or (ci, k, "off", a.reps) not in C:
                continue
            old = max(dts)
            key = (ci, k, "on", a.reps)
            add = C[key] if (have_on and key in C) else a.mult * C[(ci, k, "off", a.reps)]
            new = max(old, add)
            if new > old:
                raised += 1
            w = math.exp(ns[ci] / 12.0)
            num += w * ((new / old) ** GAMMA - 1.0)
            den += w
        if den:
            rows.append((num / den, k, raised))
    rows.sort()
    src = "measured ON dt" if have_on else f"mult={a.mult}"
    print(f"[cfg] {len(cases)} cases, {len(pool)} pool profiles, {src}")
    print(f"\n  {'profile':>7}  {'weighted dRF':>13}  raised")
    for d, k, r in rows:
        tag = "  <- tier" if k >= 86 else ""
        print(f"  {k:>7}  {100 * d:>12.4f}%  {r:>6}{tag}")
    aff = [k for d, k, _r in rows if d <= a.budget]
    print(f"\n  affordable at dRF <= {100 * a.budget:.3f}%: {len(aff)} profiles")
    print(f"  --allow {' '.join(str(k) for k in sorted(aff))}")
    return 0


def mode_rank(a):
    data, ns = load()
    pool = pools(data, ns, a.cores)
    kmax = 1 + max(k for _ci, k in data)
    rows = []
    for k in range(kmax):
        if not any((ci, k) in data for ci in pool):
            continue
        d, r = drf(data, ns, pool, [k], a.mult)
        rows.append((d, k, r))
    rows.sort()
    print(f"[cfg] mult={a.mult}  cores={a.cores}  {len(pool)} cases  "
          f"{kmax} profiles in cache")
    print(f"\n  {'rank':>4}  {'profile':>7}  {'weighted dRF':>13}  raised")
    for i, (d, k, r) in enumerate(rows[:a.top]):
        print(f"  {i + 1:>4}  {k:>7}  {100 * d:>12.4f}%  {r:>6}")
    print(f"\n  worst source: #{rows[-1][1]} {100 * rows[-1][0]:.4f}%")
    return 0


def mode_set(a):
    data, ns = load()
    pool = pools(data, ns, a.cores)
    d, r = drf(data, ns, pool, a.src, a.mult)
    print(f"[cfg] mult={a.mult}  cores={a.cores}  sources {a.src}")
    print(f"  weighted dRF {100 * d:+.4f}%   walls raised on {r} cases")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rank", "set", "measure", "afford"])
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--src", type=int, nargs="*", default=[])
    ap.add_argument("--nmin", type=int, default=101)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--beam", action="store_true",
                    help="measure: also time the beam arm (2.3x slower)")
    ap.add_argument("--budget", type=float, default=0.0005,
                    help="afford: weighted dRF ceiling per twin source")
    ap.add_argument("--use-mult", action="store_true",
                    help="afford: price a HYPOTHETICAL --mult instead of the "
                         "measured beam dt (what a cheaper beam would buy)")
    a = ap.parse_args()
    return {"rank": mode_rank, "set": mode_set, "measure": mode_measure,
            "afford": mode_afford}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
