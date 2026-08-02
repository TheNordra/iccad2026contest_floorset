"""M82 wall levers on the TRUE shipped configuration (OFFLINE, never shipped).

Two questions, one data source (audit_cache_m71_kband.pkl = M71 overlay + the
shipped K=8/K=4 band overlay = what the grader actually runs).  Pure cache
arithmetic: no C++ run, no new wall-clock produced.

G1  tier5   -- is tier-5 still FREE under M71?
    tier-5 puts M42's 22 profiles back when the box reports >=40 cores, i.e. it
    FIRES ON THE GRADER and not on this 32-core box.  The "free" argument
    (M67-E: those 22 all have dt <= the max-setter, so dW=+0.00%) was measured
    with the M71 knobs OFF.  M71 makes every profile do more work, so the
    argument may no longer hold -- and if it does not, we are paying wall on the
    grader right now without knowing.  Pre-registered bar (M67-F Phase 1,
    reused verbatim, NOT re-tuned): per-case wall increase > 2% => the
    free-restore argument is bankrupt.

G2  parallel -- upper bound on "spread one case's work over the idle cores".
    At 48 cores a case runs ~35 profiles on 35 cores => 13 idle, and the wall is
    the SLOWEST profile (everyone else finished and is waiting).  Inside that
    profile, constructive.cpp:1756-1829 tries max_trials frames SEQUENTIALLY,
    and the frames are independent (items reset to items_base each round, the
    REFINE guide chain is loop-local, the only cross-frame state is a min-
    reduction).  If they ran concurrently the critical path per profile would
    drop ~K-fold with BIT-IDENTICAL output.

    Makespan model (list-scheduling lower bound, so an OPTIMISTIC bound):
        wall = max( max_i task_i , sum_i dt_i / cores , sum_i pt_i )
    now:      task_i = dt_i
    route A:  task_i = dt_i*(1-f) + dt_i*f/K_i      (f = fraction of a profile's
              time actually inside the frame loop; swept, f=1 is the optimistic
              extreme).  K_i = 4 if n>=60 else 5 (constructive.cpp:1740).
    sum_i pt_i (the wrapper's SERIAL proxy chain) is NOT parallelised by this
    and stays as a floor.

Modes: gate0 | tier5 | parallel | report
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
if "--m71" not in sys.argv:
    sys.argv += ["--m71", "--kband"]              # this probe is ONLY meaningful there

import m67e_rf48 as e                             # noqa: E402

ALPHA_JSON = Path(r"C:\Users\.01\Downloads\cadc1075_results.json")
M80_JSON = _DIR / "results_M80_rf_remeasure.json"
OUT = _DIR / "results_M82_wall_levers.json"
KAPPA, GAMMA, FLOOR = 3.161, 0.3, 0.7
SPEEDS = (1.0, 1.5, 2.0, 2.5, 3.0)
TIER5_BAR = 0.02                                  # M67-F pre-registered, reused

CIS = sorted({k[0] for k in e.DATA})
NOF = {c: e.CASES[c]["n"] for c in CIS}
W = {c: math.exp(NOF[c] / 12.0) for c in CIS}
TOTW = sum(W.values())
KTRIALS = {c: (4 if NOF[c] >= 60 else 5) for c in CIS}   # constructive.cpp:1740


def _alpha_M():
    a = {t["test_id"]: t for t in
         json.loads(ALPHA_JSON.read_text(encoding="utf-8"))["test_results"]}
    return {c: KAPPA * a[c]["runtime_seconds"] for c in CIS}


def _quality(poolfn):
    return {c: e.cost(c, e.select(c, poolfn(c))) for c in CIS}


def score(Q, T, s, M):
    """Official-shaped total: sum w*Q*max(FLOOR,(s*t/M)^GAMMA) / sum w."""
    return sum(W[c] * Q[c] * max(FLOOR, (s * T[c] / M[c]) ** GAMMA)
               for c in CIS) / TOTW


def wall_now(c, cores=48):
    return e.wall(c, e.pool_shipped(c, cores), cores)


# ── gate0: does the wall MODEL agree with what M80 actually measured? ────────
def mode_gate0():
    assert M80_JSON.exists(), "run m80_rf_remeasure.py first"
    meas = json.loads(M80_JSON.read_text(encoding="utf-8"))["per_case_min"]
    # M80 ran on THIS 32-core box => tier-5 OFF => model the same pool
    rat = []
    for c in CIS:
        m = e.wall(c, e.pool_shipped(c, 32), 32)
        rat.append(m / meas[str(c)])
    rat.sort()
    p50, p10, p90 = rat[len(rat) // 2], rat[len(rat) // 10], rat[9 * len(rat) // 10]
    print(f"wall model vs M80 measured (same 32-core pool), model/measured:")
    print(f"  p10 {p10:.2f}x   p50 {p50:.2f}x   p90 {p90:.2f}x")
    print("  (the model is CPU-time based and omits wrapper/serialisation "
          "overhead,\n   so a constant offset is expected; what matters is that "
          "it is CONSISTENT --\n   this probe only ever uses model RATIOS, "
          "never its absolute value.)")
    spread = p90 / p10
    print(f"  spread p90/p10 = {spread:.2f}x  "
          f"[{'OK' if spread < 3 else 'WARN: model not consistent'}]")
    return {"p10": p10, "p50": p50, "p90": p90}


# ── G1 ──────────────────────────────────────────────────────────────────────
def mode_tier5():
    M = _alpha_M()
    Q_on = _quality(lambda c: e.pool_shipped(c, 48))     # tier-5 ON  (grader)
    Q_off = _quality(lambda c: e.pool_shipped(c, 32))    # tier-5 OFF
    rows, worst = [], 0.0
    for c in CIS:
        p_on, p_off = e.pool_shipped(c, 48), e.pool_shipped(c, 32)
        if set(p_on) == set(p_off):
            continue                                     # tier-5 does nothing here
        w_on, w_off = e.wall(c, p_on, 48), e.wall(c, p_off, 48)
        r = w_on / w_off
        worst = max(worst, r)
        rows.append((c, NOF[c], len(p_off), len(p_on), w_off, w_on, r))
    rows.sort(key=lambda r: -r[6])
    print(f"tier-5 touches {len(rows)} cases (n>100).  Both walls at 48 cores; "
          f"only the POOL differs.\n")
    print(f"{'case':>5} {'n':>4} {'|P|off':>7} {'|P|on':>6} {'wall off':>9} "
          f"{'wall on':>8} {'ratio':>7}")
    for c, n, a, b, wo, wn, r in rows:
        flag = "  <-- OVER 2% BAR" if r > 1 + TIER5_BAR else ""
        print(f"{c:>5} {n:>4} {a:>7} {b:>6} {wo:>9.3f} {wn:>8.3f} "
              f"{r:>7.3f}{flag}")
    over = [r for r in rows if r[6] > 1 + TIER5_BAR]
    print(f"\nworst wall ratio {worst:.3f}  ({len(over)}/{len(rows)} cases over "
          f"the pre-registered {100*TIER5_BAR:.0f}% bar)")
    print(f"VERDICT: tier-5 free-restore argument "
          f"{'BANKRUPT' if over else 'HOLDS'} under M71")

    # net: tier-5 also BUYS quality; price it
    print(f"\nnet value of tier-5 (quality gain vs wall cost):")
    print(f"{'s':>5} {'score tier5 OFF':>16} {'score tier5 ON':>15} {'delta':>9}")
    net = {}
    T_on = {c: wall_now(c, 48) for c in CIS}
    T_off = {c: e.wall(c, e.pool_shipped(c, 32), 48) for c in CIS}
    for s in SPEEDS:
        a, b = score(Q_off, T_off, s, M), score(Q_on, T_on, s, M)
        print(f"{s:>5g} {a:>16.6f} {b:>15.6f} {100*(b-a)/a:>+8.3f}%")
        net[s] = 100 * (b - a) / a
    print("  (negative = tier-5 is WINNING; it is SHIPPED and fires on the grader)")
    return {"worst_ratio": worst, "over_bar": len(over), "net_pct": net}


# ── G2 ──────────────────────────────────────────────────────────────────────
def mode_parallel():
    M = _alpha_M()
    Q = _quality(lambda c: e.pool_shipped(c, 48))        # quality does NOT change
    T_now = {c: wall_now(c, 48) for c in CIS}
    print("Route A = run a profile's max_trials frames concurrently.")
    print("Quality is UNCHANGED by construction (same frames, same arbitration),")
    print("so this is a pure wall lever.  f = fraction of a profile's time that "
          "is\nactually inside the frame loop (f=1 is the optimistic extreme).\n")
    print(f"{'f':>5} {'wall now':>10} {'wall A':>9} {'ratio':>7}   "
          + "  ".join(f"s={s:g}" for s in SPEEDS))
    out = {}
    for f in (1.0, 0.8, 0.6, 0.4):
        T_a = {}
        for c in CIS:
            pool = e.pool_shipped(c, 48)
            dts = [e.DATA[(c, k)][1] for k in pool]
            pts = [e.PT(c, k) for k in pool]
            K = KTRIALS[c]
            tasks = [d * (1 - f) + d * f / K for d in dts]
            T_a[c] = max(max(tasks), sum(dts) / 48, sum(pts))
        wn = sum(W[c] * T_now[c] for c in CIS) / TOTW
        wa = sum(W[c] * T_a[c] for c in CIS) / TOTW
        deltas = []
        for s in SPEEDS:
            base = score(Q, T_now, s, M)
            deltas.append(100 * (score(Q, T_a, s, M) - base) / base)
        print(f"{f:>5g} {wn:>10.3f} {wa:>9.3f} {wa/wn:>7.3f}   "
              + "  ".join(f"{d:>+6.2f}%" for d in deltas))
        out[f] = {"wall_ratio": wa / wn,
                  "score_delta_pct": dict(zip(map(str, SPEEDS), deltas))}
    print("\n(negative = better.  bar for opening engineering work: the project's")
    print(" new-candidate bar is 0.05%; anything here is orders above that IF the")
    print(" bit-identity audit passes -- that is codex Q1.)")
    print("\nBINDING TERM per case at f=1 (what stops it going further):")
    lim = {"max-setter": 0, "sum/48": 0, "proxy chain": 0}
    for c in CIS:
        pool = e.pool_shipped(c, 48)
        dts = [e.DATA[(c, k)][1] for k in pool]
        pts = [e.PT(c, k) for k in pool]
        K = KTRIALS[c]
        v = [max(d / K for d in dts), sum(dts) / 48, sum(pts)]
        lim[["max-setter", "sum/48", "proxy chain"][v.index(max(v))]] += 1
    print(f"  {lim}")
    return out


def mode_report():
    g0 = mode_gate0()
    print("\n" + "=" * 78 + "\nG1  tier-5\n" + "=" * 78)
    t5 = mode_tier5()
    print("\n" + "=" * 78 + "\nG2  route A upper bound\n" + "=" * 78)
    pa = mode_parallel()
    e.csave()
    json.dump({"gate0": g0, "tier5": t5, "parallel": pa}, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate0", "tier5", "parallel", "report"])
    ap.add_argument("--m71", action="store_true")
    ap.add_argument("--kband", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    e.ensure_pm()
    {"gate0": mode_gate0, "tier5": mode_tier5,
     "parallel": mode_parallel, "report": mode_report}[a.mode]()
    e.csave()
    print(f"\n[{a.mode} {time.perf_counter() - t0:.1f}s]")
