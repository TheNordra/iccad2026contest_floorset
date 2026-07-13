"""M54 LP-aware RF model (OFFLINE, never shipped).

M53 L3's winner-only LP is +0.87% quality on the shipped anchor, but a
deployed LP runs INSIDE the scored window, so the official RF term
max(0.7,(t/M)^0.3) taxes it. This tool answers: which cases (if any) are a
NET win with LP in the window, over the unknown median M x cores x solver
speed, and what gate (band / time budget / iters) to ship.

dep mode — deployable-shaped LP re-run, all 100 cases:
  Start from the shipped winner positions (results_shipped_m51.json) and run
  the m53_l3_probe LP chain, but the keep/revert decision uses ONLY what the
  wrapper has at test time (no official eval, no baseline):
    keep iff  proxy hpwl or bbox area strictly improves
          and neither hpwl / area / vrel worsens (oc._proxy_metrics semantics)
          and hard checks pass (pairwise overlap <=1e-6 both axes mirror of
              check_overlap; dims + preplaced frozen vs the anchor)
  Per pass it records t_build / t_solve(linprog) / t_guard and the OFFLINE
  official cost (reporting only). Dumps results_M54_dep_lp.json (iters=2
  chain) + m54_cache.pkl, and prints the proxy-guard vs official-guard
  (results_L3_lp_shipped_m51.json) equivalence table.

model mode — RF projection (needs dep cache):
  t_i(12c) = measured runtime_seconds from the shipped eval json; other cores
  scale by the audit-cache wall-model ratio s_i(cores) = wall(pool(ci,cores),
  cores)/wall(pool(ci,12),12) (mirrors rf_score_model's pools incl. M45
  tiers; audit dt is the K=12 counterfactual, used as a RATIO only).
  LP time per pass = t_build + beta*t_solve + t_guard, beta = vendored-solver
  slowdown sweep {1,2,4,8}. Gates: band (n>lo) x time budget cap (abort
  semantics: cap seconds wasted, no quality; skip = oracle predictor bound)
  x iters {1,2}. Projects total(M, cores, gate, beta), prints the delta
  matrix vs no-LP, the strict weak-win search and a per-case table.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

import m53_l3_probe as l3                      # noqa: E402 (loads dataset)
import optimizer_constructive as oc            # noqa: E402

CACHE = _DIR / "m54_cache.pkl"
ANCHOR_JSON = _DIR / "results_shipped_m51.json"
OFFICIAL_GUARD_JSON = _DIR / "results_L3_lp_shipped_m51.json"
EPS_OVL = 1e-6

# ── timing shim: split linprog time out of lp_pass ───────────────────────────
_SOLVE_T = [0.0]
_orig_linprog = l3.linprog


def _timed_linprog(*a, **k):
    t0 = time.perf_counter()
    r = _orig_linprog(*a, **k)
    _SOLVE_T[0] += time.perf_counter() - t0
    return r


l3.linprog = _timed_linprog
# lp_pass captured build_and_solve at def time via module globals -> patching
# l3.linprog is enough (build_and_solve reads it from the module namespace).


def hard_ok(P0, P, ci):
    """Mirror of the evaluator's hard checks the LP could conceivably touch:
    overlap (>1e-6 penetration on BOTH axes = violation, touching OK), dims
    frozen, preplaced (x,y) frozen. Areas are untouched because dims are."""
    c = l3.CASES[ci]
    n, cn = c["n"], c["cn"]
    for i in range(n):
        if P[i][2] != P0[i][2] or P[i][3] != P0[i][3]:
            return False
        if cn[i][1] != 0 and (P[i][0] != P0[i][0] or P[i][1] != P0[i][1]):
            return False
    for i in range(n):
        x1, y1, w1, h1 = P[i]
        for j in range(i + 1, n):
            x2, y2, w2, h2 = P[j]
            ox = min(x1 + w1, x2 + w2) - max(x1, x2)
            if ox > EPS_OVL:
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                if oy > EPS_OVL:
                    return False
    return True


def proxy_m(ci, P):
    c = l3.CASES[ci]
    return oc._proxy_metrics(P, c["at"], c["b2b"], c["p2b"], c["pins"],
                             c["cons"], c["n"])


def dep_case(ci, anch, iters):
    """Deployable-shaped chain for one case. Decision path = proxy only;
    official cost recorded per pass for offline reporting."""
    P0 = [tuple(p) for p in anch[ci]["positions"]]
    q_ship = anch[ci]["cost"]
    m_prev = proxy_m(ci, P0)
    P, passes = P0, []
    for _ in range(iters):
        _SOLVE_T[0] = 0.0
        t0 = time.perf_counter()
        newP, tele = l3.lp_pass(ci, P, area_obj=True)
        t_pass = time.perf_counter() - t0
        t_solve = _SOLVE_T[0]
        rec = dict(t_build=t_pass - t_solve, t_solve=t_solve, t_guard=0.0,
                   kept=False, status=tele.get("status", "?"), cost_after=None)
        if newP is None:
            passes.append(rec)
            break
        tg0 = time.perf_counter()
        m_new = proxy_m(ci, newP)
        better = (m_new["hpwl"] < m_prev["hpwl"] * (1 - 1e-12)
                  or m_new["area"] < m_prev["area"] * (1 - 1e-12))
        worse = (m_new["hpwl"] > m_prev["hpwl"] * (1 + 1e-12)
                 or m_new["area"] > m_prev["area"] * (1 + 1e-12)
                 or m_new["vrel"] > m_prev["vrel"] + 1e-12)
        keep = better and not worse and hard_ok(P0, newP, ci)
        rec["t_guard"] = time.perf_counter() - tg0
        rec["kept"] = keep
        # offline reporting only (never in the decision path)
        rec["cost_after"] = l3.cost_eval(ci, newP).cost
        passes.append(rec)
        if not keep:
            break
        P, m_prev = newP, m_new
    m_fin = l3.cost_eval(ci, P)
    q1 = passes[0]["cost_after"] if passes and passes[0]["kept"] else q_ship
    return dict(ci=ci, n=l3.CASES[ci]["n"], q_ship=q_ship,
                t12=anch[ci]["runtime_seconds"], passes=passes,
                q_dep1=q1, q_dep=m_fin.cost, feas=bool(m_fin.is_feasible),
                P=[list(p) for p in P])


def mode_dep(iters, case_ids):
    anch = {t["test_id"]: t for t in
            json.load(open(ANCHOR_JSON))["test_results"]}
    sig = repr((ANCHOR_JSON.name, iters, "area", 2))
    db = {}
    if CACHE.exists():
        try:
            c = pickle.load(open(CACHE, "rb"))
            if c.get("sig") == sig:
                db = c["db"]
        except Exception:
            pass

    def save():
        tmp = CACHE.with_suffix(".tmp")
        pickle.dump({"sig": sig, "db": db}, open(tmp, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, CACHE)

    t0 = time.perf_counter()
    for ci in case_ids:
        if ci in db:
            continue
        db[ci] = dep_case(ci, anch, iters)
        r = db[ci]
        tlp = sum(p["t_build"] + p["t_solve"] + p["t_guard"]
                  for p in r["passes"])
        print(f"case {ci:3d} n={r['n']:3d} {r['q_ship']:.6f}->{r['q_dep']:.6f}"
              f" tLP={tlp:5.2f}s passes={len(r['passes'])}"
              f" kept={sum(p['kept'] for p in r['passes'])}"
              f" feas={int(r['feas'])} ({time.perf_counter() - t0:.0f}s)",
              flush=True)
        if len(db) % 10 == 0:
            save()
    save()
    if len(db) < 100:
        print(f"[dep] partial ({len(db)}/100 cases) - no dump/summary")
        return

    # summary + equivalence vs the official-guard chain
    W, TOTW = l3.W, l3.TOTW
    tot_s = sum(W[ci] * db[ci]["q_ship"] for ci in db) / TOTW
    tot_d = sum(W[ci] * db[ci]["q_dep"] for ci in db) / TOTW
    tot_1 = sum(W[ci] * db[ci]["q_dep1"] for ci in db) / TOTW
    nfeas = sum(db[ci]["feas"] for ci in db)
    nreg = sum(1 for ci in db if db[ci]["q_dep"] > db[ci]["q_ship"] + 1e-12)
    print(f"\ndep: ship {tot_s:.10f} -> dep(iters=1) {tot_1:.10f} "
          f"({(tot_s - tot_1) / tot_s * 100:+.4f}%) -> dep(iters={iters}) "
          f"{tot_d:.10f} ({(tot_s - tot_d) / tot_s * 100:+.4f}%)  "
          f"feas {nfeas}/100  regressions {nreg}")
    tl = sorted(sum(p["t_build"] + p["t_solve"] + p["t_guard"]
                    for p in db[ci]["passes"]) for ci in db)
    print(f"tLP quartiles: p25 {tl[25]:.2f}s p50 {tl[50]:.2f}s "
          f"p75 {tl[75]:.2f}s p90 {tl[90]:.2f}s max {tl[-1]:.2f}s "
          f"sum {sum(tl):.1f}s")
    if OFFICIAL_GUARD_JSON.exists():
        og = {t["test_id"]: t["cost"] for t in
              json.load(open(OFFICIAL_GUARD_JSON))["test_results"]}
        tot_o = sum(W[ci] * og[ci] for ci in og) / TOTW
        print(f"\nequivalence vs official-guard chain "
              f"({OFFICIAL_GUARD_JSON.name}, no-area, iters=2):")
        print(f"  official-guard total {tot_o:.10f}   dep total {tot_d:.10f} "
              f"  (dep-og {(tot_d - tot_o) / tot_o * 100:+.4f}%)")
        divs = sorted(db, key=lambda ci: -abs(db[ci]["q_dep"] - og[ci]))[:8]
        for ci in divs:
            d = db[ci]["q_dep"] - og[ci]
            if abs(d) < 1e-9:
                break
            print(f"  case {ci:3d} dep {db[ci]['q_dep']:.6f} vs og "
                  f"{og[ci]:.6f} (d={d:+.6f})")

    trs = []
    for ci in range(100):
        r = db[ci]
        m = l3.cost_eval(ci, [tuple(p) for p in r["P"]])
        trs.append(dict(test_id=ci, block_count=r["n"],
                        is_feasible=bool(m.is_feasible), hpwl_gap=m.hpwl_gap,
                        area_gap=m.area_gap,
                        violations_relative=m.violations_relative,
                        runtime_seconds=0.0, cost=m.cost, positions=r["P"],
                        error=None))
    out = dict(submission_name="M54_dep_lp",
               timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
               total_score=tot_d, test_results=trs,
               summary=dict(num_tests=100, num_feasible=nfeas,
                            avg_cost=sum(t["cost"] for t in trs) / 100,
                            avg_runtime=0.0))
    dump = _DIR / "results_M54_dep_lp.json"
    json.dump(out, open(dump, "w"))
    print(f"[dump] {dump}")


# ── model mode ────────────────────────────────────────────────────────────────
GAMMA, FLOOR = 0.3, 0.7
MS = (6, 8, 11, 14, 20)
CS = (4, 8, 12, 16)
BETAS = (1.0, 2.0, 4.0, 8.0)
CAPS = (None, 4.0, 2.0, 1.0, 0.5)
BAND_LOS = (0, 60, 100, 110)                    # gate: LP only when n > lo

OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}


def load_walls():
    """audit-cache wall model, mirroring the deployed pool at each cores."""
    profiles = list(oc._PROFILES) + [OM16]
    c = pickle.load(open(_DIR / "audit_cache.pkl", "rb"))
    assert c.get("profiles") == repr(profiles), \
        "audit_cache signature != current pool -> re-run profile_audit.py"
    data = c["data"]
    n_live = len(oc._PROFILES)
    swap = {k for k in range(n_live)
            if "ICCAD_ORDER_SWAP" in profiles[k]
            or "ICCAD_ORDER_MOVE" in profiles[k]}
    big = set(oc._BIG_REDUNDANT_IDX)

    def pool(ci, cores):
        n = l3.CASES[ci]["n"]
        p = [k for k in range(n_live)
             if k not in swap and not (n > 100 and k in big)]
        for lo, hi, D in oc._M45_BAND_DROP:
            if lo < n <= hi:
                p = [k for k in p if k not in D]
        if cores <= oc._M45_CORES_MAX:
            for lo, hi, D in oc._M45_LOWCORE_DROP:
                if lo < n <= hi:
                    p = [k for k in p if k not in D]
        return p

    def wall(ci, cores):
        ts = [data[(ci, k)][1] for k in pool(ci, cores)]
        return max(max(ts), sum(ts) / cores)

    return wall


def mode_model():
    c = pickle.load(open(CACHE, "rb"))
    db = c["db"]
    assert len(db) == 100, f"dep cache incomplete ({len(db)}/100)"
    W, TOTW = l3.W, l3.TOTW
    anch_total = json.load(open(ANCHOR_JSON))["total_score"]
    tot_s = sum(W[ci] * db[ci]["q_ship"] for ci in db) / TOTW
    assert abs(tot_s - anch_total) < 1e-9, "anchor total mismatch"

    wall = load_walls()
    T = {ci: {cs: db[ci]["t12"] * wall(ci, cs) / wall(ci, 12) for cs in CS}
         for ci in db}

    def lp_time_q(ci, beta, iters, cap, band_lo, skip=False):
        """(added seconds, resulting quality) under the gate."""
        r = db[ci]
        if r["n"] <= band_lo:
            return 0.0, r["q_ship"]
        chain = r["passes"][:iters]
        if skip and cap is not None:
            tot = sum(p["t_build"] + beta * p["t_solve"] + p["t_guard"]
                      for p in chain)
            if tot > cap:
                return 0.0, r["q_ship"]
        t, q = 0.0, r["q_ship"]
        for p in chain:
            tp = p["t_build"] + beta * p["t_solve"] + p["t_guard"]
            if cap is not None and t + tp > cap:
                return cap, q                     # abort: budget burned
            t += tp
            if p["kept"]:
                q = p["cost_after"]
            else:
                break
        return t, q

    def total(M, cores, beta, iters, cap, band_lo, skip=False, lp_on=True):
        s = 0.0
        for ci in db:
            if lp_on:
                dt, q = lp_time_q(ci, beta, iters, cap, band_lo, skip)
            else:
                dt, q = 0.0, db[ci]["q_ship"]
            rf = max(FLOOR, ((T[ci][cores] + dt) / M) ** GAMMA)
            s += W[ci] * q * rf
        return s / TOTW

    print(f"anchor total (RF=1.0) {tot_s:.10f}; dep(iters=2) quality total "
          f"{sum(W[ci] * db[ci]['q_dep'] for ci in db) / TOTW:.10f}")
    print(f"cores-scaling s_i sanity: s(12)=1 by construction; "
          f"mean s(4)={sum(T[ci][4] / db[ci]['t12'] for ci in db) / 100:.2f} "
          f"mean s(8)={sum(T[ci][8] / db[ci]['t12'] for ci in db) / 100:.2f}")

    # main matrix: delta% vs no-LP, beta=1, iters=2, gates = band x cap(abort)
    gates = [("all", 0, None), ("all cap4", 0, 4.0), ("all cap2", 0, 2.0),
             ("all cap1", 0, 1.0), ("n>60", 60, None), ("n>60 cap2", 60, 2.0),
             ("n>100", 100, None), ("n>110", 110, None)]
    for beta in BETAS:
        print(f"\n== delta%% vs no-LP  (beta={beta:g}, iters=2, abort caps) ==")
        print(f"{'cores':>5} {'M':>3} " + "".join(f"{g[0]:>12}" for g in gates))
        for cores in CS:
            for M in MS:
                base = total(M, cores, beta, 2, None, 0, lp_on=False)
                row = f"{cores:>5} {M:>3} "
                for _, lo, cap in gates:
                    v = total(M, cores, beta, 2, cap, lo)
                    row += f"{100 * (v - base) / base:>+11.3f}%"
                print(row)

    # strict weak-win search over (band, cap, iters, abort/skip) per beta
    print("\n== gate search: worst / best cell over M x cores ==")
    print(f"{'gate':<28} " + "".join(f"{'b=' + str(int(b)):>18}" for b in BETAS))
    combos = [(lo, cap, it, sk) for lo in BAND_LOS for cap in CAPS
              for it in (1, 2) for sk in ((False, True) if cap else (False,))]
    results = []
    for lo, cap, it, sk in combos:
        tag = (f"n>{lo} cap={cap if cap else '-'} it{it}"
               f"{' skip' if sk else ''}")
        row, cells = f"{tag:<28} ", []
        for beta in BETAS:
            ds = []
            for cores in CS:
                for M in MS:
                    base = total(M, cores, beta, 2, None, 0, lp_on=False)
                    v = total(M, cores, beta, it, cap, lo, sk)
                    ds.append(100 * (v - base) / base)
            row += f" {max(ds):>+7.3f}/{min(ds):>+7.3f}"
            cells.append((max(ds), min(ds)))
        results.append((tag, cells))
        print(row)
    for bi, beta in enumerate(BETAS):
        ok = [(t, c[bi]) for t, c in results if c[bi][0] <= 1e-9]
        if ok:
            best = min(ok, key=lambda x: x[1][1])
            print(f"beta={beta:g}: strict weak-win gates: {len(ok)}; "
                  f"best {best[0]}  worst {best[1][0]:+.3f}% "
                  f"best-cell {best[1][1]:+.3f}%")
        else:
            near = min(results, key=lambda x: x[1][bi][0])
            print(f"beta={beta:g}: NO strict weak-win; closest "
                  f"{near[0]} worst {near[1][bi][0]:+.3f}% "
                  f"best {near[1][bi][1]:+.3f}%")

    # per-case oracle reference + per-case table at a representative cell
    for (M, cores, beta) in ((8, 12, 1.0), (11, 12, 1.0), (8, 12, 4.0)):
        base = total(M, cores, beta, 2, None, 0, lp_on=False)
        s = 0.0
        for ci in db:
            dt, q = lp_time_q(ci, beta, 2, None, 0)
            with_lp = q * max(FLOOR, ((T[ci][cores] + dt) / M) ** GAMMA)
            no_lp = db[ci]["q_ship"] * max(FLOOR, (T[ci][cores] / M) ** GAMMA)
            s += W[ci] * min(with_lp, no_lp)
        s /= TOTW
        print(f"oracle per-case gate @M={M} cores={cores} beta={beta:g}: "
              f"{100 * (s - base) / base:+.3f}% vs no-LP")

    M, cores, beta = 8, 12, 1.0
    print(f"\nper-case net @M={M} cores={cores} beta={beta:g} gate=all it2 "
          f"(top 14 by |weighted delta|):")
    rows = []
    for ci in db:
        dt, q = lp_time_q(ci, beta, 2, None, 0)
        rf0 = max(FLOOR, (T[ci][cores] / M) ** GAMMA)
        rf1 = max(FLOOR, ((T[ci][cores] + dt) / M) ** GAMMA)
        d = W[ci] * (q * rf1 - db[ci]["q_ship"] * rf0) / TOTW
        rows.append((ci, db[ci]["n"], db[ci]["q_ship"], q, T[ci][cores], dt,
                     rf0, rf1, d))
    rows.sort(key=lambda r: -abs(r[-1]))
    print(f"{'ci':>3} {'n':>4} {'Qship':>7} {'Qlp':>7} {'t':>5} {'tLP':>5} "
          f"{'RF0':>5} {'RF1':>5} {'wDelta%':>9}")
    for ci, n, q0, q1, t, dt, rf0, rf1, d in rows[:14]:
        print(f"{ci:>3} {n:>4} {q0:>7.4f} {q1:>7.4f} {t:>5.2f} {dt:>5.2f} "
              f"{rf0:>5.3f} {rf1:>5.3f} {100 * d / tot_s:>+8.4f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dep", "model", "all"])
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--cases", default=None)
    args = ap.parse_args()
    ids = list(range(100))
    if args.cases:
        a, b = args.cases.split("-")
        ids = list(range(int(a), int(b) + 1))
    if args.mode in ("dep", "all"):
        mode_dep(args.iters, ids)
    if args.mode in ("model", "all"):
        mode_model()
