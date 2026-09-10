"""M65 probe (OFFLINE, never shipped): the two _L1_EXTRA cells missing from
the K24 tier of the L1 quality pool.

optimizer_constructive.py:377 snapshots _L1_BASE = list(_PROFILES) BEFORE the
extras are appended, so the REFINE_ITERS=24 duplicate tier covers only the 41
base profiles: the 84-cell L1 pool (41 base + 2 OS16xfree extras + 41 K24) has
no K24 version of either extra.  This probe measures those two missing cells
by env overlay (profile dict + ICCAD_REFINE_ITERS=24) against the L1 anchor —
shipped files and the submission shape are untouched.

modes (sequential, early-stop):
  gate0   (a) with ICCAD_L1_POOL=1 the pool must be 84 cells and must NOT
          contain either K24-extra combo (the gap, frozen as an assert);
          (b) anchor sanity: cost_eval(anchor positions) must reproduce the
          results_L1_final.json cost bit-exactly on 3 spot cases
  pilot   2 profiles x heavy cases 85..99, official cost vs the anchor;
          mover = cost < anchor - 1e-6; ZERO movers -> RED stop
  full    (only after pilot movers) remaining 85 cases; weighted oracle-min
          gain bar 0.05%; plus an 86-pool proxy re-selection check reusing the
          84-pool metrics cached in m53_l3_cache.pkl (sig-verified)
  l3seed  (only if full shows meat) LP the near-baseline runs (lp_pass,
          area_obj=True, accept iff feasible & cost strictly improves) and
          score the extra weighted gain over results_L3_port_top32_area.json;
          bar 0.10% (weak baseline = conservative toward RED; if it ever
          clears the bar, re-check against the honest min(port32, l2b)
          baseline via m64_flip_probe.py l2base before calling it GREEN)
  report  aggregate cache -> verdicts + results_M65_l1cell.json (no positions)
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

# L1 env BEFORE the wrapper import: m53_l3_cache.pkl's signature was computed
# on the 84-profile pool (mode_portfull sets the same two knobs pre-import).
os.environ["ICCAD_L1_POOL"] = "1"
os.environ["ICCAD_ADAPTIVE_POOL"] = "0"
import m53_l3_probe as m53  # noqa: E402  (loads the dataset once)
import optimizer_constructive as oc  # noqa: E402
from optimizer_claude import _serialize_input, _parse_output  # noqa: E402
from proxy_analysis import build_opt_target_pos  # noqa: E402

CASES, W, TOTW = m53.CASES, m53.W, m53.TOTW
cost_eval, lp_pass = m53.cost_eval, m53.lp_pass

EXE = str(_DIR / "constructive.exe")
RH = 1.4
MOVER_EPS = 1e-6
BAR_ORACLE = 0.05   # % of the L1 anchor (profile_vs_portfolio bar)
BAR_SEED = 0.10     # % of the L3 anchor
HEAVY = list(range(85, 100))

NAMES = ("os16_ab_pin_tight_K24", "os16_fc_pin_tight_K24")
PROF = [dict(p, ICCAD_REFINE_ITERS="24") for p in oc._L1_EXTRA]

ANCH_J = json.load(open(_DIR / "results_L1_final.json"))
ANCHOR_TOTAL = ANCH_J["total_score"]
ANCH = {t["test_id"]: t for t in ANCH_J["test_results"]}
L3_J = json.load(open(_DIR / "results_L3_port_top32_area.json"))
L3_TOTAL = L3_J["total_score"]
L3ANCH = {t["test_id"]: t for t in L3_J["test_results"]}

CACHEP = _DIR / "m65_cache.pkl"
SIG = repr((repr(PROF), hashlib.md5(open(EXE, "rb").read()).hexdigest()))
DB = {}
if CACHEP.exists():
    try:
        _c = pickle.load(open(CACHEP, "rb"))
        if _c.get("sig") == SIG:
            DB = _c["db"]
        else:
            print("[cache] signature mismatch -> reset")
    except Exception:
        print("[cache] unreadable -> reset")


def _save():
    tmp = CACHEP.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"sig": SIG, "db": DB}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHEP)


def _clean_env():
    """Parent env stripped of every ICCAD_* knob (regression_suite hygiene);
    the profile dict + the K24 overlay are the ONLY knobs the exe sees."""
    return {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}


def run_pool(case_ids):
    jobs = [(ci, j) for ci in case_ids for j in range(2)
            if ("run", ci, j) not in DB]
    if not jobs:
        return
    txts = {}
    for ci in {c for c, _ in jobs}:
        c = CASES[ci]
        otp = build_opt_target_pos(c["tp"], c["cons"], c["n"])
        txts[ci] = _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"],
                                    c["pins"], c["cons"], otp, gnn_hint=None)
    jobs.sort(key=lambda t: -CASES[t[0]]["n"])   # heaviest first

    def run_one(ci, j):
        env = _clean_env()
        env.update(PROF[j])
        try:
            r = subprocess.run([EXE], input=txts[ci], capture_output=True,
                               text=True, env=env, timeout=600)
            return _parse_output(r.stdout, CASES[ci]["n"])
        except Exception as e:
            print(f"  [run] case {ci} {NAMES[j]}: FAILED ({e!r})", flush=True)
            return None

    t0, done = time.perf_counter(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as ex:
        futs = {ex.submit(run_one, *jb): jb for jb in jobs}
        for fut in concurrent.futures.as_completed(futs):
            ci, j = futs[fut]
            DB[("run", ci, j)] = fut.result()
            done += 1
            if done % 10 == 0:
                _save()
                print(f"  [pool] {done}/{len(jobs)} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    _save()
    print(f"  [pool] done {len(jobs)} runs "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)


def cost_of(ci, j):
    """Official strict cost of profile j's run on case ci (cached)."""
    key = ("cost", ci, j)
    if key not in DB:
        P = DB.get(("run", ci, j))
        if P is None:
            DB[key] = None
        else:
            m = cost_eval(ci, [tuple(p) for p in P])
            DB[key] = (m.cost, bool(m.is_feasible), m.hpwl_gap, m.area_gap,
                       m.violations_relative, m.hpwl_total)
    return DB[key]


def pm_of(ci, j):
    """Deployed shapely proxy metrics of profile j's run (cached)."""
    key = ("pm", ci, j)
    if key not in DB:
        P = DB.get(("run", ci, j))
        if P is None:
            DB[key] = None
        else:
            c = CASES[ci]
            m = oc._proxy_metrics(P, c["at"], c["b2b"], c["p2b"], c["pins"],
                                  c["cons"], c["n"])
            DB[key] = (m["area"], m["hpwl"], m["vrel"])
    return DB[key]


# ── gate0 ────────────────────────────────────────────────────────────────────
def mode_gate0():
    ok = True
    # (a) the gap itself, frozen as an assert
    npool = len(oc._PROFILES)
    print(f"[gate0] L1 pool size = {npool} (expect 84): "
          f"{'OK' if npool == 84 else 'FAIL'}")
    ok &= npool == 84
    for j in range(2):
        base_in = oc._L1_EXTRA[j] in oc._PROFILES
        k24_in = PROF[j] in oc._PROFILES
        print(f"[gate0] {NAMES[j]}: base extra in pool = {base_in} "
              f"(expect True), K24 combo in pool = {k24_in} (expect False): "
              f"{'OK' if base_in and not k24_in else 'FAIL'}")
        ok &= base_in and not k24_in
    # (b) anchor loader/scorer chain reproduces the json cost bit-exactly
    for ci in (0, 85, 99):
        m = cost_eval(ci, [tuple(p) for p in ANCH[ci]["positions"]])
        match = m.cost == ANCH[ci]["cost"]
        print(f"[gate0] case {ci:3d}: eval {m.cost:.17g} vs json "
              f"{ANCH[ci]['cost']:.17g}: {'OK' if match else 'FAIL'}")
        ok &= match
    print(f"[gate0] {'ALL GREEN' if ok else 'FAILED'}")
    return ok


# ── pilot / full ─────────────────────────────────────────────────────────────
def show_cases(case_ids):
    movers = []
    print(f"\n{'case':>4} {'n':>4} {'anchor':>11} "
          f"{NAMES[0]:>24} {NAMES[1]:>24}")
    for ci in case_ids:
        a = ANCH[ci]["cost"]
        cells = []
        for j in range(2):
            cj = cost_of(ci, j)
            if cj is None:
                cells.append(f"{'RUN-FAILED':>24}")
                continue
            cost, feas = cj[0], cj[1]
            d = a - cost
            tag = ""
            if not feas:
                tag = " INFEAS"
            elif cost < a - MOVER_EPS:
                tag = " *MOVER*"
                movers.append((ci, j, d))
            cells.append(f"{cost:11.6f} ({d:+9.6f}){tag:>8}")
        print(f"{ci:4d} {CASES[ci]['n']:4d} {a:11.6f} "
              f"{cells[0]} {cells[1]}")
    _save()
    return movers


def mode_pilot():
    run_pool(HEAVY)
    movers = show_cases(HEAVY)
    g = sum(W[ci] * d for ci, _, d in movers) / TOTW / ANCHOR_TOTAL * 100
    print(f"\n== pilot: {len(movers)} movers (bar: official cost < anchor - "
          f"{MOVER_EPS:g}) on 15 heavy cases; mover weighted gain {g:+.4f}% ==")
    if not movers:
        print("== VERDICT: RED (pre-registered pilot gate: zero movers -> "
              "stop; do NOT run full/l3seed) ==")
    else:
        for ci, j, d in sorted(movers, key=lambda t: -W[t[0]] * t[2]):
            print(f"   mover case {ci} {NAMES[j]} d={d:+.6f} "
                  f"wContr={W[ci] * d / TOTW / ANCHOR_TOTAL * 100:+.4f}%")
        print("== pilot gate PASSED -> run `full` next ==")
    return movers


def mode_full():
    run_pool(range(100))
    movers = show_cases(range(100))
    orc = 0.0
    for ci in range(100):
        a = ANCH[ci]["cost"]
        best = min((cost_of(ci, j)[0] for j in range(2)
                    if cost_of(ci, j) is not None and cost_of(ci, j)[1]),
                   default=None)
        if best is not None:
            orc += W[ci] * max(0.0, a - best)
    orc = orc / TOTW / ANCHOR_TOTAL * 100
    print(f"\n== full: oracle-min weighted gain {orc:+.4f}% of anchor "
          f"{ANCHOR_TOTAL:.6f} (bar {BAR_ORACLE}%) ==")

    # 86-pool proxy re-selection (does the deployed proxy actually pick the
    # new cells?) — needs the 84-pool metrics from m53_l3_cache.pkl
    l3p = _DIR / "m53_l3_cache.pkl"
    sig84 = repr((repr(list(oc._PROFILES)),
                  hashlib.md5(open(EXE, "rb").read()).hexdigest()))
    l3db = None
    if l3p.exists():
        _c = pickle.load(open(l3p, "rb"))
        if _c.get("sig") == sig84:
            l3db = _c["db"]
        else:
            print("[realizable] m53_l3_cache sig mismatch -> SKIPPED "
                  "(oracle number above stands alone)")
    else:
        print("[realizable] m53_l3_cache.pkl missing -> SKIPPED")
    if l3db is not None:
        import math
        real, picked = 0.0, []
        for ci in range(100):
            c = CASES[ci]
            A_hat = 1.035 * max(sum(max(0.0, float(c["at"][i]))
                                    for i in range(c["n"])), 1e-9)
            pm = {k: l3db[("pm", ci, k)] for k in range(84)}
            for j in range(2):
                if pm_of(ci, j) is not None:
                    pm[84 + j] = pm_of(ci, j)
            hmin = min(v[1] for v in pm.values()) or 1.0
            prox = {k: (pm[k][0] / A_hat + RH * pm[k][1] / hmin)
                    * math.exp(2 * pm[k][2]) for k in pm}
            ksel = min(prox, key=lambda k: prox[k])
            if ksel >= 84:
                P = [tuple(p) for p in DB[("run", ci, ksel - 84)]]
            else:
                P = [tuple(p) for p in l3db[("run", ci, ksel)]]
            if P == [tuple(p) for p in ANCH[ci]["positions"]]:
                csel = ANCH[ci]["cost"]
            else:
                key = ("selcost", ci, ksel)
                if key not in DB:
                    DB[key] = cost_eval(ci, P).cost
                csel = DB[key]
                picked.append((ci, ksel, ANCH[ci]["cost"] - csel))
            real += W[ci] * (ANCH[ci]["cost"] - csel)
        _save()
        real = real / TOTW / ANCHOR_TOTAL * 100
        print(f"== full: 86-pool proxy re-selection weighted delta "
              f"{real:+.4f}% ({len(picked)} winner changes) ==")
        for ci, k, d in picked:
            nm = NAMES[k - 84] if k >= 84 else f"#{k}"
            print(f"   winner change case {ci}: -> {nm} d={d:+.6f}")
    print(f"== VERDICT(oracle): {'ABOVE BAR' if orc >= BAR_ORACLE else 'RED'}"
          f" ({orc:+.4f}% vs {BAR_ORACLE}%) ==")


# ── l3seed ───────────────────────────────────────────────────────────────────
def mode_l3seed(iters):
    """LP the near-baseline new-cell runs and score the extra gain over the
    L3 port-top32 anchor.  Candidate filter: raw cost < baseline*1.03 (LP has
    never recovered more than ~1%/case; 3% slack is generous)."""
    cand = []
    for ci in range(100):
        base = L3ANCH[ci]["cost"]
        for j in range(2):
            cj = cost_of(ci, j)
            if cj is not None and cj[1] and cj[0] < base * 1.03:
                cand.append((ci, j))
    print(f"[l3seed] {len(cand)} (case,profile) candidates "
          f"(raw cost < L3 baseline * 1.03)")
    t0 = time.perf_counter()
    for ci, j in cand:
        key = ("lp", ci, j, iters, True)
        if key in DB:
            continue
        P = [tuple(p) for p in DB[("run", ci, j)]]
        bestc = cost_of(ci, j)[0]
        for _ in range(iters):
            newP, tele = lp_pass(ci, P, area_obj=True)
            if newP is None:
                break
            m = cost_eval(ci, newP)
            if m.is_feasible and m.cost < bestc - 1e-12:
                bestc, P = m.cost, newP
            else:
                break
        DB[key] = bestc
        _save()
        print(f"  [lp] case {ci} {NAMES[j]}: {cost_of(ci, j)[0]:.6f} -> "
              f"{bestc:.6f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    extra, movers = 0.0, []
    for ci in range(100):
        base = L3ANCH[ci]["cost"]
        best = min((DB[("lp", ci, j, iters, True)] for j in range(2)
                    if ("lp", ci, j, iters, True) in DB), default=None)
        if best is not None and best < base - MOVER_EPS:
            movers.append((ci, base - best))
            extra += W[ci] * (base - best)
    extra = extra / TOTW / L3_TOTAL * 100
    print(f"\n== l3seed: extra weighted gain {extra:+.4f}% of L3 anchor "
          f"{L3_TOTAL:.6f} (bar {BAR_SEED}%), {len(movers)} movers ==")
    for ci, d in sorted(movers, key=lambda t: -W[t[0]] * t[1]):
        print(f"   seed mover case {ci} d={d:+.6f} "
              f"wContr={W[ci] * d / TOTW / L3_TOTAL * 100:+.4f}%")
    if extra >= BAR_SEED:
        print("== above bar vs the WEAK baseline: re-check against the honest "
              "min(port32, l2b) baseline (m64_flip_probe.py l2base) before "
              "calling this GREEN ==")
    else:
        print("== VERDICT(l3seed): RED (weak baseline already below bar; the "
              "honest baseline can only be lower) ==")


# ── report ───────────────────────────────────────────────────────────────────
def mode_report():
    ran = sorted({k[1] for k in DB if k[0] == "run"})
    print(f"[report] cases run: {len(ran)}")
    trs = []
    for ci in ran:
        row = dict(test_id=ci, block_count=CASES[ci]["n"],
                   anchor_cost=ANCH[ci]["cost"],
                   l3_anchor_cost=L3ANCH[ci]["cost"])
        for j in range(2):
            cj = cost_of(ci, j)
            row[NAMES[j]] = None if cj is None else dict(
                cost=cj[0], feasible=cj[1], hpwl_gap=cj[2], area_gap=cj[3],
                vrel=cj[4])
            lp = [v for k, v in DB.items()
                  if k[0] == "lp" and k[1] == ci and k[2] == j]
            if lp:
                row[NAMES[j]]["lp_cost"] = min(lp)
        trs.append(row)
    movers = [(ci, j) for ci in ran for j in range(2)
              if cost_of(ci, j) is not None and cost_of(ci, j)[1]
              and cost_of(ci, j)[0] < ANCH[ci]["cost"] - MOVER_EPS]
    out = dict(probe="M65", sig=SIG, names=list(NAMES),
               anchor_total=ANCHOR_TOTAL, l3_anchor_total=L3_TOTAL,
               mover_eps=MOVER_EPS, movers=movers, cases=trs)
    dump = _DIR / "results_M65_l1cell.json"
    json.dump(out, open(dump, "w"), indent=1)
    print(f"[dump] {dump}  ({len(movers)} movers vs L1 anchor)")
    _save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode",
                    choices=["gate0", "pilot", "full", "l3seed", "report"])
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "gate0":
        sys.exit(0 if mode_gate0() else 1)
    elif args.mode == "pilot":
        mode_pilot()
    elif args.mode == "full":
        mode_full()
    elif args.mode == "l3seed":
        mode_l3seed(args.iters)
    else:
        mode_report()
