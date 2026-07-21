#!/usr/bin/env python3
"""OFFLINE ONLY - M67-E: 48-core RF projection anchored on the ALPHA results.

Beta runs on a dedicated 48-core ICELAKE box, but the whole RF axis (M41-M50)
was derived on a 12-core machine against an UNKNOWN median M swept in seconds.
Two new anchors change that:

  1. alpha gave PER-CASE grader runtimes (Downloads/cadc1075_results.json,
     `runtime_seconds`) plus RF-less raw 1.45278763 and official 1.0286
     => cost-weighted RF = 0.70802 (floor 0.70) => median calibration.
  2. M67-D measured the OOS quality tax of the adaptive cuts: +2.825% on the
     heavy band (in-set only +0.106%), and handed over: the 48c projection MUST
     use the OOS number.

CALIBRATION CORRECTION (the reason this session exists). M67-D assumed the
alpha runtime was OUR CURRENT shipped runtime, giving "median ~3.28x t_shipped,
8.2x safety margin". But alpha submitted M10 (commit 8565e38) whose pool was 14
CHEAP profiles (base/aspect/frame; no FREE/FC/OS16 stacks): its measured grader
runtimes (p50 0.673s, heavy cases 2.4-4.3s) are the SAME ORDER as our current
shipped local 12c runtimes (p50 1.547s, heavy 1.8-2.4s). So M_i must be anchored
per case to t_i^alpha, not to t_i^shipped, and the floor margin is ~1.7x on the
heavy band while the MID band is not floored at all.

Wall model at 48 cores (verified in gate0): every shipped pool (13/26/35) is
smaller than the core count, so sum/48 << max_i; the binding terms are the
max-setter and the SERIAL proxy chain (M47: one _proxy_metrics at a time on the
main thread, overlapped with the still-running profiles):

    W(pool, cores) = max( max_k dt_k , sum_k dt_k / cores , sum_k pt_k )

with dt from audit_cache.pkl and pt MEASURED here (per (case,profile) proxy
seconds). Absolute scale comes from the per-case MEASURED 12c runtimes
(results_shipped_m51.json, REFINE-correct) via gamma_i = meas12_i / W(ship,12);
the full-pool side additionally has measured 12c runtimes for n>100 in
m67_oos_cache.pkl["pool0"] (M67-D), so its REFINE=12 penalty is measured too.

Modes:
  gate0    5 gates: rf_score_model subprocess (inherits its drift asserts) /
           audit cache signature / alpha JSON alignment (weighted raw == json
           total) / 48c tier fail-open on the SHIPPED code path / 48c wall
           composition (max-bound + crossover cores c*)
  calib    alpha anchor: kappa (proportional M_i = kappa*t_i^alpha) and the
           constant-M bracket; grader machine-speed estimate via the M10 pool
  fit      wall-model calibration vs the 100 measured 12c runtimes
  project  the Final grid: variants x median models x machine speed, with the
           M67-D OOS tax, floor headroom, and the 48c free-restore budget
  report   all of the above + results_M67E_rf48.json

Run:  C:/Users/Nordra/.conda/envs/iccadv/python.exe m67e_rf48.py gate0
      ... calib | fit | project | report
"""
import argparse
import ast
import hashlib
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# Shipped defaults only (regression_suite.py:66 doctrine); recorded for gate0.
_STRIPPED = sorted(k for k in os.environ if k.startswith("ICCAD_"))
for _k in _STRIPPED:
    del os.environ[_k]

from iccad2026_evaluate import (ContestEvaluator,                    # noqa: E402
                                compute_total_score, evaluate_solution)
from proxy_analysis import build_opt_target_pos                      # noqa: E402
import optimizer_constructive as oc                                  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────
RH, GAMMA, FLOOR = 1.4, 0.3, 0.7
CORES_BETA = 48                       # Beta: dedicated 48-core ICELAKE
AUDIT = _DIR / "audit_cache.pkl"
CACHE = _DIR / "m67e_cache.pkl"
ANCHOR = _DIR / "results_shipped_m51.json"
OOS_CACHE = _DIR / "m67_oos_cache.pkl"
RFMODEL_OUT = _DIR / "m67e_rfmodel_stdout.txt"
DUMP = _DIR / "results_M67E_rf48.json"
ALPHA_JSON = _DIR.parent.parent / "cadc1075_results.json"   # Downloads/

ALPHA_RAW = 1.4527876342862842        # cadc1075_results.json total_score (RF-less)
ALPHA_OFFICIAL = 1.0286               # C_Alpha_Top5.csv rank 3 = us (4 dp)
IN_SET_TOTAL = 1.326473104916827      # shipped M51 local total (RF=1.0)

# M67-D pool0 (measured, n>100): shipped vs ICCAD_ADAPTIVE_POOL=0
OOS_TAX_SHIP = 1.659884               # OOS weighted, shipped
OOS_TAX_FULL = 1.614282               # OOS weighted, full pool + full REFINE
INSET_TAX_SHIP = 1.312108             # same comparison, in-set (for the record)
INSET_TAX_FULL = 1.310721             # -> +0.106% vs the OOS +2.825%

M10_COMMIT = "8565e38"                # alpha submission (M10, 14 cheap profiles)

# audit_cache.pkl key signature: profile_audit.py's PROFILES = live + OM16
OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
PROFILES = list(oc._PROFILES) + [OM16]
N_LIVE = len(oc._PROFILES)
FPR = repr(PROFILES)
LIVE = list(range(N_LIVE))
SWAPSET = {k for k in LIVE if "ICCAD_ORDER_SWAP" in PROFILES[k]
           or "ICCAD_ORDER_MOVE" in PROFILES[k]}
BIGSET = set(oc._BIG_REDUNDANT_IDX)
BANDS = ((0, 40), (40, 60), (60, 100), (100, 110), (110, 10 ** 9))


def band_name(lo, hi):
    return f"({lo},{'inf' if hi >= 10 ** 9 else hi}]"


def pname(k):
    p = PROFILES[k]
    if not p:
        return "base"
    short = {"ICCAD_WIRE_MULT": "W", "ICCAD_ANCHOR_W": "anc", "ICCAD_LR_ASPECT": "LR",
             "ICCAD_TB_ASPECT": "TB", "ICCAD_FRAME_ASPECTS": "fa", "ICCAD_FRAME_SCALES": "fs",
             "ICCAD_WIRE_TIEBREAK": "WT", "ICCAD_WIRE_BFS": "BFS", "ICCAD_BFS_PIN": "PIN",
             "ICCAD_ORDER_SWAP": "OS", "ICCAD_ORDER_MOVE": "OM", "ICCAD_FREE_ASPECT": "FREE",
             "ICCAD_GUIDE_MED": "GM", "ICCAD_FREE_CLUSTER": "FC", "ICCAD_FREE_ANCHORED": "FA",
             "ICCAD_FREE_ANCHORED_BND": "FAbnd", "ICCAD_MIB_ASPECT": "MIB",
             "ICCAD_CLUSTER_ASPECT": "CA"}
    parts = []
    for key, v in p.items():
        s = short.get(key, key)
        if s in ("WT", "BFS", "PIN", "FREE", "GM", "FC", "FA", "FAbnd"):
            parts.append(s)
        elif s in ("fa", "fs"):
            parts.append(f"{s}{v.split(',')[0]}")
        elif "RATIOS" in key:
            continue
        else:
            parts.append(f"{s}{v}")
    return "+".join(parts)


# ── dataset prep (mirrors rf_score_model.py / m49_refine_probe.py) ───────────
def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "missing"


print("[m67e] loading dataset + audit cache ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
CASES = []
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _at, _b2b, _p2b, _pins, _cons = _s["input"]
    _n = int((_at != -1).sum().item())
    _base, _tp = _ev._extract_baseline(_idx, _s["label"], _b2b, _p2b, _pins, _n)
    build_opt_target_pos(_tp, _cons, _n)          # parity with the other probes
    _sumA = sum(max(0.0, float(_at[i])) for i in range(_n))
    CASES.append(dict(idx=_idx, n=_n, A_hat=1.035 * max(_sumA, 1e-9),
                      w=math.exp(_n / 12.0), base=_base, tp=_tp, at=_at,
                      b2b=_b2b, p2b=_p2b, pins=_pins, cons=_cons))
TOTW = sum(c["w"] for c in CASES)
NOF = {c["idx"]: c["n"] for c in CASES}

if not AUDIT.exists():
    sys.exit("audit_cache.pkl missing -> run profile_audit.py first")
_ac = pickle.load(open(AUDIT, "rb"))
if _ac.get("profiles") != FPR:
    sys.exit("audit cache signature != current pool -> re-run profile_audit.py")
DATA = _ac["data"]                                # (ci,k) -> (positions, dt)

SIG = hashlib.md5((FPR + _md5(_DIR / "constructive.exe")).encode()).hexdigest()
_C = {"sig": SIG, "pm": {}, "pt": {}, "cost": {}}
if CACHE.exists():
    try:
        _c0 = pickle.load(open(CACHE, "rb"))
        if _c0.get("sig") == SIG:
            _C = _c0
    except Exception:
        pass


def csave():
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_C, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE)


def ensure_pm(verbose=True):
    """PM + measured proxy seconds for all (case, profile) combos in the cache.

    pt is what makes the 48c wall model honest: with 48 cores the subprocesses
    all run at once, so the serial proxy chain (M47) can become the binding
    term for a large pool.
    """
    todo = [(c["idx"], k) for c in CASES for k in range(len(PROFILES))
            if (c["idx"], k) not in _C["pm"]]
    if not todo:
        return
    if verbose:
        print(f"[m67e] measuring proxy metrics + proxy time for {len(todo)} "
              f"combos (one-off, cached) ...", flush=True)
    t0 = time.perf_counter()
    for j, (ci, k) in enumerate(todo):
        c = CASES[ci]
        ps, _dt = DATA[(ci, k)]
        t1 = time.perf_counter()
        m = oc._proxy_metrics(ps, c["at"], c["b2b"], c["p2b"], c["pins"],
                              c["cons"], c["n"])
        _C["pt"][(ci, k)] = time.perf_counter() - t1
        _C["pm"][(ci, k)] = (m["area"], m["hpwl"], m["vrel"])
        if verbose and (j + 1) % 500 == 0:
            print(f"[m67e]   {j + 1}/{len(todo)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            csave()
    csave()
    if verbose:
        print(f"[m67e] proxy pass done in {time.perf_counter() - t0:.0f}s",
              flush=True)


def PM(ci, k):
    return _C["pm"][(ci, k)]


def PT(ci, k):
    return _C["pt"][(ci, k)]


def cost(ci, k):
    """Official RF=1.0 cost of profile k on case ci (lazy, cached)."""
    key = (ci, k)
    if key not in _C["cost"]:
        c = CASES[ci]
        ps, _ = DATA[key]
        tc = evaluate_solution({'positions': ps, 'runtime': 1.0}, c["base"],
                               c["cons"][:c["n"]], c["b2b"], c["p2b"], c["pins"],
                               c["at"][:c["n"]], target_positions=c["tp"][:c["n"]],
                               median_runtime=1.0)
        _C["cost"][key] = tc.cost
    return _C["cost"][key]


def select(ci, pool):
    """Deployed _RH=1.4 proxy selector (wrapper parity)."""
    hmin = min(PM(ci, k)[1] for k in pool) or 1.0
    A = CASES[ci]["A_hat"]
    return min(pool, key=lambda k: (PM(ci, k)[0] / A + RH * PM(ci, k)[1] / hmin)
               * math.exp(2 * PM(ci, k)[2]))


# ── pools ────────────────────────────────────────────────────────────────────
def pool_shipped(ci, cores=CORES_BETA):
    """The ACTUAL shipped pool at a given cores count (mirrors rf_score_model's
    m46_pool; tier-4 / M50 low-core only fire at cores <= _M45_CORES_MAX)."""
    n = NOF[ci]
    pool = [k for k in LIVE if k not in SWAPSET and not (n > 100 and k in BIGSET)]
    for lo, hi, d in oc._M45_BAND_DROP:
        if lo < n <= hi:
            pool = [k for k in pool if k not in d]
    if cores <= oc._M45_CORES_MAX:
        for lo, hi, d in oc._M45_LOWCORE_DROP:
            if lo < n <= hi:
                pool = [k for k in pool if k not in d]
    return pool


def pool_full(ci, cores=CORES_BETA):
    """ICCAD_ADAPTIVE_POOL=0: all 41 live profiles (and full REFINE)."""
    return list(LIVE)


def restore_candidates(ci, cores=CORES_BETA, slack=1.0):
    """Profiles dropped by the adaptive tiers whose OWN runtime does not exceed
    the current max-setter: at 48 cores the wall is the max-setter, so adding
    them back cannot move the max term (they still cost proxy seconds)."""
    pool = pool_shipped(ci, cores)
    mx = max(DATA[(ci, k)][1] for k in pool)
    return [k for k in LIVE if k not in pool
            and DATA[(ci, k)][1] <= mx * slack]


def pool_restore(ci, cores=CORES_BETA, slack=1.0):
    return sorted(pool_shipped(ci, cores) + restore_candidates(ci, cores, slack))


def wall(ci, pool, cores):
    """max-setter / sum-over-cores / SERIAL proxy chain, whichever binds."""
    dts = [DATA[(ci, k)][1] for k in pool]
    pts = [PT(ci, k) for k in pool]
    return max(max(dts), sum(dts) / cores, sum(pts))


# ── measured runtimes ────────────────────────────────────────────────────────
def anchor_runtimes():
    j = json.load(open(ANCHOR))
    return {r["test_id"]: r["runtime_seconds"] for r in j["test_results"]}, j


def fullpool_runtimes():
    """Measured 12c full-pool (ICCAD_ADAPTIVE_POOL=0) runtimes for n>100,
    recorded by M67-D's pool0 mode."""
    if not OOS_CACHE.exists():
        return {}
    c = pickle.load(open(OOS_CACHE, "rb"))
    out = {}
    for key, rec in c.get("pool0", {}).items():
        if key.startswith("IN"):
            out[int(key[2:])] = rec["runtime"]
    return out


def alpha_records(path=None):
    p = Path(path) if path else ALPHA_JSON
    if not p.exists():
        sys.exit(f"alpha results json not found: {p} (pass --alpha PATH)")
    j = json.load(open(p))
    return j, {r["test_id"]: r for r in j["test_results"]}


# ── gate0 ────────────────────────────────────────────────────────────────────
def mode_gate0(args):
    ok = True

    print("=" * 72)
    print("GATE 1: rf_score_model.py subprocess (inherits its drift asserts)")
    print("=" * 72)
    if args.skip_rfmodel and RFMODEL_OUT.exists():
        out = RFMODEL_OUT.read_text(errors="replace")
        rc = 0
        print(f"  (--skip-rfmodel: reusing {RFMODEL_OUT.name})")
    else:
        env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-u", "rf_score_model.py"], cwd=str(_DIR),
                           capture_output=True, text=True, encoding="utf-8",
                           env=env, timeout=3600)
        out, rc = r.stdout + r.stderr, r.returncode
        RFMODEL_OUT.write_text(out, encoding="utf-8")
        print(f"  ran in {time.perf_counter() - t0:.0f}s")
    need = ["SANITY: full-pool RF=1.0 total",
            "shipped-chain check: refine(100) == _BIG_REDUNDANT_IDX",
            "shipped-chain check: M45 tier-3/tier-4 constants match model OK",
            "sanity: alpha=1.0 reproduces shipped projections @12c/@4c OK",
            "48-core structure", "RF MODEL DONE"]
    miss = [s for s in need if s not in out]
    print(f"  exit={rc}  missing markers: {miss if miss else 'none'}")
    ok &= (rc == 0 and not miss)

    print("\n" + "=" * 72)
    print("GATE 2: audit cache + env hygiene")
    print("=" * 72)
    missing = [(ci, k) for ci in range(100) for k in range(len(PROFILES))
               if (ci, k) not in DATA]
    print(f"  stripped ICCAD_*: {_STRIPPED if _STRIPPED else 'none'}")
    print(f"  pool={N_LIVE} live (+OM16 stand-by)  cache combos missing="
          f"{len(missing)}  sig OK")
    ok &= (not missing) and N_LIVE == 41

    print("\n" + "=" * 72)
    print("GATE 3: alpha JSON alignment")
    print("=" * 72)
    aj, arec = alpha_records(args.alpha)
    same_n = all(arec[c["idx"]]["block_count"] == c["n"] for c in CASES)
    tot = compute_total_score([arec[i]["cost"] for i in range(100)],
                              [arec[i]["block_count"] for i in range(100)])
    feas = sum(1 for i in range(100) if arec[i]["is_feasible"])
    print(f"  cases={len(arec)}  block_count matches ours: {same_n}  feasible={feas}")
    print(f"  weighted raw from per-case costs = {tot:.16f}")
    print(f"  json total_score                 = {aj['total_score']:.16f}  "
          f"|d|={abs(tot - aj['total_score']):.3e}")
    print(f"  official (leaderboard) = {ALPHA_OFFICIAL}  => cost-weighted RF = "
          f"{ALPHA_OFFICIAL / aj['total_score']:.5f}")
    g3 = same_n and feas == 100 and abs(tot - aj["total_score"]) < 1e-9
    ok &= g3

    print("\n" + "=" * 72)
    print("GATE 4: 48-core tier fail-open on the SHIPPED code path")
    print("=" * 72)
    bad = []
    for tag, cores in (("12", 12), ("48", 48)):
        os.environ["ICCAD_ADAPTIVE_CORES"] = tag
        if oc._effective_cores() != cores:
            bad.append(f"_effective_cores({tag})={oc._effective_cores()}")
    os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
    assert oc._effective_cores() > oc._M45_CORES_MAX
    p48 = {c["idx"]: oc._pool_indices(c["n"]) for c in CASES}
    b48 = {c["idx"]: oc._band_env(c["n"]) for c in CASES}
    os.environ["ICCAD_ADAPTIVE_CORES"] = "12"
    p12 = {c["idx"]: oc._pool_indices(c["n"]) for c in CASES}
    b12 = {c["idx"]: oc._band_env(c["n"]) for c in CASES}
    os.environ["ICCAD_ADAPTIVE_CORES"] = "4"
    p4 = {c["idx"]: oc._pool_indices(c["n"]) for c in CASES}
    b4 = {c["idx"]: oc._band_env(c["n"]) for c in CASES}
    del os.environ["ICCAD_ADAPTIVE_CORES"]
    for c in CASES:
        ci, n = c["idx"], c["n"]
        if p48[ci] != p12[ci]:
            bad.append(f"pool 48c != 12c on case {ci}")
        if p48[ci] != pool_shipped(ci, 48):
            bad.append(f"model pool != wrapper pool on case {ci}")
        want = {"ICCAD_REFINE_ITERS": "8"} if 60 < n <= 100 else (
            {"ICCAD_REFINE_ITERS": "4"} if n > 100 else {})
        if b48[ci] != want:
            bad.append(f"band_env(48c) {b48[ci]} != {want} on case {ci}")
        if b48[ci] != b12[ci]:
            bad.append(f"band_env 48c != 12c on case {ci}")
    lowcore_fires = sum(1 for c in CASES if p4[c["idx"]] != p48[c["idx"]])
    mid_lowcore = sum(1 for c in CASES if 60 < c["n"] <= 100
                      and b4[c["idx"]] == {"ICCAD_REFINE_ITERS": "4"})
    print(f"  _effective_cores forced 48 > _M45_CORES_MAX={oc._M45_CORES_MAX} -> "
          f"tier-4 OFF, M50 low-core OFF")
    print(f"  pools identical 48c vs 12c: {sum(1 for c in CASES if p48[c['idx']] == p12[c['idx']])}/100"
          f"   model==wrapper: {sum(1 for c in CASES if p48[c['idx']] == pool_shipped(c['idx'], 48))}/100")
    print(f"  band_env(48c) == universal tier: "
          f"{sum(1 for c in CASES if b48[c['idx']] == b12[c['idx']])}/100 "
          f"(mid K=8, n>100 K=4)")
    print(f"  control @4 cores: tier-4 changes {lowcore_fires} pools, M50 mid "
          f"K=4 fires on {mid_lowcore} cases  (proves the gate is live, not dead code)")
    print(f"  problems: {bad if bad else 'none'}")
    ok &= (not bad) and lowcore_fires > 0 and mid_lowcore > 0

    print("\n" + "=" * 72)
    print("GATE 5: 48-core wall composition")
    print("=" * 72)
    ensure_pm()
    print(f"  {'band':>12} {'#':>3} {'|P|':>4} {'maxdt':>7} {'sum/48':>7} "
          f"{'sumPT':>7} {'c*max':>6} {'binding term(s)'}")
    g5 = True
    for lo, hi in BANDS:
        bc = [c for c in CASES if lo < c["n"] <= hi]
        if not bc:
            continue
        mx, s48, spt, cst, bind = [], [], [], [], {"max": 0, "sum": 0, "pt": 0}
        for c in bc:
            ci = c["idx"]
            pool = pool_shipped(ci, 48)
            dts = [DATA[(ci, k)][1] for k in pool]
            pts = [PT(ci, k) for k in pool]
            mx.append(max(dts)); s48.append(sum(dts) / 48.0); spt.append(sum(pts))
            cst.append(sum(dts) / max(dts))
            terms = {"max": max(dts), "sum": sum(dts) / 48.0, "pt": sum(pts)}
            bind[max(terms, key=terms.get)] += 1
        if any(s > m for s, m in zip(s48, mx)):
            g5 = False
        print(f"  {band_name(lo, hi):>12} {len(bc):>3} "
              f"{len(pool_shipped(bc[0]['idx'], 48)):>4} "
              f"{sum(mx) / len(mx):>7.2f} {sum(s48) / len(s48):>7.2f} "
              f"{sum(spt) / len(spt):>7.2f} {max(cst):>6.1f}  "
              + " ".join(f"{k}x{v}" for k, v in bind.items() if v))
    print(f"  sum/48 <= max_i on every case: {g5}   (c* = sum/max = the core "
          f"count above which the max term binds)")
    ok &= g5

    print("\n" + "=" * 72)
    print(f"GATE0 {'ALL PASS' if ok else 'FAIL'}")
    print("=" * 72)
    csave()
    return 0 if ok else 1


# ── calib ────────────────────────────────────────────────────────────────────
def solve_const_median(arec, target):
    """Model B: one constant median M for every case, fitted to the official
    alpha total (per-case RF = max(0.7,(t/M)^0.3) -> the floor clamps unevenly)."""
    costs = [arec[i]["cost"] for i in range(100)]
    ts = [arec[i]["runtime_seconds"] for i in range(100)]
    ns = [arec[i]["block_count"] for i in range(100)]

    def tot(M):
        return compute_total_score(
            [q * max(FLOOR, (t / M) ** GAMMA) for q, t in zip(costs, ts)], ns)
    lo, hi = 0.05, 500.0
    if tot(hi) > target:                      # even a huge median cannot floor it
        return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if tot(mid) > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def m10_pool_indices():
    """Map the alpha submission's M10 profile list onto today's indices."""
    r = subprocess.run(["git", "show", f"{M10_COMMIT}:optimizer_constructive.py"],
                       cwd=str(_DIR), capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode != 0:
        return None, None
    txt = r.stdout
    i = txt.index("_PROFILES: List[Dict[str, str]] = [") + len(
        "_PROFILES: List[Dict[str, str]] = ")
    j = txt.index("\n]", i)
    lst = ast.literal_eval(txt[i:j + 2])
    idx, missing = [], []
    for p in lst:
        hit = [k for k in LIVE if PROFILES[k] == p]
        if hit:
            idx.append(hit[0])
        else:
            missing.append(p)
    return idx, missing


def mode_calib(args):
    ensure_pm()
    aj, arec = alpha_records(args.alpha)
    rf_bar = ALPHA_OFFICIAL / aj["total_score"]
    kappa = rf_bar ** (-1 / GAMMA)
    print("=" * 72)
    print("ALPHA ANCHOR (submission = M10, commit 8565e38, 14 cheap profiles)")
    print("=" * 72)
    print(f"  raw (RF-less)  = {aj['total_score']:.10f}")
    print(f"  official       = {ALPHA_OFFICIAL}")
    print(f"  cost-weighted RF = {rf_bar:.5f}   (floor {FLOOR}; "
          f"{'ABOVE floor' if rf_bar > FLOOR else 'AT floor'})")
    print(f"  alpha per-case grader runtime: p50 "
          f"{statistics.median(r['runtime_seconds'] for r in arec.values()):.3f}s  "
          f"max {max(r['runtime_seconds'] for r in arec.values()):.3f}s  "
          f"sum {sum(r['runtime_seconds'] for r in arec.values()):.1f}s")
    print(f"\n  MODEL A (proportional, M_i = kappa * t_i^alpha):")
    print(f"    RF is then uniform = kappa^-0.3 = {rf_bar:.5f} (self-consistent, "
          f"no floor clamp) -> kappa = {kappa:.3f}")
    Mb = solve_const_median(arec, ALPHA_OFFICIAL)
    print(f"  MODEL B (one constant median M for all cases):")
    if Mb is None:
        print("    no solution (the official total is below the all-floor value)")
    else:
        rfs = [max(FLOOR, (arec[i]["runtime_seconds"] / Mb) ** GAMMA)
               for i in range(100)]
        nfl = sum(1 for x in rfs if x > FLOOR + 1e-12)
        print(f"    M = {Mb:.2f}s  -> {nfl}/100 cases above the floor, "
              f"{100 - nfl} clamped")

    print("\n" + "=" * 72)
    print("GRADER MACHINE SPEED (via the M10 pool re-priced on today's audit)")
    print("=" * 72)
    idx, missing = m10_pool_indices()
    if idx is None:
        print("  git show failed -> skipped")
        return 0
    print(f"  M10 pool: {len(idx) + len(missing)} profiles, {len(idx)} map onto "
          f"today's _PROFILES exactly, {len(missing)} unmatched")
    for p in missing:
        print(f"    unmatched: {p}")
    print(f"  mapped idx {idx} = " + ", ".join(pname(k) for k in idx))
    # Only a few M10 dicts survive verbatim in today's pool, so bracket the M10
    # wall instead of pretending one number: LOW = the exactly-mapped subset
    # (under-prices the pool -> over-states s), HIGH = the max over every
    # KNOB-ONLY profile alive today (no FREE/FC/FA/CA/MIB/OS/OM stacks — the M10
    # pool was exactly that class, so its max cannot exceed this).
    heavy = ("ICCAD_FREE_ASPECT", "ICCAD_FREE_CLUSTER", "ICCAD_FREE_ANCHORED",
             "ICCAD_CLUSTER_ASPECT", "ICCAD_MIB_ASPECT", "ICCAD_ORDER_SWAP",
             "ICCAD_ORDER_MOVE")
    knob = [k for k in LIVE if not any(h in PROFILES[k] for h in heavy)]
    print(f"  KNOB-only profiles alive today: {knob} = "
          + ", ".join(pname(k) for k in knob))
    print(f"\n  s_eff = t^alpha(grader) / W(M10-like pool, 48c, our box)")
    print(f"  [both are UPPER-biased for pure machine speed: the M10 binary "
          f"predates the M46\n   exact speedups, so part of the ratio is binary, "
          f"not machine]")
    print(f"  {'band':>12} {'#':>3} {'alpha p50':>10} {'s(mapped) p50':>14} "
          f"{'s(knob) p50':>12}")
    all_s, all_k = [], []
    for lo, hi in BANDS:
        bc = [c for c in CASES if lo < c["n"] <= hi]
        if not bc:
            continue
        als, ss, ks = [], [], []
        for c in bc:
            ci = c["idx"]
            als.append(arec[ci]["runtime_seconds"])
            ss.append(arec[ci]["runtime_seconds"] / wall(ci, idx, 48))
            ks.append(arec[ci]["runtime_seconds"] / wall(ci, knob, 48))
        all_s.extend(ss); all_k.extend(ks)
        print(f"  {band_name(lo, hi):>12} {len(bc):>3} "
              f"{statistics.median(als):>10.2f} {statistics.median(ss):>14.2f} "
              f"{statistics.median(ks):>12.2f}")
    all_s.sort(); all_k.sort()
    print(f"  ALL: s(mapped) p50 {statistics.median(all_s):.2f} "
          f"p90 {all_s[int(0.9 * (len(all_s) - 1))]:.2f}   |   "
          f"s(knob) p50 {statistics.median(all_k):.2f} "
          f"p90 {all_k[int(0.9 * (len(all_k) - 1))]:.2f}")
    print(f"  => grader/our-box speed factor brackets roughly "
          f"[{statistics.median(all_k):.1f}, {statistics.median(all_s):.1f}]; "
          f"the projection sweeps s past both ends.")
    csave()
    return 0


# ── fit ──────────────────────────────────────────────────────────────────────
def mode_fit(args):
    ensure_pm()
    meas, _ = anchor_runtimes()
    full12 = fullpool_runtimes()
    print("=" * 72)
    print("WALL-MODEL CALIBRATION vs the 100 MEASURED 12c runtimes")
    print("=" * 72)
    print("  W(pool,cores) = max(max dt, sum dt/cores, sum proxy)  [audit dt is")
    print("  the REFINE=12 counterfactual; gamma_i absorbs the shipped K=4/K=8")
    print("  band overlay, wrapper overhead and machine, per case]")
    print(f"\n  {'band':>12} {'#':>3} {'W12 p50':>8} {'meas p50':>9} "
          f"{'gamma p50':>10} {'gamma p90':>10}")
    rows = {}
    for lo, hi in BANDS:
        bc = [c for c in CASES if lo < c["n"] <= hi]
        if not bc:
            continue
        gs, ws, ms = [], [], []
        for c in bc:
            ci = c["idx"]
            w = wall(ci, pool_shipped(ci, 12), 12)
            gs.append(meas[ci] / w); ws.append(w); ms.append(meas[ci])
        gs.sort()
        rows[(lo, hi)] = gs
        print(f"  {band_name(lo, hi):>12} {len(bc):>3} {statistics.median(ws):>8.2f} "
              f"{statistics.median(ms):>9.2f} {statistics.median(gs):>10.3f} "
              f"{gs[int(0.9 * (len(gs) - 1))]:>10.3f}")
    print("\n  gamma < 1 on the REFINE-cut bands (n>60) is exactly the K=4/K=8")
    print("  overlay the audit dt does not contain; gamma ~ 1 below n=60 says the")
    print("  wall model itself (incl. the serial proxy chain) is well calibrated.")

    # per-profile serial cost: OLS on the bands with NO REFINE overlay (n<=60),
    # where the audit dt is what the wrapper actually ran, so the residual over
    # the wall model is pure per-profile overhead (spawn + serialize + parse +
    # the M47 proxy chain) -> this is what a wall-free restore really costs.
    import numpy as np
    clean = [c for c in CASES if c["n"] <= 60]
    A = np.array([[wall(c["idx"], pool_shipped(c["idx"], 12), 12),
                   len(pool_shipped(c["idx"], 12)), 1.0] for c in clean])
    y = np.array([meas[c["idx"]] for c in clean])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    print(f"\n  per-profile overhead OLS on the {len(clean)} REFINE-free cases "
          f"(n<=60):  meas = a*W12 + b*|P| + c")
    print(f"    a = {coef[0]:.4f}   b = {coef[1] * 1000:.2f} ms/profile   "
          f"c = {coef[2] * 1000:.1f} ms   |resid| p50 {np.median(np.abs(resid)) * 1000:.0f} ms"
          f"  max {np.max(np.abs(resid)) * 1000:.0f} ms")
    print(f"    proxy chain alone: sum(pt) p50 "
          f"{statistics.median(sum(PT(c['idx'], k) for k in pool_shipped(c['idx'], 12)) for c in clean) * 1000:.0f} ms"
          f"  (M47 made it ~free; the pre-M47 2.9s tail is gone)")
    print(f"    => restoring D profiles at 48c costs ~ b*D = "
          f"{coef[1] * 1000:.0f} ms per profile as long as the max-setter does not move")

    if full12:
        print(f"\n  FULL-POOL side (measured by M67-D pool0 @12c, n>100):")
        print(f"  {'ci':>4} {'n':>4} {'measFull':>9} {'W12full':>8} {'gFull':>7} "
              f"{'measShip':>9} {'gShip':>7} {'meas ratio':>11}")
        gf, gr = [], []
        for ci in sorted(full12):
            w = wall(ci, pool_full(ci), 12)
            gf.append(full12[ci] / w); gr.append(full12[ci] / meas[ci])
            print(f"  {ci:>4} {NOF[ci]:>4} {full12[ci]:>9.2f} {w:>8.2f} "
                  f"{full12[ci] / w:>7.3f} {meas[ci]:>9.2f} "
                  f"{meas[ci] / wall(ci, pool_shipped(ci, 12), 12):>7.3f} "
                  f"{full12[ci] / meas[ci]:>11.2f}x")
        print(f"  gammaFull p50 {statistics.median(gf):.3f} vs gammaShip(n>100) "
              f"{statistics.median([meas[ci] / wall(ci, pool_shipped(ci, 12), 12) for ci in full12]):.3f}"
              f"  -> measured full/shipped wall ratio p50 {statistics.median(gr):.2f}x")
    csave()
    return 0


# ── projection ───────────────────────────────────────────────────────────────
def gammas():
    """Per-case calibration of the wall model to the measured 12c runtime."""
    meas, _ = anchor_runtimes()
    g_ship = {c["idx"]: meas[c["idx"]] / wall(c["idx"], pool_shipped(c["idx"], 12), 12)
              for c in CASES}
    full12 = fullpool_runtimes()
    g_full = {}
    meas_g = [full12[ci] / wall(ci, pool_full(ci), 12) for ci in full12] or [1.0]
    g_ref = statistics.median(meas_g)       # ~1.09, measured on n>100
    for c in CASES:
        ci = c["idx"]
        if ci in full12:                    # n>100: measured (M67-D pool0)
            g_full[ci] = full12[ci] / wall(ci, pool_full(ci), 12)
        elif c["n"] <= 60:                  # no REFINE overlay -> same as shipped
            g_full[ci] = g_ship[ci]
        else:                               # mid band: shipped runs K=8, full
            g_full[ci] = g_ref              # runs K=12 -> use the measured
    return g_ship, g_full                   # no-overlay calibration


VARIANTS = ("shipped", "restore", "restoreK12", "full")


def variant_times(cores, s, slack=1.0):
    """Per-case 48c grader runtime estimates for each variant.

    shipped     = the submission (13/26/35 profiles, REFINE K=4/K=8 overlay)
    restore     = shipped + every dropped profile that cannot move the 48c
                  max-setter; REFINE overlay KEPT
    restoreK12  = same pool, REFINE overlay REMOVED (full 12 passes) -> the
                  audit's own configuration, ~2x the heavy-band wall
    full        = ICCAD_ADAPTIVE_POOL=0 (41 profiles incl. the OS16/OM8 giants
                  + full REFINE), runtime measured by M67-D pool0 on n>100
    """
    g_ship, g_full = gammas()
    out = {}
    for c in CASES:
        ci = c["idx"]
        pr = pool_restore(ci, cores, slack)
        out[ci] = {
            "shipped": g_ship[ci] * wall(ci, pool_shipped(ci, cores), cores) * s,
            "restore": g_ship[ci] * wall(ci, pr, cores) * s,
            "restoreK12": g_full[ci] * wall(ci, pr, cores) * s,
            "full": g_full[ci] * wall(ci, pool_full(ci, cores), cores) * s,
        }
    return out


_TAXCACHE = {}


def oos_tax_factor(cores=CORES_BETA, slack=1.0):
    """Multiplier that moves the MODEL's shipped/full quality ratio on n>100
    from its in-sample value onto the OOS one M67-D measured.

    The audit costs are in-sample by construction (the cuts were gated to be
    selection-preserving ON THIS SET, so the model sees ~0 difference); M67-D
    re-solved the heavy band on the training corpus and measured shipped
    1.659884 vs full-pool 1.614282. Anchoring the ratio (rather than pasting
    +2.825% on top of whatever the audit says) also absorbs the REFINE band's
    quality loss, which the K=12 audit positions do not contain.
    """
    key = (cores, slack)
    if key in _TAXCACHE:
        return _TAXCACHE[key]
    hs = hf = 0.0
    for c in CASES:
        if c["n"] <= 100:
            continue
        ci = c["idx"]
        hs += c["w"] * cost(ci, select(ci, pool_shipped(ci, cores)))
        hf += c["w"] * cost(ci, select(ci, pool_full(ci)))
    model_ratio = hs / hf
    _TAXCACHE[key] = (OOS_TAX_SHIP / OOS_TAX_FULL) / model_ratio
    return _TAXCACHE[key]


def variant_quality(theta=0.0, tax_all=False, cores=CORES_BETA, slack=1.0):
    """RF=1.0 per-case cost for each variant, with the M67-D OOS tax.

    theta in [0,1] = the fraction of the OOS gap a wall-free restore would
    recover; unknown without an OOS run (M67-F), so the projection reports the
    break-even theta* instead of guessing.
    """
    tax = oos_tax_factor(cores, slack)
    out = {}
    for c in CASES:
        ci = c["idx"]
        t = tax if (tax_all or c["n"] > 100) else 1.0
        qs = cost(ci, select(ci, pool_shipped(ci, cores)))
        qf = cost(ci, select(ci, pool_full(ci)))
        qr = cost(ci, select(ci, pool_restore(ci, cores, slack)))
        out[ci] = {"shipped": qs * t,
                   "full": qf,
                   "restore": qr * (t ** (1.0 - theta)),
                   # K=12 restores the audit's own REFINE -> the pool cut is the
                   # only difference left vs full, so no OOS tax is carried
                   "restoreK12": qr}
    return out


def project(times, quals, medians, key):
    tot = 0.0
    for c in CASES:
        ci = c["idx"]
        rf = max(FLOOR, (times[ci][key] / medians[ci]) ** GAMMA)
        tot += c["w"] * quals[ci][key] * rf
    return tot / TOTW


def medians_from(arec, model, kappa=None, Mconst=None):
    if model == "A":
        return {c["idx"]: kappa * arec[c["idx"]]["runtime_seconds"] for c in CASES}
    return {c["idx"]: Mconst for c in CASES}


def mode_project(args):
    ensure_pm()
    aj, arec = alpha_records(args.alpha)
    rf_bar = ALPHA_OFFICIAL / aj["total_score"]
    kappa0 = rf_bar ** (-1 / GAMMA)
    Mb = solve_const_median(arec, ALPHA_OFFICIAL)
    cores = args.cores

    tax = oos_tax_factor(cores, args.slack)
    print("=" * 72)
    print(f"M67-E PROJECTION @ {cores} cores   (OOS tax {100 * (tax - 1):+.3f}% on "
          f"the shipped side, {'ALL bands' if args.tax_all else 'n>100 only'})")
    print("=" * 72)

    quals = variant_quality(0.0, args.tax_all, cores, args.slack)
    print(f"\nquality (RF=1.0 weighted totals, audit costs + OOS tax):")
    for key in VARIANTS:
        t = sum(c["w"] * quals[c["idx"]][key] for c in CASES) / TOTW
        print(f"  {key:<11} {t:.6f}")
    print(f"  [in-sample, untaxed shipped = {sum(c['w'] * cost(c['idx'], select(c['idx'], pool_shipped(c['idx'], cores))) for c in CASES) / TOTW:.6f}"
          f" = the 1.3265-style fiction]")

    print(f"\nper-case runtime estimate @{cores}c (s = machine-speed factor vs "
          f"our box):")
    print(f"  {'band':>12} {'shipped':>9} {'restore':>9} {'restK12':>9} {'full':>9} "
          f"{'alpha t':>9} {'M_i(A)':>9}")
    T = variant_times(cores, 1.0, args.slack)
    for lo, hi in BANDS:
        bc = [c for c in CASES if lo < c["n"] <= hi]
        if not bc:
            continue
        f = lambda key: statistics.median(T[c["idx"]][key] for c in bc)
        print(f"  {band_name(lo, hi):>12} {f('shipped'):>9.2f} {f('restore'):>9.2f} "
              f"{f('restoreK12'):>9.2f} {f('full'):>9.2f} "
              f"{statistics.median(arec[c['idx']]['runtime_seconds'] for c in bc):>9.2f} "
              f"{kappa0 * statistics.median(arec[c['idx']]['runtime_seconds'] for c in bc):>9.2f}")

    # ── the grid ────────────────────────────────────────────────────────────
    print(f"\nprojected OFFICIAL total (lower = better). Model A: M_i = kappa * "
          f"t_i^alpha; Model B: constant M = {Mb:.2f}s")
    for model, params in (("A", [2.5, kappa0, 4.0, 6.0]), ("B", [Mb])):
        for pv in params:
            meds = (medians_from(arec, "A", kappa=pv) if model == "A"
                    else medians_from(arec, "B", Mconst=pv))
            lbl = f"kappa={pv:.2f}" if model == "A" else f"M={pv:.2f}s"
            print(f"\n  [{model}] {lbl}")
            print(f"    {'s':>5} {'shipped':>9} {'restore':>9} {'restK12':>9} "
                  f"{'full':>9} | {'rest%':>7} {'restK12%':>9} {'full%':>8} "
                  f"| {'RFship':>7}")
            for s in args.speeds:
                T = variant_times(cores, s, args.slack)
                vals = {k: project(T, quals, meds, k) for k in VARIANTS}
                rfs = sum(c["w"] * quals[c["idx"]]["shipped"]
                          * max(FLOOR, (T[c["idx"]]["shipped"] / meds[c["idx"]]) ** GAMMA)
                          for c in CASES) / sum(c["w"] * quals[c["idx"]]["shipped"] for c in CASES)
                d = lambda k: 100 * (vals[k] - vals["shipped"]) / vals["shipped"]
                print(f"    {s:>5.2f} {vals['shipped']:>9.4f} {vals['restore']:>9.4f} "
                      f"{vals['restoreK12']:>9.4f} {vals['full']:>9.4f} | "
                      f"{d('restore'):>+6.2f}% {d('restoreK12'):>+8.2f}% "
                      f"{d('full'):>+7.2f}% | {rfs:>7.4f}")

    # ── floor headroom ──────────────────────────────────────────────────────
    print(f"\nRF-floor headroom  h_i = 0.3046 * M_i / t_i^shipped  (h>1 = we are "
          f"AT the floor with h-fold slack; model A, kappa={kappa0:.2f}):")
    meds = medians_from(arec, "A", kappa=kappa0)
    thr = FLOOR ** (1 / GAMMA)
    print(f"  {'band':>12} {'#':>3} {'h p10':>7} {'h p50':>7} {'#floored':>9} "
          f"{'wRF':>7}")
    for s in args.speeds:
        T = variant_times(cores, s, args.slack)
        print(f"  --- s={s:.2f} ---")
        for lo, hi in BANDS:
            bc = [c for c in CASES if lo < c["n"] <= hi]
            if not bc:
                continue
            hs = sorted(thr * meds[c["idx"]] / T[c["idx"]]["shipped"] for c in bc)
            nfl = sum(1 for h in hs if h >= 1.0)
            wrf = (sum(c["w"] * max(FLOOR, (T[c["idx"]]["shipped"] / meds[c["idx"]]) ** GAMMA)
                       for c in bc) / sum(c["w"] for c in bc))
            print(f"  {band_name(lo, hi):>12} {len(bc):>3} {hs[0]:>7.2f} "
                  f"{statistics.median(hs):>7.2f} {nfl:>6}/{len(bc):<3} {wrf:>7.4f}")

    # ── free-restore budget ─────────────────────────────────────────────────
    print(f"\n48c FREE-RESTORE budget (profiles the tiers dropped whose own dt is "
          f"<= the current max-setter):")
    print(f"  {'band':>12} {'|ship|':>7} {'|rest|':>7} {'+prof':>6} {'dW48%':>7} "
          f"{'dSumPT':>7} {'c*rest':>7} {'dW24%':>7}  in-set dQ%")
    for lo, hi in BANDS:
        bc = [c for c in CASES if lo < c["n"] <= hi]
        if not bc:
            continue
        dw, dp, qs, qr, np_, cst, dw24 = [], [], 0.0, 0.0, [], [], []
        for c in bc:
            ci = c["idx"]
            ps, pr = pool_shipped(ci, cores), pool_restore(ci, cores, args.slack)
            dw.append(wall(ci, pr, cores) / wall(ci, ps, cores) - 1.0)
            dw24.append(wall(ci, pr, 24) / wall(ci, ps, 24) - 1.0)
            dp.append(sum(PT(ci, k) for k in pr) - sum(PT(ci, k) for k in ps))
            np_.append(len(pr) - len(ps))
            cst.append(sum(DATA[(ci, k)][1] for k in pr)
                       / max(DATA[(ci, k)][1] for k in pr))
            qs += c["w"] * cost(ci, select(ci, ps))
            qr += c["w"] * cost(ci, select(ci, pr))
        print(f"  {band_name(lo, hi):>12} {len(pool_shipped(bc[0]['idx'], cores)):>7} "
              f"{len(pool_restore(bc[0]['idx'], cores, args.slack)):>7} "
              f"{sum(np_) / len(np_):>6.1f} {100 * sum(dw) / len(dw):>+6.2f}% "
              f"{sum(dp) / len(dp):>+7.2f}s {max(cst):>7.1f} "
              f"{100 * sum(dw24) / len(dw24):>+6.2f}%  {100 * (qr / qs - 1):>+7.4f}%")
    print("  (in-set dQ is blind by construction: the cuts were gated to be")
    print("   selection-preserving ON THIS SET. The real prize is the OOS tax.")
    print("   c*rest = cores needed for the restored pool to stay max-bound;")
    print("   dW24% = what the restore would cost if 48 logical = 24 physical.)")

    # break-even theta for restore
    print(f"\n  break-even theta (fraction of the {100 * (tax - 1):.2f}% OOS gap a "
          f"restore must recover to beat shipped), model A kappa={kappa0:.2f}:")
    print(f"    {'s':>5} {'theta*':>8} {'upside@theta=1':>16}")
    q1 = variant_quality(1.0, args.tax_all, cores, args.slack)
    for s in args.speeds:
        T = variant_times(cores, s, args.slack)
        up = 100 * (project(T, q1, meds, "restore")
                    - project(T, quals, meds, "shipped")) / project(T, quals, meds, "shipped")
        lo_t, hi_t, best = 0.0, 1.0, None
        for _ in range(40):
            mid = (lo_t + hi_t) / 2
            q = variant_quality(mid, args.tax_all, cores, args.slack)
            if project(T, q, meds, "restore") < project(T, q, meds, "shipped"):
                hi_t = mid; best = mid
            else:
                lo_t = mid
        print(f"    {s:>5.2f} {('%.3f' % best) if best is not None else '  >1.0':>8} "
              f"{up:>+15.2f}%")
    csave()
    return 0


def mode_report(args):
    rc = mode_project(args)
    aj, arec = alpha_records(args.alpha)
    rf_bar = ALPHA_OFFICIAL / aj["total_score"]
    kappa0 = rf_bar ** (-1 / GAMMA)
    Mb = solve_const_median(arec, ALPHA_OFFICIAL)
    meds = medians_from(arec, "A", kappa=kappa0)
    quals = variant_quality(0.0, args.tax_all, args.cores, args.slack)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "cores": args.cores, "oos_tax": oos_tax_factor(args.cores, args.slack),
           "kappa": kappa0,
           "const_median": Mb, "alpha_rf_weighted": rf_bar,
           "speeds": args.speeds, "cells": {}, "per_case": {}}
    for s in args.speeds:
        T = variant_times(args.cores, s, args.slack)
        out["cells"][f"s={s}"] = {k: project(T, quals, meds, k) for k in VARIANTS}
    T1 = variant_times(args.cores, 1.0, args.slack)
    for c in CASES:
        ci = c["idx"]
        out["per_case"][ci] = {
            "n": c["n"], "w": c["w"], "alpha_t": arec[ci]["runtime_seconds"],
            "M_A": meds[ci], "t_ship48": T1[ci]["shipped"],
            "t_full48": T1[ci]["full"], "t_restore48": T1[ci]["restore"],
            "q_ship_taxed": quals[ci]["shipped"], "q_full": quals[ci]["full"],
            "pool_ship": len(pool_shipped(ci, args.cores)),
            "pool_restore": len(pool_restore(ci, args.cores, args.slack))}
    json.dump(out, open(DUMP, "w"), indent=1)
    print(f"\nwrote {DUMP.name}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate0", "calib", "fit", "project", "report"])
    ap.add_argument("--alpha", default=None, help="path to cadc1075_results.json")
    ap.add_argument("--cores", type=int, default=CORES_BETA)
    ap.add_argument("--speeds", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--slack", type=float, default=1.0,
                    help="restore candidates with dt <= slack * max-setter")
    ap.add_argument("--tax-all", action="store_true", dest="tax_all",
                    help="apply the OOS tax to every band (n<=100 untested)")
    ap.add_argument("--skip-rfmodel", action="store_true", dest="skip_rfmodel")
    args = ap.parse_args()
    return {"gate0": mode_gate0, "calib": mode_calib, "fit": mode_fit,
            "project": mode_project, "report": mode_report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
