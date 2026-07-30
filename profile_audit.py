"""M25 audit: per-profile win tally / leave-one-out / runtime over the live pool.

Two phases, resumable (incremental pickle cache audit_cache.pkl):
  PHASE1 collect — run every profile x case on constructive.exe (11 workers, so
    per-run wall ~ cpu on the 12-core box), store (positions, dt). Interrupt-safe:
    re-running skips cached combos; cache invalidates if the profile set changes.
  PHASE2 report — proxy metrics for all combos, then selection simulation:
    - wins:  #cases the deployed _RH=1.4 proxy selects the profile
    - LOO:   Total Score delta when the profile is removed (selection re-simulated
             without it, hmin recomputed -> the proxy's hmin coupling is handled)
    - cpu:   mean / max subprocess wall over the 100 cases
    True costs are evaluated lazily (only profiles some pool actually selects),
    cutting ~5700 evaluate_solution calls to a few hundred.
  PRUNABLE iff wins==0 and LOO==0 (neither wins a case nor steers selection via
  hmin). Estimated per-case wrapper wall for a pool = max(max_i t_i, sum_i t_i/12)
  (all-parallel ThreadPool on 12 physical cores). Also evaluates the
  om16_bfs_wt_wire stand-by (M23, +0.148% verified): totals and wall for
  full / pruned / pruned+om16 / full+om16 pools.

M74 (2026-07-30) MODE argument — which env overlay the cached positions carry:
  base  (default) -> audit_cache.pkl       profile + _m71_env(), REFINE=12
  ship             -> audit_cache_ship.pkl  profile + _band_env(n) + _m71_env()
Both are needed and they are NOT interchangeable:
  - m49_refine_probe.py's determinism gate requires a REFINE=12 control that
    matches the cache bit-for-bit, so the K=12 reference (base) must survive;
  - M67-F correction B (M67F_REPORT.md:249-258): the M42/M45 strict
    selection-preserving gate must be re-derived under the SHIPPING K=4/K=8
    REFINE overlay, not under the K=12 counterfactual -> ship.
Cases with n<=60 carry an empty _band_env() so both caches must agree there
bit-for-bit; the ship run prints that cross-check.

Run detached with the machine otherwise idle (timings feed the om16 decision):
  python -u profile_audit.py base > m74_audit_base.txt
  python -u profile_audit.py ship > m74_audit_ship.txt
"""
import os, sys, math, time, pickle, hashlib, subprocess, concurrent.futures
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

# M74: pin the wrapper's cores view BEFORE importing the wrapper's helpers are
# used, so _band_env()/_effective_cores() report the 12-core regime regardless of
# the measuring box (mirrors m49_refine_probe.py:57). tier-4 / M50 low-core stay
# off, which is what every downstream drop-set model assumes.
os.environ["ICCAD_ADAPTIVE_CORES"] = "12"
import optimizer_constructive as oc

MODE = (sys.argv[1] if len(sys.argv) > 1 else "base").lower()
if MODE not in ("base", "ship"):
    sys.exit(f"unknown mode {MODE!r} (expected 'base' or 'ship')")

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")
RH = 1.4
CORES = 12          # physical cores (wrapper wall model)
WORKERS = 11        # leave one core for this process -> clean dt measurements
CACHE = _DIR / ("audit_cache.pkl" if MODE == "base" else "audit_cache_ship.pkl")
BASE_CACHE = _DIR / "audit_cache.pkl"

# M72 (2026-07-30): _PROFILES carries the appended _M55_EXTRA tier (ICCAD_M55_POOL,
# ship-RED, default OFF). It is NOT part of the shipped pool -- _pool_indices() never
# emits those indices unless the env gate is set -- and every drop-set constant is an
# index frozenset over 0.._M55_BASE_LEN-1. Key the cache on the SHIPPED pool so the
# gated-off tier cannot invalidate audit_cache.pkl, while any change to the 41 live
# profiles still does.
PROFILES = list(oc._PROFILES[:getattr(oc, "_M55_BASE_LEN", len(oc._PROFILES))])
N_LIVE = len(PROFILES)                       # live pool (selection baseline)
OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
PROFILES.append(OM16)                        # stand-by: cached but NOT in baseline


def _exe_md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# M74: the signature MUST pin the binary and the overlay, not just the profile
# list. The old key was repr(PROFILES) alone, so the 2026-07-10 cache stayed
# "valid" against the 2026-07-29 M71 constructive.exe and every offline gate
# silently kept measuring the pre-M71 placer. Stored under the same "profiles"
# key so every consumer's existing mismatch check and message still fire.
#
# The overlay CONSTANTS are part of the key too, and mode-dependently so: the
# ship cache's positions depend on _M49_REFINE_BAND / _M50_REFINE_LOWCORE, so
# retuning K (e.g. mid 8 -> 6) must invalidate it, while the base cache is always
# REFINE=12 and must NOT be invalidated by a K change it does not depend on.
_M71_SIG = repr(sorted(oc._m71_env().items()))
if MODE == "ship":
    _MODE_KEY = repr(("ship", _M71_SIG,
                      repr(sorted(oc._M49_REFINE_BAND)),
                      repr(sorted(oc._M50_REFINE_LOWCORE)),
                      oc._M45_CORES_MAX))
else:
    _MODE_KEY = repr(("base", _M71_SIG))
FPR = repr((repr(PROFILES), _MODE_KEY, _exe_md5(EXE)))


def pname(prof):
    if not prof:
        return "base"
    short = {"ICCAD_WIRE_MULT": "W", "ICCAD_ANCHOR_W": "anc", "ICCAD_LR_ASPECT": "LR",
             "ICCAD_TB_ASPECT": "", "ICCAD_FRAME_ASPECTS": "fa", "ICCAD_FRAME_SCALES": "fs",
             "ICCAD_WIRE_TIEBREAK": "WT", "ICCAD_WIRE_BFS": "BFS", "ICCAD_BFS_PIN": "PIN",
             "ICCAD_ORDER_SWAP": "OS", "ICCAD_ORDER_MOVE": "OM",
             "ICCAD_FREE_ASPECT": "FREE", "ICCAD_GUIDE_MED": "GM"}
    parts = []
    for k, v in prof.items():
        s = short.get(k, k)
        if s == "":
            continue
        if s in ("WT", "BFS", "PIN", "FREE", "GM"):
            parts.append(s)
        elif s in ("fa", "fs"):
            parts.append(f"{s}{v.split(',')[0]}-{v.split(',')[-1]}")
        else:
            parts.append(f"{s}{v}")
    return "+".join(parts)


# ── dataset prep (both phases) ──────────────────────────────────────────────
CASES = []
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    sumA = sum(max(0.0, float(at[i])) for i in range(n))
    CASES.append(dict(idx=idx, n=n, A_hat=1.035 * max(sumA, 1e-9),
                      w=math.exp(n / 12.0), txt=txt, base=base, tp=tp,
                      at=at, b2b=b2b, p2b=p2b, pins=pins, cons=cons))

# ── PHASE1: collect (resumable) ─────────────────────────────────────────────
cache = {"profiles": FPR, "data": {}}
if CACHE.exists():
    try:
        c0 = pickle.load(open(CACHE, "rb"))
        if c0.get("profiles") == FPR:
            cache = c0
        else:
            print("cache profile set changed -> rebuilding", flush=True)
    except Exception as e:
        print(f"cache load failed ({e}) -> rebuilding", flush=True)
data = cache["data"]


def save_cache():
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE)


# M74: strip ambient ICCAD_* from the child base (mirrors regression_suite.py's
# child_env) so a stray knob in this shell cannot contaminate 4200 cached runs.
# The overlays below are then the ONLY ICCAD_* the binary sees.
_CHILD_BASE = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}


def _overlay(n):
    """The wrapper's per-case env overlay, in its exact precedence order
    (optimizer_constructive.py:1058-1062: profile dict, then band, then M71)."""
    ov = dict(oc._band_env(n)) if MODE == "ship" else {}
    ov.update(oc._m71_env())
    return ov


def run_one(job):
    ci, k = job
    c = CASES[ci]
    env = dict(_CHILD_BASE); env.update(PROFILES[k]); env.update(_overlay(c["n"]))
    t0 = time.perf_counter()
    out = subprocess.run([EXE], input=c["txt"], capture_output=True, text=True, env=env).stdout
    dt = time.perf_counter() - t0
    return ci, k, _parse_output(out, c["n"]), dt


print(f"MODE={MODE}  cache={CACHE.name}  exe md5={_exe_md5(EXE)[:12]}", flush=True)
print(f"  m71 overlay: {oc._m71_env()}")
for _lo, _hi in ((0, 60), (60, 100), (100, 10**9)):
    _n = next((c["n"] for c in CASES if _lo < c["n"] <= _hi), None)
    if _n is not None:
        print(f"  band ({_lo},{'inf' if _hi >= 10**9 else _hi}] (n={_n}): "
              f"overlay {_overlay(_n)}")

todo = [(c["idx"], k) for c in CASES for k in range(len(PROFILES))
        if (c["idx"], k) not in data]
print(f"PHASE1 collect: {len(todo)} of {100*len(PROFILES)} combos to run", flush=True)
if todo:
    t0 = time.perf_counter()
    done_ct = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(run_one, j) for j in todo]):
            ci, k, ps, dt = fut.result()
            data[(ci, k)] = (ps, dt)
            done_ct += 1
            if done_ct % 250 == 0:
                save_cache()
                print(f"collect: {done_ct}/{len(todo)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    save_cache()
    print(f"PHASE1 done in {time.perf_counter()-t0:.0f}s", flush=True)
else:
    print("PHASE1 done (all cached)", flush=True)

# M74 cross-check: where _band_env() is empty the ship overlay collapses to the
# base overlay, so those combos MUST be bit-identical across the two caches.
# Anything else means the overlay plumbing (or the exe) drifted between runs.
if MODE == "ship" and BASE_CACHE.exists():
    try:
        _b = pickle.load(open(BASE_CACHE, "rb"))["data"]
    except Exception as e:
        _b = None
        print(f"cross-check skipped (base cache unreadable: {e})", flush=True)
    if _b:
        same_n = [c["idx"] for c in CASES if not oc._band_env(c["n"])]
        diff_n = [c["idx"] for c in CASES if oc._band_env(c["n"])]
        bad = [(ci, k) for ci in same_n for k in range(len(PROFILES))
               if (ci, k) in _b and data[(ci, k)][0] != _b[(ci, k)][0]]
        moved = sum(1 for ci in diff_n for k in range(len(PROFILES))
                    if (ci, k) in _b and data[(ci, k)][0] != _b[(ci, k)][0])
        print(f"cross-check vs base cache: no-overlay cases {len(same_n)} -> "
              f"{'EXACT' if not bad else f'{len(bad)} MISMATCH {bad[:6]}'}; "
              f"overlay cases {len(diff_n)} -> {moved} combos differ", flush=True)
        if bad:
            sys.exit("ship/base disagree where the band overlay is empty")

# ── PHASE2: analysis ────────────────────────────────────────────────────────
print("PHASE2 proxy metrics...", flush=True)
M = {}
for c in CASES:
    ci = c["idx"]
    for k in range(len(PROFILES)):
        ps, _ = data[(ci, k)]
        m = oc._proxy_metrics(ps, c["at"], c["b2b"], c["p2b"], c["pins"], c["cons"], c["n"])
        M[(ci, k)] = (m["area"], m["hpwl"], m["vrel"])

_cost = {}


def cost(ci, k):
    if (ci, k) not in _cost:
        c = CASES[ci]; ps, _ = data[(ci, k)]
        tc = evaluate_solution({'positions': ps, 'runtime': 1.0}, c["base"],
                               c["cons"][:c["n"]], c["b2b"], c["p2b"], c["pins"],
                               c["at"][:c["n"]], target_positions=c["tp"][:c["n"]],
                               median_runtime=1.0)
        _cost[(ci, k)] = tc.cost
    return _cost[(ci, k)]


def select(ci, pool):
    """Deployed selector: _RH=1.4 proxy, first-best in pool order (wrapper parity)."""
    c = CASES[ci]
    hmin = min(M[(ci, k)][1] for k in pool) or 1.0
    return min(pool, key=lambda k: (M[(ci, k)][0] / c["A_hat"] + RH * M[(ci, k)][1] / hmin)
               * math.exp(2 * M[(ci, k)][2]))


totW = sum(c["w"] for c in CASES)


def total(pool):
    return sum(c["w"] * cost(c["idx"], select(c["idx"], pool)) for c in CASES) / totW


def est_wall(pool):
    tot = 0.0
    for c in CASES:
        ts = [data[(c["idx"], k)][1] for k in pool]
        tot += max(max(ts), sum(ts) / CORES)
    return tot / len(CASES)


live = list(range(N_LIVE))
base_total = total(live)
wins = [0] * len(PROFILES)
for c in CASES:
    wins[select(c["idx"], live)] += 1

print(f"\nlive pool: {N_LIVE} profiles  proxy total = {base_total:.4f}  "
      f"est wall = {est_wall(live):.2f}s/case", flush=True)
print(f"{'#':>3} {'wins':>4} {'LOO dTotal':>11} {'cpu mean':>8} {'cpu max':>8}  name")
prunable = []
for k in range(N_LIVE):
    loo = total([j for j in live if j != k]) - base_total
    ts = [data[(c["idx"], k)][1] for c in CASES]
    mark = ""
    if wins[k] == 0 and abs(loo) < 1e-12:
        prunable.append(k); mark = "  <- PRUNABLE"
    print(f"{k:>3} {wins[k]:>4} {loo:>+11.6f} {sum(ts)/len(ts):>8.2f} {max(ts):>8.2f}  "
          f"{pname(PROFILES[k])}{mark}", flush=True)

kept = [k for k in live if k not in prunable]
kept_om = kept + [N_LIVE]
print(f"\nprunable (wins==0 and LOO==0): {len(prunable)} -> kept {len(kept)}")
print(f"{'pool':<22}{'total':>9}{'est wall':>10}")
for tag, pool in (("full", live), ("pruned", kept),
                  ("pruned+om16", kept_om), ("full+om16", live + [N_LIVE])):
    print(f"{tag+' ('+str(len(pool))+')':<22}{total(pool):>9.4f}{est_wall(pool):>9.2f}s", flush=True)
om_t = [data[(c["idx"], N_LIVE)][1] for c in CASES]
print(f"om16 cpu: mean {sum(om_t)/len(om_t):.2f}s  max {max(om_t):.2f}s")
print(f"prunable ids: {prunable}")
print("AUDIT DONE", flush=True)
