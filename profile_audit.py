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

Run detached with the machine otherwise idle (timings feed the om16 decision):
  python -u profile_audit.py > audit_log.txt

Modes (each writes its OWN cache; they are not interchangeable):
  (none)            audit_cache.pkl            knob-off, REFINE=12 counterfactual.
                                               The historical cache. rf_score_model.py
                                               and m67e_rf48.py both validate against
                                               repr(PROFILES) -- do not change its key.
  --m71             audit_cache_m71.pkl        + _M71_ENV on every profile except the
                                               M73 escape tier (mirrors _solve_impl).
  --kband           audit_cache_kband.pkl      + _band_env() per case, i.e. the SHIPPED
  --m71 --kband     audit_cache_m71_kband.pkl  K=8 (mid) / K=4 (n>100) overlay.

`--kband` exists ONLY for regenerating the M42/M45 drop constants (M67-F correction
B: those constants are derived from positions and were only ever gated at K=12, but
they ship under the K overlay -- in-set case 64 shows the tier-3 cut costs +0.41%
once K=8 is applied). Its dt must NOT be fed to m67e_rf48.py, whose gamma is fitted
to absorb the overlay and would double-count it.
"""
import os, sys, math, time, pickle, subprocess, concurrent.futures
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
# Shipped defaults only: strip every ICCAD_* BEFORE importing the wrapper, same as
# m67_oos_probe.py / regression_suite.py. run_one() below does dict(os.environ), so
# a stray knob in the shell would silently ride along on all 5000 runs.
_STRIPPED = sorted(k for k in os.environ if k.startswith("ICCAD_"))
for _k in _STRIPPED:
    del os.environ[_k]
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")
RH = 1.4
CORES = 12          # physical cores (wrapper wall model)
WORKERS = 11        # leave one core for this process -> clean dt measurements

# M73 (2026-07-30): `--m71` re-prices the pool the way the SHIPPED M71/M73 config
# actually runs it -- the two _M71_ENV knobs ride a per-profile overlay applied to
# every profile EXCEPT the M73 escape tier, which is deliberately left knob-OFF (that
# skip in _solve_impl is the whole escape mechanism). Without this the audit measures
# every profile knob-off, which makes the escape duplicates byte-identical to their
# hosts and their dt meaningless for the shipped configuration.
# _band_env() is deliberately still NOT applied *in the default modes*: those stay
# the REFINE=12 counterfactual so m67e_rf48.py's `fit` gamma keeps absorbing the
# K=4/K=8 overlay, and so both caches stay directly comparable.
#
# `--kband` (2026-07-31) opts INTO the overlay, in its own cache. It exists because
# those two facts are in direct conflict:
#   * m67e_rf48.py's wall model NEEDS K=12 dt -- gamma is fitted to absorb the
#     overlay, so an overlaid cache would double-count it;
#   * M67-F correction B proved the M42/M45 drop constants are DERIVED FROM
#     POSITIONS and must be regenerated under the shipped overlay -- the strict
#     selection-preserving gate was only ever verified at K=12, and in-set case 64
#     shows the cut costs +0.41% once the shipped K=8 is applied.
# One cache cannot serve both. Rather than change the default (which would silently
# invalidate gamma and every historical comparison), --kband writes a SEPARATE cache
# whose only consumer is drop-constant regeneration. Never point m67e_rf48.py at it.
M71_MODE = "--m71" in sys.argv
KBAND_MODE = "--kband" in sys.argv
_M71_OVER = dict(getattr(oc, "_M71_ENV", {})) if M71_MODE else {}
_ESCAPE = frozenset(getattr(oc, "_M73_IDX", ()))
CACHE = _DIR / ("audit_cache%s%s.pkl" % ("_m71" if M71_MODE else "",
                                         "_kband" if KBAND_MODE else ""))

PROFILES = list(oc._PROFILES)
# M72/M73 (2026-07-30): oc._PROFILES is no longer the shipped pool -- it now carries
# two default-OFF tiers appended after index 40 (M72 _M55_IDX 41-44, M73 escape
# _M73_IDX 45-48), gated at call time inside _pool_indices(). `N_LIVE =
# len(PROFILES)` would therefore fold both tiers into the SELECTION BASELINE, which
# is wrong twice over: the base_total sanity gate (must reproduce the shipped proxy
# total) would break, and any M42/M45 drop-constant recommendation would be derived
# over a pool that does not ship. The shipped baseline is indices 0.._M55_BASE_LEN-1
# (= 41); everything above it is cached and reported but kept OUT of the baseline,
# exactly the treatment OM16 already gets.
N_LIVE = getattr(oc, "_M55_BASE_LEN", len(PROFILES))   # shipped pool = 41
_TIER_IDX = sorted(set(getattr(oc, "_M55_IDX", ())) | set(getattr(oc, "_M73_IDX", ())))
OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
PROFILES.append(OM16)                        # stand-by: cached but NOT in baseline
OM16_IDX = len(PROFILES) - 1                 # NOT N_LIVE any more (tiers sit between)
# A knob-on and a knob-off cache share the same PROFILES list but have completely
# different dt, so the key must separate them. The DEFAULT mode's key stays exactly
# repr(PROFILES) -- rf_score_model.py:53 and m67e_rf48.py recompute that same
# expression to validate audit_cache.pkl, so changing it would invalidate the
# existing cache and break both downstream tools.
FPR = (repr(PROFILES) if not M71_MODE
       else repr((PROFILES, sorted(_M71_OVER.items()), sorted(_ESCAPE))))
if KBAND_MODE:
    # The cache FILE is already distinct; the KEY is made distinct too so that a
    # mis-copied or mis-renamed kband cache can never be read as the K=12
    # counterfactual. Note this is applied AFTER the expression above, so the
    # default and --m71 keys stay byte-identical to their pre-2026-07-31 values
    # (rf_score_model.py:53 and m67e_rf48.py recompute repr(PROFILES) to validate
    # audit_cache.pkl -- changing it would invalidate 500s of collected data).
    FPR = repr(("kband", FPR))
assert N_LIVE == 41, f"shipped baseline should be 41, got {N_LIVE}"
_KB_NOTE = ""
if KBAND_MODE:
    # _band_env() also has a low-core variant (M50: mid band K=8 -> K=4 when
    # _effective_cores() <= _M45_CORES_MAX). It must not fire here: the point of
    # this mode is to mirror the SHIPPED overlay, and neither this box nor the
    # 48c grader is low-core. Asserted rather than assumed.
    _lo = oc._effective_cores() <= oc._M45_CORES_MAX
    assert not _lo, (f"_effective_cores()={oc._effective_cores()} <= "
                     f"{oc._M45_CORES_MAX}: M50 low-core REFINE would fire and "
                     f"this cache would not mirror the shipped overlay")
    _KB_NOTE = (f" + _band_env overlay (mid K=8, n>100 K=4; "
                f"cores={oc._effective_cores()} -> low-core variant OFF)")
print(f"mode: {'M71-ON overlay (escape tier knob-OFF)' if M71_MODE else 'knob-off (historical)'}"
      f"{_KB_NOTE} -> {CACHE.name}", flush=True)
assert all(i >= N_LIVE for i in _TIER_IDX), "a gated tier leaked into the baseline"
print(f"pool: baseline {N_LIVE} shipped | stand-by {_TIER_IDX} (M72/M73 tiers) "
      f"+ [{OM16_IDX}] om16 | total cached {len(PROFILES)}", flush=True)
if _STRIPPED:
    print(f"stripped from env: {_STRIPPED}", flush=True)


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


def run_one(job):
    ci, k = job
    c = CASES[ci]
    env = dict(os.environ); env.update(PROFILES[k])
    # Precedence mirrors _solve_impl: profile < _band_env < _M71_ENV. (In the
    # wrapper this is `dict(_PROFILES[i], **overlay)` where overlay starts as the
    # band dict and is then update()d with the M71 knobs -- so the band beats the
    # profile's own ICCAD_REFINE_ITERS, and M71 beats both.)
    if KBAND_MODE:
        env.update(oc._band_env(c["n"]))
    if _M71_OVER and k not in _ESCAPE:        # mirrors _solve_impl's overlay skip
        env.update(_M71_OVER)
    t0 = time.perf_counter()
    out = subprocess.run([EXE], input=c["txt"], capture_output=True, text=True, env=env).stdout
    dt = time.perf_counter() - t0
    return ci, k, _parse_output(out, c["n"]), dt


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
kept_om = kept + [OM16_IDX]
print(f"\nprunable (wins==0 and LOO==0): {len(prunable)} -> kept {len(kept)}")
print(f"{'pool':<22}{'total':>9}{'est wall':>10}")
for tag, pool in (("full", live), ("pruned", kept),
                  ("pruned+om16", kept_om), ("full+om16", live + [OM16_IDX])):
    print(f"{tag+' ('+str(len(pool))+')':<22}{total(pool):>9.4f}{est_wall(pool):>9.2f}s", flush=True)
om_t = [data[(c["idx"], OM16_IDX)][1] for c in CASES]
print(f"om16 cpu: mean {sum(om_t)/len(om_t):.2f}s  max {max(om_t):.2f}s")
print(f"prunable ids: {prunable}")

# M72/M73 stand-by tiers: cpu only. Deliberately NOT scored as pool variants here --
# in the DEFAULT modes their dt is measured WITHOUT the _M71_ENV overlay and WITHOUT
# _band_env(), so those numbers are the K=12-counterfactual/knob-off figures the wall
# model wants, not the shipped-configuration dt. (--m71 and --kband each add one of
# those overlays, into their own caches; see the module docstring.) The escape tier's
# shipped-config TIMING is a separate measurement -- m73_escape_dt.py, which mirrors
# _solve_impl exactly and runs SERIALLY, because a per-profile dt cannot be taken
# 11-way parallel; do not mix them.
print(f"\n{'#':>3} {'cpu mean':>8} {'cpu max':>8}  stand-by tier profile")
for k in _TIER_IDX:
    ts = [data[(c["idx"], k)][1] for c in CASES]
    tag = "M72" if k in getattr(oc, "_M55_IDX", ()) else "M73-escape"
    print(f"{k:>3} {sum(ts)/len(ts):>8.2f} {max(ts):>8.2f}  [{tag}] {pname(PROFILES[k])}")
print("AUDIT DONE", flush=True)
