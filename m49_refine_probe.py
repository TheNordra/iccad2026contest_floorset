"""M49 refine band-cut probe (OFFLINE, never shipped).

Backlog 1(a)/(b): REFINE is 12 of every frame's 13 packs on big cases, yet the
score-vs-pass curve was never measured (cascade-RED only killed post-proc).
ICCAD_REFINE_ITERS already exists (constructive.cpp) -> a band-gated cut ships
with ZERO C++ change. Two modes:

  trace [big|mid]   - run the RTRACE-instrumented constructive_m46.exe
                      (ICCAD_REFINE_TRACE=1) over band x kept pool:
                        * last-improving-refine-pass distribution (all frames +
                          the WINNING frame -> the exact-truncation criterion:
                          truncating at K >= winner's last_imp is bit-identical,
                          losers only get worse and acceptance is strict-less)
                        * winning-frame success index (max_trials=3 evidence)
                        * refine >=2-cycle detection via per-pass FNV64 layout
                          hashes (opt-D only ruled out 1-cycles/fixpoints)
                        * position compare vs audit_cache.pkl (validates the
                          instrumented exe + gate-on non-perturbation)
  variant K1[,K2..] - run the SHIPPED constructive.exe with
                      ICCAD_REFINE_ITERS=K over band x kept pool (11 workers,
                      audit dt protocol) plus a K=12 same-session CONTROL:
                        * per-pair position compare vs audit_cache.pkl -> EXACT?
                          (control must be 260/260 exact or the session is bad)
                        * band wall table + RF projection sweep (M x cores,
                          mirrors rf_score_model's pools incl. M45 tier-4)
                        * non-exact pairs -> variant proxy selection + true
                          cost -> Path B quality-trade table

Decision gates (session plan): Path A = chosen K has ALL pairs exact -> ship
wrapper band env, zero quality risk (M45/M46 class). Path B = M41-precedent
bar: @12c M=11 real gain >= 1%, positive over M in [6,12], weighted local
loss <= +0.10%. Neither -> NO-GO, data to the dead-end ledger.
"""
import os, sys, math, time, pickle, re, subprocess, concurrent.futures
from pathlib import Path
from collections import Counter, defaultdict

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc

EXE   = str(_DIR / "constructive.exe")
EXE46 = str(_DIR / "constructive_m46.exe")
CACHE = _DIR / "audit_cache.pkl"
PMCACHE = _DIR / "m49_pm_cache.pkl"
RH, GAMMA, FLOOR = 1.4, 0.3, 0.7
WORKERS = 11
MGRID = (6, 8, 11, 14, 20)
CORESGRID = (4, 8, 12, 16)

# Pin the kept-pool view to the 12-core regime (tier-4 off) regardless of the
# probe machine's detection; harmless for constructive.exe subprocesses.
os.environ["ICCAD_ADAPTIVE_CORES"] = "12"

# must match profile_audit.py's cache signature (live pool + OM16 stand-by)
OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
PROFILES = list(oc._PROFILES) + [OM16]
N_LIVE = len(oc._PROFILES)
FPR = repr(PROFILES)
SWAPSET = {k for k in range(N_LIVE) if "ICCAD_ORDER_SWAP" in PROFILES[k]
           or "ICCAD_ORDER_MOVE" in PROFILES[k]}
BIGSET = set(oc._BIG_REDUNDANT_IDX)

# ── dataset prep (mirrors rf_score_model.py) ────────────────────────────────
ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
CASES = {}
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    sumA = sum(max(0.0, float(at[i])) for i in range(n))
    CASES[idx] = dict(idx=idx, n=n, A_hat=1.035 * max(sumA, 1e-9),
                      w=math.exp(n / 12.0),
                      txt=_serialize_input(n, at, b2b, p2b, pins, cons, otp,
                                           gnn_hint=None),
                      base=base, tp=tp, at=at, b2b=b2b, p2b=p2b, pins=pins,
                      cons=cons)
totW = sum(c["w"] for c in CASES.values())

if not CACHE.exists():
    sys.exit("audit_cache.pkl missing -> run profile_audit.py first")
_c = pickle.load(open(CACHE, "rb"))
if _c.get("profiles") != FPR:
    sys.exit("cache profile signature != current pool -> re-run profile_audit.py")
DATA = _c["data"]                                   # (ci,k) -> (positions, dt)

BAND_BIG = sorted(ci for ci, c in CASES.items() if c["n"] > 100)
BAND_MID = sorted(ci for ci, c in CASES.items() if 60 < c["n"] <= 100)
KEPT = {ci: oc._pool_indices(CASES[ci]["n"]) for ci in CASES}


def pool_at(ci, cores):
    """Shipped wrapper pool at a given cores count (mirrors rf_score_model's
    m46_pool: M41 swap + M42 big + M45 tier-3 universal + tier-4 low-core)."""
    n = CASES[ci]["n"]
    pool = [k for k in range(N_LIVE)
            if k not in SWAPSET and not (n > 100 and k in BIGSET)]
    for lo, hi, d in oc._M45_BAND_DROP:
        if lo < n <= hi:
            pool = [k for k in pool if k not in d]
    if cores <= oc._M45_CORES_MAX:
        for lo, hi, d in oc._M45_LOWCORE_DROP:
            if lo < n <= hi:
                pool = [k for k in pool if k not in d]
    return pool


for ci in BAND_BIG:                                  # sanity: 12c pool == kept
    assert pool_at(ci, 12) == KEPT[ci], f"pool mismatch case {ci}"

# ── proxy metrics / true cost on CACHE positions (lazy, disk-cached) ────────
_pm = {}
if PMCACHE.exists():
    try:
        p0 = pickle.load(open(PMCACHE, "rb"))
        if p0.get("profiles") == FPR:
            _pm = p0["pm"]
    except Exception:
        _pm = {}


def pm_cache(ci, k):
    if (ci, k) not in _pm:
        c = CASES[ci]; ps, _ = DATA[(ci, k)]
        m = oc._proxy_metrics(ps, c["at"], c["b2b"], c["p2b"], c["pins"],
                              c["cons"], c["n"])
        _pm[(ci, k)] = (m["area"], m["hpwl"], m["vrel"])
    return _pm[(ci, k)]


def save_pm():
    tmp = PMCACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"profiles": FPR, "pm": _pm}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, PMCACHE)


def pm_of(ps, ci):
    c = CASES[ci]
    m = oc._proxy_metrics(ps, c["at"], c["b2b"], c["p2b"], c["pins"],
                          c["cons"], c["n"])
    return (m["area"], m["hpwl"], m["vrel"])


def select_pm(ci, pool, PM):
    """Deployed _RH=1.4 proxy selector over per-(ci,k) metric tuples."""
    hmin = min(PM[k][1] for k in pool) or 1.0
    A = CASES[ci]["A_hat"]
    return min(pool, key=lambda k: (PM[k][0] / A + RH * PM[k][1] / hmin)
               * math.exp(2 * PM[k][2]))


_cost = {}


def cost_of(ps, ci, key):
    if key not in _cost:
        c = CASES[ci]
        tc = evaluate_solution({'positions': ps, 'runtime': 1.0}, c["base"],
                               c["cons"][:c["n"]], c["b2b"], c["p2b"],
                               c["pins"], c["at"][:c["n"]],
                               target_positions=c["tp"][:c["n"]],
                               median_runtime=1.0)
        _cost[key] = tc.cost
    return _cost[key]


# ── subprocess runner ────────────────────────────────────────────────────────
def run_case(exe, ci, k, extra_env):
    env = dict(os.environ); env.update(PROFILES[k]); env.update(extra_env)
    t0 = time.perf_counter()
    r = subprocess.run([exe], input=CASES[ci]["txt"], capture_output=True,
                       text=True, env=env, timeout=300)
    dt = time.perf_counter() - t0
    return ci, k, _parse_output(r.stdout, CASES[ci]["n"]), dt, r.stderr


def run_pool(exe, jobs, extra_env, tag):
    out = {}
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_case, exe, ci, k, extra_env) for ci, k in jobs]
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            ci, k, ps, dt, err = fut.result()
            out[(ci, k)] = (ps, dt, err)
            if (i + 1) % 100 == 0:
                print(f"  [{tag}] {i+1}/{len(jobs)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"  [{tag}] done {len(jobs)} runs in {time.perf_counter()-t0:.0f}s",
          flush=True)
    return out


# ── mode: trace ──────────────────────────────────────────────────────────────
RT_PASS = re.compile(r"^RTRACE pass f=(\d+) r=(\d+) sc=(\S+) imp=(\d) h=([0-9a-f]{16})")
RT_FDONE = re.compile(r"^RTRACE fdone f=(\d+) init=(\S+) final=(\S+) last_imp=(-?\d+)")
RT_WIN = re.compile(r"^RTRACE winner f=(-?\d+) last_imp=(-?\d+) trials=(\d+)")
RT_FIX = re.compile(r"^RTRACE fixpoint f=(\d+) r=(\d+)")


def mode_trace(band_tag):
    band = BAND_BIG if band_tag == "big" else BAND_MID
    jobs = [(ci, k) for ci in band for k in KEPT[ci]]
    print(f"TRACE band={band_tag}: {len(band)} cases x kept pool "
          f"({len(jobs)} runs) on instrumented exe", flush=True)
    res = run_pool(EXE46, jobs, {"ICCAD_REFINE_TRACE": "1"}, "trace")

    mism = []
    winner = {}                       # (ci,k) -> (win_f, win_last_imp, trials)
    all_last = Counter()              # last_imp over ALL frames
    win_last = Counter()              # last_imp of the WINNING frame
    win_fidx = Counter()              # winning frame success index
    cycles = []
    fixpoints = []
    for (ci, k), (ps, dt, err) in res.items():
        if ps != DATA[(ci, k)][0]:
            mism.append((ci, k))
        hashes = defaultdict(dict)    # f -> {h: first r}
        win = None
        for line in err.splitlines():
            m = RT_PASS.match(line)
            if m:
                f, r, h = int(m.group(1)), int(m.group(2)), m.group(5)
                if h in hashes[f]:
                    cycles.append((ci, k, f, hashes[f][h], r))
                else:
                    hashes[f][h] = r
                continue
            m = RT_FDONE.match(line)
            if m:
                all_last[int(m.group(4))] += 1
                continue
            m = RT_WIN.match(line)
            if m:
                win = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                continue
            m = RT_FIX.match(line)
            if m:
                fixpoints.append((ci, k, int(m.group(1)), int(m.group(2))))
        assert win is not None, f"no winner line for case {ci} prof {k}"
        winner[(ci, k)] = win
        win_fidx[win[0]] += 1
        win_last[win[1]] += 1

    print(f"\nposition-vs-cache: "
          f"{'ALL MATCH' if not mism else f'{len(mism)} MISMATCH {mism}'} "
          f"({len(jobs)} pairs)  <- instrumented-exe/gate-on validation")
    print(f"\nlast_imp over ALL frames ({sum(all_last.values())} frames):")
    for v in sorted(all_last):
        print(f"  last_imp={v:>3}: {all_last[v]}")
    print(f"\nlast_imp of WINNING frame ({len(winner)} pairs) "
          f"[exact-truncation criterion]:")
    for v in sorted(win_last):
        print(f"  last_imp={v:>3}: {win_last[v]}")
    maxW = max(v for _, v, _ in winner.values())
    print(f"  max winner last_imp = {maxW}")
    print(f"\nK-coverage (pairs exact if REFINE_ITERS=K):")
    for K in range(0, 13):
        nx = sum(1 for _, v, _ in winner.values() if v <= K)
        mark = "  <- ALL EXACT" if nx == len(winner) else ""
        print(f"  K={K:>2}: {nx}/{len(winner)}{mark}")
        if nx == len(winner):
            break
    print(f"\nwinning frame success idx (max_trials evidence):")
    for v in sorted(win_fidx):
        print(f"  f={v}: {win_fidx[v]}")
    tr = Counter(t for _, _, t in winner.values())
    print(f"  trials distribution: {dict(sorted(tr.items()))}")
    print(f"\nrefine cycle detection (1(b)): "
          f"{len(cycles)} >=2-cycles, {len(fixpoints)} fixpoints")
    for ci, k, f, r0, r1 in cycles[:20]:
        print(f"  CYCLE case {ci} prof {k} f={f}: pass {r0} == pass {r1} "
              f"(period {r1-r0})")
    worst = sorted(winner.items(), key=lambda kv: -kv[1][1])[:10]
    print(f"\nlatest winners (case, prof, win_f, last_imp):")
    for (ci, k), (wf, wl, t) in worst:
        print(f"  case {ci:>3} n={CASES[ci]['n']} prof #{k:<2} f={wf} "
              f"last_imp={wl}")


# ── mode: variant ────────────────────────────────────────────────────────────
def band_dt(ci, k, VD):
    return VD[(ci, k)][1]


def project(K, VD_ctrl, VD_k, exact_all):
    """Real-total projection: band cases use same-session dts (control vs K)
    and cache-Q (exact) or variant-Q; non-band cases identical on both sides
    (cache Q + cache dt) so they cancel in the delta but keep totals honest."""
    # variant per-case proxy metrics for band cases (Path B only)
    PMV = {}
    if not exact_all:
        for ci in BAND_BIG:
            PMV[ci] = {k: pm_of(VD_k[(ci, k)][0], ci) for k in KEPT[ci]}

    print(f"\n  projected REAL total (rows M, cols cores) "
          f"baseline -> K={K} (delta%):")
    hdr = f"    {'M':>3} " + "".join(f"{('c='+str(c)):>26}" for c in CORESGRID)
    print(hdr)
    bigset = set(BAND_BIG)
    for M in MGRID:
        row = f"    {M:>3} "
        for cores in CORESGRID:
            sb = sv = 0.0
            for ci, c in CASES.items():
                pool = pool_at(ci, cores)
                pmc = {k: pm_cache(ci, k) for k in pool}
                kb = select_pm(ci, pool, pmc)
                qb = cost_of(DATA[(ci, kb)][0], ci, ("c", ci, kb))
                if ci in bigset:
                    tsb = [band_dt(ci, k, VD_ctrl) for k in pool]
                    tsv = [band_dt(ci, k, VD_k) for k in pool]
                    if exact_all:
                        qv = qb
                    else:
                        kv = select_pm(ci, pool, PMV[ci])
                        qv = cost_of(VD_k[(ci, kv)][0], ci, ("v", K, ci, kv))
                else:
                    ts = [DATA[(ci, k)][1] for k in pool]
                    tsb = tsv = ts
                    qv = qb
                wb = max(max(tsb), sum(tsb) / cores)
                wv = max(max(tsv), sum(tsv) / cores)
                sb += c["w"] * qb * max(FLOOR, (wb / M) ** GAMMA)
                sv += c["w"] * qv * max(FLOOR, (wv / M) ** GAMMA)
            sb /= totW; sv /= totW
            row += f" {sb:.4f}->{sv:.4f}({100*(sv-sb)/sb:+.2f}%)"
        print(row, flush=True)


def mode_variant(ks):
    jobs = [(ci, k) for ci in BAND_BIG for k in KEPT[ci]]
    print(f"VARIANT: {len(BAND_BIG)} big cases x kept pool = {len(jobs)} runs "
          f"per K, Ks = control(12) + {ks}", flush=True)

    VD = {}
    for K in [12] + ks:
        VD[K] = {kk: (v[0], v[1]) for kk, v in
                 run_pool(EXE, jobs, {"ICCAD_REFINE_ITERS": str(K)},
                          f"K={K}").items()}

    # control must reproduce the cache positions exactly (determinism gate)
    bad = [(ci, k) for ci, k in jobs if VD[12][(ci, k)][0] != DATA[(ci, k)][0]]
    print(f"\ncontrol K=12 vs cache: "
          f"{'EXACT 100%' if not bad else f'{len(bad)} MISMATCH {bad[:8]} <- BAD SESSION'}")
    if bad:
        sys.exit(1)

    for K in ks:
        nonexact = [(ci, k) for ci, k in jobs
                    if VD[K][(ci, k)][0] != DATA[(ci, k)][0]]
        exact_all = not nonexact
        print(f"\n{'='*70}\nK={K}: exact {len(jobs)-len(nonexact)}/{len(jobs)}"
              f"{'  -> PATH A (bit-identical on validation)' if exact_all else ''}")
        if nonexact:
            print(f"  non-exact pairs: {nonexact}")

        # band wall summary at 12c kept pool
        s_ctl = s_k = 0.0
        print(f"  {'case':>4} {'n':>4} {'wall12_ctl':>10} {'wall12_K':>9} "
              f"{'sum_ctl':>8} {'sum_K':>8}")
        for ci in BAND_BIG:
            pool = KEPT[ci]
            tc = [band_dt(ci, k, VD[12]) for k in pool]
            tk = [band_dt(ci, k, VD[K]) for k in pool]
            wc = max(max(tc), sum(tc) / 12); wk = max(max(tk), sum(tk) / 12)
            s_ctl += wc; s_k += wk
            print(f"  {ci:>4} {CASES[ci]['n']:>4} {wc:>10.2f} {wk:>9.2f} "
                  f"{sum(tc):>8.1f} {sum(tk):>8.1f}")
        print(f"  band wall total @12c: {s_ctl:.1f}s -> {s_k:.1f}s "
              f"({100*(s_k-s_ctl)/s_ctl:+.1f}%)")

        if not exact_all:
            # Path B: weighted local (RF=1.0) delta from the band
            dloc = 0.0
            print(f"  Path B per-case quality check @12c kept pool:")
            print(f"  {'case':>4} {'n':>4} {'Qbase':>9} {'QK':>9} "
                  f"{'(tb/tk)^.3':>10} {'Qk/Qb':>8}  medInd-win?")
            for ci in BAND_BIG:
                pool = KEPT[ci]
                pmc = {k: pm_cache(ci, k) for k in pool}
                kb = select_pm(ci, pool, pmc)
                qb = cost_of(DATA[(ci, kb)][0], ci, ("c", ci, kb))
                pmv = {k: pm_of(VD[K][(ci, k)][0], ci) for k in pool}
                kv = select_pm(ci, pool, pmv)
                qv = cost_of(VD[K][(ci, kv)][0], ci, ("v", K, ci, kv))
                tb = [band_dt(ci, k, VD[12]) for k in pool]
                tk = [band_dt(ci, k, VD[K]) for k in pool]
                wb = max(max(tb), sum(tb) / 12); wk = max(max(tk), sum(tk) / 12)
                ratio = (wb / wk) ** GAMMA if wk > 0 else 1.0
                qrel = qv / qb if qb > 0 else 1.0
                dloc += CASES[ci]["w"] * (qv - qb)
                print(f"  {ci:>4} {CASES[ci]['n']:>4} {qb:>9.4f} {qv:>9.4f} "
                      f"{ratio:>10.4f} {qrel:>8.4f}  "
                      f"{'WIN' if qrel <= ratio + 1e-12 else 'lose'}")
            print(f"  weighted local delta (RF=1.0): {dloc/totW:+.6f} "
                  f"({100*(dloc/totW)/1.3277:+.3f}% of 1.3277)")

        project(K, VD[12], VD[K], exact_all)
    save_pm()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "trace"
    if mode == "trace":
        mode_trace(sys.argv[2] if len(sys.argv) > 2 else "big")
    elif mode == "variant":
        ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [6]
        mode_variant(ks)
    else:
        sys.exit(f"unknown mode {mode}")
