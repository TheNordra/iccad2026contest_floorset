"""M76: offline measurement of the teammate's M73 knob-OFF ESCAPE tier, under OUR
M74 baseline. OFFLINE TOOL — never shipped.

WHAT THE TIER IS (teammate 7403758): our shipped M71 applies the two cluster-item
knobs as a per-profile overlay to EVERY profile in the pool, so a case the knobs
hurt has nowhere to escape to. Their M73 appends knob-OFF DUPLICATES of a few
hosts and deliberately skips the M71 overlay on exactly those indices, so the pool
carries both variants of each host and the proxy arbitrates per case. They measured
_M73_SRC = (2, 22, 23, 25) against M71 with the PRE-M74 adaptive constants.

WHY THIS TOOL EXISTS. Two questions decide the tier and neither can be answered by
running the evaluator:

  1. Is there anything left to win under M74?  M74 regenerated every drop constant
     under M71, so the regression set the tier was fitted to may have moved. The
     2-way oracle (per case: min(shipped, knob-off portfolio)) is the ceiling of
     what ANY escape tier can buy, and it is exactly computable offline.

  2. Does a knob-off duplicate become the 48-core max-setter?  M67-E measured that
     at 48 cores the wall is the max-setter on 100/100 cases, so an appended
     profile costs wall ONLY if its dt exceeds the incumbent max. The M71 knobs
     make heavy cases faster, so the knob-OFF twins are the natural candidates to
     take that crown -- and a read-only pass over audit_cache_ship.pkl shows the
     four hosts ALREADY set the max on several heavy cases (#2 on 91/94, #23 on
     92/85/45, ratio exactly 1.00). This is the teammate's own open item 3, which
     they cannot answer because they have no audit cache.

HOW. audit_cache_ship.pkl holds (positions, dt) for every (case, profile) under the
SHIPPING overlay with the M71 knobs ON; audit_cache_esc.pkl (profile_audit.py esc)
holds the same under the identical overlay with the knobs OFF. Together they are
the knob-ON/knob-OFF pair for all 4100 combos, which lets us simulate ANY escape
source set exactly -- selection included, because the proxy is a pure function of
the cached positions.

MERGED INDEX SPACE: k in 0.._M55_BASE_LEN-1 is the knob-ON shipped profile k;
ESC0 + k is its knob-OFF duplicate. That mirrors the wrapper, where an escape index
is a copy of _PROFILES[k] that _solve_impl does not apply _m71_env() to.

Modes:
  oracle   2-way ceiling (KILL if below BAR_ORACLE) + per-case recoverable
  wall     48c max-setter delta-RF for a source set, and the 12c number for
           comparison with the teammate's own measurement
  derive   forward-greedy source set per band, k=1..KMAX, teammate's set marked
  report   all three

Usage:
  <python> m76_escape_probe.py oracle
  <python> m76_escape_probe.py wall   [--src 2,22,23,25] [--min-n 0]
  <python> m76_escape_probe.py derive [--kmax 6]
  <python> m76_escape_probe.py report
"""
import argparse
import hashlib
import math
import os
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# Shipped defaults only (regression_suite.py:66 doctrine).
_STRIPPED = sorted(k for k in os.environ if k.startswith("ICCAD_"))
for _k in _STRIPPED:
    del os.environ[_k]

from iccad2026_evaluate import ContestEvaluator, evaluate_solution   # noqa: E402
from proxy_analysis import build_opt_target_pos                      # noqa: E402
import optimizer_constructive as oc                                  # noqa: E402

RH, GAMMA = 1.4, 0.3                  # proxy hpwl weight; RuntimeFactor exponent
CORES_BETA, CORES_LOCAL = 48, 12
SHIP_CACHE = _DIR / "audit_cache_ship.pkl"
ESC_CACHE = _DIR / "audit_cache_esc.pkl"
CACHE = _DIR / "m76_cache.pkl"
TEAMMATE_SRC = (2, 22, 23, 25)
BAR_ORACLE = 0.15                     # % — pre-registered kill bar for mode oracle
BANDS = ((60, 100), (100, 110), (110, 10 ** 9))

OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
_SHIPPED = list(oc._PROFILES[:getattr(oc, "_M55_BASE_LEN", len(oc._PROFILES))])
PROFILES = _SHIPPED + [OM16]
N_LIVE = len(_SHIPPED)
ESC0 = 1000                           # merged-space offset for the knob-OFF twins


def _exe_md5():
    return hashlib.md5((_DIR / "constructive.exe").read_bytes()).hexdigest()


def _mode_key(mode):
    """profile_audit.py's signature for `mode`, reproduced exactly."""
    m71 = repr(sorted(({} if mode == "esc" else oc._m71_env()).items()))
    if mode in ("ship", "esc"):
        return repr((mode, m71, repr(sorted(oc._M49_REFINE_BAND)),
                     repr(sorted(oc._M50_REFINE_LOWCORE)), oc._M45_CORES_MAX))
    return repr(("base", m71))


def _load(path, mode):
    if not path.exists():
        sys.exit(f"{path.name} missing -> run `profile_audit.py {mode}` first")
    c = pickle.load(open(path, "rb"))
    want = repr((repr(PROFILES), _mode_key(mode), _exe_md5()))
    if c.get("profiles") != want:
        sys.exit(f"{path.name} signature != current pool/exe -> re-run "
                 f"`profile_audit.py {mode}`")
    return c["data"]


def band_name(lo, hi):
    return f"({lo},{'inf' if hi >= 10 ** 9 else hi}]"


def pname(k):
    p = PROFILES[k % ESC0]
    short = {"ICCAD_WIRE_MULT": "W", "ICCAD_ANCHOR_W": "anc", "ICCAD_LR_ASPECT": "LR",
             "ICCAD_TB_ASPECT": "TB", "ICCAD_FRAME_ASPECTS": "fa",
             "ICCAD_FRAME_SCALES": "fs", "ICCAD_WIRE_TIEBREAK": "WT",
             "ICCAD_WIRE_BFS": "BFS", "ICCAD_BFS_PIN": "PIN", "ICCAD_ORDER_SWAP": "OS",
             "ICCAD_ORDER_MOVE": "OM", "ICCAD_FREE_ASPECT": "FREE",
             "ICCAD_GUIDE_MED": "GM", "ICCAD_FREE_CLUSTER": "FC",
             "ICCAD_FREE_ANCHORED": "FA", "ICCAD_FREE_ANCHORED_BND": "FAbnd",
             "ICCAD_MIB_ASPECT": "MIB", "ICCAD_CLUSTER_ASPECT": "CA"}
    parts = []
    for key, v in p.items():
        s = short.get(key, key)
        if "RATIOS" in key:
            continue
        parts.append(s if s in ("WT", "BFS", "PIN", "FREE", "GM", "FC", "FA",
                                "FAbnd") else f"{s}{v}")
    return ("esc:" if k >= ESC0 else "") + ("+".join(parts) or "base")


# ── dataset prep (mirrors m67e_rf48.py / rf_score_model.py) ──────────────────
print("[m76] loading dataset + both audit caches ...", flush=True)
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

D_ON = _load(SHIP_CACHE, "ship")
D_OFF = _load(ESC_CACHE, "esc")

SIG = hashlib.md5((repr(PROFILES) + _exe_md5()).encode()).hexdigest()
_C = {"sig": SIG, "pm": {}, "cost": {}}
if CACHE.exists():
    try:
        _c0 = pickle.load(open(CACHE, "rb"))
        if _c0.get("sig") == SIG:
            _C = _c0
    except Exception:
        pass


def _csave():
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_C, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE)


def _pos(ci, k):
    return (D_OFF if k >= ESC0 else D_ON)[(ci, k % ESC0)][0]


def dt(ci, k):
    return (D_OFF if k >= ESC0 else D_ON)[(ci, k % ESC0)][1]


def PM(ci, k):
    """(area, hpwl, vrel) — the shapely proxy the wrapper selects on."""
    key = (ci, k)
    if key not in _C["pm"]:
        c = CASES[ci]
        m = oc._proxy_metrics(_pos(ci, k), c["at"], c["b2b"], c["p2b"], c["pins"],
                              c["cons"], c["n"])
        _C["pm"][key] = (m["area"], m["hpwl"], m["vrel"])
    return _C["pm"][key]


def cost(ci, k):
    """True per-case cost at RF=1.0 (the local-eval convention)."""
    key = (ci, k)
    if key not in _C["cost"]:
        c = CASES[ci]
        tc = evaluate_solution({"positions": _pos(ci, k), "runtime": 1.0}, c["base"],
                               c["cons"][:c["n"]], c["b2b"], c["p2b"], c["pins"],
                               c["at"][:c["n"]], target_positions=c["tp"][:c["n"]],
                               median_runtime=1.0)
        _C["cost"][key] = tc.cost
    return _C["cost"][key]


def select(ci, pool):
    """Deployed selector: _RH=1.4 proxy over the pool, hmin recomputed on the
    pool (the wrapper's hmin coupling — a different pool can move the argmin even
    for profiles it shares with another pool)."""
    c = CASES[ci]
    hmin = min(PM(ci, k)[1] for k in pool) or 1.0
    return min(pool, key=lambda k: (PM(ci, k)[0] / c["A_hat"]
                                    + RH * PM(ci, k)[1] / hmin)
               * math.exp(2 * PM(ci, k)[2]))


_POOL_MEMO = {}
_POOL_KEYS = ("ICCAD_ADAPTIVE_CORES", "ICCAD_M73_ESCAPE", "ICCAD_M73_SRC",
              "ICCAD_M73_MIN_N")


def _oc_pool(n, cores, env=None):
    """The pool the WRAPPER would build, translated into the merged index space.

    Delegating to oc._pool_indices() instead of reassembling the pool here is not
    a convenience: the M41 swap filter is CONTENT-based, so it also removes the
    escape twin of a swap profile. A hand-rolled `pool + [ESC0+s ...]` silently
    disagrees, and the disagreement is not academic -- an unfiltered greedy picks
    #37 (OS16) and #34 (OS8), whose audit cpu is 4.5-9s against a 2s max-setter,
    i.e. sets the wrapper would never actually run."""
    key = (n, cores, None if not env else tuple(sorted(env.items())))
    if key in _POOL_MEMO:
        return _POOL_MEMO[key]
    saved = {k: os.environ.pop(k, None) for k in _POOL_KEYS}
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(cores)
    if env:
        os.environ.update(env)
    try:
        idx = oc._pool_indices(n)
    finally:
        for k in _POOL_KEYS:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]
    out = [i if i < N_LIVE else ESC0 + (i - oc._M73_BASE) for i in idx]
    _POOL_MEMO[key] = out
    return out


def _pool_at(n, cores):
    """The shipped knob-ON pool for this case size at a forced core count."""
    return _oc_pool(n, cores)


def pool_esc(n, cores, src, min_n=0):
    """Shipped pool + the escape duplicates, exactly as _pool_indices builds it."""
    src = tuple(src)
    if not src:
        return _pool_at(n, cores)
    return _oc_pool(n, cores, {"ICCAD_M73_ESCAPE": "1",
                               "ICCAD_M73_SRC": ",".join(str(s) for s in src),
                               "ICCAD_M73_MIN_N": str(min_n)})


def weighted(percase):
    return sum(CASES[ci]["w"] * v for ci, v in percase.items()) / TOTW


def portfolio(pool_fn):
    """{case idx: cost} under a pool-building function of (n)."""
    return {c["idx"]: cost(c["idx"], select(c["idx"], pool_fn(c["n"])))
            for c in CASES}


def case_wall(ci, pool, cores):
    """The wrapper's per-case wall model: all-parallel ThreadPool over `cores`."""
    ts = [dt(ci, k) for k in pool]
    return max(max(ts), sum(ts) / cores)


def drf_cost(src, min_n, cores, cis=None):
    """Weighted sum of w_i*((t_new/t_old)^0.3 - 1) / totW, i.e. the RuntimeFactor
    cost of the tier as a fraction of the total score. Median-independent: the
    cross-submission median cancels in the ratio (M41). `cis` restricts the sum to
    a band (the rest of the corpus contributes exactly 0 when the tier is band-
    scoped, so the restricted sum IS the whole cost of a band-scoped tier)."""
    acc = 0.0
    for c in (CASES if cis is None else [CASES[i] for i in cis]):
        ci, n = c["idx"], c["n"]
        r = (case_wall(ci, pool_esc(n, cores, src, min_n), cores)
             / case_wall(ci, _pool_at(n, cores), cores)) ** GAMMA
        acc += c["w"] * (r - 1.0)
    return 100 * acc / TOTW


# ── mode: oracle ────────────────────────────────────────────────────────────
def mode_oracle(args):
    print("=" * 78)
    print("M76 mode=oracle — is there anything left for an escape tier under M74?")
    print("=" * 78)
    ship = portfolio(lambda n: _pool_at(n, CORES_LOCAL))
    # The knob-off endpoint uses the SAME pool cuts, only the overlay differs:
    # that isolates the knobs, which is what the tier arbitrates over.
    off = portfolio(lambda n: [ESC0 + i for i in _pool_at(n, CORES_LOCAL)])
    _csave()

    # The REALIZABLE ceiling of the mechanism: escape tier with all N_LIVE hosts,
    # arbitrated by the same proxy the wrapper uses. Unlike the 2-way oracle this
    # is a deployable configuration, and it can beat both endpoints because the
    # proxy chooses over the union (a knob-off host that loses inside the knob-off
    # portfolio can still win inside the mixed pool).
    uni = portfolio(lambda n: pool_esc(n, CORES_LOCAL, range(N_LIVE)))
    _csave()

    t_ship, t_off, t_uni = weighted(ship), weighted(off), weighted(uni)
    orac = {ci: min(ship[ci], off[ci]) for ci in ship}
    t_or = weighted(orac)
    print(f"\n  shipped (M74, knob-ON everywhere)  {t_ship:.9f}")
    print(f"  knob-OFF portfolio (same cuts)     {t_off:.9f}   "
          f"({(t_off / t_ship - 1) * 100:+.3f}%)")
    print(f"  per-case 2-way ORACLE              {t_or:.9f}   "
          f"({(t_or / t_ship - 1) * 100:+.3f}%)")
    print(f"  FULL-UNION escape tier (41 srcs)   {t_uni:.9f}   "
          f"({(t_uni / t_ship - 1) * 100:+.3f}%)  <- realizable, proxy-arbitrated")
    gain = (1 - t_or / t_ship) * 100
    print(f"\n  ORACLE CEILING     = {gain:.3f}%   (bar {BAR_ORACLE:.2f}%)")
    print(f"  REALIZABLE CEILING = {(1 - t_uni / t_ship) * 100:.3f}%")

    rec = sorted(((CASES[ci]["w"] * (ship[ci] - orac[ci]) / TOTW, ci)
                  for ci in ship if orac[ci] < ship[ci] - 1e-12), reverse=True)
    tot = sum(r for r, _ in rec) or 1e-18
    print(f"\n  {len(rec)} recoverable cases; weighted recoverable = {tot:.6f} "
          f"score points")
    print(f"  {'case':>5} {'n':>4} {'shipped':>10} {'knob-off':>10} "
          f"{'w*d':>10} {'cum%':>7}")
    cum = 0.0
    for r, ci in rec[:15]:
        cum += r
        print(f"  {ci:>5} {CASES[ci]['n']:>4} {ship[ci]:>10.5f} {off[ci]:>10.5f} "
              f"{r:>10.6f} {100 * cum / tot:>6.1f}%")
    bnd = {band_name(lo, hi): sum(r for r, ci in rec if lo < CASES[ci]["n"] <= hi)
           for lo, hi in BANDS}
    bnd["(0,60]"] = sum(r for r, ci in rec if CASES[ci]["n"] <= 60)
    print("\n  by band: " + "  ".join(f"{k} {100 * v / tot:.1f}%"
                                      for k, v in bnd.items()))
    # Gate on the REALIZABLE ceiling, not the oracle: a tier can only ever deliver
    # what the proxy will actually pick, and the 2-way oracle is not deployable.
    real = (1 - t_uni / t_ship) * 100
    verdict = ("PROCEED" if real >= BAR_ORACLE
               else f"RED (realizable ceiling {real:.3f}% < bar {BAR_ORACLE:.2f}%)")
    print(f"\n  VERDICT: {verdict}")
    return 0 if real >= BAR_ORACLE else 2


# ── mode: wall ──────────────────────────────────────────────────────────────
def mode_wall(args):
    src = args.src
    print("=" * 78)
    print(f"M76 mode=wall — 48c max-setter cost of src={src} (min_n={args.min_n})")
    print("=" * 78)
    print("\nper-profile knob-OFF / knob-ON dt ratio (does removing the knobs cost"
          " time?)")
    print(f"  {'band':>12} {'src':>5} {'p50':>7} {'p90':>7} {'max':>7}")
    for lo, hi in BANDS:
        cis = [c["idx"] for c in CASES if lo < c["n"] <= hi]
        for s in src:
            rs = sorted(dt(ci, ESC0 + s) / max(dt(ci, s), 1e-9) for ci in cis)
            print(f"  {band_name(lo, hi):>12} {('#' + str(s)):>5} "
                  f"{rs[len(rs) // 2]:>7.3f} {rs[int(0.9 * (len(rs) - 1))]:>7.3f} "
                  f"{rs[-1]:>7.3f}")

    for cores, label in ((CORES_BETA, "48c (Beta grader, max-setter regime)"),
                         (CORES_LOCAL, "12c (teammate's box, sum-bound regime)")):
        print(f"\n{label}")
        print(f"  {'band':>12} {'cases':>6} {'wall old':>9} {'wall new':>9} "
              f"{'dWall':>8} {'dRF':>8} {'worst dRF':>10}")
        wsum = 0.0
        for lo, hi in BANDS:
            cis = [c["idx"] for c in CASES if lo < c["n"] <= hi]
            if not cis:
                continue
            o_t = n_t = 0.0
            drf_w, worst = 0.0, 1.0
            for ci in cis:
                n = CASES[ci]["n"]
                po, pn = _pool_at(n, cores), pool_esc(n, cores, src, args.min_n)
                to = max(max(dt(ci, k) for k in po),
                         sum(dt(ci, k) for k in po) / cores)
                tn = max(max(dt(ci, k) for k in pn),
                         sum(dt(ci, k) for k in pn) / cores)
                o_t += to
                n_t += tn
                r = (tn / to) ** GAMMA
                worst = max(worst, r)
                drf_w += CASES[ci]["w"] * (r - 1.0)
            wsum += drf_w
            print(f"  {band_name(lo, hi):>12} {len(cis):>6} {o_t / len(cis):>9.3f} "
                  f"{n_t / len(cis):>9.3f} {100 * (n_t / o_t - 1):>+7.2f}% "
                  f"{100 * drf_w / TOTW:>+7.3f}% {100 * (worst - 1):>+9.2f}%")
        # small band carries no escapes when min_n >= 60 but still holds weight
        small = [c["idx"] for c in CASES if c["n"] <= 60]
        drf_s = 0.0
        for ci in small:
            n = CASES[ci]["n"]
            po, pn = _pool_at(n, cores), pool_esc(n, cores, src, args.min_n)
            to = max(max(dt(ci, k) for k in po),
                     sum(dt(ci, k) for k in po) / cores)
            tn = max(max(dt(ci, k) for k in pn),
                     sum(dt(ci, k) for k in pn) / cores)
            drf_s += CASES[ci]["w"] * ((tn / to) ** GAMMA - 1.0)
        print(f"  {'(0,60]':>12} {len(small):>6} {'':>9} {'':>9} {'':>8} "
              f"{100 * drf_s / TOTW:>+7.3f}%")
        print(f"  WEIGHTED dRF COST @{cores}c = "
              f"{100 * (wsum + drf_s) / TOTW:+.3f}% of total score")
    return 0


# ── mode: derive ────────────────────────────────────────────────────────────
def mode_derive(args):
    print("=" * 78)
    print(f"M76 mode=derive — forward-greedy escape source set (kmax={args.kmax})")
    print("=" * 78)
    print("  IN-SAMPLE FIT. M72/M74/M75 all measured that in-sample equality does "
          "not\n  transfer, so treat the result as a second candidate, not a "
          "conclusion.\n")
    cores = args.cores
    base = portfolio(lambda n: _pool_at(n, cores))
    t_base = weighted(base)
    print(f"  pools @{cores}c   baseline (M74 shipped)  {t_base:.9f}\n")

    for lo, hi in (args.band or BANDS):
        cis = [c["idx"] for c in CASES if lo < c["n"] <= hi]
        if not cis:
            continue

        def band_total(src):
            out = dict(base)
            for ci in cis:
                p = pool_esc(CASES[ci]["n"], cores, src)
                out[ci] = cost(ci, select(ci, p))
            return weighted(out)

        chosen, cur = [], t_base
        print(f"  band {band_name(lo, hi)}  ({len(cis)} cases)")
        for _ in range(args.kmax):
            cand = [(band_total(chosen + [s]), s) for s in range(N_LIVE)
                    if s not in chosen]
            best, s = min(cand)
            if best >= cur - 1e-15:
                print("    (no further improvement)")
                break
            chosen.append(s)
            q = (1 - best / t_base) * 100
            w = drf_cost(chosen, 0, CORES_BETA, cis)
            print(f"    +#{s:<3} -> {best:.9f}  q {q:+.4f}%  dRF@48c {w:+.4f}%  "
                  f"NET {q - w:+.4f}%   {pname(s)}")
            cur = best
        tm = band_total(list(TEAMMATE_SRC))
        qt = (1 - tm / t_base) * 100
        wt = drf_cost(TEAMMATE_SRC, 0, CORES_BETA, cis)
        qd = (1 - cur / t_base) * 100
        wd = drf_cost(chosen, 0, CORES_BETA, cis)
        print(f"    teammate {TEAMMATE_SRC} q {qt:+.4f}%  dRF {wt:+.4f}%  "
              f"NET {qt - wt:+.4f}%")
        print(f"    derived  {tuple(chosen)} q {qd:+.4f}%  dRF {wd:+.4f}%  "
              f"NET {qd - wd:+.4f}%\n")
    _csave()

    print("  all-band comparison (same set applied to every band):")
    for tag, src in (("teammate", list(TEAMMATE_SRC)),):
        t = weighted(portfolio(lambda n: pool_esc(n, cores, src)))
        print(f"    {tag:<10} {tuple(src)} -> {t:.9f}  "
              f"({(1 - t / t_base) * 100:+.4f}%)  "
              f"dRF@48c {drf_cost(src, 0, CORES_BETA):+.3f}%")
    _csave()
    return 0


# ── mode: score ─────────────────────────────────────────────────────────────
def mode_score(args):
    """Quality and 48c wall cost of one concrete (src, min_n) on the same page.
    This is the number that decides the tier: local eval forces RF=1.0, so the
    wall side is invisible there and has to come from the audit dt."""
    print("=" * 78)
    print("M76 mode=score — in-set quality vs 48c RuntimeFactor cost")
    print("=" * 78)
    variants = [(tuple(args.src), m) for m in (0, 60, 100)]
    if args.min_n not in (0, 60, 100):
        variants.append((tuple(args.src), args.min_n))
    # Quality is reported under BOTH pool regimes on purpose: the local evaluator
    # runs the <=16-core pools (tier-3 on, tier-5 off) so that column is what an
    # official eval on this box will print, while the 48c column is the one the
    # grader actually scores. They are different pools, so the tier's value can
    # differ between them -- exactly the tier-3/tier-5 asymmetry M74 found.
    for cores in (CORES_LOCAL, CORES_BETA):
        base = portfolio(lambda n: _pool_at(n, cores))
        t_base = weighted(base)
        print(f"\n  pools @{cores}c   baseline {t_base:.9f}")
        print(f"  {'src':>18} {'min_n':>6} {'total':>13} {'quality':>9} "
              f"{'dRF@48c':>9} {'dRF@12c':>9} {'NET@48c':>9} {'mv':>4}")
        for src, min_n in variants:
            p = portfolio(lambda n: pool_esc(n, cores, src, min_n))
            t = weighted(p)
            q = (1 - t / t_base) * 100
            w48 = drf_cost(src, min_n, CORES_BETA)
            w12 = drf_cost(src, min_n, CORES_LOCAL)
            mv = sum(1 for ci in p if abs(p[ci] - base[ci]) > 1e-12)
            bad = sum(1 for ci in p if p[ci] > base[ci] + 1e-12)
            print(f"  {str(src):>18} {min_n:>6} {t:>13.9f} {q:>+8.3f}% "
                  f"{w48:>+8.3f}% {w12:>+8.3f}% {q - w48:>+8.3f}% "
                  f"{mv:>3}{'!' + str(bad) if bad else ''}")
        _csave()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oracle", "wall", "derive", "score", "report"])
    ap.add_argument("--src", default=",".join(str(s) for s in TEAMMATE_SRC))
    ap.add_argument("--min-n", type=int, default=0, dest="min_n")
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--band", default="", help="derive: custom band 'lo,hi' "
                                               "(default: the three shipped bands)")
    ap.add_argument("--cores", type=int, default=CORES_BETA,
                    help="core count whose POOL COMPOSITION derive fits against "
                         "(default 48 = the grader; 12 = this box's local eval)")
    args = ap.parse_args()
    args.src = [int(x) for x in str(args.src).split(",") if x.strip() != ""]
    if args.band:
        _lo, _hi = (int(x) for x in args.band.split(","))
        args.band = ((_lo, _hi if _hi > 0 else 10 ** 9),)
    else:
        args.band = None
    if args.mode == "report":
        rc = mode_oracle(args)
        mode_wall(args)
        mode_derive(args)
        mode_score(args)
        return rc
    return {"oracle": mode_oracle, "wall": mode_wall, "derive": mode_derive,
            "score": mode_score}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
