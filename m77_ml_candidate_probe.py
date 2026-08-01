"""M77: what is an external (ML) placer's output WORTH inside our portfolio?

OFFLINE TOOL — never shipped.

WHY THIS EXISTS. The teammate's 2026-07-31 plan ("目前計畫.md") commits to
ML-as-placer with M67-G kept in the pool as a fallback, i.e. the deployed form is
`proxy(41 classical candidates + 1 ML candidate)` per case. But its pre-registered
kill gate is the ML model's OWN weighted total (rung 2 must be < 1.6). Those are
different questions:

  * a model with an ML-only total of 2.0 that wins 10 heavy cases IS worth
    shipping in a portfolio;
  * a model with an ML-only total of 1.6 that wins nothing is worth exactly 0.

So the gate has to be measured on the PORTFOLIO delta, not on the model's solo
score. That is what this tool reports.

M76 supplies the missing justification for trusting it: on the in-set 100 the
full-union escape pool (41 knob-ON profiles + 41 knob-OFF twins — two groups with
visibly different hpwl/area scales) selected by our shapely proxy landed EXACTLY
on the per-case 2-way oracle (1.288443844 both). Selection is not the bottleneck
even for a heterogeneous candidate set, so an ML candidate that is genuinely
better on some case will be picked up.

The three numbers that decide an ML candidate:
  1. portfolio delta      = what it is actually worth (the gate)
  2. oracle delta         = what it would be worth with perfect selection
     -> efficiency = 1 - 2 means the proxy is failing, not the model
  3. dRF@48c              = what its runtime costs, since at 48 cores the wall is
                            the max-setter (M67-E, 100/100) so a candidate is FREE
                            iff its dt is below the incumbent max

INPUT is an official results json (`iccad2026_evaluate.py --output`), i.e. exactly
what any optimizer produces. `test_results[i]` carries positions / cost /
block_count / runtime_seconds.

Usage:
  <python> m77_ml_candidate_probe.py score <ml_results.json> [--cores 48]
                                           [--dt SECONDS] [--name TAG]
  <python> m77_ml_candidate_probe.py selftest
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

_STRIPPED = sorted(k for k in os.environ if k.startswith("ICCAD_"))
for _k in _STRIPPED:
    del os.environ[_k]

from iccad2026_evaluate import ContestEvaluator, evaluate_solution   # noqa: E402
from proxy_analysis import build_opt_target_pos                      # noqa: E402
import optimizer_constructive as oc                                  # noqa: E402

RH, GAMMA = 1.4, 0.3
CORES_BETA, CORES_LOCAL = 48, 12
SHIP_CACHE = _DIR / "audit_cache_ship.pkl"
CACHE = _DIR / "m77_cache.pkl"
ML = 2000                              # merged-space index of the ML candidate
BAR_PORTFOLIO = 0.05                   # % — profile_vs_portfolio.py's house bar

OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
_SHIPPED = list(oc._PROFILES[:oc._M55_BASE_LEN])
PROFILES = _SHIPPED + [OM16]
N_LIVE = len(_SHIPPED)


def _exe_md5():
    return hashlib.md5((_DIR / "constructive.exe").read_bytes()).hexdigest()


print("[m77] loading dataset + ship audit cache ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
CASES = []
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _at, _b2b, _p2b, _pins, _cons = _s["input"]
    _n = int((_at != -1).sum().item())
    _base, _tp = _ev._extract_baseline(_idx, _s["label"], _b2b, _p2b, _pins, _n)
    build_opt_target_pos(_tp, _cons, _n)
    _sumA = sum(max(0.0, float(_at[i])) for i in range(_n))
    CASES.append(dict(idx=_idx, n=_n, A_hat=1.035 * max(_sumA, 1e-9),
                      w=math.exp(_n / 12.0), base=_base, tp=_tp, at=_at,
                      b2b=_b2b, p2b=_p2b, pins=_pins, cons=_cons))
TOTW = sum(c["w"] for c in CASES)

_want = repr((repr(PROFILES),
              repr(("ship", repr(sorted(oc._m71_env().items())),
                    repr(sorted(oc._M49_REFINE_BAND)),
                    repr(sorted(oc._M50_REFINE_LOWCORE)), oc._M45_CORES_MAX)),
              _exe_md5()))
if not SHIP_CACHE.exists():
    sys.exit("audit_cache_ship.pkl missing -> run `profile_audit.py ship` first")
_ac = pickle.load(open(SHIP_CACHE, "rb"))
if _ac.get("profiles") != _want:
    sys.exit("audit_cache_ship.pkl signature != current pool/exe -> re-run "
             "`profile_audit.py ship`")
D_ON = _ac["data"]

SIG = hashlib.md5((repr(PROFILES) + _exe_md5()).encode()).hexdigest()
_C = {"sig": SIG, "pm": {}, "cost": {}}
if CACHE.exists():
    try:
        _c0 = pickle.load(open(CACHE, "rb"))
        if _c0.get("sig") == SIG:
            _C = _c0
    except Exception:
        pass

_EXT = {}          # (ci, ML) -> positions of the candidate under test
_SELFTEST = {}     # selftest invariants, filled by mode_score


def _csave():
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_C, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE)


def _pos(ci, k):
    return _EXT[(ci, k)] if k >= ML else D_ON[(ci, k)][0]


def PM(ci, k, key=None):
    """(area, hpwl, vrel) via the shapely proxy — the wrapper's selector.
    External candidates are keyed by a content hash so two different ML runs
    cannot collide in the cache."""
    ck = (ci, k if k < ML else f"ml:{key}")
    if ck not in _C["pm"]:
        c = CASES[ci]
        m = oc._proxy_metrics(_pos(ci, k), c["at"], c["b2b"], c["p2b"], c["pins"],
                              c["cons"], c["n"])
        _C["pm"][ck] = (m["area"], m["hpwl"], m["vrel"])
    return _C["pm"][ck]


def cost(ci, k, key=None):
    ck = (ci, k if k < ML else f"ml:{key}")
    if ck not in _C["cost"]:
        c = CASES[ci]
        tc = evaluate_solution({"positions": _pos(ci, k), "runtime": 1.0}, c["base"],
                               c["cons"][:c["n"]], c["b2b"], c["p2b"], c["pins"],
                               c["at"][:c["n"]], target_positions=c["tp"][:c["n"]],
                               median_runtime=1.0)
        _C["cost"][ck] = (tc.cost, bool(tc.is_feasible))
    return _C["cost"][ck]


def select(ci, pool, key=None):
    """First-best over the pool in pool order; hmin recomputed on the pool, so an
    ML candidate with a very small hpwl re-scales every other candidate too —
    that coupling is real and is why this is not a 2-way comparison."""
    c = CASES[ci]
    hmin = min(PM(ci, k, key)[1] for k in pool) or 1.0
    return min(pool, key=lambda k: (PM(ci, k, key)[0] / c["A_hat"]
                                    + RH * PM(ci, k, key)[1] / hmin)
               * math.exp(2 * PM(ci, k, key)[2]))


_POOL_MEMO = {}


def _pool_at(n, cores):
    if (n, cores) in _POOL_MEMO:
        return _POOL_MEMO[(n, cores)]
    old = os.environ.get("ICCAD_ADAPTIVE_CORES")
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(cores)
    try:
        out = [i for i in oc._pool_indices(n) if i < N_LIVE]
    finally:
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
        if old is not None:
            os.environ["ICCAD_ADAPTIVE_CORES"] = old
    _POOL_MEMO[(n, cores)] = out
    return out


def _load_ml(path):
    """-> {case idx: dict(positions, cost, runtime, n)}, plus a content hash."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = j["test_results"]
    out = {}
    for r in rows:
        if r.get("positions") is None:
            continue
        out[int(r["test_id"])] = dict(positions=r["positions"],
                                      cost=r.get("cost"),
                                      runtime=r.get("runtime_seconds"),
                                      n=int(r["block_count"]),
                                      feasible=bool(r.get("is_feasible")))
    key = hashlib.md5(json.dumps([out[k]["positions"] for k in sorted(out)],
                                 sort_keys=True).encode()).hexdigest()[:12]
    return out, key, j.get("total_score"), j.get("submission_name", Path(path).name)


def mode_score(args):
    ml, key, ml_total, name = _load_ml(args.results)
    tag = args.name or name
    cores = args.cores
    print("=" * 78)
    print(f"M77 — portfolio value of an external candidate: {tag}")
    print("=" * 78)
    print(f"  file      {Path(args.results).name}   content md5 {key}")
    print(f"  cases     {len(ml)}/100 with positions")
    print(f"  pool      @{cores}c (tier-5 {'ON' if cores >= 40 else 'off'}, "
          f"tier-3 {'off' if cores > 16 else 'ON'})")
    if ml_total is not None:
        print(f"  ML solo   total_score {ml_total:.9f}   <- the teammate's gate")

    for ci, rec in ml.items():
        _EXT[(ci, ML)] = rec["positions"]
        if len(rec["positions"]) != CASES[ci]["n"]:
            sys.exit(f"case {ci}: {len(rec['positions'])} positions but n="
                     f"{CASES[ci]['n']} -> wrong dataset or truncated file")

    base, withml, orac = {}, {}, {}
    ml_cost, ml_feas = {}, {}
    for c in CASES:
        ci = c["idx"]
        pool = _pool_at(c["n"], cores)
        base[ci] = cost(ci, select(ci, pool))[0]
        if ci in ml:
            mc, mf = cost(ci, ML, key)
            ml_cost[ci], ml_feas[ci] = mc, mf
            # ML goes LAST so a proxy tie keeps the incumbent (no cosmetic movers)
            withml[ci] = cost(ci, select(ci, pool + [ML], key), key)[0]
            orac[ci] = min(base[ci], mc)
        else:
            withml[ci] = orac[ci] = base[ci]
    _csave()

    def wsum(d):
        return sum(CASES[ci]["w"] * v for ci, v in d.items()) / TOTW

    t_b, t_w, t_o = wsum(base), wsum(withml), wsum(orac)
    g_w = (1 - t_w / t_b) * 100
    g_o = (1 - t_o / t_b) * 100
    eff = (g_w / g_o * 100) if abs(g_o) > 1e-12 else float("nan")

    print(f"\n  M74 portfolio (41 profiles)        {t_b:.9f}")
    print(f"  + ML candidate, proxy-arbitrated   {t_w:.9f}   {g_w:+.3f}%  "
          f"<- THE GATE (bar {BAR_PORTFOLIO:.2f}%)")
    print(f"  + ML candidate, per-case ORACLE    {t_o:.9f}   {g_o:+.3f}%")
    print(f"  selection efficiency               {eff:.1f}% of the oracle delta")

    picked = [ci for ci in withml if abs(withml[ci] - base[ci]) > 1e-12]
    worse = [ci for ci in picked if withml[ci] > base[ci] + 1e-12]
    wins = [ci for ci in orac if ci in ml and ml_cost[ci] < base[ci] - 1e-12]
    infeas = [ci for ci in ml_feas if not ml_feas[ci]]
    _SELFTEST.update(picked=len(picked), wins=len(wins), gain=g_w)
    print(f"\n  ML beats the portfolio on          {len(wins)}/{len(ml)} cases")
    print(f"  proxy actually switched to ML on   {len(picked)} "
          f"({len(worse)} of them WORSE — proxy mistakes)")
    print(f"  ML infeasible on                   {len(infeas)} cases"
          + (f"  {infeas[:8]}" if infeas else ""))

    if wins:
        rows = sorted(((CASES[ci]["w"] * (base[ci] - ml_cost[ci]) / TOTW, ci)
                       for ci in wins), reverse=True)
        print(f"\n  {'case':>5} {'n':>4} {'portfolio':>10} {'ML':>10} "
              f"{'w*d':>10} {'taken':>6}")
        for r, ci in rows[:15]:
            print(f"  {ci:>5} {CASES[ci]['n']:>4} {base[ci]:>10.5f} "
                  f"{ml_cost[ci]:>10.5f} {r:>10.6f} "
                  f"{'yes' if ci in picked else 'NO':>6}")

    # ---- runtime: at 48 cores the wall is the max-setter, so the candidate is
    # free iff its dt stays under the incumbent max on that case.
    dts = ({} if args.no_rf else
           {ci: (args.dt if args.dt is not None else ml[ci]["runtime"])
            for ci in ml})
    if dts and all(v is not None for v in dts.values()):
        src = ("--dt override" if args.dt is not None
               else "runtime_seconds in the json  ** SEE THE WARNING **")
        print(f"\n  runtime source: {src}")
        if args.dt is None:
            print("  ⚠ runtime_seconds in an official results json is the WHOLE "
                  "solve() wall,\n    not one candidate's, so it is ALWAYS >= the "
                  "incumbent max-setter and the\n    dRF below is meaningless. It "
                  "is printed only so the shape is visible.\n    Pass --dt with "
                  "the model's own per-case inference time for a real\n    number "
                  "(--dt 0 if inference is negligible), or --no-rf to skip.")
        acc, over = 0.0, []
        for c in CASES:
            ci = c["idx"]
            if ci not in dts:
                continue
            pool = _pool_at(c["n"], cores)
            ts = [D_ON[(ci, k)][1] for k in pool]
            old = max(max(ts), sum(ts) / cores)
            new = max(max(max(ts), dts[ci]), (sum(ts) + dts[ci]) / cores)
            if new > old + 1e-12:
                over.append((ci, c["n"], old, dts[ci]))
            acc += c["w"] * ((new / old) ** GAMMA - 1.0)
        print(f"  dRF@{cores}c = {100 * acc / TOTW:+.3f}% of total score   "
              f"({len(over)}/{len(dts)} cases where ML sets the new wall)")
        for ci, n, old, d in sorted(over, key=lambda x: -x[3])[:5]:
            print(f"    case {ci:>3} n={n:>4}  incumbent wall {old:.2f}s  "
                  f"ML {d:.2f}s")
        print(f"\n  NET@{cores}c = {g_w - 100 * acc / TOTW:+.3f}%")
    else:
        print("\n  runtime: skipped (--no-rf, or missing in the json with no --dt)")

    verdict = ("GREEN" if g_w >= BAR_PORTFOLIO and not worse
               else "RED (below the portfolio bar)" if g_w < BAR_PORTFOLIO
               else "AMBER (gains, but the proxy also picked losers)")
    print(f"\n  VERDICT: {verdict}")
    return 0 if g_w >= BAR_PORTFOLIO else 2


def mode_selftest(args):
    """Feeding the portfolio its OWN output must be worth exactly nothing: the
    candidate is bit-identical to the incumbent winner on every case, so hmin
    cannot move and a proxy tie keeps the incumbent. Any drift here means the
    offline selector has diverged from the wrapper."""
    args.results = str(_DIR / "results_M74_default.json")
    args.name = "SELFTEST (M74's own output)"
    args.dt, args.no_rf = None, True
    global _SELFTEST
    _SELFTEST = {}
    mode_score(args)
    ok = (_SELFTEST.get("picked") == 0 and _SELFTEST.get("wins") == 0
          and abs(_SELFTEST.get("gain", 1.0)) < 1e-9)
    print(f"\n  SELFTEST: {'PASS' if ok else 'FAIL'} "
          f"(expected 0 wins / 0 switches / +0.000%; got "
          f"{_SELFTEST.get('wins')} / {_SELFTEST.get('picked')} / "
          f"{_SELFTEST.get('gain', float('nan')):+.6f}%)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "selftest"])
    ap.add_argument("results", nargs="?", help="official results json")
    ap.add_argument("--cores", type=int, default=CORES_BETA)
    ap.add_argument("--dt", type=float, default=None,
                    help="per-case seconds for the ML candidate (overrides the "
                         "json's runtime_seconds, which is the whole solve wall)")
    ap.add_argument("--no-rf", action="store_true", dest="no_rf",
                    help="skip the RuntimeFactor section entirely")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.mode == "score" and not args.results:
        ap.error("score needs a results json")
    return {"score": mode_score, "selftest": mode_selftest}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
