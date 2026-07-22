#!/usr/bin/env python3
"""OFFLINE ONLY - M67-D out-of-sample generalization pre-check (never shipped).

Every tuning decision M1..M51 saw only the 100 local validation cases
(LiteTensorDataTest, proven bit-identical to the alpha test set). Beta runs REAL
hidden cases. This probe estimates how much of the shipped 1.3265 is
overfitting, by scoring the SHIPPED M51 portfolio on cases drawn from the
TRAINING corpus (floorset_lite, 30240 layouts, never used for any tuning).

Design (measured facts, see M67_PLAN.md / M67D_REPORT.md):
  * validation holds EXACTLY ONE case per n in [21,120] -> the sample mirrors it
    by drawing K cases per n (K=2, K=4 for n>100 where the weight sits), and the
    headline averages per n BEFORE the official weighting, so unequal K cannot
    tilt the weight profile away from validation's.
  * training .th = 7-tuple; d[0][L][:,0] area_target, d[0][L][:,1:] the same
    5 constraint columns, d[5][L] fp_sol (W,H,X,Y), d[6][L] metrics_sol(8).
    metrics_sol[0]/[-2]/[-1] == recomputed area/hpwl_b2b/hpwl_p2b, so
    ContestEvaluator._extract_baseline (iccad2026_evaluate.py:806) can be
    mirrored exactly, stored-metrics branch included.
  * scoring is the official evaluate_solution with target_positions set (hard
    checks ON, like m52_phase0_probe._cost_strict) and median_runtime=1.0 =>
    RuntimeFactor 1.0, exactly the local harness semantics behind 1.3265.
  * raw-vs-raw is only a coarse comparison: 1.3265 contains the validation label
    floor (fp_sol verbatim = 1.1079 = exp(2*vrel_label)); training cases carry a
    DIFFERENT floor. The generalization number is the floor-relative ratio.

Modes:
  gate0   env/pool hygiene + 3 in-set cases reproduced BIT-EXACTLY vs
          results_shipped_m51.json + training-side baseline sanity
  run     the OOS sweep (resumable; cache m67_oos_cache.pkl)
  report  tables + results_M67D_oos.json
  ref     single-profile (ICCAD_CONSTRUCTIVE_SINGLE=1) reference on both sets,
          only needed if the headline lands yellow/red

Run:  C:/Users/Nordra/.conda/envs/iccadv/python.exe m67_oos_probe.py gate0
      C:/Users/Nordra/.conda/envs/iccadv/python.exe m67_oos_probe.py run
      C:/Users/Nordra/.conda/envs/iccadv/python.exe m67_oos_probe.py report
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import math
import os
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# Shipped defaults only: strip every ICCAD_* BEFORE importing the wrapper
# (regression_suite.py:66 does the same for its children). Recorded for gate0.
_STRIPPED = sorted(k for k in os.environ if k.startswith("ICCAD_"))
for _k in _STRIPPED:
    del os.environ[_k]

import torch                                                       # noqa: E402

from iccad2026_evaluate import (calculate_bbox_area,               # noqa: E402
                                calculate_hpwl_b2b, calculate_hpwl_p2b,
                                compute_total_score, evaluate_solution)
from proxy_analysis import build_opt_target_pos                     # noqa: E402
import optimizer_constructive as oc                                 # noqa: E402

SEED = 67
VERSION = 1
CACHE_PATH = _DIR / "m67_oos_cache.pkl"
ANCHOR_JSON = _DIR / "results_shipped_m51.json"
DUMP_JSON = _DIR / "results_M67D_oos.json"

IN_SET_TOTAL = 1.326473104916827        # results_shipped_m51.json total_score
BANDS = (("S", 20, 60), ("M", 60, 100), ("B", 100, 130))
HEAVY_LO = 100                          # n > HEAVY_LO gets --heavy-per-n draws
GATE_INSET_IDS = (0, 50, 99)
BAR_GREEN, BAR_YELLOW = 1.40, 1.45      # M67_PLAN.md section M67-D step 3
BAR_RATIO_PCT = 3.0                     # |delta| on the floor-relative ratio


def _exe_md5():
    p = Path(oc._BIN)
    if not p.exists():
        return "MISSING"
    return hashlib.md5(p.read_bytes()).hexdigest()[:12]


def _sig(args):
    """Solver identity only. Sample knobs (--per-n/--heavy-per-n/--workers) stay
    OUT: the sample is prefix-stable in K and cases are keyed individually, so a
    wider sample reuses everything already solved."""
    return repr(("m67d", VERSION, SEED, _exe_md5(), len(oc._PROFILES),
                 repr(oc._PROFILES)))


# --------------------------------------------------------------------------- #
# cache (atomic, sig-guarded; the file index survives a sig change)            #
# --------------------------------------------------------------------------- #
_C = {"sig": None, "index": {}, "cases": {}, "inset": {}, "ref": {}}


def _cload(sig):
    global _C
    if CACHE_PATH.exists():
        try:
            c = pickle.load(open(CACHE_PATH, "rb"))
            if c.get("sig") == sig:
                _C = c
            else:
                _C = {"sig": sig, "index": c.get("index", {}), "cases": {},
                      "inset": c.get("inset", {}), "ref": {}}
                print("[cache] sig changed -> cases/ref reset "
                      "(file index + in-set floor kept)")
        except Exception as e:
            print(f"[cache] unreadable ({e!r}); starting fresh")
    _C["sig"] = sig


def _csave():
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_C, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE_PATH)


def _shuffled(seq, seedstr):
    lst = list(seq)
    random.Random(seedstr).shuffle(lst)
    return lst


def _band_of(n):
    for b, lo, hi in BANDS:
        if lo < n <= hi:
            return b
    return "?"


# --------------------------------------------------------------------------- #
# training corpus: index -> n-mirrored sample -> case load                     #
# --------------------------------------------------------------------------- #
def _key_of(path):
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def _path_of(key):
    return _DIR / "floorset_lite" / (key + ".th")


def _seed_index_from_m52():
    """workers 0-2 were already scanned by m52_phase0_probe (270 files)."""
    src = _DIR / "m52_phase0_cache.pkl"
    if not src.exists():
        return 0
    try:
        files = pickle.load(open(src, "rb")).get("index", {}).get("files", {})
    except Exception:
        return 0
    got = 0
    for path, (n, nl) in files.items():
        k = _key_of(path)
        if k not in _C["index"] and _path_of(k).exists():
            _C["index"][k] = (int(n), int(nl))
            got += 1
    return got


def _index(workers):
    """{worker/stem: (n, n_layouts)} for worker_0..worker_{workers-1}."""
    got = _seed_index_from_m52()
    if got:
        print(f"[index] seeded {got} files from m52_phase0_cache.pkl")
    todo = []
    for w in range(workers):
        for f in sorted(glob.glob(str(_DIR / "floorset_lite" / f"worker_{w}"
                                      / "layouts_*.th"))):
            if _key_of(f) not in _C["index"]:
                todo.append(f)
    if todo:
        print(f"[index] scanning {len(todo)} new files (one-time, cached) ...")
        t0 = time.time()
        for i, f in enumerate(todo):
            d = torch.load(f)
            _C["index"][_key_of(f)] = (int((d[0][0][:, 0] != -1).sum().item()),
                                       int(d[0].shape[0]))
            if (i + 1) % 200 == 0:
                print(f"[index]   {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
        _csave()
    idx = {k: v for k, v in _C["index"].items()
           if int(k.split("/")[0].split("_")[1]) < workers}
    return idx


def _sample(index, per_n, heavy_per_n):
    """K cases for every n in [21,120]; distinct FILES first, same file only
    when the corpus runs out. Deterministic and prefix-stable in K."""
    by_n = {}
    for key, (n, nl) in index.items():
        by_n.setdefault(n, []).append((key, nl))
    specs, missing = [], []
    for n in range(21, 121):
        want = heavy_per_n if n > HEAVY_LO else per_n
        fl = sorted(by_n.get(n, []))
        if not fl:
            missing.append(n)
            continue
        order = _shuffled(fl, f"{SEED}:file:{n}")
        picks = {k: _shuffled(range(nl), f"{SEED}:L:{k}") for k, nl in order}
        take, rounds = [], 0
        while len(take) < want and rounds < 200:
            progressed = False
            for k, nl in order:
                if len(take) >= want:
                    break
                used = sum(1 for t in take if t[0] == k)
                if used >= nl:
                    continue
                take.append((k, picks[k][used], n))
                progressed = True
            rounds += 1
            if not progressed:
                break
        specs.extend(take)
    return specs, missing


def _load_case(d, L):
    at_all = d[0][L][:, 0]
    n = int((at_all != -1).sum().item())
    fp = d[5][L][:n]
    tp = [(float(fp[i][2]), float(fp[i][3]), float(fp[i][0]), float(fp[i][1]))
          for i in range(n)]                      # (x, y, w, h) from (W,H,X,Y)
    return dict(n=n, at=at_all[:n], cons=d[0][L][:n, 1:], b2b=d[1][L],
                p2b=d[2][L], pins=d[3][L], tp=tp, met=d[6][L])


# --------------------------------------------------------------------------- #
# official scoring (mirrors ContestEvaluator exactly)                          #
# --------------------------------------------------------------------------- #
def _baseline_official(lay):
    """ContestEvaluator._extract_baseline (:806) on a training layout: derive
    from fp_sol, then prefer the stored metrics when valid. Returns the baseline
    dict plus the max relative stored-vs-recomputed deviation (fidelity gate)."""
    pos = lay["tp"]
    hb = calculate_hpwl_b2b(pos, lay["b2b"])
    hp = calculate_hpwl_p2b(pos, lay["p2b"], lay["pins"])
    ar = calculate_bbox_area(pos)
    dev = 0.0
    met = lay.get("met")
    if met is not None and len(met) >= 8:
        def _rel(a, b):
            return abs(a - b) / max(abs(b), 1e-9)
        if met[0] > 0:
            dev = max(dev, _rel(float(met[0]), ar)); ar = float(met[0])
        if met[-2] > 0:
            dev = max(dev, _rel(float(met[-2]), hb)); hb = float(met[-2])
        if met[-1] >= 0:
            dev = max(dev, _rel(float(met[-1]), hp)); hp = float(met[-1])
    return {"hpwl_baseline": hb + hp, "area_baseline": ar}, dev


def _cost(positions, lay):
    """Official strict scoring: hard checks ON (target_positions), RF = 1.0."""
    return evaluate_solution({"positions": positions, "runtime": 1.0},
                             lay["base"], lay["cons"], lay["b2b"], lay["p2b"],
                             lay["pins"], lay["at"], target_positions=lay["tp"],
                             median_runtime=1.0)


def _mt(m):
    return dict(cost=float(m.cost), feasible=bool(m.is_feasible),
                hgap=float(m.hpwl_gap), agap=float(m.area_gap),
                vrel=float(m.violations_relative),
                vb=int(m.boundary_violations), vg=int(m.grouping_violations),
                vm=int(m.mib_violations), nsoft=int(m.max_possible_violations),
                overlaps=int(m.overlap_violations),
                dimviol=int(m.dimension_violations),
                areaviol=int(m.area_violations))


def _features(lay):
    cons, n = lay["cons"], lay["n"]
    cn = [[int(v) for v in cons[i].tolist()] for i in range(n)]
    return dict(fixed=sum(1 for c in cn if c[0] != 0),
                pre=sum(1 for c in cn if c[1] != 0),
                mibG=max([c[2] for c in cn] + [0]),
                cluG=max([c[3] for c in cn] + [0]),
                bnd=sum(1 for c in cn if c[4] != 0),
                b2bE=int((lay["b2b"][:, 0] != -1).sum().item()),
                p2bE=int((lay["p2b"][:, 0] != -1).sum().item()))


def _solve_one(opt, lay):
    """One shipped-portfolio solve, exactly as the harness calls it. stderr is
    captured so the M48 fallback chain becomes observable."""
    otp = build_opt_target_pos(lay["tp"], lay["cons"], lay["n"])
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stderr(buf):
        pos = opt.solve(lay["n"], lay["at"], lay["b2b"], lay["p2b"],
                        lay["pins"], lay["cons"], otp)
    return pos, time.perf_counter() - t0, buf.getvalue().strip()


# --------------------------------------------------------------------------- #
# validation-side helpers (anchor + label floor + features)                    #
# --------------------------------------------------------------------------- #
def _anchor():
    j = json.load(open(ANCHOR_JSON))
    return j, {r["test_id"]: r for r in j["test_results"]}


def _inset_dataset():
    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    return ev


def _inset_lay(ev, idx):
    s = ev.dataset[idx]
    at, b2b, p2b, pins, cons = s["input"]
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, s["label"], b2b, p2b, pins, n)
    return dict(n=n, at=at[:n], cons=cons[:n], b2b=b2b, p2b=p2b, pins=pins,
                tp=tp[:n], base=base, met=None)


def _inset_floor(ev):
    """fp_sol verbatim cost per validation case (the in-set label floor)."""
    st = _C["inset"].setdefault("floor", {})
    ft = _C["inset"].setdefault("feat", {})
    todo = [i for i in range(100) if i not in st]
    if todo:
        print(f"[in-set] scoring label floor for {len(todo)} cases ...")
        for i in todo:
            lay = _inset_lay(ev, i)
            st[i] = _mt(_cost(lay["tp"], lay))
            st[i]["n"] = lay["n"]
            ft[i] = _features(lay)
        _csave()
    return st, ft


# --------------------------------------------------------------------------- #
# aggregation                                                                  #
# --------------------------------------------------------------------------- #
def _per_n_total(rows, field="cost"):
    """Average within each n first, then apply the official weighting once per
    n -> the weight profile matches validation's (one case per n) regardless of
    how many draws each n got."""
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r[field])
    ns = sorted(by_n)
    means = [sum(by_n[n]) / len(by_n[n]) for n in ns]
    return compute_total_score(means, ns), ns, means


def _weights(ns):
    mx = max(ns)
    return [math.exp((n - mx) / 12.0) for n in ns]


def _sel(args):
    """(lo, hi] case-size window for pool0/restore. --pool0-hi 0 = no upper
    bound (M67-F Phase 1 semantics, n>100). M67-F mid-band top-up uses
    --pool0-lo 60 --pool0-hi 100 so the M45 tier-3 band is scored on its own;
    cases are cached individually, so a run with a wider window reuses every
    solve a narrower one already did."""
    return args.pool0_lo, (args.pool0_hi or 10 ** 9)


def _selname(lo, hi):
    return f"n>{lo}" if hi >= 10 ** 9 else f"{lo}<n<={hi}"


def _in_sel(n, lo, hi):
    return lo < n <= hi


# --------------------------------------------------------------------------- #
# modes                                                                        #
# --------------------------------------------------------------------------- #
def mode_gate0(args):
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' ' + extra) if extra else ''}")

    print("=" * 78)
    print("GATE 0-a  env / pool hygiene")
    print("=" * 78)
    print(f"  stripped from env at import: {_STRIPPED or '(none)'}")
    left = sorted(k for k in os.environ if k.startswith("ICCAD_"))
    chk("no ICCAD_* remains", not left, str(left))
    chk("_PROFILES == 41 (shipped pool, not L1)", len(oc._PROFILES) == 41,
        f"got {len(oc._PROFILES)}")
    for n in (30, 80, 120):
        print(f"    n={n:3d}  pool={len(oc._pool_indices(n)):2d}  "
              f"band_env={oc._band_env(n) or '{}'}")
    chk("adaptive pool active on n=120", len(oc._pool_indices(120)) < 41)
    chk("constructive binary present", Path(oc._BIN).exists(), f"md5 {_exe_md5()}")

    print()
    print("=" * 78)
    print("GATE 0-b  in-set bit-exact reproduction vs results_shipped_m51.json")
    print("=" * 78)
    ev = _inset_dataset()
    _j, arec = _anchor()
    opt = oc.MyOptimizer(verbose=False)
    for idx in GATE_INSET_IDS:
        lay = _inset_lay(ev, idx)
        pos, dt, err = _solve_one(opt, lay)
        m = _cost(pos, lay)
        a = arec[idx]
        same_cost = float(m.cost) == float(a["cost"])
        apos = a.get("positions") or []
        same_pos = (len(apos) == len(pos) and
                    all(float(x) == float(y)
                        for pr, ar in zip(pos, apos) for x, y in zip(pr, ar)))
        chk(f"case {idx:3d} (n={lay['n']:3d}) cost bit-exact",
            same_cost, f"{m.cost!r} vs {a['cost']!r}  [{dt:.2f}s]")
        chk(f"case {idx:3d} positions bit-exact", same_pos)
        if err:
            chk(f"case {idx:3d} no fallback on stderr", False, err.splitlines()[0])

    print()
    print("=" * 78)
    print("GATE 0-c  training-side baseline / label-floor sanity")
    print("=" * 78)
    idx = _index(args.workers)
    specs, missing = _sample(idx, 1, 1)
    for key, L, n in specs[:5]:
        d = torch.load(_path_of(key))
        lay = _load_case(d, L)
        lay["base"], dev = _baseline_official(lay)
        m = _cost(lay["tp"], lay)
        good = (m.is_feasible and abs(m.hpwl_gap) < 1e-6
                and abs(m.area_gap) < 1e-9 and dev < 1e-5)
        chk(f"{key}/L{L} (n={n:3d}) fp_sol verbatim", good,
            f"cost {m.cost:.4f} feas {m.is_feasible} hgap {m.hpwl_gap:+.2e} "
            f"agap {m.area_gap:+.2e} vrel {m.violations_relative:.4f} "
            f"metdev {dev:.1e}")
    print(f"  n values with no training file (workers<{args.workers}): "
          f"{missing or '(none)'}")

    print()
    print("=" * 78)
    print(f"GATE 0 VERDICT: {'ALL PASS' if ok else 'FAIL'}")
    print("=" * 78)
    _csave()
    return 0 if ok else 1


def mode_run(args):
    idx = _index(args.workers)
    specs, missing = _sample(idx, args.per_n, args.heavy_per_n)
    if missing:
        print(f"[warn] no training file for n={missing} "
              f"(raise --workers to cover them)")
    if args.limit:
        specs = specs[:args.limit]
    todo = [s for s in specs if f"{s[0]}/L{s[1]}" not in _C["cases"]]
    print(f"[run] sample {len(specs)} cases "
          f"(per_n={args.per_n}, heavy_per_n={args.heavy_per_n}); "
          f"{len(specs) - len(todo)} cached, {len(todo)} to solve")
    if not todo:
        print("[run] nothing to do")
        return 0

    opt = oc.MyOptimizer(verbose=False)
    byf = {}
    for key, L, n in todo:
        byf.setdefault(key, []).append(L)
    t0, done = time.time(), 0
    for key in sorted(byf):
        d = torch.load(_path_of(key))
        for L in sorted(byf[key]):
            lay = _load_case(d, L)
            lay["base"], dev = _baseline_official(lay)
            ck = f"{key}/L{L}"
            try:
                pos, dt, err = _solve_one(opt, lay)
                rec = _mt(_cost(pos, lay))
                rec["error"] = None
            except Exception as e:                       # never expected (M48)
                pos, dt, err = None, 0.0, ""
                rec = dict(cost=10.0, feasible=False, hgap=0.0, agap=0.0,
                           vrel=1.0, vb=0, vg=0, vm=0, nsoft=0, overlaps=0,
                           dimviol=0, areaviol=0, error=repr(e))
            floor = _mt(_cost(lay["tp"], lay))
            rec.update(n=lay["n"], key=ck, runtime=dt, stderr=err,
                       metdev=dev, floor_cost=floor["cost"],
                       floor_vrel=floor["vrel"], floor_feasible=floor["feasible"],
                       feat=_features(lay), positions=pos)
            _C["cases"][ck] = rec
            done += 1
            if done % 10 == 0:
                _csave()
                el = time.time() - t0
                print(f"[run] {done}/{len(todo)}  ({el:.0f}s, "
                      f"eta {el / done * (len(todo) - done):.0f}s)  "
                      f"last n={rec['n']} cost={rec['cost']:.4f} "
                      f"t={rec['runtime']:.2f}s", flush=True)
    _csave()
    print(f"[run] done {done} cases in {time.time() - t0:.0f}s")
    return 0


def _rows(args):
    idx = _index(args.workers)
    specs, _ = _sample(idx, args.per_n, args.heavy_per_n)
    if args.limit:
        specs = specs[:args.limit]
    out = []
    for key, L, n in specs:
        r = _C["cases"].get(f"{key}/L{L}")
        if r is not None:
            out.append(r)
    return out


def mode_report(args):
    rows = _rows(args)
    if not rows:
        print("[report] cache empty - run `run` first")
        return 1
    ev = _inset_dataset()
    floor_in, feat_in = _inset_floor(ev)
    _j, arec = _anchor()

    oos_total, ns, means = _per_n_total(rows)
    naive = compute_total_score([r["cost"] for r in rows],
                                [r["n"] for r in rows])
    floor_total, _, _ = _per_n_total(rows, "floor_cost")
    in_floor = compute_total_score([floor_in[i]["cost"] for i in range(100)],
                                   [floor_in[i]["n"] for i in range(100)])
    ratio_oos = oos_total / floor_total
    ratio_in = IN_SET_TOTAL / in_floor
    dratio = (ratio_oos / ratio_in - 1.0) * 100.0

    infeas = [r for r in rows if not r["feasible"]]
    fb = [r for r in rows if r["stderr"]]
    errs = [r for r in rows if r.get("error")]
    rts = sorted(r["runtime"] for r in rows)
    in_rts = sorted(arec[i]["runtime_seconds"] for i in range(100))

    def pct(v, p):
        return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]

    print("=" * 78)
    print(f"M67-D  OOS PRE-CHECK   {len(rows)} training cases, "
          f"{len(set(r['n'] for r in rows))} distinct n, seed {SEED}")
    print("=" * 78)
    if infeas or fb or errs:
        print("!! HARD FLAGS")
        print(f"   infeasible cases : {len(infeas)}  "
              f"{[r['key'] for r in infeas][:5]}")
        print(f"   fallback/stderr  : {len(fb)}  {[r['key'] for r in fb][:5]}")
        print(f"   solve exceptions : {len(errs)} {[r['key'] for r in errs][:5]}")
        for r in fb[:5]:
            print(f"     {r['key']}: {r['stderr'].splitlines()[0]}")
    else:
        print("   hard flags: NONE (100% feasible, zero fallback, zero exception)")
    print()
    print(f"  OOS raw total (per-n mean, official weighting) : {oos_total:.4f}")
    print(f"  OOS raw total (naive over all cases)           : {naive:.4f}")
    print(f"  in-set (results_shipped_m51.json)              : {IN_SET_TOTAL:.4f}")
    print(f"  delta raw                                      : "
          f"{oos_total - IN_SET_TOTAL:+.4f} "
          f"({(oos_total / IN_SET_TOTAL - 1) * 100:+.2f}%)")
    print()
    print(f"  OOS label floor (fp_sol verbatim)              : {floor_total:.4f}")
    print(f"  in-set label floor                             : {in_floor:.4f}")
    print(f"  ratio  OOS  = total/floor                      : {ratio_oos:.4f}")
    print(f"  ratio in-set= total/floor                      : {ratio_in:.4f}")
    print(f"  delta ratio (the generalization number)        : {dratio:+.2f}%")
    print()

    print("-" * 78)
    print("BAND DECOMPOSITION (weights from the per-n means)")
    print("-" * 78)
    w = dict(zip(ns, _weights(ns)))
    tw = sum(w.values())
    by_band = {}
    for n, c in zip(ns, means):
        by_band.setdefault(_band_of(n), []).append((n, c))
    in_cost = {arec[i]["block_count"]: arec[i]["cost"] for i in range(100)}
    print(f"{'band':<6}{'n range':<12}{'#n':>4}{'wContr%':>9}"
          f"{'OOS mean':>10}{'in-set':>9}{'delta':>9}{'ratio':>8}")
    for b, lo, hi in BANDS:
        items = by_band.get(b, [])
        if not items:
            continue
        ww = sum(w[n] for n, _ in items)
        oc_ = sum(w[n] * c for n, c in items) / ww
        ic_ = sum(w[n] * in_cost[n] for n, _ in items if n in in_cost) / \
            sum(w[n] for n, _ in items if n in in_cost)
        fr = _per_n_total([r for r in rows if _band_of(r["n"]) == b], "floor_cost")[0]
        oo = _per_n_total([r for r in rows if _band_of(r["n"]) == b])[0]
        print(f"{b:<6}({lo:3d},{hi:3d}]{'':<2}{len(items):>4}"
              f"{100 * ww / tw:>8.1f}%{oc_:>10.4f}{ic_:>9.4f}"
              f"{oc_ - ic_:>+9.4f}{oo / fr:>8.4f}")
    print()

    print("-" * 78)
    print("PER-n PAIRED DELTA (validation has exactly one case per n) - top 15")
    print("-" * 78)
    pair = []
    for n, c in zip(ns, means):
        if n in in_cost:
            pair.append((w[n] * (c - in_cost[n]) / tw * 100, n, c, in_cost[n]))
    pair.sort(key=lambda t: -abs(t[0]))
    print(f"{'n':>5}{'OOS mean':>11}{'in-set':>10}{'delta':>10}{'wContr%':>10}")
    for wc, n, c, ic_ in pair[:15]:
        print(f"{n:>5}{c:>11.4f}{ic_:>10.4f}{c - ic_:>+10.4f}{wc:>+10.3f}")
    print()

    print("-" * 78)
    print("WORST OOS CASES  (by weighted contribution, then by raw cost)")
    print("-" * 78)
    for r in rows:
        r["_w"] = math.exp((r["n"] - 120) / 12.0) * r["cost"]
    print(f"{'case':<28}{'n':>4}{'cost':>9}{'floor':>8}{'R':>7}"
          f"{'hgap':>8}{'agap':>8}{'vrel':>8}{'vb/vg/vm':>11}{'t(s)':>7}")
    for r in sorted(rows, key=lambda r: -r["_w"])[:15]:
        viol = f"{r['vb']}/{r['vg']}/{r['vm']}"
        print(f"{r['key']:<28}{r['n']:>4}{r['cost']:>9.4f}{r['floor_cost']:>8.4f}"
              f"{r['cost'] / r['floor_cost']:>7.3f}{r['hgap']:>8.3f}"
              f"{r['agap']:>8.3f}{r['vrel']:>8.4f}{viol:>11}"
              f"{r['runtime']:>7.2f}")
    print()
    print(f"{'case':<28}{'n':>4}{'cost':>9}   (by raw cost)")
    for r in sorted(rows, key=lambda r: -r["cost"])[:10]:
        print(f"{r['key']:<28}{r['n']:>4}{r['cost']:>9.4f}")
    print()

    print("-" * 78)
    print("QUALITY DECOMPOSITION (weighted by exp(n/12), per-n means)")
    print("-" * 78)
    for field in ("hgap", "agap", "vrel"):
        o, _, om = _per_n_total(rows, field)
        iv = sum(math.exp((arec[i]['block_count'] - 120) / 12.0) *
                 arec[i][{"hgap": "hpwl_gap", "agap": "area_gap",
                          "vrel": "violations_relative"}[field]]
                 for i in range(100))
        iw = sum(math.exp((arec[i]['block_count'] - 120) / 12.0)
                 for i in range(100))
        print(f"  {field:<6} OOS {o:>9.4f}   in-set {iv / iw:>9.4f}   "
              f"delta {o - iv / iw:+.4f}")
    fl_v = _per_n_total(rows, "floor_vrel")[0]
    in_fv = sum(math.exp((floor_in[i]['n'] - 120) / 12.0) * floor_in[i]['vrel']
                for i in range(100)) / sum(math.exp((floor_in[i]['n'] - 120) / 12.0)
                                           for i in range(100))
    print(f"  label vrel (floor): OOS {fl_v:.4f}   in-set {in_fv:.4f}")
    print()

    print("-" * 78)
    print("RUNTIME (this 12-core box, portfolio wall per case)")
    print("-" * 78)
    print(f"  OOS    p50 {pct(rts, .5):.2f}s  p90 {pct(rts, .9):.2f}s  "
          f"max {rts[-1]:.2f}s  mean {statistics.mean(rts):.2f}s")
    print(f"  in-set p50 {pct(in_rts, .5):.2f}s  p90 {pct(in_rts, .9):.2f}s  "
          f"max {in_rts[-1]:.2f}s  mean {statistics.mean(in_rts):.2f}s")
    print()

    print("-" * 78)
    print("STRUCTURAL FEATURES (n-matched: both sets span n=21..120)")
    print("-" * 78)
    keys = ("fixed", "pre", "mibG", "cluG", "bnd", "b2bE", "p2bE")
    print(f"{'feature':<8}{'OOS mean':>10}{'OOS med':>9}{'in mean':>10}"
          f"{'in med':>9}{'OOS max':>9}{'in max':>8}")
    for k in keys:
        a = [r["feat"][k] for r in rows]
        b = [feat_in[i][k] for i in range(100)]
        print(f"{k:<8}{statistics.mean(a):>10.2f}{statistics.median(a):>9.1f}"
              f"{statistics.mean(b):>10.2f}{statistics.median(b):>9.1f}"
              f"{max(a):>9}{max(b):>8}")
    print()

    verdict = ("GREEN" if oos_total <= BAR_GREEN else
               "YELLOW" if oos_total <= BAR_YELLOW else "RED")
    rverdict = "GREEN" if abs(dratio) <= BAR_RATIO_PCT else "CHECK"
    print("=" * 78)
    print(f"VERDICT  raw {oos_total:.4f} vs bars {BAR_GREEN}/{BAR_YELLOW} "
          f"-> {verdict}")
    print(f"         floor-relative ratio delta {dratio:+.2f}% vs "
          f"+-{BAR_RATIO_PCT}% -> {rverdict}")
    print(f"         hard flags: "
          f"{'NONE' if not (infeas or fb or errs) else 'PRESENT (see top)'}")
    print("=" * 78)

    dump = dict(
        submission_name="M67D_oos_probe",
        total_score=oos_total,
        naive_total=naive,
        floor_total=floor_total,
        in_set_total=IN_SET_TOTAL,
        in_set_floor=in_floor,
        ratio_oos=ratio_oos, ratio_in_set=ratio_in, delta_ratio_pct=dratio,
        verdict=verdict, ratio_verdict=rverdict,
        sample=dict(seed=SEED, per_n=args.per_n, heavy_per_n=args.heavy_per_n,
                    workers=args.workers, n_cases=len(rows)),
        summary=dict(num_tests=len(rows),
                     num_feasible=sum(1 for r in rows if r["feasible"]),
                     num_fallback=len(fb), num_error=len(errs),
                     avg_cost=statistics.mean(r["cost"] for r in rows),
                     avg_runtime=statistics.mean(rts)),
        test_results=[{k: v for k, v in r.items()
                       if k not in ("positions", "_w")} for r in rows],
    )
    json.dump(dump, open(DUMP_JSON, "w"), indent=1)
    print(f"[report] wrote {DUMP_JSON.name}")
    return 0


def mode_ref(args):
    """Single-profile reference on both sets: portfolio/single isolates
    'the OOS cases are simply harder' from 'the portfolio+proxy overfit'."""
    os.environ["ICCAD_CONSTRUCTIVE_SINGLE"] = "1"
    opt = oc.MyOptimizer(verbose=False)
    assert opt._single, "single-profile mode did not engage"
    ref = _C.setdefault("ref", {})

    ev = _inset_dataset()
    todo = [i for i in range(100) if f"IN{i}" not in ref]
    if todo:
        print(f"[ref] in-set single profile: {len(todo)} cases")
        for k, i in enumerate(todo):
            lay = _inset_lay(ev, i)
            pos, dt, _e = _solve_one(opt, lay)
            m = _cost(pos, lay)
            ref[f"IN{i}"] = dict(_mt(m), n=lay["n"], runtime=dt)
            if (k + 1) % 20 == 0:
                _csave(); print(f"[ref]   {k + 1}/{len(todo)}", flush=True)
        _csave()

    rows = _rows(args)
    byf = {}
    for r in rows:
        key, L = r["key"].rsplit("/L", 1)
        if f"OOS{r['key']}" not in ref:
            byf.setdefault(key, []).append(int(L))
    if byf:
        print(f"[ref] OOS single profile: {sum(len(v) for v in byf.values())} cases")
        for key in sorted(byf):
            d = torch.load(_path_of(key))
            for L in sorted(byf[key]):
                lay = _load_case(d, L)
                lay["base"], _dev = _baseline_official(lay)
                pos, dt, _e = _solve_one(opt, lay)
                ref[f"OOS{key}/L{L}"] = dict(_mt(_cost(pos, lay)), n=lay["n"],
                                             runtime=dt)
            _csave()

    ir = [ref[f"IN{i}"] for i in range(100)]
    orr = [ref[f"OOS{r['key']}"] for r in rows if f"OOS{r['key']}" in ref]
    in_single = compute_total_score([r["cost"] for r in ir],
                                    [r["n"] for r in ir])
    oos_single = _per_n_total(orr)[0]
    oos_port = _per_n_total(rows)[0]
    print("=" * 78)
    print("M67-D  SINGLE-PROFILE REFERENCE")
    print("=" * 78)
    print(f"  in-set: single {in_single:.4f}  portfolio {IN_SET_TOTAL:.4f}  "
          f"ratio {IN_SET_TOTAL / in_single:.4f}")
    print(f"  OOS   : single {oos_single:.4f}  portfolio {oos_port:.4f}  "
          f"ratio {oos_port / oos_single:.4f}")
    print(f"  portfolio gain in-set {(1 - IN_SET_TOTAL / in_single) * 100:.2f}%  "
          f"vs OOS {(1 - oos_port / oos_single) * 100:.2f}%")
    print("  (similar gains => the portfolio/proxy generalizes; the raw delta "
          "is set difficulty, not overfit)")
    _csave()
    return 0


def mode_pool0(args):
    """Are the adaptive cuts overfit to the 100 validation cases? Every tier
    (M41/M42/M45 pool + M49/M50 REFINE band) was gated on THAT set under a
    strict selection-preserving rule. Re-solve the heavy band on both corpora
    with ICCAD_ADAPTIVE_POOL=0 (full 41 profiles, full REFINE) and compare:
    in-set the trade is known to cost +0.13% overall (1.3248 -> 1.3265)."""
    os.environ["ICCAD_ADAPTIVE_POOL"] = "0"
    os.environ["ICCAD_PROFILE_TIMEOUT"] = "600"
    opt = oc.MyOptimizer(verbose=False)
    st = _C.setdefault("pool0", {})
    lo, hi = _sel(args)
    sel = _selname(lo, hi)

    ev = _inset_dataset()
    _j, arec = _anchor()
    in_ids = [i for i in range(100) if _in_sel(arec[i]["block_count"], lo, hi)]
    todo = [i for i in in_ids if f"IN{i}" not in st]
    if todo:
        print(f"[pool0] in-set full-pool: {len(todo)} cases ({sel})")
        for k, i in enumerate(todo):
            lay = _inset_lay(ev, i)
            pos, dt, _e = _solve_one(opt, lay)
            st[f"IN{i}"] = dict(_mt(_cost(pos, lay)), n=lay["n"], runtime=dt)
            _csave()
            print(f"[pool0]   in {k + 1}/{len(todo)} n={lay['n']} "
                  f"cost={st[f'IN{i}']['cost']:.4f} t={dt:.1f}s", flush=True)

    rows = [r for r in _rows(args) if _in_sel(r["n"], lo, hi)]
    byf = {}
    for r in rows:
        if f"OOS{r['key']}" not in st:
            key, L = r["key"].rsplit("/L", 1)
            byf.setdefault(key, []).append(int(L))
    if byf:
        tot = sum(len(v) for v in byf.values())
        print(f"[pool0] OOS full-pool: {tot} cases ({sel})")
        k = 0
        for key in sorted(byf):
            d = torch.load(_path_of(key))
            for L in sorted(byf[key]):
                lay = _load_case(d, L)
                lay["base"], _dev = _baseline_official(lay)
                pos, dt, _e = _solve_one(opt, lay)
                st[f"OOS{key}/L{L}"] = dict(_mt(_cost(pos, lay)), n=lay["n"],
                                            runtime=dt)
                k += 1
            _csave()
            print(f"[pool0]   oos {k}/{tot}", flush=True)

    print("=" * 78)
    print(f"M67-D  ADAPTIVE-CUT OVERFIT TEST  ({sel}: shipped pool vs "
          f"ICCAD_ADAPTIVE_POOL=0)")
    print("=" * 78)
    ship_in = [dict(cost=arec[i]["cost"], n=arec[i]["block_count"])
               for i in in_ids]
    full_in = [st[f"IN{i}"] for i in in_ids]
    a = compute_total_score([r["cost"] for r in ship_in],
                            [r["n"] for r in ship_in])
    b = compute_total_score([r["cost"] for r in full_in],
                            [r["n"] for r in full_in])
    mv_in = sum(1 for x, y in zip(ship_in, full_in)
                if abs(x["cost"] - y["cost"]) > 1e-9)
    print(f"  in-set {sel}: shipped {a:.6f}   full-pool {b:.6f}   "
          f"tax {(a / b - 1) * 100:+.3f}%   movers {mv_in}/{len(in_ids)}")
    ship_o = [r for r in rows if f"OOS{r['key']}" in st]
    full_o = [dict(st[f"OOS{r['key']}"], key=r["key"]) for r in ship_o]
    a2 = _per_n_total(ship_o)[0]
    b2 = _per_n_total(full_o)[0]
    mv_o = sum(1 for x, y in zip(ship_o, full_o)
               if abs(x["cost"] - y["cost"]) > 1e-9)
    worse = sum(1 for x, y in zip(ship_o, full_o) if x["cost"] > y["cost"] + 1e-9)
    print(f"  OOS    {sel}: shipped {a2:.6f}   full-pool {b2:.6f}   "
          f"tax {(a2 / b2 - 1) * 100:+.3f}%   movers {mv_o}/{len(ship_o)} "
          f"(shipped worse on {worse})")
    rt_s = statistics.mean(r["runtime"] for r in ship_o)
    rt_f = statistics.mean(r["runtime"] for r in full_o)
    print(f"  wall  : shipped {rt_s:.2f}s   full-pool {rt_f:.2f}s "
          f"({rt_f / rt_s:.2f}x)")
    d = sorted(((y["cost"] - x["cost"], x["key"], x["n"], x["cost"], y["cost"])
                for x, y in zip(ship_o, full_o)), key=lambda t: t[0])
    print("  biggest per-case regressions from the cuts (full-pool better):")
    for dd, k_, n_, cs, cf in d[:8]:
        if dd >= -1e-9:
            break
        print(f"    {k_:<28} n={n_:3d} shipped {cs:.4f} full {cf:.4f} "
              f"{(cs / cf - 1) * 100:+.2f}%")
    _csave()
    return 0


# --------------------------------------------------------------------------- #
# M67-F: theta = the share of the OOS adaptive-cut tax owned by the POOL cuts   #
# --------------------------------------------------------------------------- #
#   shipped   = ICCAD_ADAPTIVE_POOL=1  (M41 swap + M42/M45 pool + M49/M50 REFINE)
#   restore   = ICCAD_M67F_RESTORE=1   (drop ONLY the M42/M45 pool layers)
#   norefine  = ICCAD_ADAPTIVE_REFINE=0 (drop ONLY the M49/M50 REFINE band)
#   full      = ICCAD_ADAPTIVE_POOL=0  (drop everything; mode_pool0's cache)
# theta_pool = (S - R_pool) / (S - F);  theta_refine = (S - R_norefine) / (S - F).
# M67-E: at 48 cores the wall is the max-setter and every M42/M45-dropped profile
# is cheaper than it, so restoring them is wall-free there (dW=+0.00%) => any
# theta_pool > 0 is a pure score gain (break-even theta* = 0, upper bound -2.11%).
_ARMS = {"pool": {"ICCAD_M67F_RESTORE": "1"},
         "refine": {"ICCAD_ADAPTIVE_REFINE": "0"}}


def _theta_gate_a(arm):
    """Knob self-check: the arm must change EXACTLY the intended layer. Runs
    before any solve (a wrong knob would burn 10 minutes of solves)."""
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' ' + extra) if extra else ''}")

    for k in _ARMS[arm]:
        os.environ.pop(k, None)
    off = {n: len(oc._pool_indices(n)) for n in (30, 50, 80, 105, 120)}
    off_be = {n: dict(oc._band_env(n)) for n in (80, 120)}
    os.environ.update(_ARMS[arm])
    on = {n: len(oc._pool_indices(n)) for n in (30, 50, 80, 105, 120)}
    on_be = {n: dict(oc._band_env(n)) for n in (80, 120)}
    print(f"  arm={arm}  env={_ARMS[arm]}")
    print(f"    pool  off {off}\n          on  {on}")
    print(f"    band  off {off_be}\n          on  {on_be}")
    chk("knob-off pool == shipped (35/35/26/13/13)",
        off == {30: 35, 50: 35, 80: 26, 105: 13, 120: 13}, str(off))
    chk("knob-off REFINE band == shipped (mid 8, big 4)",
        off_be == {80: {"ICCAD_REFINE_ITERS": "8"},
                   120: {"ICCAD_REFINE_ITERS": "4"}}, str(off_be))
    if arm == "pool":
        chk("restore pool == 35 on every band (41 - 6 swap)",
            set(on.values()) == {35}, str(on))
        chk("restore keeps the M49/M50 REFINE band", on_be == off_be, str(on_be))
        chk("restore adds exactly the two drop sets @n=120",
            set(oc._pool_indices(120)) - set(_shipped_pool(120))
            == set(oc._BIG_REDUNDANT_IDX))
        # mid band: the layer the (60,100] top-up measures is M45 tier-3 alone
        mid_drop = set()
        for _lo, _hi, _d in oc._M45_BAND_DROP:
            if _lo < 80 <= _hi:
                mid_drop = set(_d)
        chk("restore adds exactly _M45_BAND_DROP @n=80",
            set(oc._pool_indices(80)) - set(_shipped_pool(80)) == mid_drop,
            f"(tier-3 = {sorted(mid_drop)})")
    else:
        chk("norefine keeps the shipped pool", on == off, str(on))
        chk("norefine clears the REFINE band",
            on_be == {80: {}, 120: {}}, str(on_be))
    for k in _ARMS[arm]:
        os.environ.pop(k, None)
    return ok


def _shipped_pool(n):
    for k in ("ICCAD_M67F_RESTORE", "ICCAD_ADAPTIVE_REFINE", "ICCAD_ADAPTIVE_POOL"):
        v = os.environ.pop(k, None)
        if v is not None:
            os.environ[f"_SAVE_{k}"] = v
    p = oc._pool_indices(n)
    for k in ("ICCAD_M67F_RESTORE", "ICCAD_ADAPTIVE_REFINE", "ICCAD_ADAPTIVE_POOL"):
        v = os.environ.pop(f"_SAVE_{k}", None)
        if v is not None:
            os.environ[k] = v
    return p


def _theta(S, R, F):
    """(S-R)/(S-F): the fraction of the shipped-vs-full OOS gap that this arm
    recovers. None when the denominator is degenerate."""
    den = S - F
    return None if abs(den) < 1e-12 else (S - R) / den


def mode_restore(args):
    """M67-F Phase 1. Same 80 OOS heavy cases, same estimator and cache as
    mode_pool0; only the middle point (one layer restored) is new."""
    arm = args.arm
    print("=" * 78)
    print(f"GATE A  knob self-check (arm={arm})")
    print("=" * 78)
    if not _theta_gate_a(arm):
        print("GATE A FAILED - refusing to spend solves on a wrong knob")
        return 1

    os.environ.update(_ARMS[arm])
    os.environ["ICCAD_PROFILE_TIMEOUT"] = "600"       # wider pool oversubscribes
    opt = oc.MyOptimizer(verbose=False)
    st = _C.setdefault("m67f", {}).setdefault(arm, {})
    lo, hi = _sel(args)
    sel = _selname(lo, hi)

    # ---- Gate B: in-set window under the restored pool ---------------------
    # n>100 covers the M42 gate; a window reaching into (60,100] ALSO re-checks
    # the M45 tier-3 gate, since that is the layer the knob restores there.
    ev = _inset_dataset()
    _j, arec = _anchor()
    in_ids = [i for i in range(100) if _in_sel(arec[i]["block_count"], lo, hi)]
    todo = [i for i in in_ids if f"IN{i}" not in st]
    if todo:
        print(f"\n[{arm}] in-set: {len(todo)} cases ({sel})")
        for k, i in enumerate(todo):
            lay = _inset_lay(ev, i)
            pos, dt, _e = _solve_one(opt, lay)
            st[f"IN{i}"] = dict(_mt(_cost(pos, lay)), n=lay["n"], runtime=dt)
            _csave()
            print(f"[{arm}]   in {k + 1}/{len(todo)} n={lay['n']} "
                  f"cost={st[f'IN{i}']['cost']:.4f} t={dt:.1f}s", flush=True)

    print()
    print("=" * 78)
    gates = ("M42" if lo >= 100 else
             ("M45 tier-3" if hi <= 100 else "M42 + M45 tier-3"))
    expect = (f"{gates} strict gate: restore must never be WORSE" if arm == "pool"
              else "M49/M50 trade => movers expected")
    print(f"GATE B  in-set {sel}: {arm} vs shipped ({expect})")
    print("=" * 78)
    # Two different claims, and only one of them is an invariant:
    #  * restore WORSE than shipped  -> impossible (the proxy is oracle-min in
    #    sample, M31, so a superset pool is weakly better) => knob bug, STOP.
    #  * restore BETTER than shipped -> the drop set's strict selection-
    #    preserving gate has drifted. Measured 2026-07-22: it HAS, on the mid
    #    band. _M45_BAND_DROP (and _BIG_REDUNDANT_IDX) were derived from
    #    audit_cache.pkl = REFINE K=12 positions, but M49/M50 ship a K=4/K=8
    #    overlay on exactly those bands, so the gate was never re-proven under
    #    the config that ships. In-set case 64 (n=85): equal at K=12
    #    (1.3558352796522921 both sides), but at the shipped K=8 the cut costs
    #    +0.41%. Heavy band is still 20/20 equal. Reported, not fatal - it is a
    #    finding about the shipped constants, not a fault in this measurement.
    worse, better = [], []
    for i in in_ids:
        cs, cr = float(arec[i]["cost"]), float(st[f"IN{i}"]["cost"])
        if cr > cs + 1e-9 * max(abs(cs), 1.0):
            worse.append((i, arec[i]["block_count"], cs, cr))
        elif cr < cs - 1e-9 * max(abs(cs), 1.0):
            better.append((i, arec[i]["block_count"], cs, cr))
    bad = worse + better
    if arm == "pool":
        print(f"  [{'PASS' if not worse else 'FAIL'}] restore never worse "
              f"({len(worse)} worse) - the invariant")
        print(f"  [{'PASS' if not better else 'DRIFT'}] "
              f"{len(in_ids) - len(bad)}/{len(in_ids)} cases cost-equal "
              f"(rel 1e-9) - the {gates} gate as originally proven")
        for tag, lst in (("WORSE", worse), ("DRIFT", better)):
            for i, n, cs, cr in lst:
                print(f"    [{tag}] case {i:3d} n={n:3d} shipped {cs!r} "
                      f"restore {cr!r} {(cr / cs - 1) * 100:+.4f}%")
        if worse:
            print("  => restore lost to a subset of its own pool: the knob is "
                  "wrong. STOP.")
            _csave()
            return 1
        if better:
            wt = {i: math.exp(arec[i]["block_count"] / 12.0) for i in range(100)}
            dw = sum(wt[i] * (cr - cs) for i, _n, cs, cr in better)
            print(f"  => the {gates} gate has DRIFTED (derived at REFINE K=12, "
                  f"ships at K=4/K=8).\n     In-set cost of the drift: "
                  f"{100 * dw / sum(wt.values()) / IN_SET_TOTAL:+.4f}% of the "
                  f"local total. Continuing:\n     the OOS theta below is "
                  f"exactly the quantity this drift makes interesting.")
    else:
        a = compute_total_score([arec[i]["cost"] for i in in_ids],
                                [arec[i]["block_count"] for i in in_ids])
        b = compute_total_score([st[f"IN{i}"]["cost"] for i in in_ids],
                                [st[f"IN{i}"]["n"] for i in in_ids])
        print(f"  in-set {sel}: shipped {a:.6f}  {arm} {b:.6f}  "
              f"tax {(a / b - 1) * 100:+.3f}%  movers {len(bad)}/{len(in_ids)}")

    # ---- OOS arm ----------------------------------------------------------
    rows = [r for r in _rows(args) if _in_sel(r["n"], lo, hi)]
    byf = {}
    for r in rows:
        if f"OOS{r['key']}" not in st:
            key, L = r["key"].rsplit("/L", 1)
            byf.setdefault(key, []).append(int(L))
    if byf:
        tot = sum(len(v) for v in byf.values())
        print(f"\n[{arm}] OOS: {tot} cases ({sel})")
        t0, k = time.time(), 0
        for key in sorted(byf):
            d = torch.load(_path_of(key))
            for L in sorted(byf[key]):
                lay = _load_case(d, L)
                lay["base"], _dev = _baseline_official(lay)
                pos, dt, _e = _solve_one(opt, lay)
                st[f"OOS{key}/L{L}"] = dict(_mt(_cost(pos, lay)), n=lay["n"],
                                            runtime=dt)
                k += 1
            _csave()
            el = time.time() - t0
            print(f"[{arm}]   oos {k}/{tot} ({el:.0f}s, "
                  f"eta {el / max(k, 1) * (tot - k):.0f}s)", flush=True)
    _csave()
    return _theta_report(args, arm)


def _theta_report(args, arm):
    lo, hi = _sel(args)
    sel = _selname(lo, hi)
    p0 = _C.get("pool0", {})
    st = _C.get("m67f", {}).get(arm, {})
    rows = [r for r in _rows(args) if _in_sel(r["n"], lo, hi)
            and f"OOS{r['key']}" in p0 and f"OOS{r['key']}" in st]
    if not rows:
        print("[theta] no overlapping cases - run pool0 first")
        return 1
    trip = [dict(n=r["n"], key=r["key"], S=r["cost"], R=st[f"OOS{r['key']}"]["cost"],
                 F=p0[f"OOS{r['key']}"]["cost"], tS=r["runtime"],
                 tR=st[f"OOS{r['key']}"]["runtime"], tF=p0[f"OOS{r['key']}"]["runtime"])
            for r in rows]

    def tot(sub, field):
        return _per_n_total([dict(n=t["n"], cost=t[field]) for t in sub])[0]

    print()
    print("=" * 78)
    print(f"M67-F  THETA  (arm={arm}, {sel}, per-n averaged then officially "
          f"weighted)")
    print("=" * 78)
    # pilot subset = first draw per n (the sample is prefix-stable in K)
    first = {}
    for key, L, n in _sample(_index(args.workers), args.per_n, 1)[0]:
        first[f"{key}/L{L}"] = n
    pilot = [t for t in trip if t["key"] in first]
    out = {}
    for tag, sub in ((f"theta_{len(pilot)} (pilot, checkpoint only)", pilot),
                     (f"theta_{len(trip)} (VERDICT SAMPLE)", trip)):
        if not sub:
            continue
        S, R, F = tot(sub, "S"), tot(sub, "R"), tot(sub, "F")
        th = _theta(S, R, F)
        mv = sum(1 for t in sub if abs(t["S"] - t["R"]) > 1e-9)
        better = sum(1 for t in sub if t["R"] < t["S"] - 1e-9)
        print(f"  {tag}   [{len(sub)} cases]")
        print(f"    shipped {S:.6f}   {arm} {R:.6f}   full {F:.6f}")
        print(f"    denominator (shipped vs full) {(S / F - 1) * 100:+.3f}%   "
              f"arm recovers {(S / R - 1) * 100:+.3f}%")
        print(f"    theta = {th if th is None else round(th, 4)}   "
              f"movers {mv}/{len(sub)} ({better} better, {mv - better} worse)")
        out[tag.split()[0]] = dict(cases=len(sub), S=S, R=R, F=F, theta=th,
                                   movers=mv, better=better)
    # verdict on theta_80 (pre-registered in the plan BEFORE the run)
    S, R, F = tot(trip, "S"), tot(trip, "R"), tot(trip, "F")
    th, den = _theta(S, R, F), (S / F - 1) * 100
    if den < 1.0:
        verdict = "UNRELIABLE (denominator < 1.0%)"
    elif th is None:
        verdict = "UNRELIABLE (degenerate denominator)"
    elif th < 0:
        verdict = "RED-PLUS (negative: the bigger pool mis-selects OOS)"
    elif th <= 0.10:
        verdict = "RED (<=0.10) - close it, keep M42/M45"
    elif th < 0.30:
        verdict = "YELLOW (0.10..0.30) - record, no Phase 2"
    else:
        verdict = "GREEN (>=0.30) - Phase 2 (multi-core wall)"
    print(f"\n  PRE-REGISTERED VERDICT (theta_{len(trip)}): {verdict}")
    if lo != 100 or hi < 10 ** 9:
        print("  NOTE: the >=0.30 bar was pre-registered for n>100, where the "
              "index-based\n        restore is wall-free @48c. It is NOT free "
              "on (60,100] (dW +2.34%),\n        so the ship decision there is "
              "m67e_rf48.py's wall-aware break-even\n        theta*, not this "
              "bar. theta here only measures the quality debt.")

    # ---- per-band decomposition -------------------------------------------
    # theta is a ratio inside a band; wContr says how much of the OFFICIAL total
    # that band can move (validation weight profile: one case per n in [21,120],
    # independent of how many draws the sample took).
    wall_n = sum(math.exp(n / 12.0) for n in range(21, 121))
    bands = [b for b in BANDS if any(_in_sel(t["n"], b[1], b[2]) for t in trip)]
    if len(bands) > 1 or lo != 100 or hi < 10 ** 9:
        print(f"\n  per-band decomposition ({arm}):")
        print(f"    {'band':>12} {'cases':>5} {'wContr':>7} {'shipped':>9} "
              f"{'restore':>9} {'full':>9} {'S/F-1':>8} {'S/R-1':>8} "
              f"{'theta':>7} {'movers':>7}")
        for _b, blo, bhi in bands:
            sub = [t for t in trip if _in_sel(t["n"], blo, bhi)]
            if not sub:
                continue
            bS, bR, bF = tot(sub, "S"), tot(sub, "R"), tot(sub, "F")
            bth = _theta(bS, bR, bF)
            bmv = sum(1 for t in sub if abs(t["S"] - t["R"]) > 1e-9)
            bbet = sum(1 for t in sub if t["R"] < t["S"] - 1e-9)
            wc = sum(math.exp(n / 12.0) for n in range(21, 121)
                     if _in_sel(n, blo, bhi)) / wall_n
            out[f"band_{blo}_{bhi}"] = dict(
                cases=len(sub), wcontr=wc, S=bS, R=bR, F=bF, theta=bth,
                movers=bmv, better=bbet, worse=bmv - bbet,
                denominator_pct=(bS / bF - 1) * 100,
                recovered_pct=(bS / bR - 1) * 100)
            print(f"    {_selname(blo, bhi):>12} {len(sub):>5} {100 * wc:>6.1f}% "
                  f"{bS:>9.4f} {bR:>9.4f} {bF:>9.4f} "
                  f"{(bS / bF - 1) * 100:>+7.3f}% {(bS / bR - 1) * 100:>+7.3f}% "
                  f"{'  n/a' if bth is None else f'{bth:>7.4f}'} "
                  f"{f'{bbet}/{bmv - bbet}':>7}")
        print("    (movers column = better/worse; theta = (S-R)/(S-F) inside "
              "the band)")
    print(f"  wall @12c (NOT extrapolable to 48c, see report): shipped "
          f"{statistics.mean(t['tS'] for t in trip):.2f}s  {arm} "
          f"{statistics.mean(t['tR'] for t in trip):.2f}s  full "
          f"{statistics.mean(t['tF'] for t in trip):.2f}s")
    d = [((t["R"] - t["S"]) / t["S"] * 100, t)
         for t in sorted(trip, key=lambda t: (t["R"] - t["S"]) / t["S"])]
    print(f"\n  biggest per-case moves ({arm} vs shipped, - = {arm} better):")
    for sign, sl in ((-1, d[:6]), (1, list(reversed(d[-6:])))):
        for pc, t in sl:
            if sign * pc <= 1e-7:
                continue
            print(f"    {t['key']:<28} n={t['n']:3d} shipped {t['S']:.4f} "
                  f"{arm} {t['R']:.4f} full {t['F']:.4f}  {pc:+.2f}%")
    # cross-arm consistency (needs both arms measured)
    other = "refine" if arm == "pool" else "pool"
    ost = _C.get("m67f", {}).get(other, {})
    cross = None
    if all(f"OOS{t['key']}" in ost for t in trip):
        R2 = tot([dict(n=t["n"], R2=ost[f"OOS{t['key']}"]["cost"]) for t in trip],
                 "R2")
        th2 = _theta(S, R2, F)
        cross = {other: th2, "sum": None if (th is None or th2 is None) else th + th2}
        print(f"\n  cross-arm: theta_{arm} {th:.4f} + theta_{other} {th2:.4f} "
              f"= {th + th2:.4f}  (vs 1.0 => "
              f"{'additive' if abs(th + th2 - 1) < 0.15 else 'INTERACTION'})")
    dump = dict(arm=arm, lo=lo, hi=(None if hi >= 10 ** 9 else hi), window=sel,
                cases=len(trip), theta=th, denominator_pct=den,
                verdict=verdict, totals=dict(shipped=S, arm=R, full=F),
                summary=out, cross_arm=cross,
                wall12c=dict(shipped=statistics.mean(t["tS"] for t in trip),
                             arm=statistics.mean(t["tR"] for t in trip),
                             full=statistics.mean(t["tF"] for t in trip)),
                per_case=[dict(key=t["key"], n=t["n"], shipped=t["S"], arm=t["R"],
                               full=t["F"], t_shipped=t["tS"], t_arm=t["tR"],
                               t_full=t["tF"]) for t in trip])
    # default window (n>100) keeps the Phase 1 filename; any other window gets
    # its own file so the heavy-band verdict is never clobbered
    suffix = "" if (lo == 100 and hi >= 10 ** 9) else f"_{lo}_{'inf' if hi >= 10 ** 9 else hi}"
    path = _DIR / f"results_M67F_theta_{arm}{suffix}.json"
    json.dump(dump, open(path, "w"), indent=1)
    print(f"\n  wrote {path.name}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate0", "run", "report", "ref", "pool0",
                                     "restore"])
    ap.add_argument("--arm", choices=sorted(_ARMS), default="pool")
    ap.add_argument("--pool0-lo", type=int, default=100, dest="pool0_lo")
    ap.add_argument("--pool0-hi", type=int, default=0, dest="pool0_hi",
                    help="upper case-size bound for pool0/restore (0 = none); "
                         "--pool0-lo 60 --pool0-hi 100 scores the M45 tier-3 "
                         "mid band on its own")
    ap.add_argument("--per-n", type=int, default=2, dest="per_n")
    ap.add_argument("--heavy-per-n", type=int, default=4, dest="heavy_per_n")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    _cload(_sig(args))
    return {"gate0": mode_gate0, "run": mode_run, "report": mode_report,
            "ref": mode_ref, "pool0": mode_pool0,
            "restore": mode_restore}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
