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
          results_M74_default.json + training-side baseline sanity
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
import itertools
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
# M75: the in-set anchor was two generations stale. It pointed at
# results_shipped_m51.json (a misleading filename: its CONTENT is M71,
# 1.305389893450635) while IN_SET_TOTAL still held the pre-M71 1.326473...,
# and the tree has been M74 since 2026-07-30. The OOS baseline in
# m67_oos_cache.pkl IS already M74 (its _sig() pins exe md5 a576feb61079 plus
# the regenerated drop constants), so leaving the anchor behind would have made
# Gate B report M74's own 14 movers -- 47/49/51/55/62/67/68/77/79/85/87/89/91/96,
# i.e. exactly the cluster-heavy cases an M71-knob arm is supposed to move -- as
# if the arm had caused them. Cache-neutral: _sig() does not read either name.
ANCHOR_JSON = _DIR / "results_M74_default.json"
DUMP_JSON = _DIR / "results_M67D_oos.json"

IN_SET_TOTAL = 1.293461035226291        # results_M74_default.json total_score
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
    # M74: the cached records are whole opt.solve() outputs, so EVERY constant
    # that steers the shipped portfolio is part of the solver identity. The old
    # key covered only the binary and _PROFILES, so regenerating the adaptive
    # drop sets or retuning the REFINE band would have silently reused positions
    # from the previous constants (the same failure mode audit_cache.pkl had).
    # M76: anchored on the SHIPPED PREFIX, not the whole list. _PROFILES carries
    # two appended, gated-off tiers (M72's 4, M76's 41 escape twins) and neither
    # can change a single cached solve while its gate is off -- _pool_indices()
    # never emits those indices. Keying on the whole list made every port of a
    # gated tier throw away 240 cached cases for no reason (the same fix M73's
    # ship chain applied to audit_cache.pkl). Anything that DOES steer the shipped
    # portfolio is still in the key below.
    _ship = oc._PROFILES[:oc._M55_BASE_LEN]
    return repr(("m67d", VERSION, SEED, _exe_md5(), len(_ship),
                 repr(_ship),
                 sorted(oc._BIG_REDUNDANT_IDX),
                 repr(sorted(oc._M45_BAND_DROP)),
                 repr(sorted(oc._M45_LOWCORE_DROP)),
                 oc._M45_CORES_MAX, oc._M67F_CORES_MIN,
                 repr(sorted(oc._M49_REFINE_BAND)),
                 repr(sorted(oc._M50_REFINE_LOWCORE)),
                 repr(sorted(oc._M71_ENV.items()))))


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
    """Atomic cache write, with a retry on the Windows replace.

    M76: os.replace() raised PermissionError [WinError 5] once mid-run and killed
    an arm 64 solves into its in-set pass. On Windows the destination can be
    transiently locked by a scanner or an indexer; the write itself had succeeded,
    so the only correct response is to wait and retry rather than lose the run.
    Re-raises after the last attempt so a REAL permission problem is still loud."""
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_C, f, protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(6):
        try:
            os.replace(tmp, CACHE_PATH)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))


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
    # M74: anchor on the SHIPPED prefix, not on len(_PROFILES). M72 appended the
    # four gated-off _M55_EXTRA profiles (41 -> 45), which made the old literal
    # `== 41` report a spurious FAIL. What this actually guards is "the shipped
    # pool is the 41-profile one, not the 84-profile L1 pool".
    chk("_M55_BASE_LEN == 41 (shipped pool, not L1)", oc._M55_BASE_LEN == 41,
        f"got {oc._M55_BASE_LEN} (len(_PROFILES)={len(oc._PROFILES)})")
    for n in (30, 80, 120):
        print(f"    n={n:3d}  pool={len(oc._pool_indices(n)):2d}  "
              f"band_env={oc._band_env(n) or '{}'}")
    chk("adaptive pool active on n=120",
        len(oc._pool_indices(120)) < oc._M55_BASE_LEN)
    chk("constructive binary present", Path(oc._BIN).exists(), f"md5 {_exe_md5()}")

    print()
    print("=" * 78)
    print(f"GATE 0-b  in-set bit-exact reproduction vs {ANCHOR_JSON.name}")
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
    print(f"  in-set ({ANCHOR_JSON.name})              : {IN_SET_TOTAL:.4f}")
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
#
# M72 arms (2026-07-30). The teammate's b716753 tier: the same six cluster-boundary
# knobs our shipped M71 exports GLOBALLY, packaged as 4 extra PROFILES instead, so
# a case that M71 hurts can escape to a knob-free profile. Two arms because our
# shipped baseline already carries M71 globally, unlike theirs:
#   m55  = tier ON, M71 global OFF  -> reproduces their M72 exactly (vs a knob-free
#                                     base, so it answers "does their number hold")
#   m55x = tier ON, M71 global ON   -> the union; the only form that is a candidate
#                                     for OUR tar, since M71 is already shipped
#
# M75 arms (2026-07-31). The four cluster-boundary knobs M71 ported but never
# measured, each as a GLOBAL OVERLAY on top of shipped M71 -- deliberately NOT
# the pool-tier form, which M72 already measured at -1.418% OOS for us. These
# are pure C++ env vars, so they reach the binary a different way than the other
# arms: _run_profile does `env = dict(os.environ); env.update(env_over)`, and no
# shipped profile (_PROFILES[:_M55_BASE_LEN]) sets any of these four keys, so
# putting them in os.environ IS a global overlay. Gate A asserts exactly that.
_M75_KNOBS = {"m71corner": "ICCAD_CLUSTER_BND_CORNER",
              "m71permute": "ICCAD_CLUSTER_BND_PERMUTE",
              "m71repack": "ICCAD_ANCHORED_BND_REPACK",
              "m71slide": "ICCAD_HPWL_SAFE_CLUSTER_SLIDE"}

# M76 arms (2026-08-01). The teammate's M73 knob-OFF ESCAPE tier (7403758): the
# MIRROR of M72 -- instead of adding knob-ON profiles to a knob-off base, it adds
# knob-OFF twins to our knob-ON base, so the 17 cases M71 regressed can escape.
# Offline (audit_cache_ship.pkl x audit_cache_esc.pkl, m76_escape_probe.py) the
# in-set picture at 48c pools is:
#   src=(2,22,23,25) all bands  quality +0.243%  dRF@48c +0.088%  NET +0.155%
#   src=(2,22,23,25) n>100      quality +0.191%  dRF@48c +0.020%  NET +0.171%
#   src=(21,23,2,22) n>100      quality +0.318%  dRF@48c +0.020%  NET +0.298%
# The third is our own greedy under M74 and is therefore an IN-SAMPLE fit on 20
# cases; these arms exist to find out which of the three survives out of sample.
_M76_SRC_X = "21,23,2,22"

_ARMS = {"pool": {"ICCAD_M67F_RESTORE": "1"},
         "refine": {"ICCAD_ADAPTIVE_REFINE": "0"},
         "m55": {"ICCAD_M55_POOL": "1", "ICCAD_M71": "0"},
         "m55x": {"ICCAD_M55_POOL": "1"},
         "m73": {"ICCAD_M73_ESCAPE": "1"},
         "m73big": {"ICCAD_M73_ESCAPE": "1", "ICCAD_M73_MIN_N": "100"},
         "m73x": {"ICCAD_M73_ESCAPE": "1", "ICCAD_M73_MIN_N": "100",
                  "ICCAD_M73_SRC": _M76_SRC_X}}
_M76_ARMS = frozenset({"m73", "m73big", "m73x"})
_ARMS.update({a: {k: "1"} for a, k in _M75_KNOBS.items()})
# Combination arms for the M75 phase-2 pairwise matrix + full union. Named by
# joining the single-arm names with "_" so the knob set is readable from the
# filename of results_M72_ab_<arm>_<lo>_<hi>.json.
for _n in range(2, len(_M75_KNOBS) + 1):
    for _combo in itertools.combinations(sorted(_M75_KNOBS), _n):
        _ARMS["_".join(_combo)] = {_M75_KNOBS[a]: "1" for a in _combo}
# M78 arms (2026-08-03). The candidate-set second path: M71 enriched the
# candidate set + ordering of the PURE-MOVABLE cluster path (make_group_item);
# these knobs do the same species of thing to the two paths it never touched --
# the MIXED (preplaced+movable) anchored first-pass and the generic
# item_candidates.  They live in constructive_m78.cpp only, so these arms are
# usable ONLY with --arm-bin constructive_m78.exe; without it the shipped binary
# ignores the env var and the arm silently reports a flat zero (gate A's liveness
# check catches that).  Same C++-only global-overlay shape as the M75 arms.
_M78_KNOBS = {"m78anchord1":   "ICCAD_M78_ANCH_ORD",
              "m78anchcenter": "ICCAD_M78_ANCH_CENTER",
              "m78anchcross":  "ICCAD_M78_ANCH_CROSS",
              "m78itemcenter": "ICCAD_M78_ITEM_CENTER",
              "m78itemcross":  "ICCAD_M78_ITEM_CROSS",
              "m78tb1":        "ICCAD_M78_TIEBREAK"}
_ARMS.update({a: {k: "1"} for a, k in _M78_KNOBS.items()})
_ARMS["m78anchord2"] = {"ICCAD_M78_ANCH_ORD": "2"}
_ARMS["m78anchord3"] = {"ICCAD_M78_ANCH_ORD": "3"}
_ARMS["m78anchord4"] = {"ICCAD_M78_ANCH_ORD": "4"}
_ARMS["m78tb2"] = {"ICCAD_M78_TIEBREAK": "2"}
_ARMS["m78tb3"] = {"ICCAD_M78_TIEBREAK": "3"}

_M75_ARMS = frozenset(_ARMS) - {"pool", "refine", "m55", "m55x"} - _M76_ARMS

# Liveness screen cases: a spread over the three weight bands, biased to cases
# that carry the cluster structures these knobs act on. The screen is PER
# PROFILE, not per portfolio -- see _m75_liveness.
_M75_LIVE_IDS = (3, 26, 54, 76, 89, 99)


def _m75_liveness(env_over):
    """Does the knob change what the BINARY emits, for any profile in the pool?

    Deliberately not a portfolio-level comparison. A knob can change several
    candidates and still lose the proxy argmin, so comparing final positions
    reports 'no difference' for a knob that is demonstrably doing something
    (measured 2026-07-31: CLUSTER_BND_PERMUTE changes profile #26 on in-set
    case 89 and 5 profiles on case 76, yet the portfolio winner is unchanged on
    both). Per-profile difference is the NECESSARY condition: if no profile's
    output moves, the portfolio result is bit-identical and an OOS arm on that
    knob is guaranteed to measure exactly 0.000%.

    Passing the knob through env_over (not os.environ) keeps this independent
    of the propagation path, so a FAIL here means the mechanism is inert, not
    that the plumbing is broken.
    """
    from optimizer_claude import _serialize_input
    ev = _inset_dataset()
    # _theta_gate_a has already pushed the arm into os.environ, and _run_profile
    # starts from `dict(os.environ)` -- so without this the knob would be ON for
    # BOTH sides of the comparison and every arm would report a flat zero.
    saved = {k: os.environ.pop(k, None) for k in env_over}
    detail, nlive, npair = {}, 0, 0
    try:
        for cid in _M75_LIVE_IDS:
            lay = _inset_lay(ev, cid)
            n = lay["n"]
            # M78 FIX (2026-08-03): this used to pass target_positions=None, which
            # strips EVERY preplaced (x,y,w,h) and fixed (w,h) from the binary's
            # input. With no preplaced blocks there are no MIXED clusters at all,
            # so any knob acting on the anchored first-pass has an EMPTY antecedent
            # here BY CONSTRUCTION and reports a false 0/210 -- exactly the "gate
            # measured a different configuration than the one deployed" family.
            # The M75 knobs were cluster-internal so it never showed. Use the same
            # masked otp _solve_one() feeds the real harness (preplaced/fixed are
            # hard-constraint INPUTS, not labels).
            otp = build_opt_target_pos(lay["tp"], lay["cons"], n)
            inp = _serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp,
                                   gnn_hint=None)
            band = dict(oc._band_env(n))
            moved = 0
            idxs = oc._pool_indices(n)
            for i in idxs:
                base = dict(oc._PROFILES[i], **band)
                base.update(oc._m71_env())
                a = oc._run_profile(base, inp, n)
                b = oc._run_profile(dict(base, **env_over), inp, n)
                # A launch failure returns None on BOTH sides and would otherwise
                # be indistinguishable from an inert knob (measured 2026-08-03: a
                # relative --arm-bin produced a clean-looking 0/210).
                assert a is not None and b is not None, (
                    f"_run_profile returned None on case {cid} profile {i} "
                    f"(binary {oc._BIN}) -- this is a plumbing failure, not a "
                    f"dead knob")
                moved += (a != b)
            detail[cid] = f"{moved}/{len(idxs)}p"
            npair += len(idxs)
            nlive += moved
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    return {"nlive": nlive, "npair": npair, "detail": detail}


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
    # M74: mid pool 26 -> 20 (tier-3 regenerated, 9 -> 15 dropped) and mid REFINE
    # K 8 -> 6 (K=6 weakly dominates K=8 at every projected cell). tier-3 is also
    # cores-gated now, so the expected mid size depends on THIS box: 20 where the
    # tier fires (<= _M45_MID_CORES_MAX), 35 where it does not (e.g. the 48c
    # grader, which is the whole point of the gate).
    _t3_on = oc._effective_cores() <= oc._M45_MID_CORES_MAX
    _mid_exp = 20 if _t3_on else 35
    # M76: the HEAVY expectation has to be cores-aware for the same reason. It was
    # hard-coded to 13 (the M42 cut), which is only the low-core shape — at >=40
    # detected cores tier-5 puts _BIG_REDUNDANT_IDX back and the heavy pool is 35.
    # Without this, --force-cores 48 (the grader's shape, and the one that decides
    # a ship) fails Gate A on a correct pool.
    _t5_on = oc._effective_cores_hi() >= oc._M67F_CORES_MIN
    _big_exp = 35 if _t5_on else 13
    chk(f"knob-off pool == shipped (35/35/{_mid_exp}/{_big_exp}/{_big_exp}, "
        f"tier-3 {'ON' if _t3_on else 'OFF'} / tier-5 "
        f"{'ON' if _t5_on else 'OFF'} at {oc._effective_cores()} cores)",
        off == {30: 35, 50: 35, 80: _mid_exp, 105: _big_exp, 120: _big_exp},
        str(off))
    chk("knob-off REFINE band == shipped (mid 6, big 4)",
        off_be == {80: {"ICCAD_REFINE_ITERS": "6"},
                   120: {"ICCAD_REFINE_ITERS": "4"}}, str(off_be))
    if arm == "pool":
        chk("restore pool == 35 on every band (41 - 6 swap)",
            set(on.values()) == {35}, str(on))
        chk("restore keeps the M49/M50 REFINE band", on_be == off_be, str(on_be))
        chk("restore adds exactly the two drop sets @n=120",
            set(oc._pool_indices(120)) - set(_shipped_pool(120))
            == set(oc._BIG_REDUNDANT_IDX))
        # mid band: the layer the (60,100] top-up measures is M45 tier-3 alone.
        # M74: tier-3 is cores-gated, so on a box above _M45_MID_CORES_MAX it is
        # already off and `restore` has nothing left to add there.
        mid_drop = set()
        if oc._effective_cores() <= oc._M45_MID_CORES_MAX:
            for _lo, _hi, _d in oc._M45_BAND_DROP:
                if _lo < 80 <= _hi:
                    mid_drop = set(_d)
        chk("restore adds exactly _M45_BAND_DROP @n=80",
            set(oc._pool_indices(80)) - set(_shipped_pool(80)) == mid_drop,
            f"(tier-3 = {sorted(mid_drop) or 'off at this core count'})")
    elif arm in ("m55", "m55x"):
        # The tier must add exactly its own 4 indices, on every band, and must not
        # disturb the M41/M42/M45 layers or the REFINE overlay.
        chk("tier adds exactly 4 profiles on every band",
            all(on[n] == off[n] + 4 for n in off), f"{off} -> {on}")
        chk("added indices are exactly the M55 tier",
            all(set(oc._pool_indices(n)) - set(_shipped_pool(n))
                == set(range(oc._M55_BASE_LEN, len(oc._PROFILES)))
                for n in (30, 80, 120)))
        chk("tier keeps the M49/M50 REFINE band", on_be == off_be, str(on_be))
        # ADAPTIVE_POOL=0 must NOT leak the tier (the teammate's port does; ours
        # reads the gate before the early return). Checked with the knob ON, since
        # that is the only state where a leak could hide.
        os.environ["ICCAD_ADAPTIVE_POOL"] = "0"
        f_on = len(oc._pool_indices(120))
        os.environ["ICCAD_M55_POOL"] = "0"
        f_off = len(oc._pool_indices(120))
        os.environ["ICCAD_M55_POOL"] = "1"
        os.environ.pop("ICCAD_ADAPTIVE_POOL", None)
        chk("ADAPTIVE_POOL=0 pool tracks the gate (no leak when off)",
            f_off == oc._M55_BASE_LEN and f_on == len(oc._PROFILES),
            f"off {f_off} on {f_on}")
        # M71 global overlay: OFF for the m55 arm (their baseline), ON for m55x.
        want = {} if arm == "m55" else dict(oc._M71_ENV)
        chk(f"_m71_env() == {'{}' if arm == 'm55' else 'M71 knobs'}",
            oc._m71_env() == want, str(oc._m71_env()))
        # RUNTIME flip works (an import-time gate would make this arm a silent
        # no-op and the run would report a fake 'zero difference').
        os.environ["ICCAD_M55_POOL"] = "0"
        flipped_off = len(oc._pool_indices(120))
        os.environ["ICCAD_M55_POOL"] = "1"
        chk("RUNTIME flip works (gate resolves per call)",
            flipped_off == off[120] and len(oc._pool_indices(120)) == on[120])
    elif arm in _M76_ARMS:
        env = _ARMS[arm]
        src = tuple(int(x) for x in
                    env.get("ICCAD_M73_SRC",
                            ",".join(str(s) for s in oc._M73_SRC)).split(","))
        min_n = int(env.get("ICCAD_M73_MIN_N", "0"))

        def _want(n):
            """Escape indices this band should gain. The M41 swap filter is
            CONTENT-based, so it removes the twin of a swap profile too — spelling
            that out here rather than assuming len(src) is what makes this gate
            catch a source list that silently loses a member."""
            if n <= min_n:
                return set()
            return {oc._M73_BASE + h for h in src
                    if not ("ICCAD_ORDER_SWAP" in oc._PROFILES[h]
                            or "ICCAD_ORDER_MOVE" in oc._PROFILES[h])}

        chk(f"tier adds exactly its escape indices (src={src}, min_n={min_n})",
            all(set(oc._pool_indices(n)) - set(_shipped_pool(n)) == _want(n)
                for n in off),
            f"{off} -> {on}")
        chk("tier keeps the M49/M50 REFINE band", on_be == off_be, str(on_be))
        # THE MECHANISM: an escape index must NOT receive the M71 overlay, while
        # every shipped index still must. Without this the tier is a pure duplicate
        # of its host and buys nothing but wall.
        esc_i = sorted(_want(120))
        ship_i = [i for i in oc._pool_indices(120) if i < oc._M55_BASE_LEN]
        knobs = set(oc._M71_ENV)
        chk("escape indices get NO M71 knobs (this is the mechanism)",
            bool(esc_i) and all(not (knobs & set(oc._profile_env(i, 120)))
                                for i in esc_i),
            f"escape={esc_i[:4]} env={oc._profile_env(esc_i[0], 120) if esc_i else {}}")
        chk("shipped indices still get the M71 knobs",
            all(knobs <= set(oc._profile_env(i, 120)) for i in ship_i))
        # ADAPTIVE_POOL=0 must NOT leak the tier (the teammate's port does; ours
        # reads the gate before the early return).
        os.environ["ICCAD_ADAPTIVE_POOL"] = "0"
        f_on = len(oc._pool_indices(120))
        os.environ["ICCAD_M73_ESCAPE"] = "0"
        f_off = len(oc._pool_indices(120))
        os.environ["ICCAD_M73_ESCAPE"] = "1"
        os.environ.pop("ICCAD_ADAPTIVE_POOL", None)
        chk("ADAPTIVE_POOL=0 pool tracks the gate (no leak when off)",
            f_off == oc._M55_BASE_LEN and f_on == oc._M55_BASE_LEN + len(_want(120)),
            f"off {f_off} on {f_on}")
        # RUNTIME flip works (an import-time gate would make this arm a silent
        # no-op and the run would report a fake 'zero difference').
        os.environ["ICCAD_M73_ESCAPE"] = "0"
        flipped_off = len(oc._pool_indices(120))
        os.environ["ICCAD_M73_ESCAPE"] = "1"
        chk("RUNTIME flip works (gate resolves per call)",
            flipped_off == off[120] and len(oc._pool_indices(120)) == on[120])
    elif arm in _M75_ARMS:
        # A pure C++ knob must move the BINARY and nothing on the Python side.
        chk("knob leaves every pool untouched (C++-only arm)", on == off,
            f"{off} -> {on}")
        chk("knob leaves the REFINE band untouched", on_be == off_be, str(on_be))
        # We are measuring ON TOP OF shipped M71, not instead of it.
        chk("_m71_env() still exports the shipped M71 knobs",
            oc._m71_env() == dict(oc._M71_ENV), str(oc._m71_env()))
        # _run_profile does `env = dict(os.environ); env.update(env_over)`, so a
        # profile that sets the same key would WIN over os.environ and the
        # overlay would silently not be global. The four _M55_EXTRA profiles do
        # carry these keys, which is exactly why this checks the shipped prefix.
        clash = sorted({k for k in _ARMS[arm]
                        for p in oc._PROFILES[:oc._M55_BASE_LEN] if k in p})
        chk("no shipped profile overrides the knob (overlay really is global)",
            not clash, str(clash))
        # LIVENESS. Without this a knob that never fires on this corpus reports
        # "0.000%, 0 movers" -- indistinguishable from a clean RED. Cached per
        # (arm, exe md5) so re-reporting a narrower band costs nothing.
        # v3: the M78 fixes changed BOTH what this screen feeds the binary
        # (target_positions) and how --arm-bin is resolved, so any entry cached
        # under an earlier revision is a measurement of a different thing.
        # 2026-08-03: a v2 entry written by a still-broken run was silently reused
        # by the next one -- bump this tag on EVERY change to the screen's inputs.
        lk = f"{arm}@{_exe_md5()}@v3"
        live = _C.setdefault("m75_live", {}).get(lk)
        if live is None:
            live = _m75_liveness(_ARMS[arm])
            _C["m75_live"][lk] = live
            _csave()
        chk("knob is LIVE (some profile's binary output changes)",
            live["nlive"] > 0,
            f"{live['nlive']}/{live['npair']} (case,profile) pairs move; "
            f"per-case {live['detail']}")
    else:
        chk("norefine keeps the shipped pool", on == off, str(on))
        chk("norefine clears the REFINE band",
            on_be == {80: {}, 120: {}}, str(on_be))
    for k in _ARMS[arm]:
        os.environ.pop(k, None)
    return ok


def _shipped_pool(n):
    for k in ("ICCAD_M67F_RESTORE", "ICCAD_ADAPTIVE_REFINE", "ICCAD_ADAPTIVE_POOL",
              "ICCAD_M55_POOL", "ICCAD_M73_ESCAPE", "ICCAD_M73_MIN_N",
              "ICCAD_M73_SRC"):
        v = os.environ.pop(k, None)
        if v is not None:
            os.environ[f"_SAVE_{k}"] = v
    p = oc._pool_indices(n)
    for k in ("ICCAD_M67F_RESTORE", "ICCAD_ADAPTIVE_REFINE", "ICCAD_ADAPTIVE_POOL",
              "ICCAD_M55_POOL", "ICCAD_M73_ESCAPE", "ICCAD_M73_MIN_N",
              "ICCAD_M73_SRC"):
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
    if arm in ("pool", "m55x") or arm in _M76_ARMS:
        # Superset of the shipped pool with the same per-profile env => the proxy
        # (oracle-min in-sample) can only weakly improve. Worse => knob bug.
        # M76: the escape tier is such a superset — every shipped index keeps its
        # exact overlay and the knob-off twins are ADDED — so it belongs here.
        gates = ("M72 tier" if arm == "m55x"
                 else "M76 escape tier" if arm in _M76_ARMS else gates)
        expect = f"{gates} strict gate: {arm} must never be WORSE"
    elif arm == "m55":
        # NOT a superset: this arm also turns the M71 global overlay OFF, so it is
        # the teammate's knob-free base + their tier. Movers both ways are expected.
        expect = "M71 global OFF + M72 tier => movers both ways expected"
    elif arm in _M75_ARMS:
        # NOT a pool superset: a global C++ knob changes what EVERY profile in
        # the pool emits, so the proxy's oracle-min argument does not apply and
        # an in-set regression is a legitimate result, not a knob bug. This arm
        # must therefore never reach the strict "never WORSE" branch above.
        expect = "M71 residual knob (global overlay) => movers both ways expected"
    else:
        expect = "M49/M50 trade => movers expected"
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
    if arm in ("pool", "m55x"):
        print(f"  [{'PASS' if not worse else 'FAIL'}] {arm} never worse "
              f"({len(worse)} worse) - the invariant")
        # For m55x "better" is the POINT of the tier (an in-sample quality add),
        # not a drift in a gate that was supposed to be cost-neutral.
        gain_tag = "GAIN" if arm == "m55x" else "DRIFT"
        print(f"  [{'PASS' if not better else gain_tag}] "
              f"{len(in_ids) - len(bad)}/{len(in_ids)} cases cost-equal "
              f"(rel 1e-9) - the {gates} gate as originally proven")
        for tag, lst in (("WORSE", worse), (gain_tag, better)):
            for i, n, cs, cr in lst:
                print(f"    [{tag}] case {i:3d} n={n:3d} shipped {cs!r} "
                      f"restore {cr!r} {(cr / cs - 1) * 100:+.4f}%")
        if worse:
            print("  => restore lost to a subset of its own pool: the knob is "
                  "wrong. STOP.")
            _csave()
            return 1
        if better and arm == "m55x":
            wt = {i: math.exp(arec[i]["block_count"] / 12.0) for i in range(100)}
            dw = sum(wt[i] * (cr - cs) for i, _n, cs, cr in better)
            print(f"  => in-set {sel} quality add from the tier: "
                  f"{100 * dw / sum(wt.values()) / IN_SET_TOTAL:+.4f}% of the "
                  f"local total ({len(better)} cases).")
        elif better:
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


def _ab_report(args, arm):
    """Straight OOS A/B of one arm against the shipped baseline on the same cases,
    same per-n-averaged-then-officially-weighted estimator as _theta_report. Used
    when the `full` endpoint is not cached (theta undefined) -- for the M72 arms the
    ship decision is this A/B, since the shipped side already carries M71."""
    lo, hi = _sel(args)
    sel = _selname(lo, hi)
    st = _C.get("m67f", {}).get(arm, {})
    rows = [r for r in _rows(args) if _in_sel(r["n"], lo, hi)
            and f"OOS{r['key']}" in st]
    if not rows:
        print(f"[ab] arm {arm} has no cached OOS cases in {sel}")
        return 1
    pair = [dict(n=r["n"], key=r["key"], S=r["cost"], R=st[f"OOS{r['key']}"]["cost"],
                 tS=r["runtime"], tR=st[f"OOS{r['key']}"]["runtime"],
                 feas=st[f"OOS{r['key']}"].get("feasible", True))
            for r in rows]

    def tot(sub, field):
        return _per_n_total([dict(n=t["n"], cost=t[field]) for t in sub])[0]

    print()
    print("=" * 78)
    print(f"M72  OOS A/B  (arm={arm}, {sel}, per-n averaged then officially "
          f"weighted)")
    print("=" * 78)
    S, R = tot(pair, "S"), tot(pair, "R")
    better = [t for t in pair if t["R"] < t["S"] - 1e-9]
    worse = [t for t in pair if t["R"] > t["S"] + 1e-9]
    tS = sum(t["tS"] for t in pair) / len(pair)
    tR = sum(t["tR"] for t in pair) / len(pair)
    nfeas = sum(1 for t in pair if not t["feas"])
    print(f"  cases {len(pair)}   shipped {S:.6f}   {arm} {R:.6f}   "
          f"gain {(S / R - 1) * 100:+.3f}%")
    print(f"  movers {len(better) + len(worse)}/{len(pair)} "
          f"({len(better)} better, {len(worse)} worse)   infeasible {nfeas}")
    print(f"  runtime/case (this 12c box, sum/cores-bound - NOT 48c): "
          f"{tS:.2f}s -> {tR:.2f}s ({tR / tS:.2f}x)")
    # median-independent RF-aware sign check: cost * t^0.3 per case (no floor).
    num = den = 0.0
    for t in pair:
        w = math.exp(t["n"] / 12.0)
        num += w * t["R"] * (t["tR"] / t["tS"]) ** 0.3
        den += w * t["S"]
    print(f"  RF-aware @12c wall ratio, no floor (sign check only): "
          f"{(num / den - 1) * 100:+.3f}%")
    print("    ^ 12c is sum/cores-bound; at 48c M67-E says wall = max-setter, so "
          "this\n      is an UPPER bound on the runtime cost, not the projection.")
    if worse:
        print(f"\n  worst regressions:")
        for t in sorted(worse, key=lambda t: t["S"] - t["R"])[:5]:
            print(f"    n={t['n']:3d} {t['key']}  {t['S']:.4f} -> {t['R']:.4f} "
                  f"({(t['R'] / t['S'] - 1) * 100:+.2f}%)")
    if better:
        print(f"  best improvements:")
        for t in sorted(better, key=lambda t: t["R"] - t["S"])[:5]:
            print(f"    n={t['n']:3d} {t['key']}  {t['S']:.4f} -> {t['R']:.4f} "
                  f"({(t['R'] / t['S'] - 1) * 100:+.2f}%)")
    out = {"arm": arm, "sel": sel, "cases": len(pair), "S": S, "R": R,
           "gain_pct": (S / R - 1) * 100, "better": len(better),
           "worse": len(worse), "infeasible": nfeas, "tS": tS, "tR": tR,
           "rf_sign_pct": (num / den - 1) * 100,
           "rows": [{k: t[k] for k in ("n", "key", "S", "R", "tS", "tR")}
                    for t in pair]}
    # band in the filename: the same arm is measured on more than one window
    # (heavy n>100 vs mid 60<n<=100) and a bare arm name silently overwrites.
    p = Path(f"results_M72_ab_{arm}_{lo}_{hi if hi < 10 ** 9 else 'inf'}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {p}")
    return 0


def _theta_report(args, arm):
    lo, hi = _sel(args)
    sel = _selname(lo, hi)
    p0 = _C.get("pool0", {})
    st = _C.get("m67f", {}).get(arm, {})
    rows = [r for r in _rows(args) if _in_sel(r["n"], lo, hi)
            and f"OOS{r['key']}" in p0 and f"OOS{r['key']}" in st]
    if not rows:
        # theta needs the `full` endpoint (mode_pool0). Without it the arm can
        # still be judged as a straight A/B against the shipped baseline, which is
        # the whole decision for the M72 arms (theta's denominator only matters
        # when the question is "what share of the full-pool gap does this recover").
        print("[theta] no `full` endpoint cached (mode_pool0 not run for this "
              "signature) -> A/B report vs shipped instead")
        return _ab_report(args, arm)
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
    ap.add_argument("--force-cores", type=int, default=0, dest="force_cores",
                    help="M76: re-export ICCAD_ADAPTIVE_CORES AFTER the ICCAD_* "
                         "strip above, so the run uses the grader's pool shape "
                         "(48 => tier-5 on, tier-3 off) instead of this box's. "
                         "Uses a SEPARATE cache file: _sig() does not include the "
                         "core count, so sharing one would silently reuse solves "
                         "from the other pool shape.")
    ap.add_argument("--arm-bin", default="", dest="arm_bin",
                    help="M78: run the ARM SIDE ONLY against this binary (e.g. "
                         "constructive_m78.exe). The shipped endpoint keeps "
                         "coming from constructive.exe, which is sound because "
                         "m78-flags-off is bit-identical (m78_probe.py gate0). "
                         "The binary's md5 is appended to the arm name so a "
                         "rebuilt probe binary cannot reuse stale arm solves.")
    args = ap.parse_args()
    if args.force_cores:
        global CACHE_PATH
        os.environ["ICCAD_ADAPTIVE_CORES"] = str(args.force_cores)
        CACHE_PATH = _DIR / f"m67_oos_cache_c{args.force_cores}.pkl"
        print(f"[m76] forced cores={args.force_cores} -> cache {CACHE_PATH.name}")
    _cload(_sig(args))
    if args.arm_bin:
        # M78: the arm's knobs exist only in the probe binary. _sig() is computed
        # ABOVE on the SHIPPED exe, so the cached shipped endpoint keeps its
        # identity and stays reusable -- sound only because m78-flags-off is
        # proven bit-identical to constructive.exe (m78_probe.py gate0). The arm
        # name carries the probe binary's md5 so rebuilding it cannot silently
        # reuse the old arm's solves.
        # resolve(): subprocess on Windows hands a bare filename to CreateProcess,
        # which searches python.exe's directory BEFORE the cwd -> a relative
        # --arm-bin silently fails to launch, _run_profile swallows the exception
        # and returns None for BOTH sides, and the liveness gate reports a
        # perfectly plausible 0/210. Absolute path or nothing.
        p = Path(args.arm_bin).resolve()
        assert p.exists(), f"--arm-bin {p} not found"
        h = hashlib.md5(p.read_bytes()).hexdigest()[:8]
        base_arm, newarm = args.arm, f"{args.arm}@{h}"
        _ARMS[newarm] = dict(_ARMS[base_arm])
        # The derived name must inherit the base arm's gate-A branch, or a
        # C++-only arm silently falls through to the "norefine" checks.
        if base_arm in _M75_ARMS:
            globals()["_M75_ARMS"] = _M75_ARMS | {newarm}
        args.arm = newarm
        oc._BIN = p
        print(f"[m78] arm binary {p.name} md5 {h} -> arm '{newarm}'")
    return {"gate0": mode_gate0, "run": mode_run, "report": mode_report,
            "ref": mode_ref, "pool0": mode_pool0,
            "restore": mode_restore}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
