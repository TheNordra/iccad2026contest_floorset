"""M77-OOS: what is an external (ML) placer's output WORTH out of sample?

OFFLINE TOOL — never shipped.

`m77_ml_candidate_probe.py` answers that question on the in-set 100. This file is
its out-of-sample half, and it exists because M76 measured two things that make an
in-set-only verdict useless for a ship decision:

  * the in-sample advantage of a candidate transfers at ~5% (m73x vs m73big:
    +0.127pp in set -> +0.006pp out of set), and
  * the OOS number itself depends on the POOL SHAPE: the same mechanism was
    +0.294% at the 16-core shape and +0.107% at the grader's 48-core shape,
    a factor of 2.7. Anything that interacts with the adaptive tiers must be
    judged at `--cores 48`.

Why a per-profile cache and not a 2-way comparison against the cached portfolio
winner: the selector is
    argmin_k (area_k/A_hat + 1.4*hpwl_k/hmin) * exp(2*vrel_k)
with `hmin` the MINIMUM HPWL OVER THE WHOLE POOL (optimizer_constructive.py:1249).
An ML candidate that undercuts hmin re-scales the hpwl term of every incumbent, so
the pool can re-order — it is not a two-way comparison. And the third number,
dRF@48c, needs PER-PROFILE dt (at 48 cores the wall is the max-setter, M67-E
100/100); m67_oos_cache.pkl only holds the whole portfolio's 12-core wall.

Two facts measured before building this (they are what make it cheap):
  * _band_env(n) is IDENTICAL at 12/16/48 cores (mid K=6, big K=4; the low-core
    K=4 tier needs <=8), so ONE cache of positions serves every pool shape —
    only _pool_indices() differs.
  * the union of _pool_indices() over cores in {12,48} and n in [21,120] is
    exactly 35 profiles (0..33 + 40); the six ORDER_SWAP/MOVE profiles are
    dropped by M41 on every case, so they never need to be run.

TWO SAMPLES, on purpose:
  s1 = the M67-D sample (seed 67, worker_0..9) — the SAME 240 cases as every
       historical OOS number (M67-D/F, M72, M75, M76), so an ML candidate is
       directly comparable to our own classical arms.
  s2 = disjoint (seed 67, worker_10..19) — because s1 is drawn from
       floorset_lite, which IS the teammate's ML training corpus. For a model
       that trained on worker_0..9, s1 is in-sample and s2 is the honest OOS.

Modes:
  manifest --sample s1|s2   write m77_oos_manifest_<s>.json (the interface the
                            external model is run against). s1 asserts its 240
                            keys equal m67_oos_cache.pkl's, BEFORE any solving.
  build    --sample s1|s2   fill m77_oos_audit.pkl (240 cases x 35 profiles:
                            positions + dt + proxy metrics). Resumable.
  selftest --sample s1      bit-exact cross-check of the offline simulator
                            against the real wrapper's cached solves
                            (--cores 12 -> m67_oos_cache.pkl,
                             --cores 48 -> m67_oos_cache_c48.pkl)
  selfdump --sample s -c N  dump the portfolio's OWN winners as a results json
  score <results.json>      the five numbers + VERDICT

Run (PowerShell):
  <python> m77_oos_probe.py manifest --sample s1
  <python> -u m77_oos_probe.py build --sample s1 > m77_build_s1.txt 2>&1
  <python> m77_oos_probe.py selftest --sample s1 --cores 48
  <python> m77_oos_probe.py score ml_oos_s2.json --sample s2 --cores 48 --dt 0.4
"""
import argparse
import concurrent.futures
import glob
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# Shipped defaults only, before anything imports the wrapper (same discipline as
# m67_oos_probe.py:61 and regression_suite.py's child_env).
_STRIPPED = sorted(k for k in os.environ if k.startswith("ICCAD_"))
for _k in _STRIPPED:
    del os.environ[_k]

import torch                                                        # noqa: E402

import m67_oos_probe as m67                                         # noqa: E402
# HARD read-only guard on m67's cache. _index() calls _csave() and m67's module
# level _C starts EMPTY, so importing it and touching the index would otherwise
# overwrite m67_oos_cache.pkl (240 solved cases) with a blank dict.
m67._csave = lambda *a, **k: None                                   # noqa: E731

import optimizer_constructive as oc                                 # noqa: E402
from optimizer_claude import _serialize_input, _parse_output        # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402

RH, GAMMA = 1.4, 0.3
EXE = _DIR / "constructive.exe"
CACHE = _DIR / "m77_oos_audit.pkl"
ML = 2000                          # merged-space index of the external candidate
BAR_OOS_NET = 0.30                 # % — the M75/M76 OOS ship bar (pre-registered)
BAR_INSET = 0.05                   # % — profile_vs_portfolio's in-set house bar
WORKERS = 11                       # leave a core for this process (dt fidelity)

# The known shipped OOS anchors on the M74 tree, printed as a sanity line. s1 only:
# these are the M76-era numbers for THAT sample, and pinning them to s2 would
# invite a comparison between two different corpora.
ANCHOR_OOS = {"s1": {12: 1.576749, 16: 1.576749, 48: 1.555855}, "s2": {}}

SAMPLES = {
    "s1": dict(lo=0, hi=10, seed=m67.SEED, per_n=2, heavy_per_n=4,
               xref={12: _DIR / "m67_oos_cache.pkl",
                     16: _DIR / "m67_oos_cache.pkl",
                     48: _DIR / "m67_oos_cache_c48.pkl"},
               note="M67-D sample (worker_0..9) — comparable to every historical "
                    "OOS number, but drawn from the ML training corpus"),
    "s2": dict(lo=10, hi=20, seed=m67.SEED, per_n=2, heavy_per_n=4, xref={},
               note="disjoint sample (worker_10..19) — the honest OOS for a model "
                    "trained on worker_0..9"),
}


def _exe_md5():
    h = hashlib.md5()
    with open(EXE, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _union_pool():
    """Every profile index the shipped wrapper can emit at 12 or 48 cores. Derived
    live from _pool_indices(), so a change to the drop constants simply adds work
    instead of silently scoring a pool this cache cannot serve."""
    out = set()
    old = os.environ.get("ICCAD_ADAPTIVE_CORES")
    try:
        for cores in (12, 48):
            os.environ["ICCAD_ADAPTIVE_CORES"] = str(cores)
            for n in range(21, 121):
                out |= {i for i in oc._pool_indices(n) if i < oc._M55_BASE_LEN}
    finally:
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
        if old is not None:
            os.environ["ICCAD_ADAPTIVE_CORES"] = old
    return sorted(out)


POOL_UNION = _union_pool()

# Signature: everything that steers the cached POSITIONS. Deliberately NOT the
# drop constants (_BIG_REDUNDANT_IDX / _M45_*), which only pick WHICH cached
# profiles a pool uses at score time — the same split profile_audit.py's `ship`
# mode makes. Anchored on the shipped prefix (M76): a gated-off appended tier
# cannot change a single cached solve.
SIG = repr(("m77oos", 1, _exe_md5(), oc._M55_BASE_LEN,
            repr(oc._PROFILES[:oc._M55_BASE_LEN]),
            repr(sorted(oc._M49_REFINE_BAND)),
            repr(sorted(oc._M50_REFINE_LOWCORE)),
            oc._M45_CORES_MAX,
            repr(sorted(oc._m71_env().items()))))

# Child env for every constructive.exe call: ambient ICCAD_* already stripped, so
# the overlays below are the ONLY ICCAD_* the binary ever sees (profile_audit.py:180).
_CHILD_BASE = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}

_C = {"sig": SIG, "index": {}, "data": {}, "pm": {}, "cost": {}, "ahat": {}}


def _cload():
    global _C
    if not CACHE.exists():
        return
    try:
        c0 = pickle.load(open(CACHE, "rb"))
    except Exception as e:
        print(f"[cache] unreadable ({e!r}); starting fresh")
        return
    if c0.get("sig") == SIG:
        _C = c0
        for k in ("index", "data", "pm", "cost", "ahat"):
            _C.setdefault(k, {})
    else:
        # The file index (path -> n) is exe-independent: keep it, drop the rest.
        _C = {"sig": SIG, "index": c0.get("index", {}), "data": {}, "pm": {},
              "cost": {}, "ahat": {}}
        print("[cache] sig changed -> positions/proxy/cost reset (file index kept)")


def _csave():
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_C, f, protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(6):                      # Windows: transient AV lock
        try:
            os.replace(tmp, CACHE)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))


# --------------------------------------------------------------------------- #
# sample: index -> n-mirrored draw -> manifest                                  #
# --------------------------------------------------------------------------- #
def _worker_of(filekey):
    return int(filekey.split("/")[0].split("_")[1])


def _m67_index():
    """m67_oos_cache.pkl's file index, read-only."""
    p = _DIR / "m67_oos_cache.pkl"
    if not p.exists():
        return {}
    try:
        return pickle.load(open(p, "rb")).get("index", {})
    except Exception:
        return {}


def _index_range(lo, hi, verbose=True):
    """{worker_x/layouts_y: (n, n_layouts)} for workers [lo, hi). Seeded from
    m67's index (free) and from ours; anything missing is scanned once."""
    for src in (_m67_index(), ):
        for k, v in src.items():
            if k not in _C["index"] and lo <= _worker_of(k) < hi:
                _C["index"][k] = (int(v[0]), int(v[1]))
    todo = []
    for w in range(lo, hi):
        for f in sorted(glob.glob(str(_DIR / "floorset_lite" / f"worker_{w}"
                                      / "layouts_*.th"))):
            if m67._key_of(f) not in _C["index"]:
                todo.append(f)
    if todo:
        if verbose:
            print(f"[index] scanning {len(todo)} files in worker_{lo}..{hi - 1} "
                  f"(one-time, cached) ...", flush=True)
        t0 = time.time()
        for i, f in enumerate(todo):
            d = torch.load(f)
            _C["index"][m67._key_of(f)] = (
                int((d[0][0][:, 0] != -1).sum().item()), int(d[0].shape[0]))
            if verbose and (i + 1) % 200 == 0:
                print(f"[index]   {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)",
                      flush=True)
        _csave()
    return {k: v for k, v in _C["index"].items() if lo <= _worker_of(k) < hi}


def _specs(sample, verbose=True):
    """[(case_key, file_key, layout, n)] — m67's own sampler on a worker range,
    so s1 reproduces the historical 240 exactly."""
    s = SAMPLES[sample]
    idx = _index_range(s["lo"], s["hi"], verbose)
    specs, missing = m67._sample(idx, s["per_n"], s["heavy_per_n"])
    if missing and verbose:
        print(f"[warn] no training file for n={missing} in "
              f"worker_{s['lo']}..{s['hi'] - 1}")
    return [(f"{k}/L{L}", k, L, n) for k, L, n in specs]


def _manifest_path(sample):
    return _DIR / f"m77_oos_manifest_{sample}.json"


def _keys_md5(specs):
    return hashlib.md5("\n".join(ck for ck, _f, _L, _n in specs)
                       .encode()).hexdigest()[:12]


def mode_manifest(args):
    sample = args.sample
    s = SAMPLES[sample]
    specs = _specs(sample)
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' ' + extra) if extra else ''}")

    print("=" * 78)
    print(f"M77-OOS manifest   sample={sample}  ({s['note']})")
    print("=" * 78)
    print(f"  seed {s['seed']}  workers {s['lo']}..{s['hi'] - 1}  "
          f"per_n {s['per_n']}  heavy_per_n {s['heavy_per_n']} (n>100)")
    chk("240 cases", len(specs) == 240, f"got {len(specs)}")
    chk("100 distinct n in [21,120]",
        len({n for _c, _f, _L, n in specs}) == 100)
    chk("keys unique", len({c for c, _f, _L, _n in specs}) == len(specs))
    chk("every file present",
        all(m67._path_of(f).exists() for _c, f, _L, _n in specs))

    # s1 must BE the historical sample. Checked here, before any solving.
    if sample == "s1":
        p = _DIR / "m67_oos_cache.pkl"
        have = set()
        if p.exists():
            have = set(pickle.load(open(p, "rb")).get("cases", {}))
        mine = {c for c, _f, _L, _n in specs}
        chk("keys == m67_oos_cache.pkl's 240 cases", have == mine,
            f"m67 {len(have)} / mine {len(mine)} / symdiff "
            f"{len(have ^ mine)}")
    else:
        other = {c for c, _f, _L, _n in _specs("s1", verbose=False)}
        mine = {c for c, _f, _L, _n in specs}
        chk("disjoint from s1", not (other & mine), f"overlap {len(other & mine)}")

    cases = [dict(oos_id=i, key=ck, file=f"floorset_lite/{fk}.th", layout=L, n=n)
             for i, (ck, fk, L, n) in enumerate(specs)]
    doc = dict(
        sample=sample, seed=s["seed"], workers=[s["lo"], s["hi"]],
        per_n=s["per_n"], heavy_per_n=s["heavy_per_n"],
        n_cases=len(cases), keys_md5=_keys_md5(specs), note=s["note"],
        how_to_run=("load floorset_lite/<file>.th with torch.load; the 7-tuple's "
                    "d[0][layout][:,0] is area_target (-1 padded, n = count of "
                    "non -1), d[0][layout][:n,1:] the 5 constraint columns, "
                    "d[1]/d[2]/d[3] b2b/p2b/pins. Run YOUR placer on it and "
                    "return positions as [[x,y,w,h], ...] of length n."),
        expected_results_schema={
            "submission_name": "<your model tag>", "sample": sample,
            "_note": ("one row per case; `key` OR `oos_id` (alias `test_id`) "
                      "identifies it, both are checked against this manifest. "
                      "Partial coverage is allowed — uncovered cases keep the "
                      "classical winner. runtime_seconds should be the MODEL's "
                      "own per-case inference time."),
            "test_results": [{"oos_id": 0, "key": cases[0]["key"],
                              "positions": [["x", "y", "w", "h"]],
                              "runtime_seconds": 0.0}]},
        cases=cases)
    path = _manifest_path(sample)
    json.dump(doc, open(path, "w"), indent=1)
    print(f"\n  keys md5 {doc['keys_md5']}   wrote {path.name}")
    if sample == "s1":
        print("\n  !! s1 is drawn from floorset_lite worker_0..9 = the ML training\n"
              "     corpus. It is clean OOS for OUR classical tuning, but a model\n"
              "     that trained on those workers has SEEN these layouts. Use s2\n"
              "     for the ML verdict.")
    print("=" * 78)
    print(f"MANIFEST {'OK' if ok else 'FAILED'}")
    _csave()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# case context (tensors + serialized input), loaded one .th file at a time      #
# --------------------------------------------------------------------------- #
def _ctx(d, L, n_expect):
    lay = m67._load_case(d, L)
    assert lay["n"] == n_expect, f"n mismatch {lay['n']} != {n_expect}"
    lay["base"], lay["metdev"] = m67._baseline_official(lay)
    otp = build_opt_target_pos(lay["tp"], lay["cons"], lay["n"])
    lay["txt"] = _serialize_input(lay["n"], lay["at"], lay["b2b"], lay["p2b"],
                                  lay["pins"], lay["cons"], otp, gnn_hint=None)
    return lay


def _band_overlay(n):
    """The wrapper's per-case overlay for a SHIPPED index, in _profile_env()'s
    precedence order. Pinned to the 12-core regime like profile_audit.py:56 —
    measured identical at 12/16/48, and the low-core tier (<=8) must stay off."""
    old = os.environ.get("ICCAD_ADAPTIVE_CORES")
    os.environ["ICCAD_ADAPTIVE_CORES"] = "12"
    try:
        ov = dict(oc._band_env(n))
        ov.update(oc._m71_env())
    finally:
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
        if old is not None:
            os.environ["ICCAD_ADAPTIVE_CORES"] = old
    return ov


def _run_one(job):
    ck, k, txt, n, ov = job
    env = dict(_CHILD_BASE)
    env.update(oc._PROFILES[k])
    env.update(ov)
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(EXE)], input=txt, capture_output=True, text=True,
                           env=env)
        dt = time.perf_counter() - t0
        pos = _parse_output(r.stdout, n) if r.returncode == 0 and r.stdout.strip() \
            else None
    except Exception:
        dt, pos = time.perf_counter() - t0, None
    return ck, k, pos, dt


def mode_build(args):
    specs = _specs(args.sample)
    if args.limit:                    # smoke-test knob; the cache is per-case so
        specs = specs[:args.limit]    # a limited run is just a partial fill
    todo_cases = [s for s in specs
                  if any((s[0], k) not in _C["data"] for k in POOL_UNION)]
    print("=" * 78)
    print(f"M77-OOS build   sample={args.sample}  exe md5 {_exe_md5()[:12]}")
    print("=" * 78)
    print(f"  {len(POOL_UNION)} profiles x {len(specs)} cases = "
          f"{len(POOL_UNION) * len(specs)} combos; "
          f"{sum(1 for s in specs for k in POOL_UNION if (s[0], k) in _C['data'])}"
          f" cached, {len(todo_cases)} cases to (re)visit")
    print(f"  overlay: n=80 {_band_overlay(80)}   n=120 {_band_overlay(120)}")
    if not todo_cases:
        print("  nothing to do")
        return 0

    byfile = {}
    for ck, fk, L, n in todo_cases:
        byfile.setdefault(fk, []).append((ck, L, n))
    t0, done, total = time.time(), 0, sum(
        1 for s in todo_cases for k in POOL_UNION if (s[0], k) not in _C["data"])
    fails = []
    for fi, fk in enumerate(sorted(byfile)):
        d = torch.load(m67._path_of(fk))
        for ck, L, n in sorted(byfile[fk]):
            lay = _ctx(d, L, n)
            ov = _band_overlay(n)
            jobs = [(ck, k, lay["txt"], n, ov) for k in POOL_UNION
                    if (ck, k) not in _C["data"]]
            if not jobs:
                continue
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(_run_one, j) for j in jobs]
                for fut in concurrent.futures.as_completed(futs):
                    _ck, k, pos, dt = fut.result()
                    _C["data"][(_ck, k)] = (pos, dt)
                    if pos is None:
                        fails.append((_ck, k))
                    else:
                        # M47 pattern: the proxy runs on THIS thread while the
                        # remaining subprocesses are still out, so it is ~free.
                        _C["pm"][(_ck, k)] = _pm_of(pos, lay)
                    done += 1
            _csave()
        el = time.time() - t0
        print(f"[build] file {fi + 1}/{len(byfile)}  {done}/{total} combos  "
              f"({el:.0f}s, eta {el / max(done, 1) * (total - done):.0f}s)",
              flush=True)
    _csave()
    print(f"[build] done {done} combos in {time.time() - t0:.0f}s; "
          f"failed profiles: {len(fails)} {fails[:6]}")
    return 0


# --------------------------------------------------------------------------- #
# selection / scoring                                                           #
# --------------------------------------------------------------------------- #
def _pm_of(pos, lay):
    m = oc._proxy_metrics(pos, lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                          lay["cons"], lay["n"])
    return (m["area"], m["hpwl"], m["vrel"])


_POOL_MEMO = {}


def _pool_at(n, cores):
    """The shipped pool for this case size at `cores` detected cores
    (m77_ml_candidate_probe.py:174's helper, verbatim semantics)."""
    if (n, cores) in _POOL_MEMO:
        return _POOL_MEMO[(n, cores)]
    old = os.environ.get("ICCAD_ADAPTIVE_CORES")
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(cores)
    try:
        out = [i for i in oc._pool_indices(n) if i < oc._M55_BASE_LEN]
    finally:
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
        if old is not None:
            os.environ["ICCAD_ADAPTIVE_CORES"] = old
    _POOL_MEMO[(n, cores)] = out
    return out


def _A_hat(lay):
    s = sum(max(0.0, float(lay["at"][i])) for i in range(lay["n"]))
    return 1.035 * max(s, 1e-9)


def _select(pm, A_hat, pool):
    """First-best in pool order, hmin over the WHOLE pool — optimizer_constructive
    .py:1249-1255. The external candidate is passed LAST by the caller so a proxy
    tie keeps the incumbent (no cosmetic movers)."""
    hmin = min(pm[k][1] for k in pool) or 1.0
    best, bestv = pool[0], float("inf")
    for k in pool:
        a, h, v = pm[k]
        val = (a / A_hat + RH * h / hmin) * math.exp(2.0 * v)
        if val < bestv:
            bestv, best = val, k
    return best


def _cost_cached(ck, k, pos, lay):
    key = (ck, k)
    if key not in _C["cost"]:
        m = m67._cost(pos, lay)
        _C["cost"][key] = (float(m.cost), bool(m.is_feasible))
    return _C["cost"][key]


def _load_ml(path, specs, sample):
    """-> {case_key: dict(positions, runtime)}. A key that is not in the manifest
    is fatal: a silently misaligned sample would produce a confident wrong number."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {i: ck for i, (ck, _f, _L, _n) in enumerate(specs)}
    nmap = {ck: n for ck, _f, _L, n in specs}
    if j.get("sample") not in (None, sample):
        sys.exit(f"results json says sample={j['sample']!r} but --sample "
                 f"{sample} was requested")
    out = {}
    for r in j["test_results"]:
        if r.get("positions") is None:
            continue
        # `test_id` is accepted as an alias so a runner cloned from the official
        # evaluator works unchanged; it means the manifest's oos_id, not a
        # validation case id.
        rid = r.get("oos_id", r.get("test_id"))
        ck = r.get("key")
        if ck is None and rid is not None:
            ck = by_id.get(int(rid))
        if ck is None or ck not in nmap:
            sys.exit(f"results row {rid}/{r.get('key')!r} is not in the {sample} "
                     f"manifest — wrong sample or stale file")
        if rid is not None and by_id.get(int(rid)) != ck:
            sys.exit(f"row {rid} has key {ck!r} but the manifest has "
                     f"{by_id.get(int(rid))!r}")
        if len(r["positions"]) != nmap[ck]:
            sys.exit(f"{ck}: {len(r['positions'])} positions but n={nmap[ck]}")
        out[ck] = dict(positions=[tuple(float(v) for v in p)
                                  for p in r["positions"]],
                       runtime=r.get("runtime_seconds"))
    return out, j.get("submission_name", Path(path).name), j.get("total_score")


def _walk(specs, need, fn):
    """Load each .th once and call fn(ck, lay) for the cases in `need`."""
    byfile = {}
    for ck, fk, L, n in specs:
        if ck in need:
            byfile.setdefault(fk, []).append((ck, L, n))
    for fk in sorted(byfile):
        d = torch.load(m67._path_of(fk))
        for ck, L, n in sorted(byfile[fk]):
            fn(ck, _ctx(d, L, n))


def _evaluate(specs, cores, ml=None):
    """Per-case: incumbent winner, winner-with-ML, oracle. Returns rows + flags."""
    rows, missing, need = [], [], set()
    ml = ml or {}
    for ck, _fk, _L, n in specs:
        want = _pool_at(n, cores)
        pool = [k for k in want if _C["data"].get((ck, k), (None,))[0] is not None]
        if len(pool) != len(want):
            missing.append(ck)
        if not pool:
            sys.exit(f"{ck}: no cached profile at cores={cores} -> run `build` "
                     f"for this sample first")
        gap = [k for k in pool if (ck, k) not in _C["pm"]]
        if gap:
            sys.exit(f"{ck}: positions cached but proxy metrics missing for "
                     f"{gap[:4]} -> re-run `build` (it fills both together)")
        pm = {k: _C["pm"][(ck, k)] for k in pool}
        rows.append(dict(key=ck, n=n, pool=pool, pm=pm))
    by_key = {r["key"]: r for r in rows}
    # A_hat and the costs need the tensors; walk the files once.
    for r in rows:
        ck = r["key"]
        if ck not in _C["ahat"] or ck in ml:
            need.add(ck)
        else:
            r["A_hat"] = _C["ahat"][ck]

    def fn(ck, lay):
        r = by_key[ck]
        r["A_hat"] = _A_hat(lay)
        _C["ahat"][ck] = r["A_hat"]
        r["lay"] = lay

    _walk(specs, need, fn)
    # winners (cost may need the layout; those cases are already loaded or cached)
    late = set()
    for r in rows:
        ck, pool, pm = r["key"], r["pool"], r["pm"]
        w = _select(pm, r["A_hat"], pool)
        r["win"] = w
        if (ck, w) not in _C["cost"] and "lay" not in r:
            late.add(ck)
    if late:
        def fn2(ck, lay):
            by_key[ck]["lay"] = lay
        _walk(specs, late, fn2)
    for r in rows:
        ck, w = r["key"], r["win"]
        r["base"] = _cost_cached(ck, w, _C["data"][(ck, w)][0], r.get("lay"))[0]
        if ck in ml:
            lay = r["lay"]
            pmml = _pm_of(ml[ck]["positions"], lay)
            m = m67._cost(ml[ck]["positions"], lay)
            r["ml_cost"], r["ml_feas"] = float(m.cost), bool(m.is_feasible)
            pm2 = dict(r["pm"])
            pm2[ML] = pmml
            w2 = _select(pm2, r["A_hat"], r["pool"] + [ML])
            r["win2"] = w2
            if w2 == ML:
                r["withml"] = r["ml_cost"]
            else:
                r["withml"] = _cost_cached(ck, w2, _C["data"][(ck, w2)][0], lay)[0]
            r["oracle"] = min(r["base"], r["ml_cost"])
        else:
            r["win2"], r["withml"], r["oracle"] = r["win"], r["base"], r["base"]
        r.pop("lay", None)
    _csave()
    return rows, missing


def _tot(rows, field):
    return m67._per_n_total([dict(n=r["n"], cost=r[field]) for r in rows])[0]


def mode_selftest(args):
    """The offline simulator must reproduce the REAL wrapper bit for bit on all
    240 cases. Without this every number below is a simulation of a simulation."""
    sample, cores = args.sample, args.cores
    xref = SAMPLES[sample]["xref"].get(cores)
    if xref is None or not Path(xref).exists():
        sys.exit(f"no cross-reference cache for sample={sample} cores={cores} "
                 f"(s1 only: 12/16 -> m67_oos_cache.pkl, 48 -> _c48)")
    ref = pickle.load(open(xref, "rb")).get("cases", {})
    specs = _specs(sample)
    rows, missing = _evaluate(specs, cores)

    print("=" * 78)
    print(f"M77-OOS selftest   sample={sample}  cores={cores}  vs {Path(xref).name}")
    print("=" * 78)
    same = diff = absent = 0
    bad = []
    for r in rows:
        ref_r = ref.get(r["key"])
        if ref_r is None or not ref_r.get("positions"):
            absent += 1
            continue
        a = _C["data"][(r["key"], r["win"])][0]
        b = ref_r["positions"]
        eq = (len(a) == len(b) and
              all(float(x) == float(y) for pa, pb in zip(a, b)
                  for x, y in zip(pa, pb)))
        if eq:
            same += 1
        else:
            diff += 1
            bad.append((r["key"], r["n"], r["base"], float(ref_r["cost"])))
    ours = _tot(rows, "base")
    theirs = m67._per_n_total([dict(n=ref[r["key"]]["n"], cost=ref[r["key"]]["cost"])
                               for r in rows if r["key"] in ref])[0]
    anchor = ANCHOR_OOS.get(sample, {}).get(cores)
    print(f"  positions bit-exact : {same}/{same + diff}"
          + (f"   (not in ref cache: {absent})" if absent else ""))
    print(f"  simulated total     : {ours:.9f}")
    print(f"  wrapper total       : {theirs:.9f}"
          + (f"   (known anchor {anchor:.6f})" if anchor else ""))
    print(f"  profiles missing from cache on {len(missing)} cases")
    for k, n, c1, c2 in bad[:8]:
        print(f"    MISMATCH {k:<28} n={n:3d} ours {c1:.6f} wrapper {c2:.6f}")
    ok = diff == 0 and abs(ours - theirs) < 1e-9 and not missing
    print(f"\n  SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def mode_selfdump(args):
    """The portfolio's own winners as a results json. Feeding it back into `score`
    must be worth exactly nothing (the candidate is bit-identical to the incumbent
    on every case, so hmin cannot move and a proxy tie keeps the incumbent)."""
    specs = _specs(args.sample)
    rows, _missing = _evaluate(specs, args.cores)
    out = dict(submission_name=f"SELFDUMP {args.sample} @{args.cores}c",
               sample=args.sample,
               test_results=[dict(oos_id=i, key=r["key"],
                                  positions=[list(p) for p in
                                             _C["data"][(r["key"], r["win"])][0]],
                                  runtime_seconds=None)
                             for i, r in enumerate(rows)])
    p = _DIR / f"results_oos_selfdump_{args.sample}_c{args.cores}.json"
    json.dump(out, open(p, "w"))
    print(f"wrote {p.name}  ({len(rows)} cases, total {_tot(rows, 'base'):.9f})")
    return 0


def mode_score(args):
    sample, cores = args.sample, args.cores
    specs = _specs(sample)
    ml, tag, ml_total = _load_ml(args.results, specs, sample)
    if args.name:
        tag = args.name
    rows, missing = _evaluate(specs, cores, ml)

    print("=" * 78)
    print(f"M77-OOS — portfolio value of an external candidate: {tag}")
    print("=" * 78)
    if sample == "s1":
        print("  !! s1 is floorset_lite worker_0..9 = the ML TRAINING corpus.")
        print("     Clean OOS for our classical tuning; NOT out of sample for a")
        print("     model trained on those workers. The ML verdict belongs on s2.")
    print(f"  sample    {sample} ({SAMPLES[sample]['note']})")
    print(f"  file      {Path(args.results).name}")
    print(f"  cases     {len(ml)}/{len(specs)} with positions")
    print(f"  pool      @{cores}c (tier-5 {'ON' if cores >= 40 else 'off'}, "
          f"tier-3 {'off' if cores > 16 else 'ON'})")
    if ml_total is not None:
        print(f"  ML solo   total_score {ml_total:.9f}  <- the teammate's gate "
              f"(not this one)")
    if missing:
        print(f"  !! {len(missing)} cases are missing cached profiles")

    t_b, t_w, t_o = _tot(rows, "base"), _tot(rows, "withml"), _tot(rows, "oracle")
    g_w = (1 - t_w / t_b) * 100
    g_o = (1 - t_o / t_b) * 100
    eff = (g_w / g_o * 100) if abs(g_o) > 1e-12 else float("nan")
    anchor = ANCHOR_OOS.get(sample, {}).get(cores)
    print(f"\n  shipped portfolio                  {t_b:.9f}"
          + (f"   (known anchor {anchor:.6f})" if anchor else ""))
    print(f"  + ML candidate, proxy-arbitrated   {t_w:.9f}   {g_w:+.3f}%  "
          f"<- THE GATE")
    print(f"  + ML candidate, per-case ORACLE    {t_o:.9f}   {g_o:+.3f}%")
    print(f"  selection efficiency               {eff:.1f}% of the oracle delta")

    picked = [r for r in rows if r["win2"] == ML]
    worse = [r for r in picked if r["withml"] > r["base"] + 1e-12]
    wins = [r for r in rows if "ml_cost" in r and r["ml_cost"] < r["base"] - 1e-12]
    infeas = [r["key"] for r in rows if r.get("ml_feas") is False]
    print(f"\n  ML beats the portfolio on          {len(wins)}/{len(ml)} cases")
    print(f"  proxy actually switched to ML on   {len(picked)} "
          f"({len(worse)} of them WORSE — proxy mistakes)")
    print(f"  ML infeasible on                   {len(infeas)} cases"
          + (f"  {infeas[:6]}" if infeas else ""))

    if wins:
        print(f"\n  {'case':<28}{'n':>4}{'portfolio':>11}{'ML':>11}{'d%':>8}"
              f"{'taken':>7}")
        for r in sorted(wins, key=lambda r: (r["ml_cost"] - r["base"])
                        * math.exp(r["n"] / 12.0))[:15]:
            print(f"  {r['key']:<28}{r['n']:>4}{r['base']:>11.5f}"
                  f"{r['ml_cost']:>11.5f}"
                  f"{(r['ml_cost'] / r['base'] - 1) * 100:>+8.2f}"
                  f"{('yes' if r['win2'] == ML else 'NO'):>7}")

    # per-band decomposition (the weight profile mirrors validation's one-per-n)
    print(f"\n  {'band':<10}{'cases':>6}{'shipped':>11}{'+ML':>11}{'delta':>9}")
    for b, lo, hi in m67.BANDS:
        sub = [r for r in rows if lo < r["n"] <= hi]
        if not sub:
            continue
        a, c = _tot(sub, "base"), _tot(sub, "withml")
        print(f"  ({lo:3d},{hi:3d}]{'':<1}{len(sub):>6}{a:>11.5f}{c:>11.5f}"
              f"{(1 - c / a) * 100:>+9.3f}%")

    drf = 0.0
    dts = {r["key"]: (args.dt if args.dt is not None else ml[r["key"]]["runtime"])
           for r in rows if r["key"] in ml} if not args.no_rf else {}
    if dts and all(v is not None for v in dts.values()):
        print(f"\n  runtime source: "
              + ("--dt override" if args.dt is not None else
                 "runtime_seconds in the json  ** SEE THE WARNING **"))
        if args.dt is None:
            print("  ⚠ runtime_seconds is the WHOLE solve wall, not one "
                  "candidate's, so it is\n    ALWAYS >= the incumbent max-setter "
                  "and the dRF below is meaningless.\n    Pass --dt with the "
                  "model's own inference time (--dt 0 if negligible).")
        over = []
        for r in rows:
            ts = [_C["data"][(r["key"], k)][1] for k in r["pool"]]
            old = max(max(ts), sum(ts) / cores)
            d = dts.get(r["key"])
            if d is None:
                r["rf"] = 1.0
                continue
            new = max(max(max(ts), d), (sum(ts) + d) / cores)
            r["rf"] = (new / old) ** GAMMA
            if new > old + 1e-12:
                over.append((r["key"], r["n"], old, d))
        for r in rows:
            r["costrf"] = r["withml"] * r.get("rf", 1.0)
        drf = (_tot(rows, "costrf") / _tot(rows, "withml") - 1) * 100
        print(f"  dRF@{cores}c = {drf:+.3f}% of total   "
              f"({len(over)}/{len(dts)} cases where ML sets the new wall)")
        for k, n, old, d in sorted(over, key=lambda x: -x[3])[:5]:
            print(f"    {k:<28} n={n:>4}  incumbent wall {old:.2f}s  ML {d:.2f}s")
    else:
        print("\n  runtime: skipped (--no-rf, or missing in the json with no --dt)")

    net = g_w - drf
    print(f"\n  NET@{cores}c = {net:+.3f}%   (bar {BAR_OOS_NET:.2f}% = the M75/M76 "
          f"OOS ship bar;\n              in-set house bar is {BAR_INSET:.2f}%, "
          f"and M76 died at +0.10% net)")
    verdict = ("GREEN" if net >= BAR_OOS_NET and not worse else
               "AMBER (over the bar, but the proxy also picked losers)"
               if net >= BAR_OOS_NET else
               "RED (below the OOS bar)")
    print(f"\n  VERDICT: {verdict}")

    dump = dict(sample=sample, cores=cores, candidate=tag,
                results_file=Path(args.results).name,
                cases=len(rows), covered=len(ml),
                shipped=t_b, with_ml=t_w, oracle=t_o,
                gain_pct=g_w, oracle_pct=g_o, efficiency_pct=eff,
                drf_pct=drf, net_pct=net, bar_pct=BAR_OOS_NET, verdict=verdict,
                ml_wins=len(wins), proxy_switched=len(picked),
                proxy_losers=len(worse), ml_infeasible=len(infeas),
                per_case=[{k: r[k] for k in
                           ("key", "n", "base", "withml", "oracle")
                           if k in r} for r in rows])
    p = _DIR / f"results_M77_oos_{sample}_c{cores}.json"
    json.dump(dump, open(p, "w"), indent=1)
    print(f"  wrote {p.name}")
    return 0 if net >= BAR_OOS_NET else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "build", "selftest", "selfdump",
                                     "score"])
    ap.add_argument("results", nargs="?", help="external results json (score)")
    ap.add_argument("--sample", choices=sorted(SAMPLES), default="s1")
    ap.add_argument("--cores", type=int, default=48,
                    help="pool shape to score at (48 = the grader's; M76 measured "
                         "a 2.7x difference vs 16)")
    ap.add_argument("--dt", type=float, default=None,
                    help="per-case seconds for the ML candidate (overrides the "
                         "json's runtime_seconds, which is the whole solve wall)")
    ap.add_argument("--no-rf", action="store_true", dest="no_rf")
    ap.add_argument("--limit", type=int, default=0,
                    help="build only: stop after N cases (smoke test)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.mode == "score" and not args.results:
        ap.error("score needs a results json")
    _cload()
    return {"manifest": mode_manifest, "build": mode_build,
            "selftest": mode_selftest, "selfdump": mode_selfdump,
            "score": mode_score}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
