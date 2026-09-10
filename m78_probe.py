"""M78 — candidate-set second path.  OFFLINE PROBE, NEVER SHIPPED.

Runs against `constructive_m78.exe` (built from constructive_m78.cpp), never the
shipped `constructive.exe`.  That is deliberate: since M74 every offline cache
signature pins the shipped exe's md5, so recompiling constructive.exe would
invalidate audit_cache*.pkl / m67_oos_cache*.pkl / m77_oos_audit.pkl and force a
~30-minute rebuild plus four gates.  We pay that only if an arm goes GREEN.

Modes
  gate0            all M78 flags OFF must be BIT-IDENTICAL to constructive.exe,
                   over the 100 in-set cases x the 12/48-core pool union.
  live <arm>...    per-(case,profile) binary-output liveness.  M75's lesson: a
                   flag can change candidates without changing the proxy argmin,
                   so judging liveness on PORTFOLIO output manufactures fake REDs.
                   A case whose whole pool is bit-identical provably cannot move.
  score <arm>...   full 100-case portfolio through the real wrapper (pointed at
                   the m78 exe), official per-case cost, diffed against the M74
                   anchor 1.293461035226291.

Usage:
  python m78_probe.py gate0
  python m78_probe.py live all
  python m78_probe.py score anch_ord1 item_center
"""
import concurrent.futures
import hashlib
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
# m67_oos_probe.py:61 / m77_oos_probe.py:79).
_STRIPPED = sorted(k for k in os.environ if k.startswith("ICCAD_"))
for _k in _STRIPPED:
    del os.environ[_k]

SHIP_EXE = _DIR / "constructive.exe"
M78_EXE = _DIR / "constructive_m78.exe"
# The wrapper reads ICCAD_CONSTRUCTIVE_BIN at import time (optimizer_constructive.py:60),
# so `score` must have it set BEFORE the import below.
os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(M78_EXE)

import optimizer_constructive as oc                                  # noqa: E402
from optimizer_claude import _serialize_input, _parse_output         # noqa: E402
from proxy_analysis import build_opt_target_pos                      # noqa: E402
from iccad2026_evaluate import ContestEvaluator, evaluate_solution   # noqa: E402

del os.environ["ICCAD_CONSTRUCTIVE_BIN"]     # children get it explicitly instead

CACHE = _DIR / "m78_cache.pkl"
WORKERS = 11                     # leave a core for this process
ANCHOR_TOTAL = 1.293461035226291  # M74 default (results_M74_default.json)

# ---------------------------------------------------------------------------- #
# arms                                                                          #
# ---------------------------------------------------------------------------- #
ARMS = {
    # A1 -- the mixed (preplaced+movable) cluster path.  M71's direct sibling.
    "anch_ord1":   {"ICCAD_M78_ANCH_ORD": "1"},     # corner-first
    "anch_ord2":   {"ICCAD_M78_ANCH_ORD": "2"},     # L,R,B,T rank
    "anch_ord3":   {"ICCAD_M78_ANCH_ORD": "3"},     # R,L,T,B rank
    "anch_ord4":   {"ICCAD_M78_ANCH_ORD": "4"},     # area-only (control)
    "anch_center": {"ICCAD_M78_ANCH_CENTER": "1"},
    "anch_cross":  {"ICCAD_M78_ANCH_CROSS": "1"},
    # A2 -- the generic item path (77-79% of weighted blocks)
    "item_center": {"ICCAD_M78_ITEM_CENTER": "1"},
    "item_cross":  {"ICCAD_M78_ITEM_CROSS": "1"},
    # A3 -- the hard-coded bottom-left tie-break
    "tb1":         {"ICCAD_M78_TIEBREAK": "1"},
    "tb2":         {"ICCAD_M78_TIEBREAK": "2"},
    "tb3":         {"ICCAD_M78_TIEBREAK": "3"},
    # --- M75 re-test (2026-08-03) -------------------------------------------
    # M75 judged these two "exactly 0.0000% -- 340 cases, zero profile output
    # changed".  Its liveness screen fed the binary target_positions=None, which
    # deletes every preplaced block -> no MIXED clusters exist -> REPACK's
    # antecedent is empty BY CONSTRUCTION.  REPACK acts on the same anchored
    # first-pass where M78's anch_cross just found value, so its RED has to be
    # re-measured with the masked otp the real harness uses.  CORNER is the
    # control: it lives in make_group_item, which the None input makes MORE
    # reachable (every cluster becomes pure-movable), so its 0 should stand.
    "repack":      {"ICCAD_ANCHORED_BND_REPACK": "1"},
    "corner":      {"ICCAD_CLUSTER_BND_CORNER": "1"},
}
OFF = "__off__"          # m78 exe, all flags default
SHIP = "__ship__"        # the shipped binary


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _union_pool():
    """Every profile index the shipped wrapper can emit at 12 or 48 cores
    (m77_oos_probe._union_pool).  Never hand-assembled: M41's swap filter is
    content-dependent, so a hand-built pool can contain profiles the wrapper
    would never run ([[m76-escape-tier-red]])."""
    out = set()
    for cores in (12, 48):
        os.environ["ICCAD_ADAPTIVE_CORES"] = str(cores)
        for n in range(21, 121):
            out |= {i for i in oc._pool_indices(n) if i < oc._M55_BASE_LEN}
    os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
    return sorted(out)


POOL_UNION = _union_pool()

SIG = repr(("m78", 1, _md5_file(M78_EXE), _md5_file(SHIP_EXE),
            oc._M55_BASE_LEN, repr(oc._PROFILES[:oc._M55_BASE_LEN]),
            repr(sorted(oc._M49_REFINE_BAND)),
            repr(sorted(oc._M50_REFINE_LOWCORE)),
            oc._M45_CORES_MAX,
            repr(sorted(oc._m71_env().items()))))

_CHILD_BASE = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
_C = {"sig": SIG, "out": {}, "dt": {}}


def _cload():
    global _C
    if CACHE.exists():
        try:
            d = pickle.load(open(CACHE, "rb"))
        except Exception:
            d = None
        if d and d.get("sig") == SIG:
            _C = d
            return
        print("[cache] signature mismatch (exe or pool changed) -> starting empty")


def _csave():
    pickle.dump(_C, open(CACHE, "wb"))


def _band_overlay(n):
    """The wrapper's per-case overlay for a SHIPPED index, in _profile_env()'s
    precedence order.  Pinned to 12 cores like profile_audit.py:56 / m77:357 —
    _band_env is measured identical at 12/16/48 and the <=8 tier must stay off."""
    os.environ["ICCAD_ADAPTIVE_CORES"] = "12"
    try:
        ov = dict(oc._band_env(n))
        ov.update(oc._m71_env())
    finally:
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
    return ov


# ---------------------------------------------------------------------------- #
# corpus                                                                        #
# ---------------------------------------------------------------------------- #
_EV = None


def cases():
    """[(idx, n, txt, ctx)] for the 100 in-set cases."""
    global _EV
    if _EV is None:
        _EV = ContestEvaluator(data_path=str(_DIR), verbose=False)
        _EV._load_dataset()
    out = []
    for idx in range(len(_EV.dataset)):
        s = _EV.dataset[idx]
        at, b2b, p2b, pins, cons = s["input"]
        n = int((at != -1).sum().item())
        base, tp = _EV._extract_baseline(idx, s["label"], b2b, p2b, pins, n)
        otp = build_opt_target_pos(tp, cons, n)
        txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
        out.append((idx, n, txt,
                    dict(at=at, b2b=b2b, p2b=p2b, pins=pins, cons=cons,
                         base=base, tp=tp, otp=otp)))
    return out


# ---------------------------------------------------------------------------- #
# single-profile runs                                                           #
# ---------------------------------------------------------------------------- #
def _run_one(job):
    idx, k, txt, n, ov, arm = job
    exe = SHIP_EXE if arm == SHIP else M78_EXE
    env = dict(_CHILD_BASE)
    env.update(oc._PROFILES[k])
    env.update(ov)
    if arm not in (OFF, SHIP):
        env.update(ARMS[arm])
    t0 = time.time()
    r = subprocess.run([str(exe)], input=txt, capture_output=True, text=True,
                       timeout=300, env=env)
    dt = time.time() - t0
    if r.returncode != 0 or not r.stdout.strip():
        return (idx, k, arm, None, dt)
    # md5 of the RAW stdout: %.17g text, so byte equality == bit equality.
    return (idx, k, arm, hashlib.md5(r.stdout.encode()).hexdigest(), dt)


def fill(arm, cs, verbose=True):
    """Populate _C['out'][(idx,k,arm)] for every (case, pool profile)."""
    jobs = []
    for idx, n, txt, _ctx in cs:
        ov = _band_overlay(n)
        for k in POOL_UNION:
            if (idx, k, arm) not in _C["out"]:
                jobs.append((idx, k, txt, n, ov, arm))
    if not jobs:
        return
    if verbose:
        print(f"[fill] {arm}: {len(jobs)} runs on {WORKERS} workers", flush=True)
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for idx, k, a, h, dt in ex.map(_run_one, jobs):
            _C["out"][(idx, k, a)] = h
            _C["dt"][(idx, k, a)] = dt
            done += 1
            if verbose and done % 500 == 0:
                print(f"[fill]   {done}/{len(jobs)} ({time.time() - t0:.0f}s)",
                      flush=True)
    _csave()
    if verbose:
        print(f"[fill] {arm}: done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------- #
# gate 0                                                                        #
# ---------------------------------------------------------------------------- #
def gate0():
    cs = cases()
    fill(SHIP, cs)
    fill(OFF, cs)
    bad, nones = [], []
    for idx, _n, _txt, _ctx in cs:
        for k in POOL_UNION:
            a, b = _C["out"][(idx, k, SHIP)], _C["out"][(idx, k, OFF)]
            if a is None or b is None:
                nones.append((idx, k))
            elif a != b:
                bad.append((idx, k))
    tot = len(cs) * len(POOL_UNION)
    print("=" * 78)
    print(f"GATE 0  {len(cs)} cases x {len(POOL_UNION)} pool profiles = {tot} pairs")
    print(f"  failed runs        : {len(nones)}")
    print(f"  differing outputs  : {len(bad)}")
    if bad:
        for idx, k in bad[:20]:
            print(f"    case{idx} profile{k}")
    ok = not bad and not nones
    print(f"GATE 0: {'PASS (m78 flags-off is bit-identical)' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------- #
# liveness                                                                      #
# ---------------------------------------------------------------------------- #
def live(arms):
    cs = cases()
    fill(OFF, cs)
    W = {idx: math.exp(n / 12.0) for idx, n, _t, _c in cs}
    Wtot = sum(W.values())
    print("=" * 78)
    print(f"LIVENESS  (per-PROFILE binary output; portfolio output cannot be "
          f"used — [[m75-m71-residual-knobs-red]])")
    print(f"\n  {'arm':<13} {'live (case,prof)':>17} {'live cases':>11} "
          f"{'live wt%':>9} {'max n':>6}")
    res = {}
    for arm in arms:
        fill(arm, cs, verbose=True)
        pairs, livecases = 0, set()
        for idx, _n, _t, _c in cs:
            for k in POOL_UNION:
                if _C["out"][(idx, k, OFF)] != _C["out"][(idx, k, arm)]:
                    pairs += 1
                    livecases.add(idx)
        wt = sum(W[i] for i in livecases)
        mx = max((n for idx, n, _t, _c in cs if idx in livecases), default=0)
        res[arm] = sorted(livecases)
        print(f"  {arm:<13} {pairs:>8}/{len(cs) * len(POOL_UNION):<8} "
              f"{len(livecases):>7}/{len(cs)} {100 * wt / Wtot:8.2f}% {mx:>6}")
    print("\n  live case ids (these are the only ones `score` has to move):")
    for arm, ids in res.items():
        head = ", ".join(str(i) for i in ids[:24])
        print(f"    {arm:<13} {head}{' ...' if len(ids) > 24 else ''}")
    print("\n  Arms with 0 live pairs are CLOSED: with every pool profile "
          "bit-identical,\n  the portfolio provably cannot move (M75's exact "
          "argument, no sampling).")
    return 0


# ---------------------------------------------------------------------------- #
# full portfolio score                                                          #
# ---------------------------------------------------------------------------- #
def score(arms):
    cs = cases()
    print("=" * 78)
    print(f"SCORE  full 100-case portfolio via the wrapper, exe = "
          f"{M78_EXE.name}\n  anchor (M74 default) = {ANCHOR_TOTAL:.15f}")
    for arm in arms:
        for k, v in ARMS.get(arm, {}).items():
            os.environ[k] = v
        os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(M78_EXE)
        opt = oc.MyOptimizer(verbose=False)
        rows, totW, totWC, feas = [], 0.0, 0.0, 0
        t0 = time.time()
        for idx, n, _txt, c in cs:
            # otp, NOT tp: the official harness masks target_positions to -1
            # except preplaced (x,y,w,h) and fixed (w,h) — those are hard-constraint
            # INPUTS, not the answer.  Passing tp here would leak fp_sol.
            ps = opt.solve(n, c["at"], c["b2b"], c["p2b"], c["pins"], c["cons"],
                           c["otp"])
            m = evaluate_solution({"positions": ps, "runtime": 1.0}, c["base"],
                                  c["cons"][:n], c["b2b"], c["p2b"], c["pins"],
                                  c["at"][:n], target_positions=c["tp"][:n],
                                  median_runtime=1.0)
            w = math.exp(n / 12.0)
            totW += w
            totWC += w * m.cost
            feas += 1 if m.is_feasible else 0
            rows.append((idx, n, w, m.cost))
        tot = totWC / totW
        d = 100.0 * (tot - ANCHOR_TOTAL) / ANCHOR_TOTAL
        print(f"\n  {arm:<13} total {tot:.15f}   delta {d:+.4f}%   "
              f"feasible {feas}/{len(cs)}   {time.time() - t0:.0f}s")
        _dump(arm, rows)
        for k in ARMS.get(arm, {}):
            os.environ.pop(k, None)
        os.environ.pop("ICCAD_CONSTRUCTIVE_BIN", None)
    return 0


def _dump(arm, rows):
    import json
    p = _DIR / f"results_M78_{arm}.json"
    json.dump({"test_results": [{"test_id": i, "block_count": n,
                                 "weight": w, "cost": c}
                                for i, n, w, c in rows]}, open(p, "w"))
    print(f"    wrote {p.name}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    _cload()
    args = sys.argv[2:]
    if args == ["all"]:
        args = list(ARMS)
    if mode == "gate0":
        return gate0()
    if mode == "live":
        return live(args or list(ARMS))
    if mode == "score":
        return score(args or list(ARMS))
    print(f"unknown mode {mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
