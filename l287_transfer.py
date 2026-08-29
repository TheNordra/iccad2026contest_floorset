"""L287/L289: does the in-set gain TRANSFER to a held-out corpus, and if not,
WHICH component fails to transfer?

THE QUESTION.  The rank projection (L285) assumes the in-set 48c improvement
1.295548 (M73) -> 1.226325 (now), i.e. -5.34 %, shows up on the hidden set too.
The package is deterministic -- nothing "degrades"; what is unknown is how much
of the measured gain is MECHANISM (transfers everywhere) versus corpus-specific.

THE TEST.  Run the FULL deployable pipeline -- the real wrapper, portfolio
selection and `_shape_lp_maybe` included, via m67._solve_one -- on a held-out
corpus, once per arm.  `m73` is the shipped code with the post-beta additions
switched off through the project's own kill switches; the leave-one-out arms
remove one component at a time, so each arm's delta against `ship` is that
component's held-out value.

MEASURED (s1, 240 cases): ship 1.470262 vs m73-like 1.507783 = -2.4885 %,
against the in-set -5.3431 % => TRANSFER 46.6 %.  Difficulty does NOT explain
it: a per-case OLS on this corpus predicts -2.51 % even AT in-set difficulty
while the in-set itself shows -5.34 %.  So it is a corpus effect, and the
decomposition is what says which component carries it.

⚠️ WHAT THIS CORPUS IS AND IS NOT.  L275: the OOS heavy band is 22-24 % HARDER
than the in-set while beta hidden sits ~2.4 % from it.  OOS comes from
floorset_lite; the in-set, alpha and beta sets are the contest's own.  A gap
here is EITHER over-fitting OR a distribution difference, and this harness
cannot separate those two on its own.

🚨 m67_oos_probe STRIPS every ICCAD_* at import ("shipped defaults only").  A
core count set before the import is silently deleted, `_effective_cores_hi()`
falls back to this box's 32 -- below the >=40 gate that arms the shape LP,
route A and the M80 tier -- so every arm becomes a no-op and all of them score
identically.  That is exactly what the first run of this probe reported.
Re-inject AFTER the import and assert it, which is what m77's --force-cores does.
"""
import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

import m67_oos_probe as m67                                      # noqa: E402
import m77_oos_probe as m77                                      # noqa: E402
import optimizer_constructive as oc                              # noqa: E402

CORES = "48"
os.environ["ICCAD_ADAPTIVE_CORES"] = CORES
assert oc._effective_cores_hi() >= 40, (
    "cores gate not armed: %r" % oc._effective_cores_hi())

ARMS = {
    "ship": {},
    "m73": {"ICCAD_SHAPE_LP": "0", "ICCAD_ROUTE_A": "0",
            "ICCAD_M80_TIER": "0", "ICCAD_HINT_MODE": "0",
            "ICCAD_L223_REFINE_HEAVY": "4", "ICCAD_L231_REFINE_MID": "8"},
    # L288: shape-LP depth 2.  L276 priced it RED (-0.9536 %) against the BETA
    # runtime vector and with LOCAL dt seconds added to GRADER seconds.
    # Corrected for both it is +0.17..+0.28 % NET, so L275's rule applies: it
    # must be positive here too before it is a candidate.
    "lp2": {"ICCAD_SHAPE_LP_ITERS": "2"},
    # L289 leave-one-out: shipped MINUS one component.
    "noLP": {"ICCAD_SHAPE_LP": "0"},
    "noHint": {"ICCAD_HINT_MODE": "0"},
    "noM80": {"ICCAD_M80_TIER": "0"},
    "refOld": {"ICCAD_L223_REFINE_HEAVY": "4", "ICCAD_L231_REFINE_MID": "8"},
    # L294: the per-case LP GATE off, i.e. pre-L196 behaviour -- the shape LP
    # runs on all 100 block counts instead of 71.  In set it is +2.2282 % of
    # quality (29/29 gated-off cases moved, 29 better / 0 worse, 100/100
    # feasible) for +15.78 s local, NET +0.91..+1.42 %.  L275's rule: it has to
    # be positive on BOTH samples here before it is a candidate.
    "gate0": {"ICCAD_LP_GATE": "0"},
    # L296: the COMPOSITION.  The two knobs are independent -- the gate decides
    # WHETHER the LP runs on a case (71 -> 100 block counts), the depth decides
    # HOW MANY passes where it does.  So `both` is not the sum of the two: it
    # also buys a SECOND pass on the 29 cases gate0 newly admits, which neither
    # arm alone can see.
    "both": {"ICCAD_LP_GATE": "0", "ICCAD_SHAPE_LP_ITERS": "2"},
    # L299: the `mix` arm is NOT reachable by environment variable.  It is
    # "LP everywhere at k=1, a second pass only on the 71 where L196 judged the
    # case can afford it", which needs two TABLE defaults -- `_L196_LPGATE` all
    # 1s and `_L157_DEPTH` 2 on the old 1-set.  `l296_mix_optimizer.py` is that
    # wrapper, a copy of the tree's with those two tables changed and nothing
    # else, so this arm swaps the MODULE and sets no flags at all.
    "mix": {},
    # L312: the RF-SAFE half of `gate0`.  Same shape as `mix` -- not reachable
    # by environment variable, because it is a per-block-count TABLE: the
    # shipped 71 plus the 12 counts whose MEASURED added grader time fits inside
    # that case's own slack to the RF floor.  Depth is untouched, so this arm
    # buys coverage only.  Selection used NO quality information (the L157
    # shape), which is the whole reason it is worth an OOS run: there is nothing
    # here that could have been fitted to the in-set, so a failure to transfer
    # would have to be a distribution difference rather than over-fitting.
    "rfsafe": {},
}
# arm -> wrapper module.  Everything but `mix`/`rfsafe` runs the tree wrapper.
MIX_ARMS = {"mix": "l296_mix_optimizer", "rfsafe": "l312_rfsafe_optimizer"}
_MODCACHE = {}


def mod_of(name):
    m = MIX_ARMS.get(name)
    if m is None:
        return oc
    if m not in _MODCACHE:
        _MODCACHE[m] = __import__(m)
    return _MODCACHE[m]
ALLKEYS = sorted({k for v in ARMS.values() for k in v})
INSET_DELTA = 100.0 * (1.226325 / 1.295548 - 1.0)


def set_arm(name):
    """Arm `name` and return the wrapper module it runs on.

    L294/L299 liveness: `_shape_lp_maybe` never raises by design, so a dead flag
    and a decision not to act are indistinguishable downstream -- assert at the
    source, on the module this arm actually uses.  n=38 is in the gated-off set,
    n=21 is not.  For `mix` the arm lives in the module's TABLES rather than in
    the environment, so the tables themselves are the thing to check.
    """
    for k in ALLKEYS:
        os.environ.pop(k, None)
    os.environ["ICCAD_ADAPTIVE_CORES"] = CORES     # must survive every arm
    os.environ.update(ARMS[name])
    m = mod_of(name)
    assert m._effective_cores_hi() >= 40
    assert (m is not oc) == (name in MIX_ARMS), "wrong module for %r" % name

    want_gate = (ARMS[name].get("ICCAD_LP_GATE") == "0"
                 or bool(m._L196_LPGATE.get(38, 1)))
    assert m._lp_gate_ok(38) is want_gate, (
        "LP gate not armed for arm %r: _lp_gate_ok(38)=%r, wanted %r"
        % (name, m._lp_gate_ok(38), want_gate))
    assert m._lp_gate_ok(21) is True
    want_k = int(ARMS[name].get("ICCAD_SHAPE_LP_ITERS",
                                max(m._L157_DEPTH.values())))
    assert m._shape_lp_depth(True)[0] == want_k, (
        "LP depth not armed for arm %r: %r, wanted %d"
        % (name, m._shape_lp_depth(True), want_k))
    if name == "mix":
        # the two table edits that ARE this arm; nothing else defines it
        assert sum(m._L196_LPGATE.values()) == 100, "mix gate table not all 1s"
        h = {v: sum(1 for x in m._L157_DEPTH.values() if x == v)
             for v in set(m._L157_DEPTH.values())}
        assert h == {1: 29, 2: 71}, "mix depth table is %r" % h
        assert m._depth_ok(38, 2, 0.0) is False   # no 2nd pass on the old 0-set
        assert m._depth_ok(21, 2, 0.0) is True    # yes on the old 1-set
    if name == "rfsafe":
        # the ONE table edit that IS this arm, plus the thing it must NOT touch
        assert sum(m._L196_LPGATE.values()) == 83, (
            "rfsafe gate table is %d on, wanted 83"
            % sum(m._L196_LPGATE.values()))
        for n in (38, 40, 56, 76, 79, 81, 94, 95, 107, 108, 114, 120):
            assert m._lp_gate_ok(n) is True, "rfsafe: %d should be ungated" % n
        for n in (39, 47, 49, 52, 60, 69, 78, 83, 87, 92, 99, 101, 112, 117, 118):
            assert m._lp_gate_ok(n) is False, "rfsafe: %d should still be off" % n
        assert max(m._L157_DEPTH.values()) == 1, (
            "rfsafe must not change depth; got %r" % sorted(set(m._L157_DEPTH.values())))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=240)
    ap.add_argument("--arms", default="")
    ap.add_argument("--cache", default="l287_cache.pkl")
    args = ap.parse_args()

    want = [a for a in ARMS if not args.arms or a in args.arms.split(",")]
    cp = DIR / args.cache
    db = pickle.load(open(cp, "rb")) if cp.exists() else {}
    specs = m77._specs(args.sample)[:args.limit]
    print("[l287] %s: %d cases, arms %s" % (args.sample, len(specs), want),
          flush=True)

    import torch
    loaded, t0, ndone = {}, time.perf_counter(), 0
    for ck, fk, L, n in specs:
        if all((args.sample, ck, a) in db for a in want):
            continue
        if fk not in loaded:
            loaded.clear()
            loaded[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(loaded[fk], L)
        lay["base"], _ = m67._baseline_official(lay)
        for a in want:
            key = (args.sample, ck, a)
            if key in db:
                continue
            _m = set_arm(a)
            try:
                pos = m67._solve_one(_m.MyOptimizer(verbose=False), lay)[0]
                c = m67._cost(pos, lay)
                db[key] = dict(n=n, cost=float(c.cost),
                               feas=bool(c.is_feasible),
                               hg=float(c.hpwl_gap), ag=float(c.area_gap),
                               vr=float(c.violations_relative))
            except Exception as e:
                db[key] = dict(n=n, err="%s: %s" % (type(e).__name__, e))
        ndone += 1
        if ndone % 10 == 0:
            pickle.dump(db, open(cp, "wb"))
            print("  %d cases (%.0fs)" % (ndone, time.perf_counter() - t0),
                  flush=True)
    pickle.dump(db, open(cp, "wb"))
    set_arm("ship")

    rows = []
    for ck, fk, L, n in specs:
        e = {a: db.get((args.sample, ck, a)) for a in want}
        if any(v is None or "err" in v or not v["feas"] for v in e.values()):
            continue
        rows.append((n, e))
    if not rows:
        print("no complete rows")
        return

    def W(n):
        return math.exp(n / 12.0)

    ws = sum(W(n) for n, _e in rows)

    def tot(a):
        return sum(W(n) * e[a]["cost"] for n, e in rows) / ws

    T = {a: tot(a) for a in want}
    print()
    print("== %s: %d cases feasible in EVERY arm, weighted exp(n/12) =="
          % (args.sample, len(rows)))
    print("   %-9s %-12s %-13s %s" % ("arm", "total", "ship vs arm", "movers"))
    for a in want:
        mv = sum(1 for _n, e in rows if e[a]["cost"] != e["ship"]["cost"])
        print("   %-9s %-12.6f %+12.4f%% %6d/%d"
              % (a, T[a], 100.0 * (T["ship"] / T[a] - 1.0), mv, len(rows)))

    if "m73" in T:
        d = 100.0 * (T["ship"] / T["m73"] - 1.0)
        print()
        print("   in-set reference %+.4f %%   OOS %+.4f %%   TRANSFER %.1f %%"
              % (INSET_DELTA, d, 100.0 * d / INSET_DELTA))
    if "lp2" in T:
        d2 = 100.0 * (T["lp2"] / T["ship"] - 1.0)
        print()
        print("   LP k=2 quality here: %+.4f %% (negative = k=2 BETTER); "
              "in-set was -0.3075 %%" % d2)

    for lo, hi, lbl in ((0, 60, "light n<=60"), (61, 100, "mid 61-100"),
                        (101, 999, "heavy n>=101")):
        sub = [r for r in rows if lo <= r[0] <= hi]
        if not sub:
            continue
        w2 = sum(W(n) for n, _e in sub)
        line = "   %-14s" % lbl
        for a in want:
            if a == "ship":
                continue
            b = sum(W(n) * e[a]["cost"] for n, e in sub) / w2
            s_ = sum(W(n) * e["ship"]["cost"] for n, e in sub) / w2
            line += " %s %+.3f%%" % (a, 100.0 * (s_ / b - 1.0))
        print(line)


if __name__ == "__main__":
    main()
