"""OFFLINE (never shipped): L124 R3 — how many profiles have to carry MIB-ON.

R1 measured the twin as a 2-WAY arbitration: two candidates, each already the
winner of a FULL pool run at its own setting. That is an upper bound on the
deployable form. Shipping cannot afford two full pools:

  * replacing K profiles with MIB-ON versions keeps the pool size (and so the
    48-core max-setter, and so RF) but weakens BOTH sides -- the ON side only
    gets K candidates, and the cases that preferred those profiles' OFF output
    lose it;
  * appending all 43 twins (43 -> 86) blows M67-E's free-restore budget, which
    puts the sum-bound crossover at 75-80 profiles.

Appending a SMALL K is the one option that is both interference-free and inside
the budget. R3 finds the K.

Method: capture every pool profile's output under MIB_BUCKET 0 and 1 for each
case, then simulate any subset offline. The arbitration replicates
`_solve_impl`'s exactly -- same A_hat, same pool-wide hmin, same _RH, same
`proxy = (area/A_hat + _RH*hpwl/hmin) * exp(2*vrel)` -- because the whole
question is whether the REAL selector picks the twin when it should, and an
approximated selector would answer a different question.

⚠️ Runs the L124 probe binary. The shipping exe is never touched.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

import m67_oos_probe as m67                                  # noqa: E402
import m77_oos_probe as m77                                  # noqa: E402

CACHE = _DIR / "l124_r3_cache.pkl"
# L126: the screen generalised. `--flag NAME` makes this the twin-screening
# machine for ANY live C++ knob, which is the reusable half of what L124 found:
# a mechanism's RED may only be a RED for its deployment form, and the cheap way
# to tell is the per-case 2-way oracle.
FLAG = "ICCAD_MIB_BUCKET"


def _capture(oc, lay, bucket):
    """{pool_index: (positions, metrics)} for every profile in the shipped pool.

    Patches `_run_profile` rather than re-deriving the pool by hand: the pool is
    the product of six tiers and a hand-built copy is exactly the mistake
    m76_escape_probe records ("池一律走 oc._pool_indices(), 不可自己拼").
    """
    n = lay["n"]
    idxs = list(oc._pool_indices(n))
    profiles = []
    for i in idxs:
        ov = oc._profile_env(i, n)
        profiles.append(dict(oc._PROFILES[i], **ov) if ov else oc._PROFILES[i])
    key_of = {}
    for k, p in enumerate(profiles):
        key_of.setdefault(tuple(sorted(p.items())), idxs[k])

    got = {}
    orig = oc._run_profile

    def spy(p, inp, block_count):
        pos = orig(p, inp, block_count)
        kk = key_of.get(tuple(sorted(p.items())))
        if kk is not None and pos is not None:
            got[kk] = pos
        return pos

    os.environ[FLAG] = bucket
    oc._run_profile = spy
    try:
        opt = oc.MyOptimizer(verbose=False)
        m67._solve_one(opt, lay)
    finally:
        oc._run_profile = orig

    out = {}
    for i, pos in got.items():
        try:
            m = oc._proxy_metrics(pos, lay["at"], lay["b2b"], lay["p2b"],
                                  lay["pins"], lay["cons"], n)
        except Exception:
            continue
        out[i] = ([tuple(map(float, q)) for q in pos],
                  dict(area=m["area"], hpwl=m["hpwl"], vrel=m["vrel"]))
    return out


def mode_build(a):
    exe = _DIR / a.bin
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(exe)
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    import optimizer_constructive as oc
    import torch

    specs = [(ck, fk, L, n) for ck, fk, L, n in m77._specs(a.sample)
             if n >= a.nmin]
    specs.sort(key=lambda t: -t[3])
    if a.limit:
        specs = specs[:a.limit]
    C = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    byf = collections.defaultdict(list)
    for ck, fk, L, n in specs:
        if (a.sample, ck) not in C:
            byf[fk].append((ck, L, n))
    todo = sum(len(v) for v in byf.values())
    print(f"[cfg] {a.sample} @{a.cores}c  {len(specs)} cases, {todo} to capture")
    t0, done = time.time(), 0
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, L, n in byf[fk]:
            lay = m67._load_case(d, L)
            lay["base"], _ = m67._baseline_official(lay)
            rec = {}
            for b in ("0", "1"):
                rec[b] = _capture(oc, lay, b)
            C[(a.sample, ck)] = dict(n=n, fk=fk, L=L, cap=rec)
            done += 1
            if done % 5 == 0:
                el = time.time() - t0
                print(f"  {done}/{todo} ({el:.0f}s, eta "
                      f"{el / done * (todo - done):.0f}s)  last n={n} "
                      f"profiles {len(rec['0'])}/{len(rec['1'])}")
                pickle.dump(C, open(CACHE, "wb"))
    pickle.dump(C, open(CACHE, "wb"))
    print(f"[build] cache has {len(C)} (sample,case) records")


def _arbitrate(cap_off, cap_on, flip, at, n):
    """Exactly `_solve_impl`'s selection over pool ∪ {ON twin of i for i in flip}."""
    pool = [(i, cap_off[i]) for i in sorted(cap_off)]
    pool += [(("on", i), cap_on[i]) for i in sorted(flip) if i in cap_on]
    if not pool:
        return None
    sumA = sum(max(0.0, float(at[i])) for i in range(n))
    A_hat = 1.035 * max(sumA, 1e-9)
    hmin = min(m["hpwl"] for _k, (_p, m) in pool) or 1.0
    best, bp = None, float("inf")
    for _k, (pos, m) in pool:
        px = (m["area"] / A_hat + oc._RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
        if px < bp:
            bp, best = px, pos
    return best


def mode_curve(a):
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    global oc
    import optimizer_constructive as oc
    import torch
    C = pickle.load(open(CACHE, "rb"))
    rows = [(k[1], v) for k, v in C.items() if k[0] == a.sample]
    print(f"[curve] {a.sample}: {len(rows)} cases")

    lays = {}
    byf = collections.defaultdict(list)
    for ck, v in rows:
        byf[v["fk"]].append((ck, v))
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, v in byf[fk]:
            lay = m67._load_case(d, v["L"])
            lay["base"], _ = m67._baseline_official(lay)
            lays[ck] = lay

    # how often each profile would be the ON-side winner if ALL twins existed
    tally = collections.Counter()
    for ck, v in rows:
        off, on = v["cap"]["0"], v["cap"]["1"]
        lay = lays[ck]
        win = _arbitrate(off, on, set(on), lay["at"], lay["n"])
        for i in on:
            if on[i][0] is win:
                tally[i] += 1
                break
    order = [i for i, _c in tally.most_common()]
    print(f"[curve] ON twins that ever win: {len(order)} profiles; "
          f"top: {tally.most_common(8)}")
    # L125: a twin that is SLOWER than its host has an RF budget, and at 48 cores
    # the wall is the max-setter, so a twin set costs exactly what its slowest
    # member costs -- restricting the sources to an affordable subset is the whole
    # lever. (For L124/L126 the twin cost the same as its host and `--allow` is a
    # no-op.)  Filtering here rather than in the tally keeps the ordering
    # comparable to an unrestricted run.
    allow = None
    if a.allow:
        allow = set(a.allow)
    elif a.allow_max:
        allow = {i for i in order if isinstance(i, int) and i <= a.allow_max}
    if allow is not None:
        order = [i for i in order if i in allow]
        print(f"[curve] sources restricted to {len(order)} affordable twins")

    def total(f):
        byn = collections.defaultdict(list)
        for ck, v in rows:
            byn[v["n"]].append(f(ck, v))
        num = den = 0.0
        for n, vv in byn.items():
            w = math.exp(n / 12.0)
            num += w * (sum(vv) / len(vv))
            den += w
        return num / den

    def cost_for(flip):
        def g(ck, v):
            lay = lays[ck]
            pos = _arbitrate(v["cap"]["0"], v["cap"]["1"], flip,
                             lay["at"], lay["n"])
            return float(m67._cost(pos, lay).cost)
        return total(g)

    base = cost_for(set())
    print(f"\n  K=0 (today, pool unchanged)   {base:.6f}")
    for K in a.ks:
        flip = set(order[:K])
        c = cost_for(flip)
        print(f"  K={K:<3} append {K:>2} ON twins      {c:.6f}   "
              f"{100 * (1 - c / base):+.4f}%   pool {43 + K}")
    every = set().union(*[set(v['cap']['1']) for _ck, v in rows])
    if allow is not None:
        every &= allow
    allf = cost_for(every)
    print(f"  K=all ({len(every)})                    {allf:.6f}   "
          f"{100 * (1 - allf / base):+.4f}%   (upper bound, blows the RF budget)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "curve"])
    ap.add_argument("--sample", default="s2", choices=["s1", "s2"])
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--bin", default="constructive_l124.exe")
    ap.add_argument("--nmin", type=int, default=101)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--flag", default="ICCAD_MIB_BUCKET",
                    help="the C++ knob to twin-screen")
    ap.add_argument("--cache", default="l124_r3_cache.pkl")
    ap.add_argument("--allow", type=int, nargs="*", default=None,
                    help="curve: restrict twin sources to these pool indices")
    ap.add_argument("--allow-max", type=int, default=0,
                    help="curve: restrict twin sources to indices <= this")
    a = ap.parse_args()
    global FLAG, CACHE
    FLAG = a.flag
    CACHE = _DIR / a.cache
    (mode_build if a.mode == "build" else mode_curve)(a)


if __name__ == "__main__":
    main()
