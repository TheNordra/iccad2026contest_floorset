"""L257 step 1 -- capture the L256-ON pool so any twin set can be priced offline.

L256's mechanism is positive in isolation (-0.2506% on the layouts it touches)
but its GLOBAL-OVERLAY deployment is NET +0.505%, because changing candidates
moves the proxy's pool-wide hmin and re-ranks all 51 (the M80 coupling). The twin
form removes that failure mode by construction: the originals stay in the pool, so
the proxy can only ever be offered MORE, never have its inputs swapped.

This captures positions + proxy metrics for every profile with ICCAD_L256=1, into
the same (sample, case) keying l252_cache.pkl already uses for the baseline. The
two caches then form one index space -- i = original, 1000+i = its twin -- and
l257_twin.py can evaluate any source set exactly, with no further solving.

⚠️ Pool must be the shipped 51 (asserted). ⚠️ Input is built on the deployment
path (_l137_env() is non-empty at >=40 cores).

  <python> l257_capture.py --limit 40
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l256.exe"
BASE = DIR / "l252_cache.pkl"
CACHE = DIR / "l257_cache.pkl"

_ARGV = list(sys.argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--ruin", default="0.12")
    ap.add_argument("--step", default="0.99")
    ap.add_argument("--iters", default="40")
    a = ap.parse_args(_ARGV[1:])

    sys.argv = ["x"]
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
    import optimizer_constructive as oc

    if len(list(oc._pool_indices(120))) != 51:
        print("!! not the shipped pool")
        return 1
    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)
    L256ENV = {"ICCAD_L256": "1", "ICCAD_L256_RUIN": a.ruin,
               "ICCAD_L256_STEP": a.step, "ICCAD_L256_ITERS": a.iters,
               "ICCAD_L256_MODE": "1"}
    print("[l257] {}".format(L256ENV))

    B = pickle.load(open(BASE, "rb"))
    C = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs(a.sample)}
    keys = sorted([k for k in B if k[0] == a.sample], key=lambda k: -B[k]["n"])
    keys = keys[:a.limit]
    lock = threading.Lock()
    loaded = {}

    for kn, key in enumerate(keys):
        if key in C:
            continue
        ck = key[1]
        fk, L, n = spec_of[ck]
        if fk not in loaded:
            loaded.clear()
            loaded[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(loaded[fk], L)
        lay["base"], _ = m67._baseline_official(lay)

        idxs = list(oc._pool_indices(n))
        profiles = []
        for i in idxs:
            ov = oc._profile_env(i, n)
            profiles.append(dict(oc._PROFILES[i], **ov) if ov else oc._PROFILES[i])
        key_of = {}
        for k, p in enumerate(profiles):
            key_of.setdefault(tuple(sorted(p.items())), idxs[k])
        got = {}

        def spy(p, inp, bc):
            env = dict(os.environ)
            env.update(p)
            env.update(L256ENV)
            r = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                               text=True, env=env, timeout=TO)
            pos = oc._parse_output(r.stdout, bc)
            kk = key_of.get(tuple(sorted(p.items())))
            if kk is not None and pos is not None:
                with lock:
                    got[kk] = pos
            return pos

        orig = oc._run_profile
        oc._run_profile = spy
        try:
            m67._solve_one(oc.MyOptimizer(verbose=False), lay)
        finally:
            oc._run_profile = orig

        recs = {}
        for i, pos in got.items():
            try:
                m = oc._proxy_metrics(pos, lay["at"], lay["b2b"], lay["p2b"],
                                      lay["pins"], lay["cons"], n)
                c = m67._cost(pos, lay)
            except Exception:
                continue
            recs[i] = dict(pos=[tuple(map(float, q)) for q in pos],
                           area=m["area"], hpwl=m["hpwl"], vrel=m["vrel"],
                           cost=float(c.cost), feas=bool(c.is_feasible))
        # the baseline's own true cost, on the SAME scorer call, so the two sides
        # of every later comparison come from one code path
        bre = {}
        for i, r0 in B[key]["recs"].items():
            try:
                c = m67._cost(r0["pos"], lay)
            except Exception:
                continue
            bre[i] = dict(cost=float(c.cost), feas=bool(c.is_feasible))
        C[key] = dict(n=n, sumA=B[key]["sumA"], recs=recs, basecost=bre)
        pickle.dump(C, open(CACHE, "wb"))
        nmv = sum(1 for i in recs
                  if i in B[key]["recs"] and recs[i]["pos"] != B[key]["recs"][i]["pos"])
        print("   {}/{}  n={:3d}  captured {} profiles, {} moved".format(
            kn + 1, len(keys), n, len(recs), nmv))

    print("[l257] cache holds {} cases".format(len(C)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
