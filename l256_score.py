"""L256 gate -- does the shrink actually lower TRUE cost through the portfolio?

layout_score improving proves nothing: it is the C++ internal proxy with the
150000xbv weight, and the deployed answer is chosen by the PYTHON proxy across 51
profiles. So this re-captures the whole pool with ICCAD_L256=1 and scores the
proxy-selected layout with the official strict scorer, against the identical
computation on the shipped positions already in l252_cache.pkl.

Baseline is not re-run: l252_cache.pkl holds the shipped positions for the same
40 cases x 51 profiles, and l253_editdist.py already showed this reconstruction
reproduces L250 exactly (1.511619 / 1.511432 / 1.245233).

  <python> l256_score.py --limit 12 --ruin 0.12 --step 0.99 --iters 40
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
CACHE = DIR / "l252_cache.pkl"

_ARGV = list(sys.argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--ruin", default="0.12")
    ap.add_argument("--step", default="0.99")
    ap.add_argument("--iters", default="40")
    ap.add_argument("--mode", default="1")
    ap.add_argument("--out", default="l256_score.pkl")
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
    RH = oc._RH
    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)
    L256ENV = {"ICCAD_L256": "1", "ICCAD_L256_RUIN": a.ruin,
               "ICCAD_L256_STEP": a.step, "ICCAD_L256_ITERS": a.iters,
               "ICCAD_L256_MODE": a.mode}
    print("[l256] {}".format(L256ENV))

    C = pickle.load(open(CACHE, "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs(a.sample)}
    keys = sorted([k for k in C if k[0] == a.sample], key=lambda k: -C[k]["n"])
    keys = keys[:a.limit]
    lock = threading.Lock()

    def pick(recs, sumA):
        idxs = sorted(recs)
        met = [recs[i] for i in idxs]
        A_hat = 1.035 * max(sumA, 1e-9)
        hmin = min(m["hpwl"] for m in met) or 1.0
        prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
                for m in met]
        return idxs[min(range(len(idxs)), key=lambda t: prox[t])]

    rows = []
    per = []
    loaded = {}
    for kn, key in enumerate(keys):
        ck = key[1]
        e = C[key]
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

        new = {}
        for i, pos in got.items():
            try:
                m = oc._proxy_metrics(pos, lay["at"], lay["b2b"], lay["p2b"],
                                      lay["pins"], lay["cons"], n)
            except Exception:
                continue
            new[i] = dict(pos=[tuple(map(float, q)) for q in pos],
                          area=m["area"], hpwl=m["hpwl"], vrel=m["vrel"])
        if len(new) < 2:
            continue
        # ISOLATE THE MECHANISM FROM SELECTION. Every case that moved in the first
        # run had flipped profile, and every case that kept its profile moved by
        # exactly 0.000 -- so the portfolio delta was measuring the proxy
        # re-ranking on a shifted hmin (the M80 coupling), not the shrink. This
        # compares the SAME profile before and after.
        for i in sorted(set(new) & set(e["recs"])):
            try:
                c0 = float(m67._cost(e["recs"][i]["pos"], lay).cost)
                c1 = float(m67._cost(new[i]["pos"], lay).cost)
            except Exception:
                continue
            per.append(dict(n=n, prof=i, base=c0, new=c1,
                            moved=(new[i]["pos"] != e["recs"][i]["pos"])))
        kb = pick(e["recs"], e["sumA"])
        kn2 = pick(new, e["sumA"])
        cb = m67._cost(e["recs"][kb]["pos"], lay)
        cn = m67._cost(new[kn2]["pos"], lay)

        def util(pos):
            x0 = min(q[0] for q in pos); y0 = min(q[1] for q in pos)
            x1 = max(q[0] + q[2] for q in pos); y1 = max(q[1] + q[3] for q in pos)
            sa = sum(q[2] * q[3] for q in pos)
            return sa / max((x1 - x0) * (y1 - y0), 1e-18)

        rows.append(dict(n=n, base=float(cb.cost), new=float(cn.cost),
                         bfeas=bool(cb.is_feasible), nfeas=bool(cn.is_feasible),
                         bu=util(e["recs"][kb]["pos"]), nu=util(new[kn2]["pos"]),
                         bh=float(cb.hpwl_gap), nh=float(cn.hpwl_gap),
                         ba=float(cb.area_gap), na=float(cn.area_gap),
                         bv=float(cb.violations_relative),
                         nv=float(cn.violations_relative),
                         pb=kb, pn=kn2))
        print("   n={:3d}  base {:.6f} -> {:.6f}  ({:+.3f}%)  util {:.1f}%->{:.1f}%"
              .format(n, rows[-1]["base"], rows[-1]["new"],
                      100 * (rows[-1]["new"] / rows[-1]["base"] - 1),
                      100 * rows[-1]["bu"], 100 * rows[-1]["nu"]))

    if not rows:
        print("nothing scored")
        return 1
    pickle.dump({"rows": rows, "per": per}, open(DIR / a.out, "wb"))
    mv = [x for x in per if x["moved"]]
    if mv:
        SWp = sum(math.exp(x["n"] / 12.0) for x in mv)
        pb = sum(math.exp(x["n"] / 12.0) * x["base"] for x in mv) / SWp
        pn = sum(math.exp(x["n"] / 12.0) * x["new"] for x in mv) / SWp
        print()
        print("  MECHANISM, isolated (same profile, before vs after):")
        print("    profiles whose layout actually changed: {}/{}".format(
            len(mv), len(per)))
        print("    weighted true cost over those          {:.6f} -> {:.6f}  {:+.4f}%"
              .format(pb, pn, 100 * (pn - pb) / pb))
        print("    better {}  worse {}".format(
            sum(1 for x in mv if x["new"] < x["base"] - 1e-12),
            sum(1 for x in mv if x["new"] > x["base"] + 1e-12)))
    SW = sum(math.exp(r["n"] / 12.0) for r in rows)

    def wm(f):
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rows) / SW

    wb, wn = wm(lambda r: r["base"]), wm(lambda r: r["new"])
    print()
    print("=" * 66)
    print("L256 portfolio gate -- {} cases, TRUE cost, official strict scorer".format(
        len(rows)))
    print("=" * 66)
    print("  weighted base   {:.6f}".format(wb))
    print("  weighted L256   {:.6f}   {:+.4f}%".format(wn, 100 * (wn - wb) / wb))
    print()
    print("  better {}  worse {}  same {}".format(
        sum(1 for r in rows if r["new"] < r["base"] - 1e-12),
        sum(1 for r in rows if r["new"] > r["base"] + 1e-12),
        sum(1 for r in rows if abs(r["new"] - r["base"]) <= 1e-12)))
    print("  feasible base {}/{}   L256 {}/{}".format(
        sum(1 for r in rows if r["bfeas"]), len(rows),
        sum(1 for r in rows if r["nfeas"]), len(rows)))
    print()
    print("  utilisation   {:.1f}% -> {:.1f}%".format(
        100 * wm(lambda r: r["bu"]), 100 * wm(lambda r: r["nu"])))
    print("  hpwl_gap      {:.4f} -> {:.4f}".format(
        wm(lambda r: r["bh"]), wm(lambda r: r["nh"])))
    print("  area_gap      {:.4f} -> {:.4f}".format(
        wm(lambda r: r["ba"]), wm(lambda r: r["na"])))
    print("  vrel          {:.4f} -> {:.4f}".format(
        wm(lambda r: r["bv"]), wm(lambda r: r["nv"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
