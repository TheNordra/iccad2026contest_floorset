"""L263 -- does L262's eviction produce SCORE? Three arms, official strict scorer.

L262 lowers s_min (util 81.6% -> 82.5%, area -1.04%) but only on a DENSE ladder --
the shipped ladder has no rung near the cliff edge. So a fair test needs three
arms, or the ladder change and the eviction get conflated:

  base    shipped ladder, no eviction      (from l252_cache.pkl, already captured)
  dense   dense ladder, no eviction        <- isolates the ladder
  evict   dense ladder + ICCAD_L262=1      <- isolates the eviction

Reported both ways: the PORTFOLIO delta (what would ship) and the ISOLATED
per-profile delta (the mechanism, free of the hmin re-ranking that made L256's
portfolio A/B meaningless).

  <python> l263_quality.py --limit 12
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
PROBE = DIR / "constructive_l262.exe"
CACHE = DIR / "l252_cache.pkl"
_ARGV = list(sys.argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--evmax", default="24")
    ap.add_argument("--arms", default="dense,evict")
    ap.add_argument("--out", default="l263_quality.pkl")
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
    LADDER = ",".join("{:.4f}".format(1.00 + i * 0.01) for i in range(26))
    COARSE = ",".join("{:.4f}".format(1.04 + i * 0.02) for i in range(7))
    # The shipped ladder is 1.00,1.05,1.10,1.20. s_min sits at ~1.11 (L252/L262),
    # so rung 1.00 ALWAYS fails and 1.20 is far looser than anything ever selected:
    # two of the four rungs are dead weight. "tuned" keeps the SAME RUNG COUNT --
    # therefore the same number of pack attempts, i.e. ~zero wall delta -- and just
    # puts them where the cliff is.
    TUNED = "1.0600,1.0900,1.1200,1.1600"
    ALL = {
        "tuned": {"ICCAD_FRAME_SCALES": TUNED},
        "dense": {"ICCAD_FRAME_SCALES": LADDER},
        "coarse": {"ICCAD_FRAME_SCALES": COARSE},
        "evict": {"ICCAD_FRAME_SCALES": LADDER, "ICCAD_L262": "1",
                  "ICCAD_L262_MAX": a.evmax},
    }
    want = [x for x in a.arms.split(",") if x]
    ARMS = {k: ALL[k] for k in want}
    print("[l263] arms: {}   coarse ladder = {}".format(want, COARSE))
    print("[l263] dense ladder 26 rungs; evict budget {}".format(a.evmax))

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

    rows, per = [], []
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

        def capture(extra):
            got.clear()

            def spy(p, inp, bc):
                env = dict(os.environ)
                env.update(p)
                env.update(extra)
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
            out = {}
            for i, pos in got.items():
                try:
                    m = oc._proxy_metrics(pos, lay["at"], lay["b2b"], lay["p2b"],
                                          lay["pins"], lay["cons"], n)
                    c = m67._cost(pos, lay)
                except Exception:
                    continue
                out[i] = dict(pos=[tuple(map(float, q)) for q in pos],
                              area=m["area"], hpwl=m["hpwl"], vrel=m["vrel"],
                              cost=float(c.cost), feas=bool(c.is_feasible),
                              hg=float(c.hpwl_gap), ag=float(c.area_gap),
                              vr=float(c.violations_relative))
            return out

        caps = {k: capture(v) for k, v in ARMS.items()}
        if any(len(v) < 2 for v in caps.values()):
            continue
        # baseline true cost per profile, same scorer call
        bc = {}
        for i, r0 in e["recs"].items():
            try:
                c = m67._cost(r0["pos"], lay)
            except Exception:
                continue
            bc[i] = dict(cost=float(c.cost), feas=bool(c.is_feasible))
        kb = pick(e["recs"], e["sumA"])
        row = dict(n=n, base=bc[kb]["cost"], bfeas=bc[kb]["feas"])
        for nm, cp in caps.items():
            kk = pick(cp, e["sumA"])
            row[nm] = cp[kk]["cost"]
            row[nm + "_feas"] = cp[kk]["feas"]
            row[nm + "_hg"] = cp[kk]["hg"]
            row[nm + "_ag"] = cp[kk]["ag"]
            row[nm + "_vr"] = cp[kk]["vr"]
        rows.append(row)
        if "dense" in caps and "evict" in caps:
          for i in sorted(set(caps["dense"]) & set(caps["evict"]) & set(bc)):
            per.append(dict(n=n, prof=i, base=bc[i]["cost"],
                            dense=caps["dense"][i]["cost"],
                            evict=caps["evict"][i]["cost"],
                            moved=(caps["evict"][i]["pos"] != caps["dense"][i]["pos"])))
        print("   n={:3d}  base {:.6f}   ".format(n, row["base"]) +
              "   ".join("{} {:.6f} ({:+.3f}%)".format(
                  nm, row[nm], 100 * (row[nm] / row["base"] - 1)) for nm in ARMS))

    if not rows:
        print("nothing scored")
        return 1
    pickle.dump({"rows": rows, "per": per}, open(DIR / a.out, "wb"))
    SW = sum(math.exp(r["n"] / 12.0) for r in rows)

    def wm(f, rs=None):
        rs = rs if rs is not None else rows
        sw = sum(math.exp(r["n"] / 12.0) for r in rs)
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rs) / max(sw, 1e-18)

    wb = wm(lambda r: r["base"])
    print()
    print("=" * 72)
    print("L263 -- PORTFOLIO, true cost, {} cases".format(len(rows)))
    print("=" * 72)
    print("  base (shipped ladder)      {:.6f}".format(wb))
    for nm in ARMS:
        w = wm(lambda r: r[nm])
        print("  {:24s}   {:.6f}   {:+.4f}%   feasible {}/{}".format(
            nm, w, 100 * (w - wb) / wb,
            sum(1 for r in rows if r[nm + "_feas"]), len(rows)))
    if "dense" in ARMS and "evict" in ARMS:
        wd = wm(lambda r: r["dense"])
        we = wm(lambda r: r["evict"])
        print()
        print("  eviction alone (evict vs dense)   {:+.4f}%".format(100 * (we - wd) / wd))
    print()
    print("  gaps at the picked layout:")
    for nm in ARMS:
        print("    {:6s} hpwl {:.4f}  area {:.4f}  vrel {:.4f}".format(
            nm, wm(lambda r: r[nm + "_hg"]), wm(lambda r: r[nm + "_ag"]),
            wm(lambda r: r[nm + "_vr"])))

    mv = [x for x in per if x["moved"]]
    if mv:
        sw = sum(math.exp(x["n"] / 12.0) for x in mv)
        pd = sum(math.exp(x["n"] / 12.0) * x["dense"] for x in mv) / sw
        pe = sum(math.exp(x["n"] / 12.0) * x["evict"] for x in mv) / sw
        print()
        print("  MECHANISM, isolated (same profile, dense vs dense+evict):")
        print("    layouts eviction actually changed   {}/{}".format(len(mv), len(per)))
        print("    weighted true cost   {:.6f} -> {:.6f}   {:+.4f}%".format(
            pd, pe, 100 * (pe - pd) / pd))
        print("    better {}  worse {}".format(
            sum(1 for x in mv if x["evict"] < x["dense"] - 1e-12),
            sum(1 for x in mv if x["evict"] > x["dense"] + 1e-12)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
