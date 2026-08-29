"""L267/L268 -- true cost, generalised arms, plus the deterministic pack bill.

Same framework as l263_quality.py (official strict scorer, portfolio delta AND
isolated per-profile delta), with two changes:

  * arms are given on the command line, so one run can price the adaptive frame
    search and big-first ordering side by side against the same base;
  * every run's L267PACKS is captured. It is a DETERMINISTIC signal, which the
    stopwatch is not -- but 🚨 it is NOT a wall proxy, and this run is what
    proved it: the adaptive arm spends 1.063x the packs of the shipped ladder on
    the max-setter and 1.2417x the WALL. A pack is not a unit of cost. Packs
    near the cliff are the expensive ones (fewer legal origins to find, more
    candidates generated and rejected, and a FAILING pack has already placed
    ~91% of the blocks before it gives up -- L254). Read pack counts as "how
    many attempts", never as "how long".

Base comes from l252_cache.pkl (the shipped ladder, already captured), and the
key list is the same one l264/l265 used -- sorted by -n, limit 40 -- so the rows
join by index for a free split-half.

  <python> l267_quality.py --limit 40 --arms adapt,big1,big2
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
PROBE = DIR / "constructive_l267.exe"
CACHE = DIR / "l252_cache.pkl"
_ARGV = list(sys.argv)

DENSE = ",".join("{:.4f}".format(1.00 + i * 0.01) for i in range(26))

ALL = {
    # the adaptive frame search: no rung is a constant, every one is derived per
    # case from where that case's own cliff actually is.
    "adapt":   {"ICCAD_L267": "1"},
    "adapt6":  {"ICCAD_L267": "1", "ICCAD_L267_PROBES": "6"},
    "adapt3":  {"ICCAD_L267": "1", "ICCAD_L267_RUNGS": "3"},
    "adaptbw": {"ICCAD_L267": "1", "ICCAD_L267_STEP": "0"},
    "adapt02": {"ICCAD_L267": "1", "ICCAD_L267_STEP": "0.02"},
    # big-first commitment order, shipped ladder untouched
    "big1":    {"ICCAD_L268": "1"},
    "big2":    {"ICCAD_L268": "2"},
    "l269":   {"ICCAD_L269": "1"},
    "l269b":  {"ICCAD_L269": "2"},
    "adaptp2": {"ICCAD_L267": "1", "ICCAD_L267_PROBES": "2"},
    "adaptp3": {"ICCAD_L267": "1", "ICCAD_L267_PROBES": "3"},
    "l269p1": {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "1"},
    "l269p4": {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "4"},
    "l269p2": {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "2"},
    "l269p3": {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "3"},
    # L268=4 keeps the bscore CLASS order -- which is what protects vrel, and what
    # M78's anch_ord4 (+1.069%) and WIRE_ORDER (vBd 390) both measured the cost of
    # dropping -- and removes ONLY the compound-item-first tie-break. The screen
    # says that is where the density is: s_min 1.0857 vs big1's 1.0837 against a
    # 1.1088 base, i.e. 92% of the reachability gain, and 25 tighter / 0 LOOSER.
    "nosize":    {"ICCAD_L268": "4"},
    "nosize269": {"ICCAD_L268": "4", "ICCAD_L269": "1"},
    "hoist1":    {"ICCAD_L268": "3"},
    "bigfree":   {"ICCAD_L268": "6"},
    # the two together
    "both":    {"ICCAD_L267": "1", "ICCAD_L268": "1"},
    # references
    "dense":   {"ICCAD_FRAME_SCALES": DENSE},
    "ship":    {},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--arms", default="adapt,big1,big2")
    ap.add_argument("--out", default="l267_quality.pkl")
    ap.add_argument("--probe", default="constructive_l267.exe")
    ap.add_argument("--gate", default="",
                    help="per-profile wall pkl; apply each arm only to profiles "
                         "whose armed time fits under the shipped max-setter")
    a = ap.parse_args(_ARGV[1:])

    global PROBE
    PROBE = DIR / a.probe
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
    want = [x for x in a.arms.split(",") if x]
    for w in want:
        if w not in ALL:
            print("!! unknown arm {}".format(w))
            return 1
    ARMS = {k: ALL[k] for k in want}
    print("[l267q] probe {}  arms: {}".format(PROBE.name, want))

    # AFFORDABLE-SET deployment. The grader's profile phase is MAX-bound at n=120,
    # so a mechanism that only slows down profiles which are NOT the max-setter
    # costs nothing on the number that decides the score. Enable the arm only on
    # profiles whose armed time still fits under the shipped max-setter; every other
    # profile keeps the shipped behaviour, so the pool max is unchanged BY
    # CONSTRUCTION. 🚨 The set is fitted to measured timings, which is L258's exact
    # failure mode -- it has to be shown to transfer before it means anything.
    GATE = {}
    if a.gate:
        wt = pickle.load(open(DIR / a.gate, "rb"))
        mx = max(wt["ship"].values())
        for nm in ARMS:
            if nm in wt:
                GATE[nm] = set(p for p, t in wt[nm].items() if t <= mx)
                print("[l267q]   gate {:8s} |A| = {}/{}  (ship max-setter {:.3f}s)"
                      .format(nm, len(GATE[nm]), len(wt["ship"]), mx))
            else:
                print("[l267q]   gate {:8s} NOT TIMED -> ungated".format(nm))

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
    for key in keys:
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

        def capture(extra, allow=None):
            got.clear()

            def spy(p, inp, bc):
                env = dict(os.environ)
                env.update(p)              # !! the profile dict beats the shell
                env["ICCAD_L252"] = "1"    # stderr only -- gated 102/102 PASS
                kk0 = key_of.get(tuple(sorted(p.items())))
                if allow is None or kk0 in allow:
                    env.update(extra)
                r = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, env=env, timeout=TO)
                pos = oc._parse_output(r.stdout, bc)
                packs = 0
                for line in r.stderr.splitlines():
                    if line.startswith("L267PACKS "):
                        packs = int(line.split()[1])
                kk = key_of.get(tuple(sorted(p.items())))
                if kk is not None and pos is not None:
                    with lock:
                        got[kk] = (pos, packs)
                return pos
            orig = oc._run_profile
            oc._run_profile = spy
            try:
                m67._solve_one(oc.MyOptimizer(verbose=False), lay)
            finally:
                oc._run_profile = orig
            out = {}
            for i, (pos, packs) in got.items():
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
                              vr=float(c.violations_relative), packs=packs)
            return out

        caps = {k: capture(v, GATE.get(k)) for k, v in ARMS.items()}
        if any(len(v) < 2 for v in caps.values()):
            continue
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
            # the grader's profile phase is MAX-bound at n=120, so the number that
            # decides the bill is the heaviest profile, not the mean.
            row[nm + "_pkmax"] = max(v["packs"] for v in cp.values())
            row[nm + "_pksum"] = sum(v["packs"] for v in cp.values())
        # Per-profile PROXY METRICS as well as cost, for both the base and every
        # arm. Without them a mixed pool (arm on some profiles, base on the rest)
        # cannot be simulated offline -- the selector's hmin is a whole-pool
        # quantity, so a 2-way approximation is not the same thing.
        for i in sorted(set(bc).intersection(*[set(c) for c in caps.values()])):
            d = dict(n=n, prof=i, base=bc[i]["cost"], sumA=e["sumA"],
                     base_area=e["recs"][i]["area"], base_hpwl=e["recs"][i]["hpwl"],
                     base_vrel=e["recs"][i]["vrel"])
            for nm, cp in caps.items():
                d[nm] = cp[i]["cost"]
                d[nm + "_pk"] = cp[i]["packs"]
                d[nm + "_area"] = cp[i]["area"]
                d[nm + "_hpwl"] = cp[i]["hpwl"]
                d[nm + "_vrel"] = cp[i]["vrel"]
                d[nm + "_feas"] = cp[i]["feas"]
            per.append(d)
        rows.append(row)
        print("   n={:3d}  base {:.6f}   ".format(n, row["base"]) +
              "   ".join("{} {:.6f} ({:+.3f}%)".format(
                  nm, row[nm], 100 * (row[nm] / row["base"] - 1)) for nm in ARMS))

    if not rows:
        print("nothing scored")
        return 1
    pickle.dump({"rows": rows, "per": per, "arms": want},
                open(DIR / a.out, "wb"))

    def wm(f, rs=None):
        rs = rs if rs is not None else rows
        sw = sum(math.exp(r["n"] / 12.0) for r in rs)
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rs) / max(sw, 1e-18)

    wb = wm(lambda r: r["base"])
    print()
    print("=" * 76)
    print("L267/L268 -- PORTFOLIO, official strict scorer, {} cases".format(len(rows)))
    print("=" * 76)
    print("  base (shipped)             {:.6f}".format(wb))
    for nm in ARMS:
        w = wm(lambda r: r[nm])
        print("  {:24s}   {:.6f}   {:+.4f}%   feasible {}/{}".format(
            nm, w, 100 * (w - wb) / wb,
            sum(1 for r in rows if r[nm + "_feas"]), len(rows)))
    print()
    print("  gaps at the picked layout (L251 ref: hpwl 0.2766 area 0.2256 vrel 0.0857):")
    for nm in ARMS:
        print("    {:9s} hpwl {:.4f}  area {:.4f}  vrel {:.4f}".format(
            nm, wm(lambda r: r[nm + "_hg"]), wm(lambda r: r[nm + "_ag"]),
            wm(lambda r: r[nm + "_vr"])))
    print()
    print("  pack ATTEMPTS (max over profiles / whole pool)  -- NOT a wall")
    print("  proxy: the max-by-packs profile is not the max-by-TIME profile.")
    print("  For a wall proxy run l269_wallproxy.py against a timing pkl.")
    if "ship" in ARMS:
        b_mx = wm(lambda r: r["ship_pkmax"])
        b_sm = wm(lambda r: r["ship_pksum"])
        for nm in ARMS:
            print("    {:9s} max {:6.1f} ({:.3f}x)   sum {:7.1f} ({:.3f}x)".format(
                nm, wm(lambda r: r[nm + "_pkmax"]),
                wm(lambda r: r[nm + "_pkmax"]) / max(b_mx, 1e-9),
                wm(lambda r: r[nm + "_pksum"]),
                wm(lambda r: r[nm + "_pksum"]) / max(b_sm, 1e-9)))
    else:
        for nm in ARMS:
            print("    {:9s} max {:6.1f}   sum {:7.1f}".format(
                nm, wm(lambda r: r[nm + "_pkmax"]), wm(lambda r: r[nm + "_pksum"])))
    print()
    print("  MECHANISM, isolated (same profile, base vs arm):")
    for nm in ARMS:
        mv = [x for x in per if abs(x[nm] - x["base"]) > 1e-12]
        if not mv:
            print("    {:9s} no layout changed".format(nm))
            continue
        sw = sum(math.exp(x["n"] / 12.0) for x in mv)
        pb = sum(math.exp(x["n"] / 12.0) * x["base"] for x in mv) / sw
        pa = sum(math.exp(x["n"] / 12.0) * x[nm] for x in mv) / sw
        print("    {:9s} changed {}/{}   {:.6f} -> {:.6f}  {:+.4f}%"
              "   better {} worse {}".format(
                  nm, len(mv), len(per), pb, pa, 100 * (pa - pb) / pb,
                  sum(1 for x in mv if x[nm] < x["base"] - 1e-12),
                  sum(1 for x in mv if x[nm] > x["base"] + 1e-12)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
