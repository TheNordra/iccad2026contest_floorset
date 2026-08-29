"""L252 step 0 -- where does the frame ladder land, and where is the cliff?

area_gap 0.2256 on the heavy band says our bbox is 22.6% bigger than the label's.
frame_candidates() builds frames of area SumA * s^2, so a frame's utilisation is
exactly 1/s^2 and the whole ladder is one number per rung:

    s = 1.00 -> 100.0%   s = 1.05 -> 90.70%   s = 1.10 -> 82.64%   s = 1.20 -> 69.44%

and the label sits at 96.6% == s 1.0174, between rung 1 and rung 2, where no rung
exists. This measures three quantities per case, on the profile the proxy actually
ships, WITHOUT reading any label:

    s_min      tightest frame that packs at all        -> the CLIFF, on our packer
    s_landed   the frame layout_score selects          -> where we land
    s_eff      sqrt(bbox / SumA) of the final layout   -> what we actually achieve

s_landed - s_min is reachable slope. s_min - 1.0174 is the cliff, and is the half
of the frame axis that no ladder change can buy.

  <python> l252_frames.py --sample s1 --nmin 101 --limit 40
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
PROBE = DIR / "constructive_l252.exe"
CACHE = DIR / "l252_cache.pkl"


def parse_l252(stderr):
    tot, frames, tries = None, [], {}
    for line in stderr.splitlines():
        if line.startswith("L252TOT "):
            tot = float(line.split()[1])
        elif line.startswith("L252FRM "):
            _, i, w, h = line.split()
            frames.append((int(i), float(w), float(h)))
        elif line.startswith("L252TRY "):
            _, i, ok, sc = line.split()
            tries[int(i)] = (int(ok), float(sc))
    return tot, frames, tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--nmin", type=int, default=101)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    sys.argv = ["x"]
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
    import optimizer_constructive as oc

    npool = len(list(oc._pool_indices(120)))
    print("[l252] pool at n=120: {} profiles".format(npool))
    if npool != 51:
        print("!! not the shipped pool -- refusing to measure it")
        return 1
    RH = oc._RH
    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)

    specs = [(ck, fk, L, n) for ck, fk, L, n in m77._specs(a.sample) if n >= a.nmin]
    specs.sort(key=lambda t: -t[3])
    specs = specs[:a.limit]
    print("[l252] {} cases n>={}  sample {}".format(len(specs), a.nmin, a.sample))

    C = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    lock = threading.Lock()

    def capture(lay):
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

        def spy(p, inp, block_count):
            env = dict(os.environ)
            env.update(p)
            env["ICCAD_L252"] = "1"
            r = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                               text=True, env=env, timeout=TO)
            pos = oc._parse_output(r.stdout, block_count)
            kk = key_of.get(tuple(sorted(p.items())))
            if kk is not None and pos is not None:
                with lock:
                    got[kk] = (pos, parse_l252(r.stderr))
            return pos

        orig = oc._run_profile
        oc._run_profile = spy
        try:
            opt = oc.MyOptimizer(verbose=False)
            m67._solve_one(opt, lay)
        finally:
            oc._run_profile = orig
        return got

    rows = []
    byf = {}
    for ck, fk, L, n in specs:
        byf.setdefault(fk, []).append((ck, L, n))
    done = 0
    for fk in sorted(byf):
        d = None
        for ck, L, n in byf[fk]:
            key = (a.sample, ck)
            if key not in C:
                if d is None:
                    d = torch.load(m67._path_of(fk))
                lay = m67._load_case(d, L)
                got = capture(lay)
                sumA = sum(max(0.0, float(lay["at"][i])) for i in range(n))
                recs = {}
                for i, (pos, l252) in got.items():
                    try:
                        m = oc._proxy_metrics(pos, lay["at"], lay["b2b"],
                                              lay["p2b"], lay["pins"],
                                              lay["cons"], n)
                    except Exception:
                        continue
                    recs[i] = dict(
                        pos=[tuple(map(float, q)) for q in pos],
                        area=m["area"], hpwl=m["hpwl"], vrel=m["vrel"],
                        tot=l252[0], frames=l252[1], tries=l252[2])
                C[key] = dict(n=n, sumA=sumA, recs=recs)
                pickle.dump(C, open(CACHE, "wb"))
            e = C[key]
            if len(e["recs"]) < 2:
                continue
            idxs = sorted(e["recs"])
            met = [e["recs"][i] for i in idxs]
            A_hat = 1.035 * max(e["sumA"], 1e-9)
            hmin = min(m["hpwl"] for m in met) or 1.0
            prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin)
                    * math.exp(2.0 * m["vrel"]) for m in met]
            k = min(range(len(idxs)), key=lambda j: prox[j])
            w = met[k]
            tot = w["tot"] or e["sumA"]
            s_of = {}
            for i, ww, hh in w["frames"]:
                s_of[i] = math.sqrt(max(ww * hh, 1e-18) / max(tot, 1e-18))
            oks = [i for i, (ok, _sc) in sorted(w["tries"].items()) if ok]
            if not oks or not s_of:
                continue
            s_min = min(s_of[i] for i in oks)
            best_i, best_sc = None, None
            for i in sorted(w["tries"]):
                ok, sc = w["tries"][i]
                if ok and (best_sc is None or sc < best_sc):
                    best_i, best_sc = i, sc
            s_landed = s_of[best_i]
            xs = w["pos"]
            x0 = min(q[0] for q in xs)
            y0 = min(q[1] for q in xs)
            x1 = max(q[0] + q[2] for q in xs)
            y1 = max(q[1] + q[3] for q in xs)
            bbox = max((x1 - x0) * (y1 - y0), 1e-18)
            s_eff = math.sqrt(bbox / max(tot, 1e-18))
            nfail = 0
            for _i, (ok, _s) in w["tries"].items():
                if not ok:
                    nfail += 1
            rows.append(dict(ck=ck, n=e["n"], prof=idxs[k], nfrm=len(w["frames"]),
                             ntry=len(w["tries"]), nfail=nfail,
                             s_min=s_min, s_landed=s_landed, s_eff=s_eff,
                             first_ok=min(oks), best_i=best_i))
            done += 1
            if done % 10 == 0:
                print("   {}/{}".format(done, len(specs)))

    if not rows:
        print("nothing measured")
        return 1
    S_LABEL = 1.0 / math.sqrt(0.966)
    SW = sum(math.exp(r["n"] / 12.0) for r in rows)

    def wm(f):
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rows) / SW

    print()
    print("=" * 78)
    print("L252 step 0 -- frame ladder, {} cases n>={}, sample {}".format(
        len(rows), a.nmin, a.sample))
    print("=" * 78)
    print("  {:>5s} {:>4s} {:>5s} {:>5s}  {:>8s} {:>8s} {:>8s}  {:>7s} {:>7s}".format(
        "n", "prof", "frms", "fail", "s_min", "s_land", "s_eff", "u_land", "u_eff"))
    for r in sorted(rows, key=lambda r: -r["n"])[:14]:
        print("  {:5d} {:4d} {:5d} {:5d}  {:8.4f} {:8.4f} {:8.4f}  {:6.1f}% {:6.1f}%".format(
            r["n"], r["prof"], r["nfrm"], r["nfail"], r["s_min"], r["s_landed"],
            r["s_eff"], 100.0 / r["s_landed"] ** 2, 100.0 / r["s_eff"] ** 2))
    if len(rows) > 14:
        print("  ... {} more".format(len(rows) - 14))
    print()
    print("  weighted by exp(n/12):")
    print("    s_label (96.6% util)       {:.4f}".format(S_LABEL))
    print("    s_min    tightest packable {:.4f}   util {:5.1f}%".format(
        wm(lambda r: r["s_min"]), 100.0 / wm(lambda r: r["s_min"]) ** 2))
    print("    s_landed what we select    {:.4f}   util {:5.1f}%".format(
        wm(lambda r: r["s_landed"]), 100.0 / wm(lambda r: r["s_landed"]) ** 2))
    print("    s_eff    final bbox        {:.4f}   util {:5.1f}%".format(
        wm(lambda r: r["s_eff"]), 100.0 / wm(lambda r: r["s_eff"]) ** 2))
    print()
    nsl = 0
    for r in rows:
        if r["s_landed"] > r["s_min"] + 1e-9:
            nsl += 1
    print("    SLOPE  s_landed > s_min in {}/{} cases  (a tighter frame packed"
          " and layout_score passed on it)".format(nsl, len(rows)))
    print("    CLIFF  weighted s_min - s_label      {:+.4f}".format(
        wm(lambda r: r["s_min"]) - S_LABEL))
    print("    area headroom if s_landed -> s_min   {:+.2f}%".format(
        100.0 * (wm(lambda r: r["s_landed"] ** 2)
                 / wm(lambda r: r["s_min"] ** 2) - 1.0)))
    print("    bbox vs frame (s_eff / s_landed)     {:.4f}".format(
        wm(lambda r: r["s_eff"]) / wm(lambda r: r["s_landed"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
