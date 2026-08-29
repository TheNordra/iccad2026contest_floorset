"""L268 -- the CHEAP screen: does an ordering move the density ceiling?

One binary run per (case, arm) on the dense 26-rung ladder, reading s_min off the
L252 emitters. This is the gate L262/L267 established: s_min falling IS the
reachability claim, and it costs ~1/50th of a quality run. An ordering that does
not move s_min cannot be worth scoring.

⚠️ s_min means "the greedy placed every block", not "the result is good" (L252 §5).
It is the right definition for a reachability bound and the wrong one for a quality
claim -- which is exactly why big-first passes this screen and still costs +4.98%.
So this is a NECESSARY-condition filter, never a verdict.

Arms are given as name=ENV=V[,ENV=V] on the command line.

  <python> l268_screen.py --cases 12 --arms big1=ICCAD_L268=1 hoist1=ICCAD_L268=3
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
_ARGV = list(sys.argv)


def parse(stderr):
    tries, frames, tot, packs = {}, [], None, None
    for line in stderr.splitlines():
        if line.startswith("L252TOT "):
            tot = float(line.split()[1])
        elif line.startswith("L252FRM "):
            _, i, w, h = line.split()
            frames.append((int(i), float(w), float(h)))
        elif line.startswith("L252TRY "):
            _, i, ok, _s = line.split()
            tries[int(i)] = int(ok)
        elif line.startswith("L267PACKS "):
            packs = int(line.split()[1])
    return tries, frames, tot, packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--probe", default="constructive_l270.exe")
    ap.add_argument("--ladder", default="dense", choices=["dense", "ship"])
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", default="l268_screen.pkl")
    a = ap.parse_args(_ARGV[1:])

    PROBE = DIR / a.probe
    if not PROBE.exists():
        print("!! missing {}".format(PROBE))
        return 1
    ARMS = [("off", {})]
    for spec in a.arms:
        nm, _, rest = spec.partition("=")
        env = {}
        for kv in rest.split(","):
            k, _, v = kv.partition("=")
            if k:
                env[k] = v
        ARMS.append((nm, env))
    print("[l268s] probe {}  ladder {}  arms {}".format(
        PROBE.name, a.ladder, [n for n, _ in ARMS]))

    sys.argv = ["x"]
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
    import optimizer_constructive as oc

    C = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs("s1")}
    keys = sorted([k for k in C if k[0] == "s1"], key=lambda k: -C[k]["n"])[:a.cases]
    LADDER = ",".join("{:.4f}".format(1.00 + i * 0.01) for i in range(26))
    RH = oc._RH
    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)
    loaded, rows = {}, []
    for key in keys:
        ck = key[1]
        e = C[key]
        fk, L, n = spec_of[ck]
        if fk not in loaded:
            loaded.clear()
            loaded[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(loaded[fk], L)
        otp = m67.build_opt_target_pos(lay["tp"], lay["cons"], n)
        hint = None
        # !! _l137_env() is non-empty at >=40 cores: building stdin with
        # gnn_hint=None builds a DIFFERENT case than the 48c deployment path.
        if bool(oc._l137_env()) or bool(oc._l137_active(n)):
            try:
                hint = oc._gordian_hint(n, lay["at"], lay["b2b"], lay["p2b"],
                                        lay["pins"], lay["cons"], otp)
            except Exception:
                hint = None
        inp = oc._serialize_input(n, lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                                  lay["cons"], otp, gnn_hint=hint)
        idxs = sorted(e["recs"])
        met = [e["recs"][i] for i in idxs]
        A_hat = 1.035 * max(e["sumA"], 1e-9)
        hmin = min(m["hpwl"] for m in met) or 1.0
        prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
                for m in met]
        widx = idxs[min(range(len(idxs)), key=lambda t: prox[t])]
        prof = dict(oc._PROFILES[widx])
        ov = oc._profile_env(widx, n)
        if ov:
            prof.update(ov)

        r = dict(n=n)
        for nm, extra in ARMS:
            env = dict(os.environ)
            env.update(prof)               # !! the profile dict beats the shell
            if a.ladder == "dense":
                env["ICCAD_FRAME_SCALES"] = LADDER
            env["ICCAD_L252"] = "1"
            env.update(extra)
            res = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                 text=True, env=env, timeout=TO)
            tries, frames, tot, packs = parse(res.stderr)
            t = tot or e["sumA"]
            s = {i: math.sqrt(max(w * h, 1e-18) / max(t, 1e-18)) for i, w, h in frames}
            oks = [i for i, o in tries.items() if o]
            r[nm] = dict(smin=min((s[i] for i in oks), default=None), packs=packs,
                         nok=len(oks), ntry=len(tries))
        rows.append(r)
        print("   n={:3d}  ".format(n) + "  ".join(
            "{} {:.4f}".format(nm, r[nm]["smin"] or 0) for nm, _ in ARMS))

    if not rows:
        print("nothing measured")
        return 1
    pickle.dump(rows, open(DIR / a.out, "wb"))
    rows = [r for r in rows if all(r[nm]["smin"] for nm, _ in ARMS)]

    def wm(f):
        sw = sum(math.exp(r["n"] / 12.0) for r in rows)
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rows) / max(sw, 1e-18)

    base = wm(lambda r: r["off"]["smin"])
    pk0 = wm(lambda r: r["off"]["packs"] or 0)
    print()
    print("=" * 78)
    print("L268 SCREEN -- {} ladder, {} cases.  s_min falling IS the claim."
          .format(a.ladder, len(rows)))
    print("=" * 78)
    print("  {:10s} {:>8s} {:>8s} {:>9s}  {:>18s} {:>8s}".format(
        "arm", "s_min", "util%", "area", "tighter/looser/same", "packs"))
    for nm, _ in ARMS:
        w = wm(lambda r: r[nm]["smin"])
        t = sum(1 for r in rows if r[nm]["smin"] < r["off"]["smin"] - 1e-9)
        l = sum(1 for r in rows if r[nm]["smin"] > r["off"]["smin"] + 1e-9)
        pk = wm(lambda r: r[nm]["packs"] or 0)
        print("  {:10s} {:8.4f} {:8.2f} {:+8.2f}%  {:>18s} {:8.3f}x".format(
            nm, w, 100 / w ** 2, 100 * (w ** 2 / base ** 2 - 1),
            "{}/{}/{}".format(t, l, len(rows) - t - l), pk / max(pk0, 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
