"""L262 -- does eviction actually lower s_min? Dense ladder, ON vs OFF.

L252 measured the packer's density ceiling as s_min = the tightest frame in the
ladder that packs. L259/L260/L261 argued that ceiling is a property of the greedy
rule, not the instance. This tests it directly on the real packer: same case, same
profile, same dense 26-rung ladder, ICCAD_L262 off then on.

s_min falling IS the claim. Everything else (quality, wall, selection) comes after.
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l262.exe"
_ARGV = list(sys.argv)


def parse(stderr):
    tries, frames, tot = {}, [], None
    for line in stderr.splitlines():
        if line.startswith("L252TOT "):
            tot = float(line.split()[1])
        elif line.startswith("L252FRM "):
            _, i, w, h = line.split()
            frames.append((int(i), float(w), float(h)))
        elif line.startswith("L252TRY "):
            _, i, ok, _s = line.split()
            tries[int(i)] = int(ok)
    return tries, frames, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--evmax", default="24")
    a = ap.parse_args(_ARGV[1:])

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
    loaded = {}
    rows = []
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

        def run(evict):
            env = dict(os.environ)
            env.update(prof)
            env["ICCAD_FRAME_SCALES"] = LADDER
            env["ICCAD_L252"] = "1"
            if evict:
                env["ICCAD_L262"] = "1"
                env["ICCAD_L262_MAX"] = a.evmax
            else:
                env.pop("ICCAD_L262", None)
            r = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                               text=True, env=env, timeout=TO)
            tries, frames, tot = parse(r.stderr)
            oks = [i for i, o in tries.items() if o]
            if not oks or not frames:
                return None, None
            t = tot or e["sumA"]
            s = {i: math.sqrt(max(w * h, 1e-18) / max(t, 1e-18)) for i, w, h in frames}
            return min(s[i] for i in oks), len(r.stdout)

        s_off, _ = run(False)
        s_on, _ = run(True)
        if s_off is None or s_on is None:
            continue
        rows.append(dict(n=n, off=s_off, on=s_on))
        print("   n={:3d}  s_min  OFF {:.4f} ({:.1f}%)  ->  ON {:.4f} ({:.1f}%)"
              "   {}".format(n, s_off, 100.0 / s_off ** 2, s_on, 100.0 / s_on ** 2,
                             "TIGHTER" if s_on < s_off - 1e-9 else "same"))

    if not rows:
        print("nothing measured")
        return 1
    pickle.dump(rows, open(DIR / "l262_smin.pkl", "wb"))
    SW = sum(math.exp(r["n"] / 12.0) for r in rows)
    wo = sum(math.exp(r["n"] / 12.0) * r["off"] for r in rows) / SW
    wn = sum(math.exp(r["n"] / 12.0) * r["on"] for r in rows) / SW
    print()
    print("=" * 62)
    print("L262 -- does eviction lower s_min?  {} cases".format(len(rows)))
    print("=" * 62)
    print("  s_min OFF   {:.4f}   util {:5.1f}%".format(wo, 100.0 / wo ** 2))
    print("  s_min ON    {:.4f}   util {:5.1f}%".format(wn, 100.0 / wn ** 2))
    print("  area if the packer landed there: {:+.2f}%".format(
        100.0 * (wn ** 2 / wo ** 2 - 1.0)))
    print("  cases tighter: {}/{}".format(
        sum(1 for r in rows if r["on"] < r["off"] - 1e-9), len(rows)))
    print()
    print("  label sits at s = 1.0174 (96.6% util).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
