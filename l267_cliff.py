"""L267/L268 -- the cliff and the pack bill, in one pass.

Two independent questions, both answered from the same per-case run:

  L268  does BIG-FIRST commitment order lower s_min?  (handoff 3.4)
        Same protocol as l262_smin.py: same case, same profile, same dense
        26-rung ladder, flag off then on. s_min falling IS the claim -- M26
        measured ordering as worth +0.005% of COST inside the existing frame
        regime, so cost is not the question here; reachability is.

  L267  where does the ADAPTIVE search land, and what does it cost?  (handoff 2.1)
        s* vs the shipped ladder's own tightest packable rung, and L267PACKS vs
        the shipped ladder's L267PACKS. The pack count is a DETERMINISTIC cost
        signal -- a wall-clock gate is run-to-run non-deterministic (L158) and
        cannot be compared byte-for-byte across arms.

  <python> l267_cliff.py --cases 12
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l267.exe"
_ARGV = list(sys.argv)


def parse(stderr):
    tries, frames, tot, packs, sel = {}, [], None, None, None
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
        elif line.startswith("L267SEL "):
            p = line.split()
            sel = (float(p[1]), float(p[2]), int(p[3]),
                   float(p[4]) if len(p) > 4 else float(p[1]),
                   float(p[5]) if len(p) > 5 else float("nan"))
    return tries, frames, tot, packs, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--probes", default="5")
    ap.add_argument("--out", default="l267_cliff.pkl")
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

        def run(extra, ladder=None):
            env = dict(os.environ)
            env.update(prof)                 # !! the profile dict beats the shell
            if ladder is not None:
                env["ICCAD_FRAME_SCALES"] = ladder
            env["ICCAD_L252"] = "1"
            env.update(extra)
            r = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                               text=True, env=env, timeout=TO)
            tries, frames, tot, packs, sel = parse(r.stderr)
            t = tot or e["sumA"]
            s = {i: math.sqrt(max(w * h, 1e-18) / max(t, 1e-18)) for i, w, h in frames}
            oks = [i for i, o in tries.items() if o]
            smin = min((s[i] for i in oks), default=None)
            return dict(smin=smin, packs=packs, sel=sel, nfrm=len(frames),
                        ntry=len(tries), nok=len(oks))

        r = dict(n=n)
        r["off"] = run({}, LADDER)                       # dense, shipped order
        r["big1"] = run({"ICCAD_L268": "1"}, LADDER)     # dense, area desc
        r["big2"] = run({"ICCAD_L268": "2"}, LADDER)     # dense, maxdim desc
        r["ship"] = run({})                              # profile's own ladder
        r["adapt"] = run({"ICCAD_L267": "1", "ICCAD_L267_PROBES": a.probes})
        rows.append(r)

        def u(x):
            return 100.0 / x ** 2 if x else float("nan")
        print("   n={:3d} | dense s_min  off {:.4f} ({:.1f}%)  big1 {:.4f}  big2 {:.4f}"
              .format(n, r["off"]["smin"] or 0, u(r["off"]["smin"]),
                      r["big1"]["smin"] or 0, r["big2"]["smin"] or 0))
        sel = r["adapt"]["sel"]
        print("        | ship  s_min {:.4f}  packs {:4d}   ->   adapt s_eff {:.4f}"
              "  packs {:4d}  (probes {}, floor {:.4f})"
              .format(r["ship"]["smin"] or 0, r["ship"]["packs"] or 0,
                      sel[3] if sel else float("nan"), r["adapt"]["packs"] or 0,
                      sel[2] if sel else -1, sel[4] if sel else float("nan")))

    if not rows:
        print("nothing measured")
        return 1
    pickle.dump(rows, open(DIR / a.out, "wb"))

    def wm(f):
        sw = sum(math.exp(r["n"] / 12.0) for r in rows)
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rows) / max(sw, 1e-18)

    print()
    print("=" * 74)
    print("L268 -- does big-first lower the cliff?   dense ladder, {} cases"
          .format(len(rows)))
    print("=" * 74)
    base = wm(lambda r: r["off"]["smin"])
    print("  shipped order   s_min {:.4f}   util {:.2f}%".format(base, 100 / base ** 2))
    for nm in ("big1", "big2"):
        w = wm(lambda r: r[nm]["smin"])
        tighter = sum(1 for r in rows if r[nm]["smin"] < r["off"]["smin"] - 1e-9)
        looser = sum(1 for r in rows if r[nm]["smin"] > r["off"]["smin"] + 1e-9)
        print("  {:14s}  s_min {:.4f}   util {:.2f}%   area {:+.2f}%"
              "   tighter {} / looser {} / same {}".format(
                  nm, w, 100 / w ** 2, 100 * (w ** 2 / base ** 2 - 1),
                  tighter, looser, len(rows) - tighter - looser))
    print()
    print("=" * 74)
    print("L267 -- adaptive vs the shipped ladder")
    print("=" * 74)
    ss = wm(lambda r: r["ship"]["smin"])
    sa = wm(lambda r: r["adapt"]["sel"][3] if r["adapt"]["sel"] else r["ship"]["smin"])
    sf = wm(lambda r: r["off"]["smin"])
    pk_s = wm(lambda r: r["ship"]["packs"] or 0)
    pk_a = wm(lambda r: r["adapt"]["packs"] or 0)
    pk_d = wm(lambda r: r["off"]["packs"] or 0)
    print("  tightest packable rung   ship {:.4f}   adapt s_eff {:.4f}  dense {:.4f}"
          .format(ss, sa, sf))
    print("    -> area vs ship        {:+.2f}%   (dense reference {:+.2f}%)"
          .format(100 * (sa ** 2 / ss ** 2 - 1), 100 * (sf ** 2 / ss ** 2 - 1)))
    print("  packs (deterministic)    ship {:.1f}   adapt {:.1f} ({:.3f}x)"
          "   dense {:.1f} ({:.3f}x)".format(
              pk_s, pk_a, pk_a / max(pk_s, 1e-9), pk_d, pk_d / max(pk_s, 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
