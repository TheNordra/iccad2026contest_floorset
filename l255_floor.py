"""L255 -- how far down does the fragmentation regime go? The packer's true floor.

L254 showed the cliff EDGE is fragmentation: the greedy jams ~3.4pp below its own
frame's allowance with >=10x the needed area free. That proves the edge is soft.
It does not say how far down it stays soft, and that distance IS the prize for
adding relocation to the packer.

The metric must not be bounded by construction (L254 burned one that was). This
uses  placed_frac = placed_area / total_block_area  at the moment the pack dies:

    placed_frac ~ 1.0   the greedy got essentially all the area in and is stuck on
                        the last block or two -> a repair pass could plausibly cross
    placed_frac  low    large fractions of the design cannot be placed at all
                        -> geometry, and relocation will not save it

Per case, over every frame BELOW the first success (the whole sub-cliff ladder),
this reports the curve and then the headline:

    s_floor   the tightest frame at which placed_frac is still >= --thresh
              -> the density this packer could reach IF the last few blocks could
                 be relocated. Compare to L252's 81.3% and the label's 96.6%.

  <python> l255_floor.py --limit 40
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l254.exe"
CACHE = DIR / "l252_cache.pkl"


def parse(stderr):
    tot, frames, tries, fails = None, [], {}, {}
    for line in stderr.splitlines():
        if line.startswith("L252TOT "):
            tot = float(line.split()[1])
        elif line.startswith("L252FRM "):
            _, i, w, h = line.split()
            frames.append((int(i), float(w), float(h)))
        elif line.startswith("L252TRY "):
            _, i, ok, sc = line.split()
            tries[int(i)] = (int(ok), float(sc))
        elif line.startswith("L254FAIL "):
            p = line.split()
            fails[int(p[1])] = dict(kind=p[2], ndone=int(p[3]), N=int(p[4]),
                                    nblk=int(p[5]), iarea=float(p[6]),
                                    parea=float(p[7]), fw=float(p[8]),
                                    fh=float(p[9]))
    return tot, frames, tries, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--lo", type=float, default=1.00)
    ap.add_argument("--hi", type=float, default=1.25)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--thresh", type=float, default=0.98,
                    help="placed_frac counted as 'only the last blocks are stuck'")
    a = ap.parse_args()

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
    nst = int(round((a.hi - a.lo) / a.step)) + 1
    LADDER = ",".join("{:.4f}".format(a.lo + i * a.step) for i in range(nst))

    C = pickle.load(open(CACHE, "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs(a.sample)}
    keys = [k for k in C if k[0] == a.sample]
    keys.sort(key=lambda k: -C[k]["n"])
    keys = keys[:a.limit]
    print("[l255] {} cases, ladder {} rungs {:.2f}..{:.2f}, thresh {:.2f}".format(
        len(keys), nst, a.lo, a.hi, a.thresh))

    def build_inp(lay):
        n = lay["n"]
        otp = m67.build_opt_target_pos(lay["tp"], lay["cons"], n)
        hint = None
        if bool(oc._l137_env()) or bool(oc._l137_active(n)):
            try:
                hint = oc._gordian_hint(n, lay["at"], lay["b2b"], lay["p2b"],
                                        lay["pins"], lay["cons"], otp)
            except Exception:
                hint = None
        return oc._serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp, gnn_hint=hint)

    rows, curve = [], {}
    loaded = {}
    gated = False
    for kn, key in enumerate(keys):
        ck = key[1]
        e = C[key]
        fk, L, n = spec_of[ck]
        if fk not in loaded:
            loaded.clear()
            loaded[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(loaded[fk], L)
        inp = build_inp(lay)
        if not gated:
            grab = {}

            def spy(p, i2, bc):
                grab.setdefault("inp", i2)
                return None
            orig = oc._run_profile
            oc._run_profile = spy
            try:
                m67._solve_one(oc.MyOptimizer(verbose=False), lay)
            except Exception:
                pass
            finally:
                oc._run_profile = orig
            if grab.get("inp") != inp:
                print("[gate] direct input == deployment input : FAIL")
                return 1
            print("[gate] direct input == deployment input : PASS")
            gated = True

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
        env = dict(os.environ)
        env.update(prof)
        env["ICCAD_FRAME_SCALES"] = LADDER
        env["ICCAD_L252"] = "1"
        env["ICCAD_L254"] = "1"
        r = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                           text=True, env=env, timeout=TO)
        tot, frames, tries, fails = parse(r.stderr)
        if not frames or not tries:
            continue
        oks = [i for i, (o, _s) in tries.items() if o]
        if not oks:
            continue
        tot = tot or e["sumA"]
        s_of = {i: math.sqrt(max(w * h, 1e-18) / max(tot, 1e-18))
                for i, w, h in frames}
        first_ok = min(oks)
        s_min = min(s_of[i] for i in oks)

        # best placed_frac achieved at each scale below the cliff
        best = {}
        for i, f in fails.items():
            if i >= first_ok:
                continue
            s = s_of[i]
            pf = f["parea"] / max(tot, 1e-18)
            b = round(s, 3)
            if pf > best.get(b, -1.0):
                best[b] = pf
            curve.setdefault(round(s, 2), []).append(pf)
        if not best:
            continue
        # s_floor: the TIGHTEST scale still placing >= thresh of the area
        good = [s for s, pf in best.items() if pf >= a.thresh]
        s_floor = min(good) if good else None
        rows.append(dict(n=n, s_min=s_min, s_floor=s_floor,
                         pf_at_100=best.get(round(min(best), 3)),
                         s_lo=min(best), pf_lo=best[min(best)],
                         nfail=len(best)))
        if (kn + 1) % 10 == 0:
            print("   {}/{}".format(kn + 1, len(keys)))

    if not rows:
        print("nothing measured")
        return 1
    pickle.dump(dict(rows=rows, curve=curve), open(DIR / "l255_floor.pkl", "wb"))

    S_LABEL = 1.0 / math.sqrt(0.966)
    print()
    print("=" * 72)
    print("L255 -- the sub-cliff curve, {} cases".format(len(rows)))
    print("=" * 72)
    print("  placed_frac = placed area / total block area, at the moment the pack died")
    print("  (median over cases; a scale is only listed if some case had a frame there)")
    print()
    print("  {:>6s} {:>6s} {:>9s} {:>9s} {:>9s}".format(
        "scale", "util", "p50 pf", "p10 pf", "cases"))
    for b in sorted(curve):
        v = sorted(curve[b])
        if len(v) < 3:
            continue
        print("  {:6.2f} {:5.1f}% {:9.4f} {:9.4f} {:9d}".format(
            b, 100.0 / b ** 2, v[len(v) // 2], v[int(0.1 * (len(v) - 1))], len(v)))

    have = [r for r in rows if r["s_floor"] is not None]
    SW = sum(math.exp(r["n"] / 12.0) for r in rows)

    def wm(f, rs=None):
        rs = rs if rs is not None else rows
        sw = sum(math.exp(r["n"] / 12.0) for r in rs)
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rs) / max(sw, 1e-18)

    print()
    print("  cases where SOME sub-cliff frame still placed >= {:.0%} of the area:"
          "  {}/{}".format(a.thresh, len(have), len(rows)))
    if have:
        sf = wm(lambda r: r["s_floor"], have)
        sm = wm(lambda r: r["s_min"], have)
        print()
        print("  weighted by exp(n/12), over those {} cases:".format(len(have)))
        print("    s_min    what the packer actually reaches   {:.4f}   util {:5.1f}%"
              .format(sm, 100.0 / sm ** 2))
        print("    s_floor  tightest frame still >= {:.0%} placed  {:.4f}   util {:5.1f}%"
              .format(a.thresh, sf, 100.0 / sf ** 2))
        print("    s_label  the ground truth                  {:.4f}   util {:5.1f}%"
              .format(S_LABEL, 100.0 / S_LABEL ** 2))
        print()
        print("    PRIZE for perfect relocation of the last {:.0%}:  area {:+.2f}%"
              .format(1 - a.thresh, 100.0 * (sf ** 2 / sm ** 2 - 1.0)))
        print("    residual cliff even then, vs the label:        area {:+.2f}%"
              .format(100.0 * (sf ** 2 / S_LABEL ** 2 - 1.0)))
    print()
    print("    tightest frame ANY case reached, and what it placed there:")
    print("      s_lo p50 {:.4f}   placed_frac there p50 {:.4f}".format(
        sorted(r["s_lo"] for r in rows)[len(rows) // 2],
        sorted(r["pf_lo"] for r in rows)[len(rows) // 2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
