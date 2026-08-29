"""L254 -- cliff anatomy: at the tightest frame that FAILS, what exactly failed?

L252 put the packer's ceiling at ~81.3% utilisation and showed 83.7% of the area
deficit sits there. This asks why. Per case, on the proxy-WINNING profile, with a
dense scale ladder, the run reports every failed pack; the one that matters is the
LAST failure below the first success -- the cliff edge.

The discriminator is not "which block" but "was there room":

    free = fw*fh - placed_area       space left in the frame when the pack gave up
    need = iarea                     area of the item that found no origin

    free >> need   the space EXISTS and the greedy could not use it -> FRAGMENTATION,
                   which a backtrack or a repair pass could in principle cross
    free <= need   the frame is genuinely full -> GEOMETRY, and the ceiling is real

  <python> l254_anatomy.py --limit 40
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
    print("[l254] {} cases, dense ladder {} rungs".format(len(keys), nst))

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

    rows = []
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
            ok = grab.get("inp") == inp
            print("[gate] direct input == deployment input : {}".format(
                "PASS" if ok else "FAIL"))
            if not ok:
                return 1
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
        first_ok = min(oks)
        below = [i for i in fails if i < first_ok]
        if not below:
            continue                       # tightest candidate packed: no cliff here
        edge = max(below)                  # the last failure before the first success
        f = fails[edge]
        tot = tot or e["sumA"]
        areas = np.asarray([max(0.0, float(lay["at"][i])) for i in range(n)])
        s_edge = math.sqrt(max(f["fw"] * f["fh"], 1e-18) / max(tot, 1e-18))
        free = f["fw"] * f["fh"] - f["parea"]
        rows.append(dict(
            n=n, kind=f["kind"], ndone=f["ndone"], N=f["N"], nblk=f["nblk"],
            frac_done=f["ndone"] / max(f["N"], 1), s_edge=s_edge,
            iarea=f["iarea"], free=free,
            util_fail=f["parea"] / max(f["fw"] * f["fh"], 1e-18),
            unplaced=f["N"] - f["ndone"],
            free_over_need=free / max(f["iarea"], 1e-18),
            # the airtight version: free space vs the area of EVERY block still
            # unplaced, not just the one that happened to fail. >1 means the frame
            # could still hold all of them and the greedy simply cannot reach it.
            slack_ratio=(f["fw"] * f["fh"] - f["parea"]) / max(tot - f["parea"], 1e-18),
            pct=float((areas < f["iarea"]).mean()) if f["nblk"] == 1 else float("nan"),
            iarea_rel=f["iarea"] / max(float(areas.mean()), 1e-18),
            nfail_below=len(below)))
        if (kn + 1) % 10 == 0:
            print("   {}/{}".format(kn + 1, len(keys)))

    if not rows:
        print("no cliff events measured")
        return 1

    SW = sum(math.exp(r["n"] / 12.0) for r in rows)

    def wm(f):
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rows) / SW

    print()
    print("=" * 80)
    print("L254 cliff anatomy -- the last FAILING frame below s_min, {} cases".format(
        len(rows)))
    print("=" * 80)
    print("  {:>5s} {:>7s} {:>8s} {:>9s} {:>7s} {:>8s} {:>9s} {:>8s}".format(
        "n", "kind", "s_edge", "placed", "dens@f", "need", "free", "free/need"))
    for r in sorted(rows, key=lambda r: -r["n"])[:16]:
        print("  {:5d} {:>7s} {:8.4f} {:4d}/{:<4d} {:6.1f}% {:8.3f} {:9.3f} {:8.2f}".format(
            r["n"], r["kind"], r["s_edge"], r["ndone"], r["N"],
            100.0 * r["util_fail"], r["iarea"], r["free"], r["free_over_need"]))
    if len(rows) > 16:
        print("  ... {} more".format(len(rows) - 16))

    print()
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("  failure kind: " + "  ".join(
        "{} {}/{}".format(k, v, len(rows)) for k, v in sorted(kinds.items())))
    print()
    print("  weighted by exp(n/12):")
    print("    blocks placed when it gave up       {:.1f}%   ({:.1f} blocks left)".format(
        100.0 * wm(lambda r: r["frac_done"]), wm(lambda r: r["unplaced"])))
    print("    DENSITY reached when it jammed      {:.1f}%   (frame would allow"
          " {:.1f}%)".format(100.0 * wm(lambda r: r["util_fail"]),
                             100.0 / wm(lambda r: r["s_edge"]) ** 2))
    print("    failing item area / mean block area {:.2f}x".format(
        wm(lambda r: r["iarea_rel"])))
    print("    FREE / NEED at the moment of failure {:.2f}x".format(
        wm(lambda r: r["free_over_need"])))
    print()
    frag = sum(1 for r in rows if r["free_over_need"] > 1.0)
    print("    cases where the space EXISTED (free > need): {}/{}".format(frag, len(rows)))
    lateish = sum(1 for r in rows if r["frac_done"] >= 0.9)
    print("    cases that failed after >=90% of blocks placed: {}/{}".format(
        lateish, len(rows)))
    big = sum(1 for r in rows if r["iarea_rel"] >= 2.0)
    print("    cases where the failing item is >=2x mean block: {}/{}".format(
        big, len(rows)))
    import pickle as _pk
    _pk.dump(rows, open(DIR / "l254_rows.pkl", "wb"))
    def q(f, t):
        v = sorted(f(r) for r in rows)
        return v[int(t * (len(v) - 1))]
    print()
    print("  medians (mean is dragged by outliers -- report both):")
    print("    blocks left unplaced      p10 {:.0f}  p50 {:.0f}  p90 {:.0f}"
          .format(q(lambda r: r["unplaced"], .1), q(lambda r: r["unplaced"], .5),
                  q(lambda r: r["unplaced"], .9)))
    print("    free / need               p10 {:.1f}x p50 {:.1f}x p90 {:.1f}x"
          .format(q(lambda r: r["free_over_need"], .1),
                  q(lambda r: r["free_over_need"], .5),
                  q(lambda r: r["free_over_need"], .9)))
    print("    density when it jammed    p10 {:.1f}% p50 {:.1f}% p90 {:.1f}%"
          .format(100*q(lambda r: r["util_fail"], .1),
                  100*q(lambda r: r["util_fail"], .5),
                  100*q(lambda r: r["util_fail"], .9)))
    print("    min free/need over ALL cases {:.2f}x".format(
        min(r["free_over_need"] for r in rows)))
    print("    free / ALL-unplaced-area  p10 {:.2f}x p50 {:.2f}x p90 {:.2f}x   min {:.2f}x"
          .format(q(lambda r: r["slack_ratio"], .1), q(lambda r: r["slack_ratio"], .5),
                  q(lambda r: r["slack_ratio"], .9),
                  min(r["slack_ratio"] for r in rows)))
    print("    cases where free > ALL unplaced blocks: {}/{}".format(
        sum(1 for r in rows if r["slack_ratio"] > 1.0), len(rows)))
    print()
    print("  VERDICT INPUT: fragmentation (free>need AND late) vs geometry")
    both = sum(1 for r in rows
               if r["free_over_need"] > 1.0 and r["frac_done"] >= 0.9)
    print("    free>need AND >=90% placed : {}/{}".format(both, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
