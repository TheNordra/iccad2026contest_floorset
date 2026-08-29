"""L257 step 3 -- what does an L256 twin cost, relative to its parent?

L248's pool-size curve was measured on PLAIN extra profiles. An L256 twin runs the
shrink loop on top of its parent, so the per-profile cost has to be measured, not
inherited. Everything here is a RATIO TAKEN INSIDE ONE BATCH (the ledger's rule:
this box's absolute wall is worthless, a same-batch ratio is not), min-of-3 both
sides, alternating order so drift cannot favour one arm.

  <python> l257_wall.py --cases 3 --profiles 6
"""
import argparse
import math
import os
import pickle
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l256.exe"
_ARGV = list(sys.argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--profiles", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args(_ARGV[1:])

    sys.argv = ["x"]
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
    import optimizer_constructive as oc

    B = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
    T = pickle.load(open(DIR / "l257_twin.pkl", "rb"))
    chosen = T["chosen"]
    print("[l257] timing the greedy's chosen parents: {}".format(chosen[:a.profiles]))

    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs(a.sample)}
    keys = sorted([k for k in B if k[0] == a.sample], key=lambda k: -B[k]["n"])[:a.cases]
    L256ENV = {"ICCAD_L256": "1", "ICCAD_L256_RUIN": "0.12",
               "ICCAD_L256_STEP": "0.99", "ICCAD_L256_ITERS": "40",
               "ICCAD_L256_MODE": "1"}
    loaded = {}
    ratios, offs, ons = [], [], []
    for key in keys:
        ck = key[1]
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
        for pi in chosen[:a.profiles]:
            prof = dict(oc._PROFILES[pi])
            ov = oc._profile_env(pi, n)
            if ov:
                prof.update(ov)
            e_off = dict(os.environ); e_off.update(prof); e_off.pop("ICCAD_L256", None)
            e_on = dict(os.environ); e_on.update(prof); e_on.update(L256ENV)
            toff, ton = [], []
            for r in range(a.reps):
                for which, envd, acc in ((0, e_off, toff), (1, e_on, ton)) if r % 2 == 0 \
                        else ((1, e_on, ton), (0, e_off, toff)):
                    t0 = time.perf_counter()
                    subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, env=envd, timeout=300)
                    acc.append(time.perf_counter() - t0)
            mo, mn = min(toff), min(ton)
            ratios.append(mn / max(mo, 1e-9))
            offs.append(mo); ons.append(mn)
            print("   n={:3d} prof={:3d}  off {:.3f}s  on {:.3f}s  x{:.3f}".format(
                n, pi, mo, mn, mn / max(mo, 1e-9)))

    ratios.sort()
    print()
    print("=" * 60)
    print("L256 twin cost, same-batch ratio, min-of-{}".format(a.reps))
    print("=" * 60)
    print("  per-profile x   p10 {:.3f}  p50 {:.3f}  p90 {:.3f}  max {:.3f}".format(
        ratios[int(.1 * (len(ratios) - 1))], ratios[len(ratios) // 2],
        ratios[int(.9 * (len(ratios) - 1))], ratios[-1]))
    print("  slowest parent  {:.3f}s   slowest twin {:.3f}s".format(max(offs), max(ons)))
    print()
    print("  The grader's profile phase is MAX-bound (wall ~ slowest profile), so")
    print("  what matters is whether a twin becomes the max-setter, not the mean.")
    print("  slowest twin / slowest parent = {:.3f}x".format(max(ons) / max(offs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
