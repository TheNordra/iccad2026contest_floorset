"""L258 step 2 -- is the shrink's benefit carried by the pool's MAX-SETTER?

If the profile phase is max-bound, running the L256 shrink only on profiles that
are NOT the max-setter costs ~0 wall. L257's greedy says 8 parents carry the whole
+0.33%: [87, 28, 20, 86, 101, 95, 94, 0]. This times all 51 profiles (shrink OFF)
to rank them, then times those 8 with the shrink ON, and reports what the new
max-setter would be.

All ratios are taken inside one batch, min-of-N, which is the only wall statement
this box can make.
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l256.exe"
_ARGV = list(sys.argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
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
    GATED = set(T["chosen"])
    print("[l258] gated set (L257 greedy): {}".format(sorted(GATED)))

    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs("s1")}
    keys = sorted([k for k in B if k[0] == "s1"], key=lambda k: -B[k]["n"])[:a.cases]
    L256ENV = {"ICCAD_L256": "1", "ICCAD_L256_RUIN": "0.12",
               "ICCAD_L256_STEP": "0.99", "ICCAD_L256_ITERS": "40",
               "ICCAD_L256_MODE": "1"}
    loaded = {}
    off_t, on_t = {}, {}
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
        for pi in list(oc._pool_indices(n)):
            prof = dict(oc._PROFILES[pi])
            ov = oc._profile_env(pi, n)
            if ov:
                prof.update(ov)
            e0 = dict(os.environ); e0.update(prof); e0.pop("ICCAD_L256", None)
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                subprocess.run([str(PROBE)], input=inp, capture_output=True,
                               text=True, env=e0, timeout=300)
                ts.append(time.perf_counter() - t0)
            off_t.setdefault(pi, []).append(min(ts))
            if pi in GATED:
                e1 = dict(os.environ); e1.update(prof); e1.update(L256ENV)
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, env=e1, timeout=300)
                    ts.append(time.perf_counter() - t0)
                on_t.setdefault(pi, []).append(min(ts))
        print("   case n={} timed".format(n))

    # per profile: mean over cases of the min-of-reps
    mo = {p: sum(v) / len(v) for p, v in off_t.items()}
    mn = {p: sum(v) / len(v) for p, v in on_t.items()}
    order = sorted(mo, key=lambda p: -mo[p])
    print()
    print("=" * 62)
    print("pool timing, shrink OFF -- slowest first")
    print("=" * 62)
    for p in order[:12]:
        tag = "  <-- GATED" if p in GATED else ""
        print("  prof {:3d}  {:.3f}s{}".format(p, mo[p], tag))
    print("  ...")
    pickle.dump(dict(off=mo, on=mn), open(DIR / "l258_times.pkl", "wb"))
    print()
    print("  gated profiles, off -> on:")
    for p2 in sorted(mn, key=lambda q: -mn[q]):
        print("    prof {:3d}  {:.3f}s -> {:.3f}s  x{:.3f}".format(
            p2, mo[p2], mn[p2], mn[p2] / max(mo[p2], 1e-9)))
    maxoff = max(mo.values())
    maxoff_p = max(mo, key=lambda p: mo[p])
    print()
    print("  pool max-setter  = prof {} at {:.3f}s".format(maxoff_p, maxoff))
    print("  is it gated?       {}".format("YES" if maxoff_p in GATED else "NO"))
    newmax = max([mo[p] for p in mo if p not in GATED] +
                 [mn.get(p, mo[p]) for p in GATED if p in mo])
    print("  max with shrink on the gated set = {:.3f}s   -> wall x{:.4f}".format(
        newmax, newmax / maxoff))
    print()
    tot_off = sum(mo.values())
    tot_on = tot_off + sum(mn[p] - mo[p] for p in mn)
    print("  sum-bound view: total work {:.2f}s -> {:.2f}s   x{:.4f}".format(
        tot_off, tot_on, tot_on / tot_off))
    print()
    print("  L248 conversion: 0.151 pp of NET per 1% of heavy-band wall")
    for lab, mult in (("max-bound", newmax / maxoff), ("sum-bound", tot_on / tot_off)):
        pct = 100 * (mult - 1)
        print("    {:10s} +{:.2f}% wall -> -{:.2f} pp  |  quality +0.331 pp"
              "  => NET {:+.2f} pp".format(lab, pct, 0.151 * pct,
                                           0.331 - 0.151 * pct))
    return 0


if __name__ == "__main__":
    sys.exit(main())
