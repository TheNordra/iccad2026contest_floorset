"""L264 -- what does a denser frame ladder cost in wall?

L263: the dense 26-rung ladder is worth -1.68% of true cost with no code change
(ICCAD_FRAME_SCALES is a shipped knob). The obvious cost is that the trial loop
walks many FAILING frames before its first success, and L254 measured those
failures as full pack attempts -- max_trials counts successes, so failures are
pure overhead.

Times all 51 profiles under three ladders, same batch, min-of-N, so only ratios
are claimed (this box's absolute wall is worthless):

    ship     each profile's own FRAME_SCALES (do NOT set the variable)
    dense    1.00 -> 1.25 step 0.01   (26 rungs)
    coarse   1.04 -> 1.16 step 0.02   ( 7 rungs)

The grader's profile phase is MAX-bound at n=120 (the research handoff's own
d_max 3.204s > sum/48 2.501s), so the number that decides it is the max-setter.

  <python> l264_wall.py --cases 3 --reps 2
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
PROBE = DIR / "constructive_l262.exe"      # arm A identical to stock; L262 stays off
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

    DENSE = ",".join("{:.4f}".format(1.00 + i * 0.01) for i in range(26))
    COARSE = ",".join("{:.4f}".format(1.04 + i * 0.02) for i in range(7))
    TUNED = "1.0600,1.0900,1.1200,1.1600"   # 4 rungs = same count as shipped
    ARMS = [("ship", None), ("dense", DENSE), ("coarse", COARSE),
            ("tuned", TUNED)]

    B = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs("s1")}
    keys = sorted([k for k in B if k[0] == "s1"], key=lambda k: -B[k]["n"])[:a.cases]
    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)
    loaded = {}
    t = {nm: {} for nm, _ in ARMS}

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
            for rep in range(a.reps):
                order = ARMS if rep % 2 == 0 else list(reversed(ARMS))
                for nm, lad in order:
                    env = dict(os.environ)
                    env.update(prof)
                    env.pop("ICCAD_L262", None)
                    if lad is None:
                        pass                      # keep the profile's own scales
                    else:
                        env["ICCAD_FRAME_SCALES"] = lad
                    t0 = time.perf_counter()
                    subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, env=env, timeout=TO)
                    dt = time.perf_counter() - t0
                    d = t[nm].setdefault(pi, {}).setdefault(n, [])
                    d.append(dt)
        print("   case n={} timed".format(n))

    # per profile: mean over cases of min-over-reps -- but we stored per rep across
    # cases, so collapse conservatively with the minimum per profile per arm
    # per profile: SUM over cases of (min over reps). Collapsing across cases with
    # a min would let the fastest case define the max-setter, which is the whole
    # quantity being measured.
    res = {}
    for nm, _ in ARMS:
        res[nm] = {p: sum(min(v) for v in bycase.values())
                   for p, bycase in t[nm].items()}
    pickle.dump(res, open(DIR / "l264_wall.pkl", "wb"))

    base = res["ship"]
    mx0 = max(base.values())
    p0 = max(base, key=lambda q: base[q])
    tot0 = sum(base.values())
    print()
    print("=" * 70)
    print("L264 -- ladder wall cost, same batch, min-of-{}".format(a.reps))
    print("=" * 70)
    print("  ship: max-setter prof {} at {:.3f}s   total work {:.2f}s".format(
        p0, mx0, tot0))
    print()
    print("  {:8s} {:>10s} {:>10s} {:>12s} {:>12s}".format(
        "arm", "max", "x max", "total", "x total"))
    out = {}
    for nm, _ in ARMS:
        r = res[nm]
        mx = max(r.values())
        tt = sum(r.values())
        out[nm] = (mx / mx0, tt / tot0)
        print("  {:8s} {:10.3f} {:10.4f} {:12.2f} {:12.4f}".format(
            nm, mx, mx / mx0, tt, tt / tot0))
    print()
    print("  L248 conversion: 0.151 pp of NET per 1% of heavy-band wall")
    print("  {:8s} {:>12s} {:>12s}".format("arm", "max-bound", "sum-bound"))
    for nm, _ in ARMS:
        if nm == "ship":
            continue
        mb = 100 * (out[nm][0] - 1)
        sb = 100 * (out[nm][1] - 1)
        print("  {:8s} {:+11.2f}% {:+11.2f}%   -> cost {:.2f} / {:.2f} pp".format(
            nm, mb, sb, 0.151 * mb, 0.151 * sb))
    print()
    print("  pair with L263's quality to get NET. Quality is +pp when cost falls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
