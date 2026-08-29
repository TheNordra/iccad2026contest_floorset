"""L267/L268 -- wall, same batch, min-of-N, all arms, over all 51 profiles.

l264_wall.py with env-flag arms instead of ladder strings. Only RATIOS are
claimed -- this box's absolute wall is worthless and its run-to-run spread is
>=20%. The grader's profile phase is MAX-bound at n=120 (d_max 3.204 s >
sum/48 2.501 s), so the max-setter is the number that decides it.

Arm "ship" runs the probe with no flags: byte-identical to the shipped binary
(gated 102/102), so it is the right zero.

Run this ALONE -- nothing else on the box.

  <python> l267_wall.py --cases 3 --reps 2
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l267.exe"
_ARGV = list(sys.argv)

DENSE = ",".join("{:.4f}".format(1.00 + i * 0.01) for i in range(26))
ALL = {
    "ship":  {},
    "adapt": {"ICCAD_L267": "1"},
    "big1":  {"ICCAD_L268": "1"},
    "both":  {"ICCAD_L267": "1", "ICCAD_L268": "1"},
    "adapt3": {"ICCAD_L267": "1", "ICCAD_L267_RUNGS": "3"},
    "adapt2": {"ICCAD_L267": "1", "ICCAD_L267_RUNGS": "2"},
    "l269":   {"ICCAD_L269": "1"},
    "l269b":  {"ICCAD_L269": "2"},
    "adaptp2": {"ICCAD_L267": "1", "ICCAD_L267_PROBES": "2"},
    "adaptp3": {"ICCAD_L267": "1", "ICCAD_L267_PROBES": "3"},
    "l269p1": {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "1"},
    "l269p4": {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "4"},
    "l269p2": {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "2"},
    "l269p3": {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "3"},
    "dense": {"ICCAD_FRAME_SCALES": DENSE},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--arms", default="ship,adapt,big1,both")
    ap.add_argument("--out", default="l267_wall.pkl")
    ap.add_argument("--probe", default="constructive_l267.exe")
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

    ARMS = [(k, ALL[k]) for k in a.arms.split(",") if k]
    print("[l267w] probe {}".format(PROBE.name))
    print("[l267w] arms {}".format([k for k, _ in ARMS]))
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
                for nm, ex in order:
                    env = dict(os.environ)
                    env.update(prof)
                    env.pop("ICCAD_L252", None)     # never time with emitters on
                    env.update(ex)
                    t0 = time.perf_counter()
                    subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, env=env, timeout=TO)
                    dt = time.perf_counter() - t0
                    t[nm].setdefault(pi, {}).setdefault(n, []).append(dt)
        print("   case n={} timed".format(n))

    # per profile: SUM over cases of (min over reps). Collapsing across cases with
    # a min would let the fastest case define the max-setter, which is the whole
    # quantity being measured.
    res = {nm: {p: sum(min(v) for v in bycase.values())
                for p, bycase in t[nm].items()} for nm, _ in ARMS}
    pickle.dump(res, open(DIR / a.out, "wb"))

    base = res["ship"]
    mx0, tot0 = max(base.values()), sum(base.values())
    p0 = max(base, key=lambda q: base[q])
    print()
    print("=" * 70)
    print("L267 -- wall, same batch, min-of-{}".format(a.reps))
    print("=" * 70)
    print("  ship: max-setter prof {} at {:.3f}s   total work {:.2f}s".format(
        p0, mx0, tot0))
    print()
    print("  {:8s} {:>10s} {:>10s} {:>12s} {:>12s}".format(
        "arm", "max", "x max", "total", "x total"))
    out = {}
    for nm, _ in ARMS:
        r = res[nm]
        mx, tt = max(r.values()), sum(r.values())
        out[nm] = (mx / mx0, tt / tot0)
        print("  {:8s} {:10.3f} {:10.4f} {:12.2f} {:12.4f}".format(
            nm, mx, mx / mx0, tt, tt / tot0))
    print()
    print("  L248 conversion: 0.151 pp of NET per 1% of heavy-band wall")
    print("  {:8s} {:>12s} {:>12s}".format("arm", "max-bound", "sum-bound"))
    for nm, _ in ARMS:
        if nm == "ship":
            continue
        rm, rt = out[nm]
        print("  {:8s} {:+11.3f}pp {:+11.3f}pp".format(
            nm, -0.151 * 100 * (rm - 1.0), -0.151 * 100 * (rt - 1.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
