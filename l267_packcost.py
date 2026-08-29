"""L267 -- what does one pack cost, as a function of how tight the frame is?

This exists because the pack counter and the stopwatch disagreed by 4x: the
adaptive search spends 1.063x the packs of the shipped ladder on the max-setter
and 1.2417x the WALL. If a pack were a unit of cost those two would match, so
they are not the same thing.

Hypothesis: a pack near the cliff is expensive. Tight frames leave fewer legal
origins, so item_candidates() generates and rejects more of them, and a FAILING
pack has already placed ~91% of the blocks (L254) before it gives up -- it pays
almost the whole bill and returns nothing.

Method: one case, one profile, a ladder with exactly ONE rung, swept across
scales. L252 reports how many packs happened, the stopwatch reports how long, so
seconds-per-pack falls out. min-of-N; only ratios are claimed.

  <python> l267_packcost.py --reps 3
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l269.exe"
_ARGV = list(sys.argv)
SCALES = [1.02, 1.05, 1.08, 1.11, 1.15, 1.25, 1.50, 2.10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args(_ARGV[1:])

    sys.argv = ["x"]
    import torch
    import pickle
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(PROBE)
    import optimizer_constructive as oc

    B = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in m77._specs("s1")}
    keys = sorted([k for k in B if k[0] == "s1"], key=lambda k: -B[k]["n"])[:a.cases]
    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)
    loaded, agg = {}, {}

    for key in keys:
        fk, L, n = spec_of[key[1]]
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
        pi = list(oc._pool_indices(n))[0]
        prof = dict(oc._PROFILES[pi])
        ov = oc._profile_env(pi, n)
        if ov:
            prof.update(ov)
        print("   n={}  profile {}".format(n, pi))
        for s in SCALES:
            best_dt, packs, ok = 1e18, 0, 0
            for _ in range(a.reps):
                env = dict(os.environ)
                env.update(prof)
                env["ICCAD_FRAME_SCALES"] = "{:.4f}".format(s)
                env["ICCAD_L252"] = "1"
                t0 = time.perf_counter()
                r = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                                   text=True, env=env, timeout=TO)
                dt = time.perf_counter() - t0
                if dt < best_dt:
                    best_dt = dt
                    packs = 0
                    ok = 0
                    for line in r.stderr.splitlines():
                        if line.startswith("L267PACKS "):
                            packs = int(line.split()[1])
                        elif line.startswith("L252TRY ") and line.split()[2] == "1":
                            ok += 1
            agg.setdefault(s, []).append((best_dt, packs, ok))
            print("      s={:.2f}  {:.3f}s  packs {:3d}  ok {:2d}  ->  {:.4f} s/pack"
                  .format(s, best_dt, packs, ok, best_dt / max(packs, 1)))

    print()
    print("=" * 66)
    print("L267 -- seconds per pack vs frame tightness ({} cases, min-of-{})"
          .format(len(keys), a.reps))
    print("=" * 66)
    ref = None
    print("  {:>6s} {:>10s} {:>8s} {:>12s} {:>10s}".format(
        "scale", "util", "packs", "s/pack", "x loosest"))
    for s in SCALES:
        v = agg.get(s) or []
        if not v:
            continue
        spp = sum(d / max(p, 1) for d, p, _ in v) / len(v)
        pk = sum(p for _, p, _ in v) / len(v)
        if ref is None or s == SCALES[-1]:
            pass
        print("  {:6.2f} {:9.1f}% {:8.1f} {:12.5f} {:>10s}".format(
            s, 100.0 / s ** 2, pk, spp, "-"))
    loose = agg.get(SCALES[-1])
    if loose:
        base = sum(d / max(p, 1) for d, p, _ in loose) / len(loose)
        print()
        print("  relative to the loosest rung (s={:.2f}):".format(SCALES[-1]))
        for s in SCALES:
            v = agg.get(s) or []
            if not v:
                continue
            spp = sum(d / max(p, 1) for d, p, _ in v) / len(v)
            print("    s={:.2f}  {:.2f}x per pack".format(s, spp / max(base, 1e-12)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
