"""L252 step 0b -- resolve the cliff edge that the shipped ladder cannot see.

l252_gap.py showed the coarse s_min is an UPPER BOUND in 27/40 cases: scale 1.05
fails, 1.10 packs, and the true edge is somewhere in between (median unresolved
width 0.048, max 0.100). This re-runs the proxy-WINNING profile of each case with a
dense scale ladder and reads the true tightest packable frame off L252TRY.

Only the scales are overridden -- aspects, and every other flag of that profile,
are left exactly as the pool builds them. The clamp in frame_candidates()
(w >= max(pre_w, max_iw) + FRAME_EPS) still applies, so a case whose floor is set
by its widest block still reports that floor: that one is real geometry, not
granularity.

Gate: the input handed to the binary is built directly here rather than through a
portfolio solve, so the first case checks that construction byte-for-byte against
the one the deployment path actually produces.

  <python> l252_fine.py --limit 40
"""
import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PROBE = DIR / "constructive_l252.exe"
CACHE = DIR / "l252_cache.pkl"


def parse_l252(stderr):
    tot, frames, tries = None, [], {}
    for line in stderr.splitlines():
        if line.startswith("L252TOT "):
            tot = float(line.split()[1])
        elif line.startswith("L252FRM "):
            _, i, w, h = line.split()
            frames.append((int(i), float(w), float(h)))
        elif line.startswith("L252TRY "):
            _, i, ok, sc = line.split()
            tries[int(i)] = (int(ok), float(sc))
    return tot, frames, tries


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

    nsteps = int(round((a.hi - a.lo) / a.step)) + 1
    LADDER = ",".join("{:.4f}".format(a.lo + i * a.step) for i in range(nsteps))
    print("[l252] dense ladder: {} rungs {:.2f}..{:.2f}".format(nsteps, a.lo, a.hi))

    C = pickle.load(open(CACHE, "rb"))
    specs = [(ck, fk, L, n) for ck, fk, L, n in m77._specs(a.sample)]
    spec_of = {ck: (fk, L, n) for ck, fk, L, n in specs}

    def winner_of(e):
        idxs = sorted(e["recs"])
        met = [e["recs"][i] for i in idxs]
        A_hat = 1.035 * max(e["sumA"], 1e-9)
        hmin = min(m["hpwl"] for m in met) or 1.0
        prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
                for m in met]
        k = min(range(len(idxs)), key=lambda j: prox[j])
        return idxs[k], met[k]

    def build_inp(lay):
        n = lay["n"]
        otp = m67.build_opt_target_pos(lay["tp"], lay["cons"], n)
        # _solve_impl's own condition -- _l137_env() is NON-empty at >=40 cores
        # (ICCAD_HINT_MODE unset + _effective_cores_hi() >= _L137_CORES_MIN), so
        # the shipped 48c input carries a GORDIAN hint. Passing gnn_hint=None
        # here builds a DIFFERENT case; the gate below is what caught it.
        hint = None
        if bool(oc._l137_env()) or bool(oc._l137_active(n)):
            try:
                hint = oc._gordian_hint(n, lay["at"], lay["b2b"], lay["p2b"],
                                        lay["pins"], lay["cons"], otp)
            except Exception:
                hint = None
        return oc._serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp, gnn_hint=hint)

    keys = [k for k in C if k[0] == a.sample]
    keys.sort(key=lambda k: -C[k]["n"])
    keys = keys[:a.limit]

    gated = False
    rows = []
    cache_d = {}
    for ki, key in enumerate(keys):
        ck = key[1]
        e = C[key]
        fk, L, n = spec_of[ck]
        if fk not in cache_d:
            cache_d.clear()
            cache_d[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(cache_d[fk], L)
        inp = build_inp(lay)

        if not gated:
            # prove the directly-built input IS the deployment input
            grab = {}

            def spy(p, i2, bc):
                grab.setdefault("inp", i2)
                return None
            orig = oc._run_profile
            oc._run_profile = spy
            try:
                opt = oc.MyOptimizer(verbose=False)
                m67._solve_one(opt, lay)
            except Exception:
                pass
            finally:
                oc._run_profile = orig
            same = grab.get("inp") == inp
            print("[gate] direct input == deployment input : {}".format(
                "PASS" if same else "FAIL"))
            if not same:
                print("!! refusing to sweep on an input the pool does not build")
                return 1
            gated = True

        widx, w = winner_of(e)
        prof = dict(oc._PROFILES[widx])
        ov = oc._profile_env(widx, n)
        if ov:
            prof.update(ov)
        env = dict(os.environ)
        env.update(prof)
        env["ICCAD_FRAME_SCALES"] = LADDER
        env["ICCAD_L252"] = "1"
        r = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                           text=True, env=env, timeout=TO)
        tot, frames, tries = parse_l252(r.stderr)
        if not frames or not tries:
            print("   n={} produced no ladder -- skipped".format(n))
            continue
        tot = tot or e["sumA"]
        s_of = {i: math.sqrt(max(ww * hh, 1e-18) / max(tot, 1e-18))
                for i, ww, hh in frames}
        oks = [i for i, (ok, _s) in tries.items() if ok]
        if not oks:
            print("   n={} nothing packed on the dense ladder -- skipped".format(n))
            continue
        fine = min(s_of[i] for i in oks)

        # the coarse numbers, recomputed from the same cache entry
        cs_of = {i: math.sqrt(max(ww * hh, 1e-18) / max(w["tot"] or tot, 1e-18))
                 for i, ww, hh in w["frames"]}
        coks = [i for i, (ok, _s) in w["tries"].items() if ok]
        coarse = min(cs_of[i] for i in coks) if coks else float("nan")
        bi, bs = None, None
        for i in sorted(w["tries"]):
            ok, sc = w["tries"][i]
            if ok and (bs is None or sc < bs):
                bi, bs = i, sc
        landed = cs_of[bi] if bi is not None else float("nan")
        rows.append(dict(n=n, prof=widx, coarse=coarse, fine=fine, landed=landed,
                         nfrm=len(frames), nfail=len(frames) - len(oks)))
        if (ki + 1) % 10 == 0:
            print("   {}/{}".format(ki + 1, len(keys)))

    if not rows:
        print("nothing measured")
        return 1

    S_LABEL = 1.0 / math.sqrt(0.966)
    SW = sum(math.exp(r["n"] / 12.0) for r in rows)

    def wm(f):
        return sum(math.exp(r["n"] / 12.0) * f(r) for r in rows) / SW

    print()
    print("=" * 72)
    print("L252 step 0b -- true cliff edge, {} cases".format(len(rows)))
    print("=" * 72)
    print("  {:>5s} {:>5s} {:>9s} {:>9s} {:>9s} {:>8s}".format(
        "n", "prof", "s_landed", "s_coarse", "s_fine", "gained"))
    for r in sorted(rows, key=lambda r: -r["n"])[:16]:
        print("  {:5d} {:5d} {:9.4f} {:9.4f} {:9.4f} {:8.4f}".format(
            r["n"], r["prof"], r["landed"], r["coarse"], r["fine"],
            r["coarse"] - r["fine"]))
    if len(rows) > 16:
        print("  ... {} more".format(len(rows) - 16))

    wl, wc, wf = wm(lambda r: r["landed"]), wm(lambda r: r["coarse"]), wm(lambda r: r["fine"])
    print()
    print("  weighted by exp(n/12):")
    print("    s_label                 {:.4f}   util {:5.1f}%".format(
        S_LABEL, 100.0 / S_LABEL ** 2))
    print("    s_landed  (shipped)     {:.4f}   util {:5.1f}%".format(wl, 100.0 / wl ** 2))
    print("    s_min     coarse ladder {:.4f}   util {:5.1f}%".format(wc, 100.0 / wc ** 2))
    print("    s_min     DENSE ladder  {:.4f}   util {:5.1f}%".format(wf, 100.0 / wf ** 2))
    print()
    print("    reachable area if s_landed -> s_fine   {:+.2f}%".format(
        100.0 * (wm(lambda r: r["fine"] ** 2) / wm(lambda r: r["landed"] ** 2) - 1.0)))
    print("    residual cliff  s_fine vs s_label      {:+.2f}% of area".format(
        100.0 * (wm(lambda r: r["fine"] ** 2) / S_LABEL ** 2 - 1.0)))
    nmoved = sum(1 for r in rows if r["fine"] < r["coarse"] - 1e-9)
    print("    cases where the dense ladder found a tighter packable frame: {}/{}".format(
        nmoved, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
