"""L156 — A/B the lazy row families against the shipped matrix.

L155 set the bar (f* = 1.75x-2.0x), found that ~97% of LP cost is proportional
to row count, and measured solve time as **rows^1.5..1.9** -- so a row removed
is worth MORE than its share of the matrix. It also gave the post-prune census:

    hpwl 59.3% | separation 15.2% | area_tangent 14.9% | envelope 9.6%

`area_tangent` and `envelope` are 24.5% between them and neither has ever been
touched. Both are pure over-supply: 4 envelope rows per block when only the
bbox frontier can bind, and 10 tangents per reshapeable unit when 1-2 can bind.

The two knobs under test (both default OFF; unset = the shipped matrix
bit-for-bit):

    ICCAD_SHAPE_LP_LAZY_ENV=m   omit a block's envelope rows when it sits
                                further than m*bbox_span inside that edge
    ICCAD_SHAPE_LP_LAZY_TAN=j   emit only the 2j+1 tangents nearest the
                                current width

🔑 THE GATE THAT MAKES THIS DECIDABLE. Both are RELAXATIONS that solve_pruned
verifies and repairs, so they cannot change the optimum -- only how many rounds
it takes. Therefore **the objective must be identical to the baseline arm on
every case.** If it moves, the mechanism is not what it claims and its speed
number is worthless. A different LAYOUT at the same objective is degeneracy
(L119) and is expected.

  <python> -u l156_lazy_ab.py --arms base,env0.05,tan1,env0.05+tan1 --minn 100
"""
import argparse
import os
import time
from collections import Counter

import l129_global_placer as L
import optimizer_constructive as oc
import l155_lp_rows as R

_l3 = oc.l3


def parse_arm(spec):
    """'env0.05+tan1' -> ('env0.05+tan1', {'lazy_env': 0.05, 'lazy_tan': 1.0})"""
    kw = {}
    if spec != "base":
        for part in spec.split("+"):
            if part.startswith("env"):
                kw["lazy_env"] = float(part[3:])
            elif part.startswith("tan"):
                kw["lazy_tan"] = float(part[3:])
            else:
                raise SystemExit(f"bad arm component {part!r}")
    return spec, kw


def one(c, P0, kw, reps):
    key = "l156"
    n = c["n"]
    sumA = sum(max(0.0, float(c["at"][i])) for i in range(n))
    try:
        hp = oc._proxy_metrics(P0, c["at"], c["b2b"], c["p2b"], c["pins"],
                               c["cons"], n)["hpwl"]
    except Exception:
        return None
    base = {"hpwl_baseline": max(float(hp), 1e-6),
            "area_baseline": max(sumA / oc._LP_UTIL, 1e-6)}
    _l3.CASES[key] = oc._lp_build_case(n, c["at"], c["b2b"], c["p2b"],
                                       c["pins"], c["cons"], base)
    saved = oc.PRUNE_B
    oc.PRUNE_B = 8.0                       # the shipped ICCAD_SHAPE_LP_B
    best = None
    try:
        for _ in range(max(1, reps)):
            rec = R.Recorder()
            omitted = [0]
            orig = oc.build_and_solve

            def wrapped(*a, **k):
                d = orig(*a, **k)
                omitted[0] += d.get("lazy_omitted", 0)
                return d

            oc.build_and_solve = wrapped
            try:
                rec._orig = oc.build_and_solve
                with rec:
                    t0 = time.perf_counter()
                    newP, tele, _B = oc.lp_pass(key, P0, 0.06, sep_trim=True, **kw)
                    wall = time.perf_counter() - t0
            finally:
                oc.build_and_solve = orig
            if newP is None:
                return dict(status=tele.get("status", "none"), wall=wall,
                            rows=dict(rec.rows), calls=rec.calls, omitted=0,
                            t_build=rec.t_build, t_solve=rec.t_solve,
                            obj=None, lay=None)
            r = dict(status="ok", wall=wall, rows=dict(rec.rows),
                     calls=rec.calls, omitted=omitted[0],
                     t_build=rec.t_build, t_solve=rec.t_solve,
                     obj=tele["lp_obj"], lay=R._hash_layout(newP),
                     attempts=tele.get("attempts", 0))
            if best is None or r["wall"] < best["wall"]:
                best = r
            omitted[0] = 0
    finally:
        oc.PRUNE_B = saved
        _l3.CASES.pop(key, None)
        oc._HARD_MASKS.pop(key, None)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,env0.05,tan1,env0.05+tan1")
    ap.add_argument("--minn", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--layouts", default="results_L153_lpoff_L137.json")
    a = ap.parse_args()

    for k in ("ICCAD_SHAPE_LP_LAZY_ENV", "ICCAD_SHAPE_LP_LAZY_TAN"):
        if os.environ.get(k):
            raise SystemExit(f"{k} is set in the environment; this tool drives "
                             "the knobs itself and an ambient value would make "
                             "every arm the same silently")

    arms = [parse_arm(s) for s in a.arms.split(",")]
    lay = R._load_layouts(a.layouts)
    kwbase = R._lpkw()
    cases = [(i, c) for i, c in enumerate(L.CASES) if i in lay]
    if a.minn:
        cases = [(i, c) for i, c in cases if c["n"] >= a.minn]
    if a.limit:
        cases = cases[:a.limit]
    print(f"[l156] {len(cases)} cases | arms {a.arms} | reps {a.reps} "
          f"| base kw {kwbase}\n")

    tot = {s: Counter() for s, _ in arms}
    twall = {s: 0.0 for s, _ in arms}
    tsolve = {s: 0.0 for s, _ in arms}
    calls = {s: 0 for s, _ in arms}
    omit = {s: 0 for s, _ in arms}
    moved, degen, ran = [], 0, 0

    print(f"{'case':>5}{'n':>5}  " + "".join(f"{s:>22}" for s, _ in arms))
    print(f"{'':>10}  " + "".join(f"{'rows omit rnd   wall':>22}" for _ in arms))
    for i, c in cases:
        P0 = lay[i]
        if len(P0) != c["n"]:
            continue
        row, ref = f"{i:>5}{c['n']:>5}  ", None
        for s, extra in arms:
            r = one(c, P0, dict(kwbase, **extra), a.reps)
            if r is None or r["status"] != "ok":
                row += f"{'--':>22}"
                continue
            nrows = sum(r["rows"].values())
            row += f"{nrows:>7}{r['omitted']:>6}{r['calls']:>4}{r['wall']:>7.3f}"
            tot[s].update(r["rows"])
            twall[s] += r["wall"]
            tsolve[s] += r["t_solve"]
            calls[s] += r["calls"]
            omit[s] += r["omitted"]
            if ref is None:
                ref, refa = (r["obj"], r["lay"]), r["attempts"]
            else:
                rel = abs(r["obj"] - ref[0]) / max(1.0, abs(ref[0]))
                if rel > 1e-9:
                    moved.append((i, s, ref[0], r["obj"], rel, refa, r["attempts"]))
                elif r["lay"] != ref[1]:
                    degen += 1
        ran += 1
        print(row)

    print(f"\n=== totals over {ran} cases ===")
    print(f"{'arm':>16}{'rows':>9}{'hpwl':>8}{'sep':>7}{'tan':>7}{'env':>7}"
          f"{'omitted':>9}{'builds':>8}{'t_solve':>9}{'wall':>9}{'speedup':>9}")
    ref_wall = None
    for s, _ in arms:
        r = tot[s]
        nrows = sum(r.values())
        if ref_wall is None:
            ref_wall = twall[s]
        sp = (ref_wall / twall[s]) if twall[s] else float("nan")
        print(f"{s:>16}{nrows:>9}{r.get('hpwl',0):>8}{r.get('separation',0):>7}"
              f"{r.get('area_tangent',0):>7}{r.get('envelope',0):>7}"
              f"{omit[s]:>9}{calls[s]:>8}{tsolve[s]:>9.2f}{twall[s]:>9.2f}"
              f"{sp:>8.2f}x")

    print("\n=== EXACTNESS GATE (relaxation + repair => optimum must not move) ===")
    print(f"  same objective, different vertex: {degen} -- degeneracy, not a defect")
    if moved:
        print(f"  ** objective MOVED on {len(moved)} (case, arm) pairs. The lazy "
              "rows are NOT being repaired; every speed number above is void.")
        for i, s, o, nv, rel, aa, bb in sorted(moved, key=lambda m: -m[4])[:8]:
            print(f"    case {i:>3} {s}: rel {rel:.2e}  {o!r} -> {nv!r}"
                  f"   lp_pass attempts {aa} -> {bb}"
                  f"{'  <- FREEZE-SET PATH, not a repair miss' if bb != aa else ''}")
        return 1
    print("  objective MOVED: none on any arm. The relaxations are exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
