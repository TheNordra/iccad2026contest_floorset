"""L155 — what is actually IN the LP matrix, and what does pruning harder cost?

HANDOFF_2026-08-20 §5.2 is the last open line: k=2 is +0.5967% of real quality
blocked only by +23.18s, so the LP has to get cheaper. It says "L112 measured
HPWL at ~80% of all LP rows ... nobody has tried to cut the HPWL rows further"
and warns that going further "probably means giving up exactness".

TWO THINGS IN THAT PARAGRAPH LOOK WRONG, AND BOTH ARE CHECKABLE HERE.

1. The 80% is the PRE-prune matrix. `lp_pass`'s own comment says that after
   pruning, "separation is the MAJORITY of the remaining rows (56-73% on the
   heavy cases)". If that holds on the shipped L147 config, then cutting HPWL
   rows harder is an attack on the minority -- the same mistake L150 made from
   the other side when it cut shape rows.

2. Exactness is NOT a property of how conservative prune_B is. `solve_pruned`
   solves the pruned program (a provable LOWER bound), checks every dropped
   term's assumed sign, forces the offenders back in, and after `max_rounds`
   falls back to the unpruned build. So prune_B is "only a HEURISTIC for which
   terms to try dropping" -- its own docstring. Pruning harder risks REPAIR
   ROUNDS, i.e. time, never the optimum. If that is true, `ICCAD_SHAPE_LP_B`
   is already a free speed knob and no new code is needed to test the idea.

So this tool measures, per case and per prune_B, on the layouts the shipped
portfolio actually hands the LP:

    rows by origin | HPWL terms kept vs dropped | repair rounds
    t_build vs t_solve | the LP objective | a hash of the resulting layout

and gates on the thing that makes the whole question decidable: **the objective
and the layout must not move with prune_B.** If they do, the prune is not exact
at that B (or it silently fell back) and any speed number from it is worthless.

Layouts come from an LP-OFF results json -- that IS the pre-LP input `_shape_lp`
receives, so this measures the deployed path rather than l129's global placer
(which is what l134 measured, and why l134's PRUNE_B numbers do not transfer).

  <python> -u l155_lp_rows.py census --minn 100
  <python> -u l155_lp_rows.py census --b 1,2,4,8,16 --limit 12 --reps 3
"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter

import l129_global_placer as L
import optimizer_constructive as oc

_l3 = oc.l3

# The shipped L147 arm. Kept here rather than read from the environment so a
# stripped-env subprocess cannot silently measure the shipped band instead --
# that is HANDOFF_2026-08-20 §4.3 (binary override ignored) in a new location.
L147_KW = {"area_R": 1.5, "area_g": 1.10, "area_tol": 0.006, "area_price": 1.0}


def _lpkw():
    """Mirror _shape_lp's env parsing so an override still works if asked for."""
    kw = dict(L147_KW)
    r = os.environ.get("ICCAD_SHAPE_LP_R", "")
    if r not in ("", "0"):
        kw["area_R"] = float(r)
        kw["area_g"] = float(os.environ.get("ICCAD_SHAPE_LP_G", "1.05"))
        kw["area_tol"] = float(os.environ.get("ICCAD_SHAPE_LP_TOL", "0.006"))
        kw["area_price"] = float(os.environ.get("ICCAD_SHAPE_LP_PRICE", "1.0"))
    return {k: v for k, v in kw.items() if v is not None}


class Recorder:
    """Wraps oc.build_and_solve and sums one lp_pass worth of builds.

    A single lp_pass can call build_and_solve several times (one per repair
    round, plus a final unpruned fallback), and the interesting cost is the
    TOTAL of those, not the last one. Counting calls also gives the repair-round
    count for free, which is the quantity that decides whether pruning harder
    is actually cheaper.
    """

    def __init__(self):
        self.reset()
        self._orig = oc.build_and_solve

    def reset(self):
        self.calls = 0
        self.rows = Counter()
        self.kept = 0
        self.dropped = 0
        self.nnz = 0
        self.t_build = 0.0
        self.t_solve = 0.0

    def __enter__(self):
        rec = self

        def wrapped(*a, **kw):
            d = rec._orig(*a, **kw)
            rec.calls += 1
            rec.rows.update(d["rows_by_origin"])
            rec.kept += d["prune_kept"]
            rec.dropped += d["prune_dropped"]
            rec.nnz += d["nnz"]
            rec.t_build += d["timing"]["t_build"]
            rec.t_solve += d["timing"]["t_solve"]
            return d

        oc.build_and_solve = wrapped
        return self

    def __exit__(self, *exc):
        oc.build_and_solve = self._orig
        return False


def _load_layouts(path):
    j = json.load(open(path))
    return {int(r["test_id"]): [tuple(float(v) for v in q) for q in r["positions"]]
            for r in j["test_results"]}


def _hash_layout(P):
    h = hashlib.md5()
    for q in P:
        h.update(b"".join(repr(v).encode() for v in q))
    return h.hexdigest()[:12]


def one(c, P0, prune_b, kw, reps):
    """One lp_pass at this prune_B, min-of-`reps`. Returns None if it did not run.

    min-of-N because a single timing cannot order anything at this size: the
    control's own whole-run spread is 2.8% p50 / 8.9% max, and L154 saw a
    single-shot pair read a strictly-added mechanism as FASTER.
    """
    key = "l155"
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
    oc.PRUNE_B = prune_b
    best = None
    try:
        for _ in range(max(1, reps)):
            rec = Recorder()
            with rec:
                t0 = time.perf_counter()
                newP, tele, _B = oc.lp_pass(key, P0, 0.06, sep_trim=True, **kw)
                wall = time.perf_counter() - t0
            if newP is None:
                return dict(status=tele.get("status", "none"), wall=wall,
                            rows=dict(rec.rows), kept=rec.kept,
                            dropped=rec.dropped, calls=rec.calls,
                            t_build=rec.t_build, t_solve=rec.t_solve,
                            obj=None, lay=None, ok=False)
            r = dict(status="ok", wall=wall, rows=dict(rec.rows),
                     kept=rec.kept, dropped=rec.dropped, calls=rec.calls,
                     t_build=rec.t_build, t_solve=rec.t_solve,
                     obj=tele["lp_obj"], lay=_hash_layout(newP),
                     ok=oc.hard_ok(P0, newP, key))
            if best is None or r["wall"] < best["wall"]:
                best = r
    finally:
        oc.PRUNE_B = saved
        _l3.CASES.pop(key, None)
        oc._HARD_MASKS.pop(key, None)
    return best


def census(args):
    kw = _lpkw()
    lay = _load_layouts(args.layouts)
    bs = [None if b in ("none", "0") else float(b) for b in args.b.split(",")]
    cases = [(i, c) for i, c in enumerate(L.CASES) if i in lay]
    if args.minn:
        cases = [(i, c) for i, c in cases if c["n"] >= args.minn]
    if args.limit:
        cases = cases[:args.limit]
    print(f"[l155] {len(cases)} cases | prune_B {args.b} | reps {args.reps} "
          f"| kw {kw} | layouts {os.path.basename(args.layouts)}\n")

    tot = {b: Counter() for b in bs}
    twall = {b: 0.0 for b in bs}
    tsolve = {b: 0.0 for b in bs}
    tbuild = {b: 0.0 for b in bs}
    calls = {b: 0 for b in bs}
    kept = {b: 0 for b in bs}
    drop = {b: 0 for b in bs}
    mismatch = []
    degenerate = []
    ran = 0

    hdr = f"{'case':>5}{'n':>5}  " + "".join(f"{('B=' + str(b)):>26}" for b in bs)
    print(hdr)
    print(f"{'':>10}  " + "".join(f"{'rows kept/drop rnd  wall':>26}" for b in bs))
    for i, c in cases:
        P0 = lay[i]
        if len(P0) != c["n"]:
            continue
        row = f"{i:>5}{c['n']:>5}  "
        ref = None
        for b in bs:
            r = one(c, P0, b, kw, args.reps)
            if r is None:
                row += f"{'--':>26}"
                continue
            nrows = sum(r["rows"].values())
            row += (f"{nrows:>7}{r['kept']:>5}/{r['dropped']:<5}"
                    f"{r['calls']:>3}{r['wall']:>7.3f}")
            if r["status"] != "ok":
                continue
            tot[b].update(r["rows"])
            twall[b] += r["wall"]
            tsolve[b] += r["t_solve"]
            tbuild[b] += r["t_build"]
            calls[b] += r["calls"]
            kept[b] += r["kept"]
            drop[b] += r["dropped"]
            if ref is None:
                ref = (r["obj"], r["lay"])
            else:
                rel = abs(r["obj"] - ref[0]) / max(1.0, abs(ref[0]))
                if rel > 1e-9:
                    mismatch.append((i, b, ref, (r["obj"], r["lay"]), rel))
                elif r["lay"] != ref[1]:
                    degenerate.append((i, b, rel))
        ran += 1
        print(row)

    print(f"\n=== totals over {ran} cases ===")
    print(f"{'prune_B':>9}{'rows':>9}{'hpwl':>8}{'sep':>8}{'other':>7}"
          f"{'kept':>7}{'drop':>7}{'builds':>8}{'t_build':>9}{'t_solve':>9}{'wall':>9}{'speedup':>9}")
    ref_wall = None
    for b in bs:
        r = tot[b]
        nrows = sum(r.values())
        hp, sp = r.get("hpwl", 0), r.get("separation", 0)
        other = nrows - hp - sp
        if ref_wall is None:
            ref_wall = twall[b]
        sp_up = (ref_wall / twall[b]) if twall[b] else float("nan")
        print(f"{str(b):>9}{nrows:>9}{hp:>8}{sp:>8}{other:>7}"
              f"{kept[b]:>7}{drop[b]:>7}{calls[b]:>8}"
              f"{tbuild[b]:>9.2f}{tsolve[b]:>9.2f}{twall[b]:>9.2f}{sp_up:>8.2f}x")

    if bs and tot[bs[0]]:
        r = tot[bs[0]]
        nrows = sum(r.values())
        print(f"\npost-prune composition at B={bs[0]}: " +
              ", ".join(f"{k} {v} ({100.0*v/nrows:.1f}%)"
                        for k, v in sorted(r.items(), key=lambda kv: -kv[1])))

    print("\n=== EXACTNESS GATE ===")
    if len(bs) < 2:
        print("  N/A: one prune_B only, nothing to compare against.")
        return 0
    # Degeneracy is NOT inexactness. This LP is massively degenerate -- L119
    # measured Windows and Linux scipy landing on different optima of the SAME
    # program -- so a different LAYOUT at the same OBJECTIVE is just the solver
    # picking a different vertex. Only a moved objective indicts the prune. The
    # first version of this gate ORed the two and reported 74 "failures" that
    # were 3e-16 apart.
    print(f"  same objective, different vertex: {len(degenerate)} (case, B) pairs"
          " -- degeneracy, not a defect")
    if not mismatch:
        print("  objective MOVED: none. prune_B is a pure speed knob.")
        return 0
    real = [m for m in mismatch if m[4] > 1e-6]
    print(f"  objective MOVED: {len(mismatch)} pairs, {len(real)} of them by"
          " more than 1e-6 relative")
    for i2, b, a, c2, rel in sorted(mismatch, key=lambda m: -m[4])[:8]:
        print(f"    case {i2:>3} B={b}: rel {rel:.2e}  {a[0]!r} -> {c2[0]!r}")
    print("  NOTE lp_pass freezes units when a cluster breaks and retries, so a"
          " different degenerate vertex can lead to a different FREEZE SET and"
          " hence a genuinely different program. A moved objective is therefore"
          " not by itself proof the prune is inexact -- but anything above 1e-6"
          " needs that path ruled out before its speed number is used.")
    return 1 if real else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("--b", default="1,2,4,8,16")
    c.add_argument("--minn", type=int, default=0)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--reps", type=int, default=3)
    c.add_argument("--layouts", default="results_L153_lpoff_L137.json")
    a = ap.parse_args()
    return census(a)


if __name__ == "__main__":
    raise SystemExit(main())
