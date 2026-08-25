"""L239 - the last open item on the LP line: the SOLVER, 75% of the wall.

L235 took the Python half from 1.00x to 1.553x and stopped, because everything
left in it is under 4% of the LP or is required by the exactness contract. What
was never touched is the other 75%: `linprog(..., method="highs")`, where
"highs" means "let scipy choose". `highs-ds` (dual simplex) and `highs-ipm`
(interior point + crossover) have never been run on this program.

⚠️ THIS IS NOT A BIT-IDENTICAL CHANGE and must not be gated as one. The optimal
OBJECTIVE is unique, but this LP is massively degenerate -- L119 has Windows and
Linux landing on different optima of the same program -- so a different method
lands on a different vertex, and `hard_ok` then adjudicates a different layout.
Quality can move either way and needs an OOS pass. What this file measures is
only the two things that decide whether an OOS pass is worth running:

    does the OBJECTIVE agree (to solver tolerance), and is it FASTER?

A method that moves the objective by more than ~1e-9 relative is not a
degeneracy story, it is a tolerance story, and it is out.

  <python> l239_solver.py [--minn 100] [--limit 12] [--reps 3]
"""
import argparse
import os
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent
METHODS = ("highs", "highs-ds", "highs-ipm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minn", type=int, default=100)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--b", type=float, default=8.0)
    ap.add_argument("--layouts", default="results_L153_lpoff_L137.json")
    a = ap.parse_args()

    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l155_lp_rows as M
    import l129_global_placer as L
    import optimizer_constructive as oc

    real = oc.linprog
    state = {"m": "highs", "t": 0.0, "n": 0}

    def wrapped(*args, **kw):
        kw = dict(kw)
        kw["method"] = state["m"]
        t0 = time.perf_counter()
        r = real(*args, **kw)
        state["t"] += time.perf_counter() - t0
        state["n"] += 1
        return r

    oc.linprog = wrapped
    kw = M._lpkw()
    lay = M._load_layouts(a.layouts)
    cs = [(i, c) for i, c in enumerate(L.CASES) if i in lay and c["n"] >= a.minn]
    if a.limit:
        cs = cs[:a.limit]
    print("[l239] {} cases n>={}  prune_B={}  reps={}"
          .format(len(cs), a.minn, a.b, a.reps))

    out = {m: {} for m in METHODS}
    for i, c in cs:
        for m in METHODS:
            state["m"] = m
            best = None
            for _ in range(a.reps):
                state["t"], state["n"] = 0.0, 0
                t0 = time.perf_counter()
                try:
                    r = M.one(c, lay[i], a.b, kw, 1)
                except Exception as e:                      # noqa: BLE001
                    r = None
                    print("   case {} {}: raised {!r}".format(i, m, e))
                w = time.perf_counter() - t0
                if r is None:
                    break
                if best is None or w < best[0]:
                    best = (w, state["t"], state["n"], r)
            out[m][i] = best
    oc.linprog = real

    print()
    print("{:>5}{:>6}".format("case", "n")
          + "".join("{:>22}".format(m) for m in METHODS))
    print("{:>11}".format("") + "".join("{:>11}{:>11}".format("wall", "obj")
                                        for m in METHODS))
    print("-" * (11 + 22 * len(METHODS)))
    tot = {m: 0.0 for m in METHODS}
    tsolve = {m: 0.0 for m in METHODS}
    bad, moved = [], {m: 0 for m in METHODS}
    for i, c in cs:
        row = "{:>5}{:>6}".format(i, c["n"])
        ref = out["highs"].get(i)
        for m in METHODS:
            r = out[m].get(i)
            if r is None:
                row += "{:>22}".format("FAILED")
                bad.append((i, m, "no result"))
                continue
            tot[m] += r[0]
            tsolve[m] += r[1]
            o = r[3]["obj"]
            row += "{:>11.3f}".format(r[0])
            if ref and ref[3]["obj"] and o:
                rel = abs(o - ref[3]["obj"]) / max(abs(ref[3]["obj"]), 1e-12)
                row += "{:>11.1e}".format(rel)
                if rel > 1e-9:
                    bad.append((i, m, "objective moved {:.2e}".format(rel)))
            else:
                row += "{:>11}".format("-")
            if ref and r[3]["lay"] != ref[3]["lay"]:
                moved[m] += 1
        print(row)
    print("-" * (11 + 22 * len(METHODS)))
    print("{:>11}".format("TOTAL")
          + "".join("{:>11.3f}{:>11}".format(tot[m], "") for m in METHODS))
    print()
    for m in METHODS:
        sp = tot["highs"] / tot[m] if tot[m] else 0.0
        print("  {:<10} whole-LP {:.3f}s  solve {:.3f}s  vs highs {:+.3f}x  "
              "layouts moved {}/{}".format(m, tot[m], tsolve[m], sp, moved[m],
                                           len(cs)))
    print()
    if bad:
        print("!! objective/failure issues ({}):".format(len(bad)))
        for i, m, why in bad[:15]:
            print("   case {} {}: {}".format(i, m, why))
        print("   an objective that MOVES is a tolerance story, not degeneracy.")
    else:
        print("objective agrees to <1e-9 relative on every case and method.")
    print()
    print("A faster method is only a candidate: the layouts DO move (that is the")
    print("degeneracy), so it needs the judge48() invariants and an OOS pass,")
    print("not the equality gate L235 used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
