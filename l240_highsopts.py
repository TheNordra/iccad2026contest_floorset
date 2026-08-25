"""L240 - the solver's own knobs, since the METHOD turned out to be a no-op.

L239: scipy's `method="highs"` already resolves to dual simplex -- `highs` and
`highs-ds` are the same wall to 0.1% and move 0/12 layouts -- and `highs-ipm` is
5% SLOWER while moving 11/12. So the method axis is closed.

What is left inside the solver is HiGHS's own configuration. Two of its knobs
are worth a measurement because they change the PIVOT PATH, not the answer:

  simplex_dual_edge_weight_strategy   dantzig / devex / steepest / steepest-devex
  presolve                            on (default) / off

Neither is bit-identical -- a different pivot path lands on a different vertex of
the same degenerate optimum -- so the objective is the gate here, exactly as in
L239, and anything that survives still needs judge48() and an OOS pass.

  <python> l240_highsopts.py [--minn 100] [--limit 12] [--reps 3]
"""
import argparse
import os
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent

CONFIGS = [
    ("shipped", {}),
    ("dantzig", {"simplex_dual_edge_weight_strategy": "dantzig"}),
    ("devex", {"simplex_dual_edge_weight_strategy": "devex"}),
    ("steepest", {"simplex_dual_edge_weight_strategy": "steepest"}),
    ("steep-devex", {"simplex_dual_edge_weight_strategy": "steepest-devex"}),
    ("no-presolve", {"presolve": False}),
]


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
    state = {"o": {}, "t": 0.0, "err": None}

    def wrapped(*args, **kw):
        kw = dict(kw)
        opt = dict(kw.get("options") or {})
        opt.update(state["o"])
        if opt:
            kw["options"] = opt
        t0 = time.perf_counter()
        try:
            r = real(*args, **kw)
        except Exception as e:                                  # noqa: BLE001
            state["err"] = repr(e)
            raise
        state["t"] += time.perf_counter() - t0
        return r

    oc.linprog = wrapped
    kw = M._lpkw()
    lay = M._load_layouts(a.layouts)
    cs = [(i, c) for i, c in enumerate(L.CASES) if i in lay and c["n"] >= a.minn]
    if a.limit:
        cs = cs[:a.limit]
    print("[l240] {} cases n>={}  prune_B={}  reps={}"
          .format(len(cs), a.minn, a.b, a.reps))

    res = {}
    for tag, opts in CONFIGS:
        state["o"] = opts
        tot = solve = 0.0
        moved = bad = 0
        ok = True
        for i, c in cs:
            best = None
            for _ in range(a.reps):
                state["t"] = 0.0
                state["err"] = None
                t0 = time.perf_counter()
                try:
                    r = M.one(c, lay[i], a.b, kw, 1)
                except Exception as e:                          # noqa: BLE001
                    print("   {} case {}: raised {!r}".format(tag, i, e))
                    ok = False
                    r = None
                w = time.perf_counter() - t0
                if r is None:
                    break
                if best is None or w < best[0]:
                    best = (w, state["t"], r)
            if best is None:
                ok = False
                continue
            tot += best[0]
            solve += best[1]
            res.setdefault(tag, {})[i] = best[2]
            ref = res.get("shipped", {}).get(i)
            if ref:
                if ref["obj"] and best[2]["obj"]:
                    rel = abs(best[2]["obj"] - ref["obj"]) / abs(ref["obj"])
                    if rel > 1e-9:
                        bad += 1
                if best[2]["lay"] != ref["lay"]:
                    moved += 1
        res.setdefault("_t", {})[tag] = (tot, solve, moved, bad, ok)
    oc.linprog = real

    base = res["_t"]["shipped"][0]
    print()
    print("{:<14}{:>10}{:>10}{:>9}{:>10}{:>10}"
          .format("config", "wall", "solve", "speed", "layouts", "obj moved"))
    print("-" * 63)
    for tag, _ in CONFIGS:
        t, s, mv, bd, ok = res["_t"][tag]
        print("{:<14}{:>10.3f}{:>10.3f}{:>8.3f}x{:>10}{:>10}{}"
              .format(tag, t, s, base / t if t else 0, mv, bd,
                      "" if ok else "   !! some cases failed"))
    print("-" * 63)
    print("'layouts' counts cases whose layout hash differs from the shipped")
    print("config -- expected and harmless on a degenerate LP as long as")
    print("'obj moved' is 0. A moved OBJECTIVE is a tolerance story and is out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
