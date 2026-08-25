"""L235 - profile and A/B the shape LP's PYTHON half.

Why this half is worth attacking, in one line: the LP is worth +4.80% of in-set
quality fully on, `_L196_LPGATE` can only afford +1.57% of it, and re-optimising
the gate at LP speed f is worth +0.20pp (1.15x) / +0.86pp (1.5x) / +1.27pp (2x)
on the post-REFINE budget. L155 priced the same speedup at exactly 0.0000%
because the package then sat on the RF floor and there was no gate to widen.

`prof`  cProfile over l155_lp_rows.one(), printed for optimizer_* frames only.
`ab`    run the SAME cases through two optimizer modules and require
        (lp objective, layout hash, rows-by-origin, kept/dropped counts) to be
        IDENTICAL, then report min-of-N wall for each. Bit-identity is the whole
        point: a change that only removes Python overhead must not be able to
        move the solution, so the gate is equality, not a quality measurement.

  <python> l235_lpbench.py prof  [--minn 105] [--limit 6]
  <python> l235_lpbench.py ab --mod optimizer_l235lp [--reps 3]
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

DIR = Path(__file__).parent


def _cases(M, L, minn, limit, layouts):
    lay = M._load_layouts(layouts)
    cs = [(i, c) for i, c in enumerate(L.CASES) if i in lay]
    if minn:
        cs = [(i, c) for i, c in cs if c["n"] >= minn]
    if limit:
        cs = cs[:limit]
    return cs, lay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prof", "ab"])
    ap.add_argument("--minn", type=int, default=105)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--b", type=float, default=8.0)
    ap.add_argument("--mod", default="optimizer_l235lp")
    ap.add_argument("--layouts", default="results_L153_lpoff_L137.json")
    a = ap.parse_args()

    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l155_lp_rows as M
    import l129_global_placer as L
    kw = M._lpkw()
    cs, lay = _cases(M, L, a.minn, a.limit, a.layouts)
    print("[l235] {} cases, n>={}, prune_B={}, kw={}"
          .format(len(cs), a.minn, a.b, kw))

    if a.mode == "prof":
        if a.mod and a.mod != "optimizer_constructive":
            import importlib
            B = importlib.import_module(a.mod)
            M.oc, M._l3 = B, B.l3
            print("[l235] profiling {}".format(a.mod))
        pr = cProfile.Profile()
        for i, c in cs:                       # warm the caches OUTSIDE the profile
            M.one(c, lay[i], a.b, kw, 1)
        pr.enable()
        for i, c in cs:
            M.one(c, lay[i], a.b, kw, 1)
        pr.disable()
        s = io.StringIO()
        st = pstats.Stats(pr, stream=s).sort_stats("tottime")
        st.print_stats("optimizer_|l129_global|highs")
        print(s.getvalue())
        return 0

    # ---- A/B ---------------------------------------------------------------
    import importlib
    import optimizer_constructive as A
    B = importlib.import_module(a.mod)
    print("[l235] A = optimizer_constructive   B = {}".format(a.mod))
    bad, rows = [], []
    for i, c in cs:
        out = {}
        for tag, mod in (("A", A), ("B", B)):
            M.oc = mod
            M._l3 = mod.l3
            out[tag] = M.one(c, lay[i], a.b, kw, a.reps)
        M.oc, M._l3 = A, A.l3
        ra, rb = out["A"], out["B"]
        if ra is None or rb is None:
            bad.append((i, "one() returned None"))
            continue
        for k in ("obj", "lay", "rows", "kept", "dropped", "calls", "status",
                  "ok"):
            if ra[k] != rb[k]:
                bad.append((i, "{}: {!r} != {!r}".format(k, ra[k], rb[k])))
        rows.append((i, c["n"], ra["wall"], rb["wall"], ra["t_solve"],
                     rb["t_solve"]))
    print()
    print("{:>5}{:>6}{:>10}{:>10}{:>9}{:>10}{:>10}{:>9}"
          .format("case", "n", "A wall", "B wall", "speed", "A solve",
                  "B solve", "pyspeed"))
    print("-" * 70)
    ta = tb = tsa = tsb = 0.0
    for i, n, wa, wb, sa, sb in rows:
        pa, pb = max(wa - sa, 1e-9), max(wb - sb, 1e-9)
        print("{:>5}{:>6}{:>10.3f}{:>10.3f}{:>9.3f}x{:>10.3f}{:>10.3f}"
              "{:>8.3f}x".format(i, n, wa, wb, wa / wb, sa, sb, pa / pb))
        ta += wa; tb += wb; tsa += sa; tsb += sb
    print("-" * 70)
    pa, pb = max(ta - tsa, 1e-9), max(tb - tsb, 1e-9)
    print("TOTAL      {:>10.3f}{:>10.3f}{:>9.3f}x{:>10.3f}{:>10.3f}{:>8.3f}x"
          .format(ta, tb, ta / tb if tb else 0, tsa, tsb, pa / pb))
    print("  'speed' is the whole LP; 'pyspeed' is wall minus solve, i.e. the")
    print("  half this change is allowed to touch. The solve columns must stay")
    print("  the same -- the program handed to HiGHS is identical.")
    print()
    if bad:
        print("!! IDENTITY GATE FAILED on {} checks:".format(len(bad)))
        for i, m in bad[:12]:
            print("   case {}: {}".format(i, m))
        return 1
    print("IDENTITY GATE PASS: objective, layout hash, rows-by-origin, "
          "kept/dropped, calls and hard_ok all identical on {} cases"
          .format(len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
