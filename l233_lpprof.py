"""L233 - where the LP's PYTHON half actually goes, function by function.

L155's census established that ~40% of the LP wall is outside the solver, and
priced it at exactly 0.0000% because we sat on the RF floor: "time we give back
is time nobody pays us for". That premise died twice over --

  * the package no longer sits on the floor (cwRF 0.70523, 41 cases above it),
  * and, more importantly, THE LP IS NOW GATED. A faster LP does not give time
    back; it lets _L196_LPGATE turn ON more block counts. Measured on the
    post-REFINE budget (l230_pool_new.json), re-optimising the gate at LP speed
    f is worth

        f     1.15x   1.30x   1.50x   2.00x   3.00x    inf
        NET  +0.20pp +0.53pp +0.86pp +1.27pp +1.47pp +2.61pp

    against a rank-2 margin of 0.12pp. Partial progress is no longer worth
    zero -- that is L155's headline inverted, by the gate it did not have.

This file does not optimise anything. It measures which Python functions own
the non-solver half, so the optimisation goes where the time is instead of
where the row COUNT is (they are not the same: the separation loop is O(n^2)
over pairs that mostly produce no row at all).

  <python> l233_lpprof.py [--minn 110] [--limit 4]
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
from pathlib import Path

DIR = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minn", type=int, default=110)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--b", type=float, default=8.0)
    ap.add_argument("--layouts", default="results_L153_lpoff_L137.json")
    a = ap.parse_args()

    os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48")
    sys.argv = ["x"]
    import l155_lp_rows as M
    import optimizer_constructive as oc

    cases = M.pick_cases(a.layouts, a.minn, a.limit) if hasattr(M, "pick_cases") \
        else None
    if cases is None:
        print("l155_lp_rows has no pick_cases(); falling back to its census "
              "entry point under the profiler")
        pr = cProfile.Profile()
        sys.argv = ["l155_lp_rows.py", "census", "--minn", str(a.minn),
                    "--limit", str(a.limit), "--reps", "1",
                    "--b", str(int(a.b))]
        import runpy
        pr.enable()
        try:
            runpy.run_path(str(DIR / "l155_lp_rows.py"), run_name="__main__")
        except SystemExit:
            pass
        pr.disable()
    else:
        pr = cProfile.Profile()
        pr.enable()
        for c in cases:
            M.run_case(c, a.b)
        pr.disable()

    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    st.sort_stats("tottime")
    st.print_stats("optimizer_constructive")
    out = s.getvalue()
    print(out)

    # the totals that matter: solver vs everything else
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(
        "_highs_wrapper|linprog|csr_matrix|coo_matrix")
    print(s2.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
