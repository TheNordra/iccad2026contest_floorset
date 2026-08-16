"""Is the large-n regression made by stage A, or spent by lp_polish?

The alternation improves hpwl_gap in every size bucket, and for n>=80 -- which
carries 96% of the weight -- area_gap and vgrp both get worse. Two candidates:

  (a) stage A hands over a genuinely worse-for-area ordering at large n, or
  (b) the ordering is fine and lp_polish, which is soft-constraint-blind, spends
      the difference.

Handoff 08-15 §3.1 is the standing warning that (b) is the likelier shape, so it
is measured before anything is redesigned. Reports the official metrics BEFORE
and AFTER the LP for both stage-A forms on the same cases.

  <python> -u l130_lp_split.py --minn 80 --limit 12
"""
import argparse
import math
import os
import time


def run(cases, gordian):
    # `place` reads the module global at call time, so the two stage-A forms can
    # be compared inside one process -- no reload, and the dataset is loaded once
    import l129_global_placer as L
    L.GORDIAN = gordian
    out = {}
    for c in cases:
        t0 = time.perf_counter()
        P = L.place(c)
        t_place = time.perf_counter() - t0
        if P is None:
            out[c["idx"]] = None
            continue
        pre = L.official(c, P)
        t1 = time.perf_counter()
        Q = L.lp_polish(c, P)
        t_lp = time.perf_counter() - t1
        post = L.official(c, Q)
        out[c["idx"]] = dict(n=c["n"], pre=pre, post=post,
                             t_place=t_place, t_lp=t_lp)
    return out


def wsum(rows, get):
    ok = [r for r in rows if r is not None]
    if not ok:
        return float("nan")
    ws = sum(math.exp(r["n"] / 12.0) for r in ok)
    return sum(math.exp(r["n"] / 12.0) * get(r) for r in ok) / ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minn", type=int, default=80)
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()

    import l129_global_placer as L
    cases = [c for c in L.CASES if c["n"] >= a.minn][:a.limit]
    print(f"cases: {[c['idx'] for c in cases]}\n")

    A = run(cases, False)
    B = run(cases, True)
    common = [k for k in A if A[k] is not None and B[k] is not None
              and A[k]["post"].is_feasible and B[k]["post"].is_feasible]
    ra = [A[k] for k in common]
    rb = [B[k] for k in common]
    print(f"coverage: base {sum(1 for k in A if A[k])}/{len(A)}  "
          f"gordian {sum(1 for k in B if B[k])}/{len(B)}   common feasible {len(common)}\n")

    def show(tag, get):
        va, vb = wsum(ra, get), wsum(rb, get)
        print(f"{tag:<28} {va:>11.5f} {vb:>11.5f} {vb - va:>+11.5f}")

    print(f"{'metric':<28} {'base':>11} {'gordian':>11} {'delta':>11}")
    for stage in ("pre", "post"):
        print(f"-- {stage}-LP --")
        for key in ("cost", "hpwl_gap", "area_gap", "violations_relative",
                    "grouping_violations", "boundary_violations"):
            show(f"  {stage} {key}", lambda r, s=stage, k=key: float(getattr(r[s], k)))
    print("-- what the LP moved --")
    for key in ("cost", "hpwl_gap", "area_gap", "grouping_violations"):
        show(f"  d{key}",
             lambda r, k=key: float(getattr(r["post"], k)) - float(getattr(r["pre"], k)))
    show("  t_place", lambda r: r["t_place"])
    show("  t_lp", lambda r: r["t_lp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
