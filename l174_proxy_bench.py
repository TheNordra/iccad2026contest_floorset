"""L174 - how much of the SERIAL _proxy_metrics cost is recomputed for nothing?

WHY THIS MATTERS. L173 measures the per-case wall as LINEAR in profile count
and fits wall(C) = a + b/C with a large core-INDEPENDENT `a`. The known source
of `a` is this function: M47 records it running once per profile on the main
thread at ~71 ms per profile on n=120, and records that parallelising it was
4x WORSE (GIL). 51 profiles ship. A core-independent cost does not shrink on
the grader's 48 cores -- it transfers in full.

Only `positions` differs between the 51 calls per case. The three
`constraints[:n, k].tolist()` conversions, `nsoft`, `ngrp`, `nmib` and every
per-group index list do not. `l174_hoisted.py` computes those once per case,
and replaces the O(n * ngrp) group scan with one O(n) bucketing pass.

This asserts the hoisted version is BIT-IDENTICAL on real inputs, then times
both. Nothing on the shipped path is modified.

    cd iccad2026contest
    python iccad2026_evaluate.py --evaluate ../_l174_capture.py --test-id 99 \
        -o ../_l174_cap.json
    cd ..
    python l174_proxy_bench.py
"""
import math
import pickle
import sys
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
sys.path.insert(0, str(DIR / "iccad2026contest"))

CASE = DIR / "_l174_case.pkl"
REPS = 40
PROFILES = 51
F = 3.17


def main():
    if not CASE.exists():
        print("missing {}. Run the capture shim first (see the docstring)."
              .format(CASE.name))
        return 1
    import optimizer_constructive as O
    import l174_hoisted as H

    d = pickle.load(open(CASE, "rb"))
    n = d["block_count"]
    ct = d["constraints"]
    args = (d["area_targets"], d["b2b"], d["p2b"], d["pins"], ct, n)

    # A layout with real geometry: identical inputs to both implementations is
    # all that matters, but a degenerate one would skip the shapely branch and
    # flatter the hoisted version, so spread the blocks out.
    pos = []
    for i in range(n):
        a = float(d["area_targets"][i])
        w = math.sqrt(a if a > 0 else 1.0)
        pos.append([(i % 11) * w * 1.05, (i // 11) * w * 1.05, w, w])

    print(__doc__)
    print("=" * 74)
    print("case n={}   shapely={}".format(n, O._SHAPELY))

    cache = H.build_case_cache(ct, n)
    a_ref = O._proxy_metrics(pos, *args)
    a_new = H.proxy_metrics_hoisted(pos, *args, cache=cache, mod=O)
    same = all(a_ref[k] == a_new[k] for k in ("area", "hpwl", "vrel"))
    print("\nEQUIVALENCE: {}".format("bit-identical" if same else "*** DIFFERS ***"))
    for k in ("area", "hpwl", "vrel"):
        print("   {:5s} shipped {!r:24s} hoisted {!r}".format(k, a_ref[k], a_new[k]))
    if not same:
        print("\nSTOP: the hoisted form is not equivalent. Nothing below is usable.")
        return 1

    O._proxy_metrics(pos, *args)
    t0 = time.perf_counter()
    for _ in range(REPS):
        O._proxy_metrics(pos, *args)
    shipped = (time.perf_counter() - t0) / REPS

    H.proxy_metrics_hoisted(pos, *args, cache=cache, mod=O)
    t0 = time.perf_counter()
    for _ in range(REPS):
        H.proxy_metrics_hoisted(pos, *args, cache=cache, mod=O)
    hoist = (time.perf_counter() - t0) / REPS

    t0 = time.perf_counter()
    for _ in range(REPS):
        H.build_case_cache(ct, n)
    build = (time.perf_counter() - t0) / REPS

    print("\n  shipped _proxy_metrics      {:8.2f} ms   (M47 quotes ~71 ms)"
          .format(1000 * shipped))
    print("  hoisted, per profile        {:8.2f} ms   {:.2f}x"
          .format(1000 * hoist, shipped / hoist if hoist else 0))
    print("  cache build, ONCE per case  {:8.2f} ms".format(1000 * build))

    old = PROFILES * shipped
    new = PROFILES * hoist + build
    print("\n  per case at {} profiles:".format(PROFILES))
    print("     shipped {:6.3f} s serial     hoisted {:6.3f} s     saves {:6.3f} s"
          " ({:.0f}%)".format(old, new, old - new, 100 * (old - new) / old))
    print("     in grader seconds (f={}):  saves {:.3f} s on THIS case"
          .format(F, (old - new) / F))
    print("\n  The entire free budget on the 2026-08-23 medians is 14.72 s")
    print("  across all 100 cases, and this is the heaviest single case.")

    import cProfile
    import io
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(REPS):
        O._proxy_metrics(pos, *args)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
    print("\nwhere the shipped version spends it:")
    print("\n".join(s.getvalue().splitlines()[4:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
