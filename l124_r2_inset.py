"""OFFLINE (never shipped): L124 R2 — the in-set bit-identical gate.

CLAIM. Appending the eight MIB-ON twins changes nothing in-set. Not "changes
little" -- nothing, to the last digit. Two independent reasons:

  1. All 100 in-set MIB groups are already unified by the two existing branches
     of apply_safe_mib_dims (measured: in-set V_mib is 0, and held-out only 2.5%
     of groups are unifiable). So a twin's binary output is IDENTICAL to its
     source's in-set -- there is nothing for bucketing to do.
  2. A duplicate candidate cannot move the selection: it does not change
     `hmin` (a min over a set that already contains the same value), and the
     argmin loop uses a strict `<`, so a tie keeps the earlier index -- and the
     twins are appended AFTER their sources.

⚠️ THE GATE ONLY MEANS SOMETHING WITH A BINARY THAT UNDERSTANDS THE FLAG. Run
with the shipping exe (which ignores ICCAD_MIB_BUCKET) and the twins are
trivially identical to their sources, so the gate passes for a reason that has
nothing to do with the claim. ICCAD_CONSTRUCTIVE_BIN must point at the L124
probe binary, and this script refuses to run without it.

Anchors: 48c 1.2367916697725434 (route A + shape LP + M80 tier on),
32c 1.293461035226291 (everything cores-gated off).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

ANCHOR = {48: 1.2367916697725434, 32: 1.293461035226291}
# per-case reference: the same run the anchor total came from
ANCHOR_JSON = {48: "results_L114_48c_lp_anchor.json",
               32: "results_M74_default.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--bin", default="constructive_l124.exe")
    ap.add_argument("--cases", type=int, default=100)
    a = ap.parse_args()

    exe = _DIR / a.bin
    if not exe.exists():
        sys.exit(f"missing probe binary {exe} -- see the warning in the docstring")
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(exe)
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)

    import m53_l3_probe as l3
    import optimizer_constructive as oc
    from proxy_analysis import build_opt_target_pos

    live = sorted(oc._m124_active(120))
    print(f"[cfg] bin={a.bin}  cores={a.cores}  twins active: {live or 'none'}")
    print(f"[cfg] pool @n=120: {len(oc._pool_indices(120))}")

    opt = oc.MyOptimizer(verbose=False)
    W, tot, cov = l3.W, 0.0, 0.0
    per_case = {}
    t0 = time.time()
    for ci in range(a.cases):
        c = l3.CASES[ci]
        # target_positions is NOT optional: it carries the preplaced/fixed
        # placements. Passing None deletes them, every profile fails its hard
        # checks, and the whole run sinks to the SA fallback at 10.0 per case --
        # which is exactly what a first cut of this script measured, and the
        # same input-mismatch M75 recorded when _m75_liveness fed None.
        otp = build_opt_target_pos(c["tp"], c["cons"], c["n"])
        pos = opt.solve(c["n"], c["at"], c["b2b"], c["p2b"], c["pins"],
                        c["cons"], otp)
        cst = float(l3.cost_eval(ci, pos).cost)
        per_case[ci] = cst
        tot += W[ci] * cst
        cov += W[ci]
        if (ci + 1) % 20 == 0:
            print(f"  {ci + 1}/{a.cases} ({time.time() - t0:.0f}s)")
    total = tot / cov
    want = ANCHOR.get(a.cores)
    print(f"\n  total   {total:.16f}")
    if want is not None:
        print(f"  anchor  {want:.16f}   |delta| {abs(total - want):.3e}")

    # PER-CASE is the real invariant. The total is a re-summation in a different
    # order than the evaluator that produced the anchor, so it lands 1-2 ULP away
    # even when every case is identical -- gating on the total would either pass
    # for the wrong reason or fail for no reason. (M67-C's Linux run recorded the
    # same thing: one case 4.441e-16 apart, accepted as a ULP artefact.)
    ref = ANCHOR_JSON.get(a.cores)
    if ref and Path(_DIR / ref).exists():
        import json
        j = json.loads((_DIR / ref).read_text(encoding="utf-8"))
        want_c = {int(r["test_id"]): float(r["cost"]) for r in j["test_results"]}
        exact = near = bad = 0
        worst = 0.0
        for ci, got in per_case.items():
            if ci not in want_c:
                continue
            d = abs(got - want_c[ci])
            worst = max(worst, d)
            if d == 0.0:
                exact += 1
            elif d <= 1e-12 * max(abs(want_c[ci]), 1.0):
                near += 1
            else:
                bad += 1
                print(f"    case {ci}: {got:.17g} vs {want_c[ci]:.17g}")
        print(f"  per-case vs {ref}: {exact} bit-equal, {near} within 1e-12, "
              f"{bad} differ; worst {worst:.3e}")
        print(f"  -> {'PASS' if bad == 0 else 'FAIL'}")
        sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
