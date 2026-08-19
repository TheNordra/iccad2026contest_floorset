"""L142 - compare violation arms against the L140 OOS baseline, on shared keys.

Every arm here is a screening run on a SUBSET (--limit), so the comparison must
be restricted to the keys the arm actually solved. Prints the official weighted
aggregate plus the boundary/grouping/MIB split, and -- first, because
`[[probe-import-time-silent-nooks]]` -- a hard "did anything actually change?"
gate: if an arm's positions are bit-identical to the baseline's, the knob never
reached the C++ and every number below it is meaningless.

  <python> l142_arm_cmp.py l140_oos_s1_c48.json l142_bpw1e6.json l142_nohpwl.json
"""
import argparse
import json
import math
from pathlib import Path


def load(f):
    return {r["key"]: r for r in json.load(open(f))["test_results"]}


def agg(rows):
    ws = sum(math.exp(r["n"] / 12.0) for r in rows)
    out = {}
    for k in ("cost", "hpwl_gap", "area_gap", "vrel"):
        out[k] = sum(math.exp(r["n"] / 12.0) * r[k] for r in rows) / ws
    for k in ("v_bnd", "v_grp", "v_mib"):
        out[k] = sum(r[k] for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()

    B = load(a.base)
    print(f"baseline {Path(a.base).name}  ({len(B)} cases)")
    for f in a.arms:
        A = load(f)
        keys = sorted(set(A) & set(B))
        moved = sum(1 for k in keys if A[k]["positions"] != B[k]["positions"])
        b = agg([B[k] for k in keys])
        x = agg([A[k] for k in keys])
        print(f"\n=== {Path(f).name}   {len(keys)} shared cases ===")
        print(f"  LIVENESS: {moved}/{len(keys)} cases changed positions"
              + ("   <-- SILENT NO-OP, knob never reached the placer"
                 if moved == 0 else ""))
        d = 100 * (b["cost"] - x["cost"]) / b["cost"]
        print(f"  cost      {b['cost']:.6f} -> {x['cost']:.6f}   {d:+.4f}%")
        for k in ("hpwl_gap", "area_gap", "vrel"):
            print(f"  {k:<9} {b[k]:.6f} -> {x[k]:.6f}")
        for k, lab in (("v_bnd", "boundary"), ("v_grp", "grouping"),
                       ("v_mib", "MIB")):
            print(f"  {lab:<9} {b[k]:>5} -> {x[k]:>5}"
                  f"   {x[k] - b[k]:+d}")
        inf = sum(1 for k in keys if not A[k]["feasible"])
        if inf:
            print(f"  !! {inf} infeasible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
