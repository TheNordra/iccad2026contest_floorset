"""Compare two L129 result jsons on the COMMON feasible set.

Coverage is a free variable for this placer (a case it cannot legalise emits
nothing), so a headline weighted cost computed over each file's own feasible set
compares two different case mixes. Every claim below is on the intersection.
"""
import json
import math
import sys


def load(p):
    d = json.load(open(p))
    return {r["test_id"]: r for r in d["test_results"]}


def wavg(rows, key):
    ws = sum(math.exp(r["block_count"] / 12.0) for r in rows)
    return sum(math.exp(r["block_count"] / 12.0) * r[key] for r in rows) / ws


def main(pa, pb):
    A, B = load(pa), load(pb)
    fa = {k for k, r in A.items() if r["is_feasible"]}
    fb = {k for k, r in B.items() if r["is_feasible"]}
    common = sorted(fa & fb)
    print(f"A={pa}  feasible {len(fa)}")
    print(f"B={pb}  feasible {len(fb)}")
    print(f"only A: {sorted(fa - fb)}")
    print(f"only B: {sorted(fb - fa)}")
    print(f"common: {len(common)}\n")
    ra = [A[k] for k in common]
    rb = [B[k] for k in common]
    print(f"{'metric':<24} {'A':>12} {'B':>12} {'B-A':>12}")
    for k in ("cost", "hpwl_gap", "area_gap", "violations_relative", "vbnd",
              "vgrp", "vmib", "runtime_seconds"):
        va, vb = wavg(ra, k), wavg(rb, k)
        print(f"{k:<24} {va:>12.6f} {vb:>12.6f} {vb - va:>+12.6f}")
    win = sum(1 for k in common if B[k]["cost"] < A[k]["cost"] - 1e-9)
    los = sum(1 for k in common if B[k]["cost"] > A[k]["cost"] + 1e-9)
    print(f"\nper-case: B better on {win}/{len(common)}, worse on {los}")
    d = sorted(common, key=lambda k: B[k]["cost"] - A[k]["cost"])
    print("\nbiggest B wins:")
    for k in d[:5]:
        print(f"  case {k:>3} n={A[k]['block_count']:>3}  "
              f"{A[k]['cost']:8.4f} -> {B[k]['cost']:8.4f}")
    print("biggest B losses:")
    for k in d[-5:]:
        print(f"  case {k:>3} n={A[k]['block_count']:>3}  "
              f"{A[k]['cost']:8.4f} -> {B[k]['cost']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
