"""OFFLINE (never shipped): derive judge48's per-case regression budget from data.

HANDOFF_2026-08-20 §5.1 asks for "none worse than the pre-LP anchor beyond
budget". The strict budget=0 form is unsatisfiable for a real package -- the
already-uploaded L136 is itself worse than results_M80_48c_anchor.json on 2
cases, and L147's own Gate 2 recorded the shipped band at 2 regressions against
its pre-LP base. So the budget has to be measured, and the honest reference is
the band that is ALREADY DEPLOYED: whatever per-case damage the shipped LP does
against the same pre-LP base is damage the team has already accepted. An arm
that stays inside it is no worse than what is running today.

    python l153_budget.py <pre-LP base.json> <control.json> [arm.json ...]
"""
import json
import sys
from pathlib import Path


def load(p):
    j = json.loads(Path(p).read_text(encoding="utf-8"))
    return j, {r["test_id"]: r for r in j["test_results"]}


def regs(base, arm):
    return sorted(((arm[i]["cost"] - base[i]["cost"], i)
                   for i in base if i in arm
                   and arm[i]["cost"] > base[i]["cost"] + 1e-12), reverse=True)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    bj, b = load(sys.argv[1])
    cj, c = load(sys.argv[2])
    print(f"pre-LP base  {Path(sys.argv[1]).name:38s} {bj['total_score']!r}")
    cr = regs(b, c)
    print(f"control      {Path(sys.argv[2]).name:38s} {cj['total_score']!r}   "
          f"{100 * (1 - cj['total_score'] / bj['total_score']):+.4f}%")
    print(f"  control regressions vs base: {len(cr)}"
          + (f"   worst {cr[0][0]:.9f} (case {cr[0][1]})" if cr else ""))
    for d, i in cr[:8]:
        print(f"    case {i:3d}: base {b[i]['cost']:.9f} -> ctrl "
              f"{c[i]['cost']:.9f}   +{d:.9f}")
    budget = cr[0][0] if cr else 0.0
    for a_path in sys.argv[3:]:
        aj, a = load(a_path)
        ar = regs(b, a)
        over = [(d, i) for d, i in ar if d > budget + 1e-12]
        print(f"arm          {Path(a_path).name:38s} {aj['total_score']!r}   "
              f"{100 * (1 - aj['total_score'] / bj['total_score']):+.4f}%")
        print(f"  arm regressions vs base: {len(ar)}   over the "
              f"{budget:.9f} budget: {len(over)}")
        for d, i in ar[:8]:
            print(f"    case {i:3d}: base {b[i]['cost']:.9f} -> arm "
                  f"{a[i]['cost']:.9f}   +{d:.9f}"
                  + ("   OVER" if d > budget + 1e-12 else ""))
    print(f"\nBUDGET {budget:.9f}")


if __name__ == "__main__":
    main()
