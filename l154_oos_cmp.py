"""OFFLINE (never shipped): L154 OOS A/B on the l140 dumps.

Weighting is the official exp(n/12), same as l140_oos_soft_audit's own
reporting, so these numbers are comparable to every historical OOS figure.

    python l154_oos_cmp.py <off.json> <on.json> [label]
"""
import json
import math
import sys
from pathlib import Path


def load(p):
    j = json.loads(Path(p).read_text(encoding="utf-8"))
    return j, {r["test_id"]: r for r in j["test_results"]}


def wavg(rows, k):
    ws = sum(math.exp(r["n"] / 12.0) for r in rows)
    return sum(math.exp(r["n"] / 12.0) * r[k] for r in rows) / ws


def main():
    oj, o = load(sys.argv[1])
    nj, n = load(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    ids = sorted(set(o) & set(n))
    ro = [o[i] for i in ids]
    rn = [n[i] for i in ids]
    co, cn = wavg(ro, "cost"), wavg(rn, "cost")
    print(f"== {label or Path(sys.argv[2]).name}   ({len(ids)} cases, "
          f"sample {oj.get('sample')})")
    print(f"   OFF  {Path(sys.argv[1]).name:28s} cost {co:.6f}")
    print(f"   ON   {Path(sys.argv[2]).name:28s} cost {cn:.6f}   "
          f"{100 * (1 - cn / co):+.4f}%")
    for k in ("hpwl_gap", "area_gap", "vrel"):
        print(f"     {k:10s} {wavg(ro, k):.6f} -> {wavg(rn, k):.6f}")
    moved = [i for i in ids if o[i]["cost"] != n[i]["cost"]]
    better = [i for i in moved if n[i]["cost"] < o[i]["cost"]]
    worse = [i for i in moved if n[i]["cost"] > o[i]["cost"]]
    print(f"   moved {len(moved)}/{len(ids)}   better {len(better)}   "
          f"worse {len(worse)}")
    print(f"   feasible OFF {sum(1 for i in ids if o[i]['feasible'])}/{len(ids)}"
          f"   ON {sum(1 for i in ids if n[i]['feasible'])}/{len(ids)}")
    if moved:
        maxn = max(o[i]["n"] for i in ids)
        W = {i: math.exp((o[i]["n"] - maxn) / 12) for i in ids}
        SW = sum(math.exp((o[i]["n"] - maxn) / 12) for i in ids)
        contrib = sorted(((o[i]["cost"] - n[i]["cost"]) * W[i] / SW, i)
                         for i in moved)
        contrib.sort(key=lambda x: -abs(x[0]))
        tot = sum(c for c, _ in contrib)
        print(f"   weighted gain from the movers {tot:+.9f}")
        for c, i in contrib[:12]:
            print(f"     case {i:4d} n={o[i]['n']:3d} w={W[i]:.4f}  "
                  f"{o[i]['cost']:.6f} -> {n[i]['cost']:.6f}  raw "
                  f"{n[i]['cost'] - o[i]['cost']:+.6f}  weighted {c:+.9f}"
                  f"   ({100 * c / tot:5.1f}%)")
    return moved


if __name__ == "__main__":
    main()
