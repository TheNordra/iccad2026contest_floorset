"""M53 L1 helper: diff two official-eval results jsons per case.
Usage: python m53_diff_results.py base.json new.json [tol]
Prints totals, weighted delta, and every case whose cost moved > tol (default 1e-4),
sorted by |weighted contribution|. Never shipped; offline analysis only."""
import json, math, sys

base_p, new_p = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
A = json.load(open(base_p)); B = json.load(open(new_p))
ca = {t["test_id"]: t for t in A["test_results"]}
cb = {t["test_id"]: t for t in B["test_results"]}
n_of = {t["test_id"]: t.get("num_blocks") or t.get("block_count") for t in A["test_results"]}
# fall back: weight from cost contribution is unknown -> recompute if n present
rows = []
totW = 0.0
for tid in sorted(ca):
    n = n_of[tid]
    w = math.exp(n / 12.0) if n else 1.0
    totW += w
    d = cb[tid]["cost"] - ca[tid]["cost"]
    rows.append((abs(d) * w, d, tid, n, ca[tid]["cost"], cb[tid]["cost"], w))
rows.sort(reverse=True)
print(f"base  total = {A['total_score']:.10f}   ({base_p})")
print(f"new   total = {B['total_score']:.10f}   ({new_p})")
print(f"delta total = {B['total_score']-A['total_score']:+.10f}  "
      f"({(B['total_score']-A['total_score'])/A['total_score']*100:+.4f}%)")
mov = [r for r in rows if abs(r[1]) > tol]
imp = sum(1 for r in mov if r[1] < 0); reg = len(mov) - imp
print(f"movers (|d|>{tol}): {len(mov)}  ({imp} improved / {reg} regressed)")
for _, d, tid, n, c0, c1, w in mov:
    print(f"  case {tid:3d} n={n!s:>4} {c0:.6f} -> {c1:.6f}  d={d:+.6f}  wContr={d*w/totW*100:+.4f}%")
feas_b = sum(1 for t in B["test_results"] if t.get("is_feasible"))
rt_a = sum(t["runtime_seconds"] for t in A["test_results"]) / len(ca)
rt_b = sum(t["runtime_seconds"] for t in B["test_results"]) / len(cb)
print(f"new is_feasible: {feas_b}/{len(cb)}   avg runtime {rt_a:.2f}s -> {rt_b:.2f}s")
