"""Ad-hoc breakdown for portfolio_results.json vs prior baselines."""
import json

d = json.load(open("portfolio_results.json"))
print("Portfolio total:", round(d["total_score"], 4))
buckets = {}
for r in d["test_results"]:
    n = r["block_count"]
    bk = (n // 20) * 20
    buckets.setdefault(bk, []).append(r["cost"])
for bk in sorted(buckets):
    costs = buckets[bk]
    print(f"  n={bk:3d}-{bk+19:3d}: count={len(costs):3d}, "
          f"avg_cost={sum(costs)/len(costs):.4f}, max={max(costs):.4f}")

# Compare with boundary_aspect_results.json (3.4255)
print()
print("Compare to boundary_aspect_results.json (was 3.4255):")
d2 = json.load(open("boundary_aspect_results.json"))
wins_hiN = 0
losses_hiN = 0
total_delta_hiN = 0.0
for r1, r2 in zip(d["test_results"], d2["test_results"]):
    if r1["test_id"] != r2["test_id"]:
        continue
    if r1["block_count"] < 80:
        continue
    delta = r1["cost"] - r2["cost"]
    total_delta_hiN += delta
    if delta < -0.1:
        wins_hiN += 1
    elif delta > 0.1:
        losses_hiN += 1
print(f"  n>=80: wins={wins_hiN}, losses={losses_hiN}, "
      f"avg_delta={total_delta_hiN / max(1, sum(1 for r in d['test_results'] if r['block_count'] >= 80)):.4f}")

# Top regressions
print()
print("Top 5 regressions (any size):")
deltas = []
for r1, r2 in zip(d["test_results"], d2["test_results"]):
    if r1["test_id"] != r2["test_id"]:
        continue
    deltas.append((r1["cost"] - r2["cost"], r1["test_id"], r1["block_count"], r1["cost"], r2["cost"]))
deltas.sort(reverse=True)
for delta, tid, n, c_new, c_old in deltas[:5]:
    print(f"  test {tid:3d} n={n:3d}: {c_old:.3f} -> {c_new:.3f} ({delta:+.3f})")

print()
print("Top 5 wins (any size):")
for delta, tid, n, c_new, c_old in deltas[-5:]:
    print(f"  test {tid:3d} n={n:3d}: {c_old:.3f} -> {c_new:.3f} ({delta:+.3f})")
