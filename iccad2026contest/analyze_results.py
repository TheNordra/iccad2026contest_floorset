import json, statistics

with open('optimizer_claude_results.json') as f:
    data = json.load(f)

results = data.get('test_results', [])
print(f"Total cases: {len(results)}")
print(f"Total Score: {data.get('total_score', '?'):.6f}")
print()

sorted_r = sorted(results, key=lambda x: x.get('cost', 0), reverse=True)

print('Top 20 highest cost cases:')
for r in sorted_r[:20]:
    idx = r.get('test_id', '?')
    cost = r.get('cost', 0)
    feasible = r.get('is_feasible', '?')
    n = r.get('block_count', '?')
    hpwl_gap = r.get('hpwl_gap', 0)
    area_gap = r.get('area_gap', 0)
    viol = r.get('violations_relative', 0)
    print(f'  Case {idx:3d} n={n:3d}: cost={cost:.4f}, feasible={feasible}, hpwl_gap={hpwl_gap:.3f}, area_gap={area_gap:.3f}, viol={viol:.3f}')

print()
print('Bottom 10 (best) cases:')
for r in sorted_r[-10:]:
    idx = r.get('test_id', '?')
    cost = r.get('cost', 0)
    feasible = r.get('is_feasible', '?')
    n = r.get('block_count', '?')
    hpwl_gap = r.get('hpwl_gap', 0)
    area_gap = r.get('area_gap', 0)
    viol = r.get('violations_relative', 0)
    print(f'  Case {idx:3d} n={n:3d}: cost={cost:.4f}, feasible={feasible}, hpwl_gap={hpwl_gap:.3f}, area_gap={area_gap:.3f}, viol={viol:.3f}')

infeasible = [r for r in results if not r.get('is_feasible', True)]
print(f"\nInfeasible: {len(infeasible)}/100")

costs = [r.get('cost', 0) for r in results]
hpwl_gaps = [r.get('hpwl_gap', 0) for r in results]
area_gaps = [r.get('area_gap', 0) for r in results]
viols = [r.get('violations_relative', 0) for r in results]

print(f"Cost   stats: min={min(costs):.4f}, max={max(costs):.4f}, median={statistics.median(costs):.4f}, mean={statistics.mean(costs):.4f}")
print(f"HPWL gap avg={statistics.mean(hpwl_gaps):.4f}, max={max(hpwl_gaps):.4f}")
print(f"Area gap avg={statistics.mean(area_gaps):.4f}, max={max(area_gaps):.4f}")
print(f"Viol avg={statistics.mean(viols):.4f}, max={max(viols):.4f}")

# Check summary
summ = data.get('summary', {})
print(f"\nSummary: {summ}")
