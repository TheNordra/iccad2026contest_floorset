"""Quick violation breakdown for top cases."""
import json, sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import torch
from iccad2026_evaluate import (get_validation_dataloader, evaluate_solution,
                                  compute_total_score)

with open('optimizer_claude_results.json') as f:
    data = json.load(f)
results = data['test_results']

val_loader = get_validation_dataloader(batch_size=1)
batches = list(val_loader)

# Sort by weighted contribution (cost * weight) to find biggest score levers
max_n = max(r.get('block_count', 0) for r in results)
import math
for r in results:
    n = r.get('block_count', 0)
    r['_weight'] = math.exp(n / 12)
    r['_contrib'] = r.get('cost', 0) * r['_weight']
sorted_r = sorted(results, key=lambda x: x['_contrib'], reverse=True)

print(f"{'tid':>4} {'n':>4} {'cost':>6} {'wt':>7} {'contr':>7} {'vBd':>4} {'vCl':>4} {'vMb':>4} {'Vrel':>5} {'hpwl':>5} {'area':>5}")
total_weight = sum(r['_weight'] for r in results)
print(f"total weight = {total_weight:.1f}")

for r in sorted_r[:15]:
    tid = r['test_id']
    cost = r['cost']
    positions = r.get('positions')
    if positions is None:
        continue

    batch = batches[tid]
    inputs, labels = batch
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    fp_sol, metrics = labels
    area_target = area_target.squeeze(0)
    b2b_conn = b2b_conn.squeeze(0)
    p2b_conn = p2b_conn.squeeze(0)
    pins_pos = pins_pos.squeeze(0)
    constraints = constraints.squeeze(0)
    metrics = metrics.squeeze(0)
    target_positions = None    # skip dimension hard check for analysis

    block_count = int((area_target != -1).sum().item())

    solution = {'positions': positions, 'runtime': r['runtime_seconds']}
    baseline_metrics = {
        'hpwl_baseline': float(metrics[0]),
        'area_baseline': float(metrics[1])
    }

    m = evaluate_solution(
        solution=solution,
        baseline_metrics=baseline_metrics,
        target_constraints=constraints,
        b2b_connectivity=b2b_conn,
        p2b_connectivity=p2b_conn,
        pins_pos=pins_pos,
        target_areas=area_target[:block_count],
        target_positions=target_positions,
        median_runtime=r['runtime_seconds'],
    )

    n_soft = m.max_possible_violations or 1
    print(f"{tid:>4} {block_count:>4} {cost:>6.3f} {r['_weight']:>7.1f} "
          f"{r['_contrib']:>7.1f} {m.boundary_violations:>4} {m.grouping_violations:>4} "
          f"{m.mib_violations:>4} {m.violations_relative:>5.3f} "
          f"{m.hpwl_gap:>5.2f} {m.area_gap:>5.2f}")
