"""Get violation breakdown for high-cost cases."""
import sys, json
sys.path.insert(0, '.')
sys.path.insert(0, '..')

from iccad2026_evaluate import get_validation_dataloader, evaluate_solution

# Load solution
with open('optimizer_claude_results.json') as f:
    data = json.load(f)
results = data['test_results']
sorted_r = sorted(results, key=lambda x: x.get('cost', 0), reverse=True)

# Load validation data
val_loader = get_validation_dataloader(batch_size=1)
batches = list(val_loader)

test_ids = [r['test_id'] for r in sorted_r[:10]]
print("Violation breakdown for top 10 high-cost cases:")

for r in sorted_r[:10]:
    tid = r['test_id']
    cost = r['cost']
    positions = r.get('positions')
    if positions is None:
        print(f"Case {tid}: no positions saved")
        continue

    batch = batches[tid]
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
    area_target = area_target.squeeze(0)
    b2b_conn = b2b_conn.squeeze(0)
    p2b_conn = p2b_conn.squeeze(0)
    pins_pos = pins_pos.squeeze(0)
    constraints = constraints.squeeze(0)
    metrics = metrics.squeeze(0)
    target_positions = fp_sol.squeeze(0) if fp_sol is not None else None

    block_count = int((area_target != -1).sum().item())

    m = evaluate_solution(
        positions=positions,
        block_count=block_count,
        area_targets=area_target[:block_count],
        b2b_connectivity=b2b_conn,
        p2b_connectivity=p2b_conn,
        pins_pos=pins_pos,
        target_constraints=constraints,
        target_positions=target_positions,
        baseline_hpwl=float(metrics[0]),
        baseline_area=float(metrics[1]),
        runtime=r['runtime_seconds'],
    )

    print(f"\nCase {tid:3d} (n={block_count}, cost={cost:.3f}):")
    print(f"  boundary_violations={m.boundary_violations}, grouping_violations={m.grouping_violations}, mib_violations={m.mib_violations}")
    print(f"  n_soft={m.max_possible_violations}, total_soft={m.total_soft_violations}, viol_rel={m.violations_relative:.3f}")
    print(f"  hpwl_gap={m.hpwl_gap:.3f}, area_gap={m.area_gap:.3f}")
