"""Analyze violation breakdown for top failing cases."""
import json, sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

with open('optimizer_claude_results.json') as f:
    data = json.load(f)

results = data['test_results']
sorted_r = sorted(results, key=lambda x: x.get('cost', 0), reverse=True)
print("Top 10 cases with violation details:")

from lite_dataset_test import FloorplanDatasetLiteTest
dataset = FloorplanDatasetLiteTest("../LiteTensorDataTest/")

for r in sorted_r[:10]:
    tid = r['test_id']
    cost = r['cost']
    n = r['block_count']
    viol = r['violations_relative']
    hpwl_g = r['hpwl_gap']
    area_g = r['area_gap']

    print(f"\nCase {tid:3d} (n={n}): cost={cost:.3f}, viol_rel={viol:.3f}, hpwl_gap={hpwl_g:.3f}, area_gap={area_g:.3f}")

    sample = dataset[tid]
    area_t, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics = sample
    block_count = int((area_t != -1).sum().item())
    constr = constraints[:block_count]

    n_mib = int((constr[:, 2] > 0).sum().item())
    n_cluster = int((constr[:, 3] > 0).sum().item())
    n_boundary = int((constr[:, 4] > 0).sum().item())

    mib_groups = sorted(set(int(constr[i, 2].item()) for i in range(block_count) if constr[i, 2] > 0))
    cluster_groups = sorted(set(int(constr[i, 3].item()) for i in range(block_count) if constr[i, 3] > 0))

    n_soft = n_boundary
    for g in mib_groups:
        gs = int((constr[:, 2] == g).sum().item())
        n_soft += max(0, gs - 1)
    for g in cluster_groups:
        gs = int((constr[:, 3] == g).sum().item())
        n_soft += max(0, gs - 1)

    mib_sizes = [int((constr[:,2]==g).sum()) for g in mib_groups]
    cluster_sizes = [int((constr[:,3]==g).sum()) for g in cluster_groups]
    boundary_flags = [int(constr[i,4].item()) for i in range(block_count) if constr[i,4] > 0]

    n_soft_mib = sum(max(0, s-1) for s in mib_sizes)
    n_soft_cluster = sum(max(0, s-1) for s in cluster_sizes)

    print(f"  Constraints: {n_boundary} boundary, {n_mib} MIB blocks ({len(mib_groups)} groups), {n_cluster} cluster blocks ({len(cluster_groups)} groups)")
    print(f"  n_soft = {n_soft} (boundary={n_boundary}, MIB={n_soft_mib}, cluster={n_soft_cluster})")
    print(f"  total_violations ≈ {viol * n_soft:.1f}")
    print(f"  MIB sizes: {mib_sizes}")
    print(f"  Cluster sizes: {cluster_sizes}")
    print(f"  Boundary flags: {boundary_flags}")
