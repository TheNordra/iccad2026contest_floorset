"""M39 liveness: count pure-movable clusters (no preplaced member, >=2 movable)
that have a FREE_CLUSTER_BND-eligible BOUNDARY member. Mirrors the C++ gate
(make_group_item line ~379 + dispatch line ~1306). constraints columns:
0=fixed 1=preplaced 2=mib 3=cluster 4=boundary."""
import sys
from collections import defaultdict
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator
ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()

tot_cases = tot_groups = tot_members = 0
for idx in range(100):
    at, b2b, p2b, pins, cons = ev.dataset[idx]["input"]
    n = int((at != -1).sum().item())
    cl_members = defaultdict(list)
    for i in range(n):
        cl = int(cons[i, 3].item())
        if cl > 0:
            cl_members[cl].append(i)
    eg = em_tot = 0
    for cl, mem in cl_members.items():
        pre = [i for i in mem if int(cons[i, 1].item()) != 0]
        mov = [i for i in mem if int(cons[i, 1].item()) == 0]
        if pre:                      # mixed -> anchored path (FREE_ANCHORED), not make_group_item
            continue
        if len(mov) < 2:             # not a compound item
            continue
        em = [i for i in mov
              if int(cons[i, 4].item()) != 0     # boundary
              and int(cons[i, 2].item()) == 0    # mib==0
              and int(cons[i, 0].item()) == 0    # not fixed
              and float(at[i]) > 0]
        if em:
            eg += 1; em_tot += len(em)
    if eg:
        tot_cases += 1; tot_groups += eg; tot_members += em_tot
        print(f"case {idx:3d} n={n:3d}: {eg} pure-movable cluster(s) w/ boundary member -> {em_tot} eligible member(s)")
print(f"\nLIVENESS: {tot_cases}/100 cases, {tot_groups} groups, {tot_members} eligible boundary members")
