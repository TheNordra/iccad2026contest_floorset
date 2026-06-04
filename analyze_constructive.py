"""Per-case violation breakdown for constructive.exe.
Re-runs each case through evaluate_solution to split vrel into vBd/vCl/vMb,
and aggregates weighted (e^(n/12)) contribution to find the real bottleneck.
"""
import os, sys, math, subprocess
from pathlib import Path
import torch
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")

rows = []
totW = 0.0; totWC = 0.0
sum_bd = sum_cl = sum_mb = 0
for idx in range(len(ev.dataset)):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    r = subprocess.run([EXE], input=txt, capture_output=True, text=True)
    ps = _parse_output(r.stdout, n)
    m = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                          b2b, p2b, pins, at[:n], target_positions=tp[:n],
                          median_runtime=1.0)
    w = math.exp(n / 12.0)
    totW += w; totWC += w * m.cost
    sum_bd += m.boundary_violations; sum_cl += m.grouping_violations; sum_mb += m.mib_violations
    rows.append((idx, n, w, m.cost, m.hpwl_gap, m.area_gap, m.violations_relative,
                 m.boundary_violations, m.grouping_violations, m.mib_violations,
                 m.max_possible_violations, m.is_feasible))

print(f"Total Score = {totWC/totW:.4f}   (sum vBd={sum_bd} vCl={sum_cl} vMb={sum_mb})")
print(f"{'case':>4} {'n':>4} {'wt%':>5} {'cost':>6} {'hgap':>6} {'agap':>6} {'vrel':>6} "
      f"{'vBd':>4} {'vCl':>4} {'vMb':>4} {'nsft':>4} {'wContr%':>7}")
# sort by weighted contribution to total score
rows.sort(key=lambda r: -(r[2]*r[3]))
for (idx,n,w,cost,hg,ag,vr,bd,cl,mb,ns,feas) in rows[:30]:
    print(f"{idx:>4} {n:>4} {100*w/totW:5.2f} {cost:6.3f} {hg:6.3f} {ag:6.3f} {vr:6.3f} "
          f"{bd:>4} {cl:>4} {mb:>4} {ns:>4} {100*w*cost/totWC:7.2f}"
          f"{'' if feas else '  INFEASIBLE'}")
