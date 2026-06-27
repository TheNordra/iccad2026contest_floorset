"""OFFLINE (never shipped): verify the python_sa_solve FALLBACK is feasible on all
100 cases. The portfolio falls back to python_sa_solve only when the binary is
missing OR every profile fails (optimizer_constructive.py solve()); the main
constructive path is 100/100 feasible, but this safety net itself was never
validated. Mirrors the harness call exactly: solve() receives opt_target_pos
(preplaced full pos + fixed shape only, rest -1 = build_opt_target_pos), while
evaluate_solution checks against the true baseline tp. Feasible iff no overlap +
soft-area within 1% + fixed/preplaced dims exact (iccad2026_evaluate.py:396)."""
import sys, time
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import python_sa_solve
from proxy_analysis import build_opt_target_pos

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
nfeas = 0; bad = []; t_all = time.time()
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)        # exactly what the harness passes to solve()
    t0 = time.time()
    ps = python_sa_solve(n, at, b2b, p2b, pins, cons, otp)
    dt = time.time() - t0
    tc = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                           b2b, p2b, pins, at[:n], target_positions=tp[:n],
                           median_runtime=1.0)
    feas = bool(tc.is_feasible)        # authoritative hard-constraint flag (iccad2026_evaluate.py:556)
    nfeas += int(feas)
    if not feas:
        bad.append((idx, n, tc.overlap_violations, tc.area_violations, tc.dimension_violations))
    # cost==10.0 means INFEASIBLE; feasible cost is capped at 9.999999 (M_PENALTY-1e-6),
    # which the SA fallback hits on most cases (terrible quality, but legal) -> flag it so
    # the saturated cost is not mistaken for infeasibility.
    sat = "  (feasible cap 9.999999, NOT infeasible)" if feas and tc.cost >= 9.999998 else ""
    print(f"  case {idx:>2} n={n:>3}  is_feasible={str(feas):>5}  cost={tc.cost:.6f}  {dt:5.1f}s{sat}",
          flush=True)

print(f"\nfallback python_sa_solve is_feasible: {nfeas}/100   ({time.time()-t_all:.0f}s total)")
if bad:
    print("INFEASIBLE (idx,n,overlap,area,dim):", bad, "  -> safety-net BUG, needs fix")
else:
    print("ALL 100 is_feasible=True -> fallback safety net OK")
    print("(cost~9.999999 on most cases = feasible-quality CEILING, not infeasibility; expected")
    print(" for a pure-SA fallback that only triggers when EVERY constructive profile fails.)")
