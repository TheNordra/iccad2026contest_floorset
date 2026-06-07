"""Run a candidate env-profile across all 100 cases; compare each case's TRUE cost
to the current portfolio (optimizer_constructive_results.json). Reports cases the
candidate would WIN in the portfolio (downside-protected) and the oracle-min gain."""
import os, sys, math, json, subprocess
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos
ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")
PORT = json.load(open(_DIR/"iccad2026contest"/"optimizer_constructive_results.json"))
pcost = {t['test_id']: t['cost'] for t in PORT['test_results']}

# candidate profile from argv: KEY=VAL pairs
extra = {}
for a in sys.argv[1:]:
    k,v=a.split("="); extra[k]=v
print("candidate env:", extra)

def run(idx, env_extra):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    env = dict(os.environ); env.update(env_extra)
    r = subprocess.run([EXE], input=txt, capture_output=True, text=True, env=env)
    ps = _parse_output(r.stdout, n)
    m = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                          b2b, p2b, pins, at[:n], target_positions=tp[:n], median_runtime=1.0)
    return n, m.cost

totW=0.0; gain=0.0; wins=[]
for idx in range(100):
    n,cC = run(idx, extra)
    w=math.exp(n/12.0); totW+=w
    pc = pcost[idx]
    if cC < pc-1e-4:
        wins.append((w*(pc-cC), idx, n, pc, cC)); gain += w*(pc-cC)
wins.sort(reverse=True)
print("=== cases candidate BEATS current portfolio ===")
for g,idx,n,pc,cc in wins:
    print(f"  case {idx:3d} n={n:3d} port={pc:.4f} cand={cc:.4f} gain={pc-cc:+.4f} wContr={g/totW*100:.3f}%")
print(f"\nportfolio Total = {PORT['total_score']:.4f}")
print(f"if added (oracle-min) gain = {gain/totW*100:.3f}%  -> ~{PORT['total_score']-gain/totW:.4f}")
