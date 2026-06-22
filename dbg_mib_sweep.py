"""M37 MIB-aspect ratio sweep (loads dataset ONCE, unlike repeated profile_vs_portfolio).
For each ratio r, runs MIB_ASPECT=r stacked on a base recipe across all 100 cases and
reports oracle-min gain vs the current portfolio (optimizer_constructive_results.json).
Usage: dbg_mib_sweep.py [recipe_name]   recipe in {pin_tight, gm_pin, bare}"""
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

RECIPES = {
    "pin_tight": {"ICCAD_FREE_ASPECT":"1","ICCAD_WIRE_BFS":"1","ICCAD_BFS_PIN":"1",
                  "ICCAD_WIRE_TIEBREAK":"1","ICCAD_FRAME_SCALES":"1.00,1.05,1.10,1.20","ICCAD_WIRE_MULT":"2.0"},
    "gm_pin":    {"ICCAD_FREE_ASPECT":"1","ICCAD_GUIDE_MED":"1","ICCAD_WIRE_BFS":"1","ICCAD_BFS_PIN":"1",
                  "ICCAD_WIRE_TIEBREAK":"1","ICCAD_WIRE_MULT":"2.0"},
    "bare":      {},
    # M36 lead recipe (ungated wide anchored + per-member cluster + free) -- where 97/98/89 are won
    "anchored":  {"ICCAD_FREE_ANCHORED":"1","ICCAD_FREE_ANCHORED_BND":"1",
                  "ICCAD_FREE_ANCHORED_RATIOS":"0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0",
                  "ICCAD_FREE_CLUSTER":"1","ICCAD_FREE_CLUSTER_RATIOS":"0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0",
                  "ICCAD_FREE_ASPECT":"1","ICCAD_WIRE_BFS":"1","ICCAD_BFS_PIN":"1","ICCAD_WIRE_TIEBREAK":"1",
                  "ICCAD_FRAME_SCALES":"1.00,1.05,1.10,1.20","ICCAD_WIRE_MULT":"2.0"},
}
recipe_name = sys.argv[1] if len(sys.argv) > 1 else "pin_tight"
base = RECIPES[recipe_name]
RATIOS = [0.25, 0.333, 0.5, 0.6, 0.6667, 1.5, 2.0, 2.5, 3.0, 4.0]
if os.environ.get("ICCAD_SWEEP_RATIOS"):
    RATIOS = [float(x) for x in os.environ["ICCAD_SWEEP_RATIOS"].split(",")]

# precompute per-case serialized input + baseline (dataset loaded once)
CASES = []
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]; at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    bse, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    CASES.append((idx, n, txt, bse, cons, b2b, p2b, pins, at, tp))

def cost_of(env_extra):
    out = {}
    for (idx,n,txt,bse,cons,b2b,p2b,pins,at,tp) in CASES:
        env = dict(os.environ); env.update(env_extra)
        r = subprocess.run([EXE], input=txt, capture_output=True, text=True, env=env)
        ps = _parse_output(r.stdout, n)
        m = evaluate_solution({'positions': ps, 'runtime': 1.0}, bse, cons[:n], b2b, p2b, pins,
                              at[:n], target_positions=tp[:n], median_runtime=1.0)
        out[idx] = (n, m.cost)
    return out

print(f"recipe={recipe_name} base={base}\nportfolio Total = {PORT['total_score']:.4f}")
for r in RATIOS:
    env = dict(base); env["ICCAD_MIB_ASPECT"] = str(r)
    cc = cost_of(env)
    totW = 0.0; gain = 0.0; wins = []
    for idx,(n,c) in cc.items():
        w = math.exp(n/12.0); totW += w
        if c < pcost[idx] - 1e-4:
            wins.append((idx, n, pcost[idx], c)); gain += w*(pcost[idx]-c)
    g = gain/totW*100
    wstr = " ".join(f"{idx}({pcost[idx]:.4f}->{c:.4f})" for idx,n,pc,c in sorted(wins, key=lambda x:-(math.exp(x[1]/12)*(x[2]-x[3]))))
    print(f"  MIB_ASPECT={r:<6} oracle-min gain={g:+.3f}%  wins={len(wins)}  {wstr}")
