"""Cache (area,hpwl,vrel,true_cost) for every profile×case ONCE (parallel), then
sweep _RH offline: for each _RH, simulate proxy selection per case and compute the
resulting Total Score. Finds the _RH that best aligns proxy selection to true cost."""
import os, sys, math, subprocess, concurrent.futures
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc
ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")
PROFILES = oc._PROFILES

def run_one(args):
    txt, prof, base, cons, b2b, p2b, pins, at, tp, n = args
    env=dict(os.environ); env.update(prof)
    ps=_parse_output(subprocess.run([EXE],input=txt,capture_output=True,text=True,env=env).stdout,n)
    m=oc._proxy_metrics(ps, at, b2b, p2b, pins, cons, n)
    tc=evaluate_solution({'positions':ps,'runtime':1.0}, base, cons[:n], b2b,p2b,pins,at[:n],
                         target_positions=tp[:n], median_runtime=1.0)
    return (m["area"], m["hpwl"], m["vrel"], tc.cost)

cases=[]
tasks=[]
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    sumA = sum(max(0.0,float(at[i])) for i in range(n)); A_hat=1.035*max(sumA,1e-9)
    cases.append({"idx":idx,"n":n,"A_hat":A_hat,"start":len(tasks)})
    for prof in PROFILES:
        tasks.append((txt,prof,base,cons,b2b,p2b,pins,at,tp,n))

print(f"running {len(tasks)} profile-case combos in parallel...", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
    results=list(ex.map(run_one, tasks))
for c in cases:
    c["metrics"]=results[c["start"]:c["start"]+len(PROFILES)]

def total_for_rh(rh):
    totW=totC=0.0
    for c in cases:
        ms=c["metrics"]; A_hat=c["A_hat"]
        hmin=min(m[1] for m in ms) or 1.0
        best=min(ms, key=lambda m:(m[0]/A_hat + rh*m[1]/hmin)*math.exp(2*m[2]))
        w=math.exp(c["n"]/12.0); totW+=w; totC+=w*best[3]
    return totC/totW

# oracle (perfect selection)
totW=oc_=0.0
for c in cases:
    w=math.exp(c["n"]/12.0); totW+=w; oc_+=w*min(m[3] for m in c["metrics"])
print(f"oracle-min (perfect selection) = {oc_/totW:.4f}")
print("RH sweep:")
for rh in [0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.5]:
    print(f"  _RH={rh:.2f}  Total={total_for_rh(rh):.4f}")
