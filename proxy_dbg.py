"""For given cases, run all portfolio profiles; print each profile's proxy value vs
TRUE cost, and show the proxy's pick vs the true-cost-optimal pick."""
import os, sys, math, subprocess
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

def short(p):
    if not p: return "base"
    fa = p.get("ICCAD_FRAME_ASPECTS","")
    tag=[]
    if fa.startswith("0.55"): tag.append("narrow")
    elif fa.startswith("0.67"): tag.append("tall")
    if "ICCAD_FRAME_SCALES" in p: tag.append("tight")
    if p.get("ICCAD_LR_ASPECT"): tag.append("LR"+p["ICCAD_LR_ASPECT"])
    if p.get("ICCAD_WIRE_MULT"): tag.append("W"+p["ICCAD_WIRE_MULT"])
    if p.get("ICCAD_ANCHOR_W"): tag.append("anc"+p["ICCAD_ANCHOR_W"])
    return ",".join(tag) or "base"

for idx in [int(a) for a in sys.argv[1:]] or [98]:
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    sumA = sum(max(0.0,float(at[i])) for i in range(n)); A_hat=1.035*max(sumA,1e-9)
    rows=[]
    for prof in PROFILES:
        env=dict(os.environ); env.update(prof)
        ps=_parse_output(subprocess.run([EXE],input=txt,capture_output=True,text=True,env=env).stdout,n)
        m=oc._proxy_metrics(ps, at, b2b, p2b, pins, cons, n)
        tc=evaluate_solution({'positions':ps,'runtime':1.0}, base, cons[:n], b2b,p2b,pins,at[:n],
                             target_positions=tp[:n], median_runtime=1.0)
        rows.append((short(prof), m["area"], m["hpwl"], m["vrel"], tc.cost))
    hmin=min(r[2] for r in rows) or 1.0
    for r in rows:
        r_proxy=(r[1]/A_hat + r[2]/hmin)*math.exp(2*r[3])
        rows[rows.index(r)] = r + (r_proxy,)
    proxy_pick=min(rows,key=lambda r:r[5])
    true_pick=min(rows,key=lambda r:r[4])
    print(f"=== case {idx} n={n}  hbase={base['hpwl_baseline']:.0f} hmin={hmin:.0f} abase={base['area_baseline']:.0f} Ahat={A_hat:.0f} ===")
    print(f"  proxy picks: {proxy_pick[0]:22s} proxy={proxy_pick[5]:.4f} TRUE={proxy_pick[4]:.4f}")
    print(f"  TRUE-opt is: {true_pick[0]:22s} proxy={true_pick[5]:.4f} TRUE={true_pick[4]:.4f}")
    print(f"  realizable gain if proxy correct: {proxy_pick[4]-true_pick[4]:+.4f}")
    rows.sort(key=lambda r:r[4])
    print("  top-6 by TRUE cost:")
    for r in rows[:6]:
        print(f"    {r[0]:22s} area={r[1]:8.0f} hpwl={r[2]:8.0f} vrel={r[3]:.4f} proxy={r[5]:.4f} TRUE={r[4]:.4f}")
