import os, sys, subprocess
from pathlib import Path
import torch
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
for idx in [int(a) for a in sys.argv[1:]] or [0]:
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    npre = sum(1 for i in range(n) if cons.dim()>1 and cons.shape[1]>1 and cons[i,1]!=0)
    nb = sum(1 for i in range(n) if cons.dim()>1 and cons.shape[1]>4 and int(cons[i,4])!=0)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    env = dict(os.environ); env["CONSTRUCTIVE_DEBUG"]="1"
    r = subprocess.run([str(_DIR/"constructive.exe")], input=txt, capture_output=True, text=True, env=env)
    ps = _parse_output(r.stdout, n)
    xs=[p[0] for p in ps]; ye=[p[1]+p[3] for p in ps]; xe=[p[0]+p[2] for p in ps]; ymn=[p[1] for p in ps]
    bw=max(xe)-min(xs); bh=max(ye)-min(ymn)
    print(f"case {idx}: n={n} npreplaced={npre} nboundary={nb} baseH={base['hpwl_baseline']:.0f} baseA={base['area_baseline']:.0f}")
    print(f"   sum_area={sum(p[2]*p[3] for p in ps):.0f} OUR bbox={bw:.1f}x{bh:.1f} area={bw*bh:.0f}")
    print("   stderr:", r.stderr.strip())
