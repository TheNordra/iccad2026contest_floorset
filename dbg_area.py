"""Decompose area_gap for constructive.exe (base profile).
For each case compare our bbox to the ground-truth baseline bbox:
  density = bbox / sum(block areas)   (1.0 = perfect tiling)
  agap    = our_bbox/base_bbox - 1
  w_ratio,h_ratio = our_dim / base_dim  (aspect/dimension mismatch)
If density_ours >> density_base, the gap is removable dead space (compaction).
If density_ours ~ density_base but agap>0, it's aspect/position mismatch.
"""
import os, sys, math, subprocess
from pathlib import Path
import torch
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")

def bbox(ps):
    xmin=min(p[0] for p in ps); ymin=min(p[1] for p in ps)
    xmax=max(p[0]+p[2] for p in ps); ymax=max(p[1]+p[3] for p in ps)
    return xmax-xmin, ymax-ymin

rows=[]; totW=0.0; wsum_agap=0.0; wsum_db=0.0; wsum_do=0.0
for idx in range(len(ev.dataset)):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    ps = _parse_output(subprocess.run([EXE], input=txt, capture_output=True, text=True).stdout, n)
    sumA = float(sum(float(at[i]) for i in range(n)))
    wo,ho = bbox(ps); wb,hb = bbox([tuple(float(v) for v in tp[i]) for i in range(n)])
    areao=wo*ho; areab=float(base['area_baseline'])
    do=areao/sumA; db=areab/sumA; agap=areao/areab-1.0
    w = math.exp(n/12.0); totW+=w
    wsum_agap+=w*agap; wsum_db+=w*db; wsum_do+=w*do
    rows.append((idx,n,w,agap,db,do,wo/wb,ho/hb))

print(f"weighted: agap={wsum_agap/totW:.3f}  density_base={wsum_db/totW:.3f}  density_ours={wsum_do/totW:.3f}")
print(f"{'case':>4} {'n':>4} {'wt%':>5} {'agap':>6} {'dBase':>6} {'dOurs':>6} {'w/wb':>5} {'h/hb':>5}")
rows.sort(key=lambda r:-(r[2]*r[3]))
for (idx,n,w,agap,db,do,wr,hr) in rows[:25]:
    print(f"{idx:>4} {n:>4} {100*w/totW:5.2f} {agap:6.3f} {db:6.3f} {do:6.3f} {wr:5.2f} {hr:5.2f}")
