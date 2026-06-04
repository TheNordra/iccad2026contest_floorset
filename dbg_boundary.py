"""Classify boundary violations for given cases: single vs cluster member,
and whether sliding the block to its required bbox edge is blocked by overlap."""
import os, sys, subprocess
from pathlib import Path
import torch
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")
L,R,T,B = 1,2,4,8

def overlaps(ps, self, x,y,w,h):
    for j,(px,py,pw,ph) in enumerate(ps):
        if j==self: continue
        ox=max(0,min(x+w,px+pw)-max(x,px)); oy=max(0,min(y+h,py+ph)-max(y,py))
        if ox>1e-6 and oy>1e-6: return j
    return -1

for idx in [int(a) for a in sys.argv[1:]] or [89]:
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    ps = _parse_output(subprocess.run([EXE], input=txt, capture_output=True, text=True).stdout, n)
    xmin=min(p[0] for p in ps); ymin=min(p[1] for p in ps)
    xmax=max(p[0]+p[2] for p in ps); ymax=max(p[1]+p[3] for p in ps)
    eps=1e-6
    print(f"=== case {idx}: n={n} bbox=[{xmin:.2f},{xmax:.2f}]x[{ymin:.2f},{ymax:.2f}] ===")
    nsingle=ncluster=nblocked=nfree=0
    for i in range(n):
        code=int(cons[i,4]) if cons.shape[1]>4 else 0
        if code==0: continue
        x,y,w,h=ps[i]; clu=int(cons[i,3]) if cons.shape[1]>3 else 0
        pre=cons[i,1]!=0 if cons.shape[1]>1 else False
        tch=[]
        if code&L: tch.append(abs(x-xmin)<eps)
        if code&R: tch.append(abs(x+w-xmax)<eps)
        if code&T: tch.append(abs(y+h-ymax)<eps)
        if code&B: tch.append(abs(y-ymin)<eps)
        if all(tch): continue
        # try slide to edge
        nx,ny=x,y
        if code&L: nx=xmin
        if code&R: nx=xmax-w
        if code&B: ny=ymin
        if code&T: ny=ymax-h
        blk=overlaps(ps,i,nx,ny,w,h)
        kind="cluster" if clu>0 else ("preplaced" if pre else "single")
        if clu>0: ncluster+=1
        else: nsingle+=1
        if blk>=0: nblocked+=1
        else: nfree+=1
        print(f"  blk {i:>3} code={code:>2} {kind:>9} clu={clu:>2} pos=({x:.1f},{y:.1f},{w:.1f},{h:.1f}) "
              f"slide->({nx:.1f},{ny:.1f}) {'BLOCKED by '+str(blk) if blk>=0 else 'FREE(nudge should fix?!)'}")
    print(f"  -> singles={nsingle} cluster={ncluster} | blocked={nblocked} free={nfree}")
