"""L336b: what HPWL does the AREA-ONLY replay land on? (measured once, not in the loop)"""
import math, sys, time
import torch
sys.path.insert(0,"iccad2026contest")
from iccad2026_evaluate import calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area
from l333_btree_sa import DAT, LAB, divpairs, pack_ref
from l334_fixed_outline import anneal
import random
from l333_btree_sa import rand_tree, detach, attach

def anneal_keep(n, shapes, iters, seed):
    rng=random.Random(seed); L,R,par,root=rand_tree(n,rng); si=[0]*n
    W,H,X,Y,_p=pack_ref(L,R,si,shapes,root,n); cur=W*H; best=cur
    bs=(W,H,X[:],Y[:],si[:]); T0=max(cur*.05,1.)
    for it in range(iters):
        T=T0*(1-it/iters)**2+1e-9
        sL,sR,sP,sS,sRoot=L[:],R[:],par[:],si[:],root
        m=rng.random()
        if m<.40:
            k=rng.randrange(n)
            if len(shapes[k])>1: si[k]=rng.randrange(len(shapes[k]))
        elif m<.70:
            a,b=rng.randrange(n),rng.randrange(n)
            if a!=b:
                for arr in (L,R):
                    for i in range(n):
                        if arr[i]==a: arr[i]=b
                        elif arr[i]==b: arr[i]=a
                par[a],par[b]=par[b],par[a]; L[a],L[b]=L[b],L[a]; R[a],R[b]=R[b],R[a]
                for c in (L[a],R[a]):
                    if c!=-1: par[c]=a
                for c in (L[b],R[b]):
                    if c!=-1: par[c]=b
                if root==a: root=b
                elif root==b: root=a
        else:
            k=rng.randrange(n); nr=detach(L,R,par,root,k)
            if nr is None:
                L,R,par,si,root=sL,sR,sP,sS,sRoot; continue
            root=nr; p=rng.randrange(n)
            while p==k: p=rng.randrange(n)
            attach(L,R,par,k,p,rng.randrange(2),rng)
        W,H,X,Y,_p=pack_ref(L,R,si,shapes,root,n); new=W*H
        if new<=cur or rng.random()<math.exp(min(0.,(cur-new)/T)):
            cur=new
            if new<best: best=new; bs=(W,H,X[:],Y[:],si[:])
        else: L,R,par,si,root=sL,sR,sP,sS,sRoot
    return bs

IT=int(sys.argv[2]) if len(sys.argv)>2 else 40000
print("== L336b area-only replay: what does it cost on BOTH axes? (%d iters) ==" % IT)
print("   %-5s %10s %11s %11s" % ("n","util","hpwl_gap","area_gap"))
for n in [int(x) for x in (sys.argv[1] if len(sys.argv)>1 else "40,80,120").split(",")]:
    d=torch.load(DAT[n],weights_only=False)[0]; meta,b2b,p2b,pins=d[0],d[1],d[2],d[3]
    m8=torch.load(LAB[n],weights_only=False)[0][0]
    nb=int((meta[:,0]>0).sum()); shapes=[divpairs(float(meta[k,0])) for k in range(nb)]
    sumA=sum(int(round(float(meta[k,0]))) for k in range(nb))
    hpb=float(m8[-2])+float(m8[-1]); arb=float(m8[0])
    W,H,X,Y,si=anneal_keep(nb,shapes,IT,7+n)
    pos=[(X[k],Y[k],shapes[k][si[k]][0],shapes[k][si[k]][1]) for k in range(nb)]
    hp=calculate_hpwl_b2b(pos,b2b)+calculate_hpwl_p2b(pos,p2b,pins)
    print("   %-5d %10.4f %11.4f %11.4f" % (n,sumA/(W*H),max(0,(hp-hpb)/hpb),
          max(0,(calculate_bbox_area(pos)-arb)/arb)))
print("\n   our shipped mix arm: util 0.877  hpwl_gap 0.2402  area_gap 0.1176")
