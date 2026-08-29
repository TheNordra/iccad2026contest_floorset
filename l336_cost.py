"""L336: utilisation is not the score. What does the B*-tree SA actually COST?

L335: the manifold reaches 0.9455 / 0.8867 / 0.8806 at n=40/80/120 with 160k
iterations and is still climbing -- above our own 0.877 at all three, which refutes
L284's 85.4 % ceiling AS A PROPERTY OF THE PROBLEM (it was a property of our packer).

But area is only 0.0458 of the cost contribution and HPWL is 0.1158 (2.5x). An
area-only SA optimises the wrong functional for OUR score even though it is the
right one for the generator. So measure both gaps, for both objectives.
"""
import math, sys, time
import torch
sys.path.insert(0, "iccad2026contest")
from iccad2026_evaluate import calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area
from l333_btree_sa import DAT, LAB, divpairs, pack_ref, rand_tree, detach, attach
import random

def anneal2(n, shapes, iters, seed, b2b, p2b, pins, hw):
    """hw = weight on HPWL relative to area. hw=0 reproduces the L333/L335 arm."""
    rng = random.Random(seed)
    L,R,par,root = rand_tree(n, rng); si=[0]*n
    def cost():
        W,H,X,Y,_p = pack_ref(L,R,si,shapes,root,n)
        a = W*H
        if hw <= 0: return a, W, H, X, Y
        pos=[(X[k],Y[k],shapes[k][si[k]][0],shapes[k][si[k]][1]) for k in range(n)]
        hp = calculate_hpwl_b2b(pos,b2b)+calculate_hpwl_p2b(pos,p2b,pins)
        return a + hw*hp, W, H, X, Y
    cur,W,H,X,Y = cost(); best=cur; bestS=(W,H,X[:],Y[:],si[:]); T0=max(cur*.05,1.)
    for it in range(iters):
        T = T0*(1-it/iters)**2 + 1e-9
        sL,sR,sP,sS,sRoot = L[:],R[:],par[:],si[:],root
        m = rng.random()
        if m < .40:
            k=rng.randrange(n)
            if len(shapes[k])>1: si[k]=rng.randrange(len(shapes[k]))
        elif m < .70:
            a,b = rng.randrange(n),rng.randrange(n)
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
                L,R,par,si,root = sL,sR,sP,sS,sRoot; continue
            root=nr; p=rng.randrange(n)
            while p==k: p=rng.randrange(n)
            attach(L,R,par,k,p,rng.randrange(2),rng)
        new,W,H,X,Y = cost()
        if new<=cur or rng.random()<math.exp(min(0.,(cur-new)/T)):
            cur=new
            if new<best: best=new; bestS=(W,H,X[:],Y[:],si[:])
        else:
            L,R,par,si,root = sL,sR,sP,sS,sRoot
    return bestS

NS=[int(x) for x in (sys.argv[1] if len(sys.argv)>1 else "40,80,120").split(",")]
IT=int(sys.argv[2]) if len(sys.argv)>2 else 40000
print("== L336 the two gaps, both objectives, %d iterations ==" % IT)
print("   %-5s %-14s %10s %10s %10s" % ("n","objective","util","hpwl_gap","area_gap"))
for n in NS:
    d = torch.load(DAT[n], weights_only=False)[0]
    meta,b2b,p2b,pins = d[0],d[1],d[2],d[3]
    lab = torch.load(LAB[n], weights_only=False)[0]; m8,poly = lab[0],lab[1]
    nb=int((meta[:,0]>0).sum())
    shapes=[divpairs(float(meta[k,0])) for k in range(nb)]
    sumA=sum(int(round(float(meta[k,0]))) for k in range(nb))
    hpb=float(m8[-2])+float(m8[-1]); arb=float(m8[0])
    # scale HPWL into the same units as area so hw is interpretable
    for tag,hw in (("area only",0.0),("area + HPWL", sumA/max(hpb,1e-9)*0.5)):
        W,H,X,Y,si = anneal2(nb,shapes,IT,7+n,b2b,p2b,pins,hw)
        pos=[(X[k],Y[k],shapes[k][si[k]][0],shapes[k][si[k]][1]) for k in range(nb)]
        hp=calculate_hpwl_b2b(pos,b2b)+calculate_hpwl_p2b(pos,p2b,pins)
        print("   %-5d %-14s %10.4f %10.4f %10.4f"
              % (n,tag,sumA/(W*H),max(0,(hp-hpb)/hpb),max(0,(calculate_bbox_area(pos)-arb)/arb)))
print("\n   our shipped mix arm: util 0.877  hpwl_gap 0.2402  area_gap 0.1176")
print("   the label:           util 0.970  hpwl_gap 0.0000  area_gap 0.0000")
