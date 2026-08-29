import json,math,sys
def wt(p):
    d=json.load(open(p)); rs=d["test_results"]
    ns=[r.get("n", r.get("block_count")) for r in rs]
    mx=max(ns); w=[math.exp((n-mx)/12) for n in ns]
    cs=[r["cost"] for r in rs]
    feas=sum(1 for r in rs if r.get("feasible"))
    return sum(c*x for c,x in zip(cs,w))/sum(w), len(rs), feas
for p in sys.argv[1:]:
    try:
        t,n,f=wt(p); print("%-42s %.16f n=%d feas=%d" % (p,t,n,f))
    except Exception as e: print(p,"ERR",e)
