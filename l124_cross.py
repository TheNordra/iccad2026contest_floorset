import pickle, collections, math, os, sys
os.environ["ICCAD_ADAPTIVE_CORES"]="48"
sys.path.insert(0,'iccad2026contest'); sys.path.insert(0,'.')
import m67_oos_probe as m67, optimizer_constructive as oc, torch
import l124_r3_scale as R
R.oc = oc
C=pickle.load(open('l124_r3_cache.pkl','rb'))
def load(sample):
    rows=[(k[1],v) for k,v in C.items() if k[0]==sample]
    lays={}; byf=collections.defaultdict(list)
    for ck,v in rows: byf[v['fk']].append((ck,v))
    for fk in sorted(byf):
        d=torch.load(m67._path_of(fk))
        for ck,v in byf[fk]:
            lay=m67._load_case(d,v['L']); lay['base'],_=m67._baseline_official(lay); lays[ck]=lay
    return rows,lays
def tally(rows,lays):
    t=collections.Counter()
    for ck,v in rows:
        on=v['cap']['1']; lay=lays[ck]
        win=R._arbitrate(v['cap']['0'],on,set(on),lay['at'],lay['n'])
        for i in on:
            if on[i][0] is win: t[i]+=1; break
    return [i for i,_ in t.most_common()]
def score(rows,lays,flip,cache={}):
    byn=collections.defaultdict(list)
    for ck,v in rows:
        lay=lays[ck]
        pos=R._arbitrate(v['cap']['0'],v['cap']['1'],flip,lay['at'],lay['n'])
        key=(id(lay),tuple(map(tuple,pos)))
        c=cache.get(key)
        if c is None: c=cache[key]=float(m67._cost(pos,lay).cost)
        byn[v['n']].append(c)
    num=den=0.0
    for n,vv in byn.items():
        w=math.exp(n/12.0); num+=w*(sum(vv)/len(vv)); den+=w
    return num/den
r1,l1=load('s1'); r2,l2=load('s2')
o1,o2=tally(r1,l1),tally(r2,l2)
print('top-8 selected on s1:',o1[:8])
print('top-8 selected on s2:',o2[:8])
print('intersection:',sorted(set(o1[:8])&set(o2[:8])))
b1,b2=score(r1,l1,set()),score(r2,l2,set())
print()
for K in (4,6,8,12):
    f1,f2=set(o1[:K]),set(o2[:K])
    a_in,a_out=100*(1-score(r1,l1,f1)/b1),100*(1-score(r2,l2,f1)/b2)
    b_out,b_in=100*(1-score(r1,l1,f2)/b1),100*(1-score(r2,l2,f2)/b2)
    print('K=%-2d  set from s1: s1 %+.4f%% (in)   s2 %+.4f%% (OUT)'%(K,a_in,a_out))
    print('      set from s2: s1 %+.4f%% (OUT)  s2 %+.4f%% (in)'%(b_out,b_in))
