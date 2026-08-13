import pickle, collections, math, os, sys
os.environ["ICCAD_ADAPTIVE_CORES"]="48"
sys.path.insert(0,'iccad2026contest'); sys.path.insert(0,'.')
import m67_oos_probe as m67, optimizer_constructive as oc, torch
import l124_r3_scale as R
R.oc = oc
C=pickle.load(open('l124_r3_cache.pkl','rb'))
tot=collections.Counter(); per={}
for s in ('s1','s2'):
    rows=[(k[1],v) for k,v in C.items() if k[0]==s]
    byf=collections.defaultdict(list)
    for ck,v in rows: byf[v['fk']].append((ck,v))
    t=collections.Counter()
    for fk in sorted(byf):
        d=torch.load(m67._path_of(fk))
        for ck,v in byf[fk]:
            lay=m67._load_case(d,v['L']); lay['base'],_=m67._baseline_official(lay)
            on=v['cap']['1']
            win=R._arbitrate(v['cap']['0'],on,set(on),lay['at'],lay['n'])
            for i in on:
                if on[i][0] is win: t[i]+=1; break
    per[s]=t; tot.update(t)
print('combined tally (s1+s2), top 14:')
for i,c in tot.most_common(14):
    print('  #%-3d  total %2d   (s1 %d / s2 %d)'%(i,c,per['s1'][i],per['s2'][i]))
top8=[i for i,_ in tot.most_common(8)]
print()
print('TOP8 =',sorted(top8))
print('both-sample support:',[i for i in sorted(top8) if per['s1'][i]>0 and per['s2'][i]>0])
