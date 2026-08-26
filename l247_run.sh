#!/bin/sh
cd /c/ICCAD_ml/ship_final || exit 1
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
export PYTHONIOENCODING=utf-8
"$PY" -u - <<'PYX'
import os, sys, statistics as st
os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48"); sys.argv = ["x"]
import l155_lp_rows as M, l129_global_placer as L, optimizer_l247probe as P
M.oc, M._l3 = P, P.l3
kw = M._lpkw(); lay = M._load_layouts("results_L153_lpoff_L137.json")
cs = [(i, c) for i, c in enumerate(L.CASES) if i in lay]
print(f"[l247] {len(cs)} cases, prune_B=8 (shipped)")
for i, c in cs:
    M.one(c, lay[i], 8.0, kw, 1)
R = P._L247
by = {}
for r in R:
    by.setdefault(r["ci"] if isinstance(r["ci"], int) else id(r["ci"]), []).append(r)
# l155 uses one key for every case, so group by call order instead
rounds = []
cur = []
for r in R:
    cur.append(r)
    if not r["bad"]:
        rounds.append(cur); cur = []
if cur: rounds.append(cur)
print(f"   solve_pruned invocations captured: {len(rounds)}")
nr = [len(g) for g in rounds]
print(f"   rounds per case: mean {sum(nr)/len(nr):.2f}  "
      f"1-round {sum(1 for x in nr if x==1)}  2 {sum(1 for x in nr if x==2)}  "
      f"3+ {sum(1 for x in nr if x>=3)}   of {len(nr)}")
tot_builds = sum(nr)
print(f"   total builds {tot_builds}; repair builds {tot_builds-len(nr)} "
      f"({100*(tot_builds-len(nr))/tot_builds:.1f}% of all builds/solves)")
print()
print("=== is the FORCED set predictable from the pre-solve margin? ===")
print("for each round-1 that needed a repair: where do the forced tids rank when")
print("all dropped terms are sorted by margin = |dC| - slack, ASCENDING")
pct, need, ndrop = [], [], []
for g in rounds:
    r0 = g[0]
    if not r0["bad"]:
        continue
    m = r0["margins"]
    order = sorted(m, key=lambda t: m[t])
    pos = {t: k for k, t in enumerate(order)}
    ranks = [pos[t] for t in r0["bad"] if t in pos]
    if not ranks:
        continue
    N = len(order)
    pct.append(100.0 * max(ranks) / N)
    need.append(len(ranks))
    ndrop.append(N)
if pct:
    pct.sort()
    print(f"   cases needing a repair: {len(pct)}")
    print(f"   forced terms per such case: median {st.median(need):.0f} of "
          f"{st.median(ndrop):.0f} dropped ({100*st.median(need)/st.median(ndrop):.2f}%)")
    print(f"   DEEPEST forced term's percentile in the margin order:")
    print(f"      p10 {pct[len(pct)//10]:.1f}%   p50 {st.median(pct):.1f}%   "
          f"p90 {pct[-max(1,len(pct)//10)]:.1f}%   max {pct[-1]:.1f}%")
    for k in (1, 2, 5, 10, 20, 50):
        hit = sum(1 for x in pct if x <= k)
        print(f"      force-keeping the smallest-margin {k:>2}% would pre-empt "
              f"{hit}/{len(pct)} repairs ({100*hit/len(pct):.0f}%)")
else:
    print("   no repairs observed")
PYX
echo L247_DONE
