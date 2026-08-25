#!/bin/sh
cd /c/ICCAD_ml/ship_final || exit 1
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
export PYTHONIOENCODING=utf-8
"$PY" -u - <<'PYX'
import os, sys, time
os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48"); sys.argv = ["x"]
import l155_lp_rows as M, l129_global_placer as L, optimizer_constructive as oc
real = oc.linprog
st = {"o": {}, "t": 0.0}
def wrapped(*a, **k):
    k = dict(k)
    if st["o"]:
        k["options"] = dict(k.get("options") or {}, **st["o"])
    t0 = time.perf_counter(); r = real(*a, **k); st["t"] += time.perf_counter()-t0
    return r
oc.linprog = wrapped
kw = M._lpkw(); lay = M._load_layouts("results_L153_lpoff_L137.json")
cs = [(i, c) for i, c in enumerate(L.CASES) if i in lay]
print("[l241] {} cases, shipped vs devex, min-of-3".format(len(cs)))
CFG = [("shipped", {}), ("devex", {"simplex_dual_edge_weight_strategy": "devex"})]
out = {}
for tag, opts in CFG:
    st["o"] = opts
    tot = solve = 0.0
    for i, c in cs:
        best = None
        for _ in range(3):
            st["t"] = 0.0; t0 = time.perf_counter()
            r = M.one(c, lay[i], 8.0, kw, 1)
            w = time.perf_counter() - t0
            if r is None: break
            if best is None or w < best[0]: best = (w, st["t"], r)
        if best is None: continue
        tot += best[0]; solve += best[1]
        out.setdefault(tag, {})[i] = (best[0], best[1], best[2], c["n"])
oc.linprog = real
import statistics as stt
a, b = out["shipped"], out["devex"]
ids = sorted(set(a) & set(b))
ta = sum(a[i][0] for i in ids); tb = sum(b[i][0] for i in ids)
sa = sum(a[i][1] for i in ids); sb = sum(b[i][1] for i in ids)
mv = sum(1 for i in ids if a[i][2]["lay"] != b[i][2]["lay"])
bad = [i for i in ids if a[i][2]["obj"] and b[i][2]["obj"]
       and abs(a[i][2]["obj"] - b[i][2]["obj"]) / abs(a[i][2]["obj"]) > 1e-9]
okd = sum(1 for i in ids if b[i][2]["ok"]); oka = sum(1 for i in ids if a[i][2]["ok"])
print()
print("cases {}   whole-LP shipped {:.2f}s  devex {:.2f}s  speed {:.3f}x"
      .format(len(ids), ta, tb, ta/tb))
print("             solve   shipped {:.2f}s  devex {:.2f}s  speed {:.3f}x"
      .format(sa, sb, sa/sb))
for lo, hi in ((20,60),(60,100),(100,121)):
    v = [i for i in ids if lo < a[i][3] <= hi]
    if v:
        x = sum(a[i][0] for i in v); y = sum(b[i][0] for i in v)
        print("   {:>3}-{:<4} shipped {:6.2f}s  devex {:6.2f}s  {:.3f}x  ({} cases)"
              .format(lo+1, hi, x, y, x/y, len(v)))
print()
print("layout hash moved on {}/{}   objective moved (>1e-9) on {}   hard_ok kept {} -> {}"
      .format(mv, len(ids), len(bad), oka, okd))
if bad: print("   objective movers:", bad[:12])
PYX
echo L241_DONE
