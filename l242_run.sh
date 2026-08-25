#!/bin/sh
cd /c/ICCAD_ml/ship_final || exit 1
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
export PYTHONIOENCODING=utf-8
"$PY" -u - <<'PYX'
import os, sys, time
os.environ.setdefault("ICCAD_ADAPTIVE_CORES", "48"); sys.argv = ["x"]
import l155_lp_rows as M, l129_global_placer as L, optimizer_constructive as oc
real = oc.linprog
st = {"o": {}}
def wrapped(*a, **k):
    k = dict(k)
    if st["o"]:
        k["options"] = dict(k.get("options") or {}, **st["o"])
    return real(*a, **k)
oc.linprog = wrapped
kw = M._lpkw(); lay = M._load_layouts("results_L153_lpoff_L137.json")
cs = {i: c for i, c in enumerate(L.CASES) if i in lay}
print("the four objective movers, in full")
for i in (10, 32, 70, 93):
    row = {}
    for tag, opts in (("shipped", {}), ("devex", {"simplex_dual_edge_weight_strategy": "devex"})):
        st["o"] = opts
        row[tag] = M.one(cs[i], lay[i], 8.0, kw, 1)
    a, b = row["shipped"], row["devex"]
    rel = abs(b["obj"] - a["obj"]) / abs(a["obj"])
    print("  case {:3d} n={:3d}  shipped {:.12f}  devex {:.12f}  rel {:.2e}  "
          "rounds shipped/devex builds {}/{}  hard_ok {}/{}"
          .format(i, cs[i]["n"], a["obj"], b["obj"], rel, a["calls"], b["calls"],
                  a["ok"], b["ok"]))
oc.linprog = real
PYX
echo L242_DONE
