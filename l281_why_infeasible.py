"""L281: the certificate keeps 94 % of the wire value, and then the LP rejects
77-85 % of what it kept.  So what, inside the LP, is doing the rejecting?

The certificate checks only acyclicity and chain-vs-bbox.  The LP additionally
carries (a) boundary equalities tying satisfied boundary blocks and the four
extreme-definers to the bbox vars, and (b) frozen/preplaced blocks that have no
delta variable at all and so are hard points in the middle of the layout.

M64 ran exactly this diagnostic on its own move and got 0/15 -- boundary
equalities were NOT its cause.  This repeats it for relocation, on
certified-coherent candidates only, so the two answers are comparable.

Read-only on the cache.
"""
import json
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import l281_reloc_probe as L                                       # noqa: E402
import m64_flip_probe as m64                                       # noqa: E402

nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 30

aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
L.ANCH = ANCH
DB = pickle.load(open(_DIR / "l281_cache.pkl", "rb"))["db"]

cen = {k[1]: v for k, v in DB.items() if k[0] == "census" and len(k) > 2}
todo = []
for k, v in DB.items():
    if k[0] != "rel2" or not str(v.get("status", "")).startswith("lp_status"):
        continue
    todo.append((k[1], k[2], k[3], v))
todo.sort(key=lambda t: (t[0], str(t[1]), t[2]))

geo = {}
res = dict(bnd=0, frozen=0, both=0, neither=0, n=0)
for ci, ku, ic, v in todo:
    if res["n"] >= nmax:
        break
    if ci not in geo:
        P = [tuple(p) for p in ANCH[ci]["positions"]]
        _u, uo, uk, box, _m = L.unit_geo(ci, P)
        geo[ci] = (P, box, sorted(box), uo, uk)
    P, box, keys, uo, uk = geo[ci]
    if "tgt" not in v:
        continue
    x, y = v["tgt"]
    ex, ey = box[ku][1] - box[ku][0], box[ku][3] - box[ku][2]
    fr, _g = L.induced_rel(ku, (x, x + ex, y, y + ey), box, keys)
    if L.n_binding(ku, box[ku], fr, box, keys) <= 0:
        continue
    # arm 1: drop the boundary equalities (M64's own diagnostic)
    s1, _t1 = m64.lp_pass_flip(ci, P, area_obj=True, force_rel=fr,
                               skip_bnd_ties=True)
    # arm 2: drop the boundary equalities AND let the bbox grow 20 %
    s2, _t2 = m64.lp_pass_flip(ci, P, area_obj=True, force_rel=fr,
                               skip_bnd_ties=True, bbox_relax=1.2)
    res["n"] += 1
    if s1 is not None:
        res["bnd"] += 1
    elif s2 is not None:
        res["both"] += 1
    else:
        res["neither"] += 1
print(f"== {res['n']} certified-coherent, BINDING, LP-infeasible relocations ==")
print(f"   feasible once boundary equalities are dropped        : "
      f"{res['bnd']}   (M64's answer on its own move was 0/15)")
print(f"   feasible only once the bbox may ALSO grow 20 %       : "
      f"{res['both']}")
print(f"   still infeasible with no boundary ties and +20 % bbox: "
      f"{res['neither']}")
