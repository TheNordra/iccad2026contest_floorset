"""L282 diagnostic: chain-shortening moves pass the certificate BY CONSTRUCTION
(they are only kept when both new chains fit the current box), so when the LP
still rejects them the cause must be one of the things the certificate does not
model.  Same three arms as `l281_why_infeasible.py`, so the two moves can be
compared directly:

  (a) drop the boundary equalities                  -- M64 got 0/15, L281 0/30
  (b) drop them AND let the bbox grow 20 %
  (c) let the bbox grow 20 % alone

Arm (c) is the interesting one here: this move is supposed to SHRINK the box, so
if it only becomes feasible when the box is allowed to GROW, the obstacle is the
displacement geometry -- the unit cannot get to its target through the layout --
rather than the destination being too small.

Read-only on l282_cache.pkl.
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
DB = pickle.load(open(_DIR / "l282_cache.pkl", "rb"))["db"]

todo = [(k[1], k[2], k[3], v) for k, v in DB.items()
        if k[0] == "rel" and str(v.get("status", "")).startswith("lp_status")
        and "tgt" in v]
todo.sort(key=lambda t: -t[3].get("pred_shrink", 0.0))

geo, res = {}, dict(bnd=0, grow=0, both=0, neither=0, n=0)
for ci, ku, ic, v in todo[:nmax]:
    if ci not in geo:
        P = [tuple(p) for p in ANCH[ci]["positions"]]
        _u, uo, uk, box, _m = L.unit_geo(ci, P)
        geo[ci] = (P, box, sorted(box))
    P, box, keys = geo[ci]
    x, y = v["tgt"]
    ex, ey = box[ku][1] - box[ku][0], box[ku][3] - box[ku][2]
    fr, _g = L.induced_rel(ku, (x, x + ex, y, y + ey), box, keys)
    a = m64.lp_pass_flip(ci, P, area_obj=True, force_rel=fr,
                         skip_bnd_ties=True)[0]
    c = m64.lp_pass_flip(ci, P, area_obj=True, force_rel=fr,
                         bbox_relax=1.2)[0]
    b = m64.lp_pass_flip(ci, P, area_obj=True, force_rel=fr,
                         skip_bnd_ties=True, bbox_relax=1.2)[0]
    res["n"] += 1
    if a is not None:
        res["bnd"] += 1
    elif c is not None:
        res["grow"] += 1
    elif b is not None:
        res["both"] += 1
    else:
        res["neither"] += 1

print(f"== {res['n']} chain-SHORTENING moves the LP rejected "
      f"(highest predicted shrink first) ==")
print(f"   feasible once boundary equalities are dropped   : {res['bnd']}"
      f"   (M64 0/15, L281 0/30)")
print(f"   feasible once the bbox may GROW 20 %            : {res['grow']}")
print(f"   feasible only with both                         : {res['both']}")
print(f"   still infeasible with both                      : {res['neither']}")
print("\n   a move that is supposed to SHRINK the box but only solves when the")
print("   box may GROW is blocked by the journey, not the destination.")
