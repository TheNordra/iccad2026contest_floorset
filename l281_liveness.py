"""L281 liveness (handoff trap #1: a mechanism that reads PASS because it was
never actually applied).  Many relocations come back with cost EXACTLY equal to
the control's.  That is either

  (a) live but non-binding / degenerate -- the LP reaches an equal-cost optimum
      in the new topology, positions differ; or
  (b) a silent no-op -- force_rel never reached the LP rows.

Only (b) is a bug, and cost alone cannot tell them apart.  This compares
POSITIONS bit by bit and reports, for each sampled relocation, how many blocks
moved and how far -- with no cache writes, so it can run beside the scan.
"""
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import l281_reloc_probe as L                                       # noqa: E402
import m64_flip_probe as m64                                       # noqa: E402

ci = int(sys.argv[1]) if len(sys.argv) > 1 else 85
nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 8

aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
L.ANCH = ANCH
P0 = [tuple(p) for p in ANCH[ci]["positions"]]
n = L.CASES[ci]["n"]

base, _t = m64.lp_pass_flip(ci, P0, area_obj=True)
mb = L.cost_eval(ci, base)
print(f"[liveness] case {ci}: control cost {mb.cost:.12f}")

ranked, G = L.rank_units(ci, P0)
box, keys, bb = G["box"], G["keys"], G["bb"]
unit_of, ukey = G["unit_of"], G["ukey"]

shown = 0
for rec in ranked:
    if shown >= nshow:
        break
    if rec["pinned"]:
        continue
    ku = rec["ku"]
    vx, vy, nb = L.wire_terms(ci, P0, unit_of, ukey, ku)
    tg = L.gen_targets(ku, box, keys, bb, nb, vx, vy, 8)
    if not tg:
        continue
    EHb, EVb = L.base_graph(ci, P0, unit_of, ukey, ku)
    cur_fr, _g = L.induced_rel(ku, box[ku], box, keys)
    ex = box[ku][1] - box[ku][0]
    ey = box[ku][3] - box[ku][2]
    for x, y in tg:
        fr, gmin = L.induced_rel(ku, (x, x + ex, y, y + ey), box, keys)
        if fr == cur_fr:
            continue
        nflip = sum(1 for pk, k in fr.items() if cur_fr.get(pk) != k)
        EHu, EVu = L.unit_edges(ci, P0, unit_of, ukey, ku, fr)
        if not L.certificate(ci, P0, EHb, EVb, EHu, EVu, bb)["ok"]:
            continue
        sol, tele = m64.lp_pass_flip(ci, P0, area_obj=True, force_rel=fr)
        if sol is None:
            continue
        m = L.cost_eval(ci, sol)
        nd = sum(1 for a, b in zip(base, sol)
                 if any(u != v for u, v in zip(a, b)))
        far = max((max(abs(u - v) for u, v in zip(a, b))
                   for a, b in zip(base, sol)), default=0.0)
        # did the RELOCATED unit itself actually end up somewhere else?
        mem = [i for i in range(n) if ukey[i] == ku]
        du = max(max(abs(base[i][0] - sol[i][0]), abs(base[i][1] - sol[i][1]))
                 for i in mem)
        print(f"  {ku} nflip={nflip:3d} cost {m.cost:.12f} "
              f"d={mb.cost - m.cost:+.3e}  blocks moved {nd:3d}/{n}  "
              f"maxmove {far:8.4f}  unit moved {du:8.4f}", flush=True)
        shown += 1
        break

print("\n[liveness] verdict: a row with 'blocks moved 0' AND 'unit moved 0' "
      "would be a silent no-op; anything else is live-but-degenerate.")
