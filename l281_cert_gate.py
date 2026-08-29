"""L281 certificate soundness gate.

The census classifies ~83 % of relocations as CYCLIC or OVERSIZED and never
gives them to the LP.  That is only legitimate if the certificate is a
NECESSARY condition for LP feasibility -- i.e. nothing it rejects can be
LP-feasible.  This feeds a sample of REJECTED candidates to the same LP and
asserts every one comes back infeasible.  One feasible row invalidates the
whole census, so this runs before any number from it is reported.

No cache writes, so it is safe to run beside a scan.
"""
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import l281_reloc_probe as L                                       # noqa: E402
import m64_flip_probe as m64                                       # noqa: E402

cases = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                          else "85,88").split(",")]
per = int(sys.argv[2]) if len(sys.argv) > 2 else 20

aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
L.ANCH = ANCH

bad, tested = [], 0
for ci in cases:
    P0 = [tuple(p) for p in ANCH[ci]["positions"]]
    ranked, G = L.rank_units(ci, P0)
    box, keys, bb = G["box"], G["keys"], G["bb"]
    unit_of, ukey = G["unit_of"], G["ukey"]
    got = 0
    for rec in ranked:
        if got >= per:
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
        ex, ey = box[ku][1] - box[ku][0], box[ku][3] - box[ku][2]
        for x, y in tg:
            if got >= per:
                break
            fr, _gm = L.induced_rel(ku, (x, x + ex, y, y + ey), box, keys)
            if fr == cur_fr:
                continue
            EHu, EVu = L.unit_edges(ci, P0, unit_of, ukey, ku, fr)
            cert = L.certificate(ci, P0, EHb, EVb, EHu, EVu, bb)
            if cert["ok"]:
                continue                       # only REJECTED ones are the test
            sol, tele = m64.lp_pass_flip(ci, P0, area_obj=True, force_rel=fr)
            tested += 1
            got += 1
            if sol is not None:
                m = L.cost_eval(ci, sol)
                bad.append((ci, ku, cert["why"], m.cost, bool(m.is_feasible)))
                print(f"  !! c{ci} {ku} certificate said {cert['why']} but the "
                      f"LP solved it: cost {m.cost:.6f} "
                      f"feas={int(m.is_feasible)}", flush=True)
    print(f"case {ci}: {got} rejected candidates re-tested", flush=True)

print(f"\n== certificate soundness: {tested} candidates the certificate "
      f"REJECTED, fed to the LP ==")
print(f"   LP found a solution for {len(bad)} of them")
if bad:
    raise SystemExit("GATE FAIL: the certificate is not a necessary condition "
                     "-- every census number is invalid")
print("   GATE PASS: nothing the certificate rejects is LP-feasible")
