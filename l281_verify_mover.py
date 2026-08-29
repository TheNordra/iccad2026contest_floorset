"""L281 final integrity check: the reported gain must be a REAL layout.

Everything else in this probe is bookkeeping on top of LP outputs.  This takes
the stored positions of the best relocation on each case, re-runs the official
strict scorer on them from scratch, and checks (a) the cost matches what the
cache claims, (b) the layout is feasible, (c) it really beats the control.
"""
import json
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import m53_l3_probe as m53                                        # noqa: E402

aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
DB = pickle.load(open(_DIR / "l281_cache.pkl", "rb"))["db"]

best = {}
for k, v in DB.items():
    if k[0] != "rel2" or "positions" not in v:
        continue
    ci = k[1]
    c = v.get("polished", v["cost"])
    if c < best.get(ci, (float("inf"),))[0]:
        best[ci] = (c, v["positions"], k[2], k[3])

bad = 0
for ci in sorted(best):
    claim, pos, ku, ic = best[ci]
    m = m53.cost_eval(ci, [tuple(p) for p in pos])
    ct = [DB[("ctrl", ci)]["cost"]] if DB.get(("ctrl", ci), {}).get("feas") \
        else []
    ct += [v["cost"] for k, v in DB.items()
           if k[0] == "ctrlp" and k[1] == ci and v.get("feas")]
    cb = min(ct + [ANCH[ci]["cost"]])
    ok = (m.cost == claim) and bool(m.is_feasible) and m.cost < cb
    bad += 0 if ok else 1
    print(f"case {ci:3d} {str(ku):12s} claim {claim:.12f}  rescored "
          f"{m.cost:.12f}  feasible {int(m.is_feasible)}  base {cb:.12f}  "
          f"gain {100.0 * (cb - m.cost) / cb:+.4f} %  {'OK' if ok else 'FAIL'}")
print(f"\n== {len(best)} best-per-case layouts re-scored from stored "
      f"positions: {bad} failures ==")
if bad:
    raise SystemExit("INTEGRITY FAIL")
print("   every reported gain is a real, feasible layout under the official "
      "strict scorer")
