"""L281 gate 0: can the m53/m64 offline scorer reproduce a given anchor json
bit-exactly?  Run for both the m64 anchor and the SHIPPED in-set-100 anchor."""
import json, sys
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import m53_l3_probe as m53
cost_eval, W, TOTW = m53.cost_eval, m53.W, m53.TOTW

for f in sys.argv[1:]:
    j = json.load(open(f)); tr = {t["test_id"]: t for t in j["test_results"]}
    bad, maxd, tot = 0, 0.0, 0.0
    for ci, t in sorted(tr.items()):
        P = [tuple(p) for p in t["positions"]]
        m = cost_eval(ci, P)
        d = abs(m.cost - t["cost"])
        if m.cost != t["cost"]:
            bad += 1; maxd = max(maxd, d)
        tot += W[ci] * m.cost
    print(f"{Path(f).name:34s} json_total={j['total_score']:.10f} "
          f"recomputed={tot/TOTW:.10f} mismatch={bad}/100 maxabs={maxd:.3e}",
          flush=True)
