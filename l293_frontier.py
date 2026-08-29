"""L293: the shape-LP depth frontier on the CURRENT shipped code, priced with
the project's own machine factor.

Two things changed since the depth axis was last judged:

  * `l276_price.py` and `l146_rf_price.py` add LOCAL dt seconds to GRADER
    seconds with no machine factor, while `l172_depthmap.py` has carried
    `F = 3.17  # dev-box LP second -> grader second (L161)` since L161.
    Restoring it cuts LP k=2's RF bill from -0.4816 % to -0.0146 % (33x).
  * L285 measured the shipped package at 0.8679x the beta runtime vector, so
    98/100 cases sit on the RF floor where the RF derivative is exactly zero.

Together those mean depth is nearly free, so the frontier has to be re-walked
rather than assumed to stop at 1.  Quality is the official evaluator; RF is
priced on the beta hidden population with the 2026-08-23 medians.
"""
import json
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402

BASE = "l285_lp_on.json"
ARMS = [("k=1 (shipped)", BASE),
        ("k=2", "l290_inset_lp2.json"),
        ("k=4", "l293_k4.json"),
        ("k=8", "l293_k8.json"),
        ("k=12", "l293_k12.json")]
F = 3.17                       # l172_depthmap.py:39, measured at L161
SCALE = 0.8679                 # shipped / beta runtime vector, measured at L285
GRADER_BETA = 52.0712
THRESH_S = 64.1                # runtime at which rank 2 is lost (L285)


def rd(f):
    p = DIR / f
    if not p.exists():
        return None
    j = json.load(open(p))
    tr = j["test_results"]
    return (j["total_score"],
            sum(t.get("runtime_seconds", 0.0) for t in tr),
            sum(1 for t in tr if t["is_feasible"]))


base = rd(BASE)
print("== shape-LP depth frontier, in-set 100, official evaluator ==")
print("   base = shipped default %.9f, local %.2f s" % (base[0], base[1]))
print()
print("   %-14s %12s %9s %9s %9s %9s %8s %10s"
      % ("arm", "total", "quality", "RF", "NET", "local dt", "feas",
         "grader s"))
for lbl, f in ARMS:
    r = rd(f)
    if r is None:
        print("   %-14s (not measured yet)" % lbl)
        continue
    if f == BASE:
        print("   %-14s %12.9f %8.4f%% %8.4f%% %8.4f%% %8.2fs %8d %9.1fs"
              % (lbl, r[0], 0.0, 0.0, 0.0, 0.0, r[2], GRADER_BETA * SCALE))
        continue
    dt = P.dt_by_n(BASE, f)
    fb = statistics.mean(dt.values())
    q = P.quality_pct(BASE, f)
    rows = [dict(x, t=x["t"] * SCALE) for x in P.load()]
    b = P.total(rows)
    sl = P.total(rows, lambda x: dt.get(x["n"], fb) / F)
    rf = 100.0 * (b - sl) / b
    dsum = sum(dt.values())
    gs = GRADER_BETA * SCALE + dsum / F
    flag = "" if gs < THRESH_S else "   <-- OVER the rank-2 runtime threshold"
    print("   %-14s %12.9f %8.4f%% %8.4f%% %8.4f%% %8.2fs %8d %9.1fs%s"
          % (lbl, r[0], q, rf, q + rf, dsum, r[2], gs, flag))

print()
print("   f = %.2f (l172_depthmap, measured L161); baseline scaled by %.4f"
      % (F, SCALE))
print("   rank-2 runtime threshold %.1f s; shipped sits at %.1f s"
      % (THRESH_S, GRADER_BETA * SCALE))
