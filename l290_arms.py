"""L290: the same seven arms on BOTH corpora, plus LP k=2 priced from a dt
measured in this session.

WHY.  L287's transfer figure (46.6 %) compared two things that are not
comparable: the in-set -5.34 % is against REAL M73, while the OOS `m73` arm is
"M73-like" -- it still carries L131/L136's correctness fixes and M74's constant
regen, because those are code, not flags.  Running the identical arm set on both
corpora removes that mismatch entirely.

It also replaces the L276-era dt for LP k=2 with one measured back to back in
this session, which is the last soft input in the corrected RF pricing.
"""
import json
import math
import pickle
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402

INSET_FILES = {
    "ship": "l285_lp_on.json",
    "noLP": "l285_lp_off.json",
    "m73": "l285_betacfg.json",
    "lp2": "l290_inset_lp2.json",
    "noHint": "l290_inset_noHint.json",
    "noM80": "l290_inset_noM80.json",
    "refOld": "l290_inset_refOld.json",
}


def rd(f):
    j = json.load(open(DIR / f))
    tr = j["test_results"]
    return (j["total_score"],
            sum(t.get("runtime_seconds", 0.0) for t in tr),
            sum(1 for t in tr if t["is_feasible"]))


INS = {a: rd(f) for a, f in INSET_FILES.items()}

db = pickle.load(open(DIR / "l287_cache.pkl", "rb"))
rows = []
for k, v in db.items():
    if k[2] != "ship" or "err" in v or not v["feas"]:
        continue
    e = {a: db.get((k[0], k[1], a)) for a in INSET_FILES}
    if any(x is None or "err" in x or not x["feas"] for x in e.values()):
        continue
    rows.append((v["n"], e))


def W(n):
    return math.exp(n / 12.0)


ws = sum(W(n) for n, _e in rows)
OOS = {a: sum(W(n) * e[a]["cost"] for n, e in rows) / ws for a in INSET_FILES}

si, so = INS["ship"][0], OOS["ship"]
print("== the SAME arms on BOTH corpora  (in-set 100 official eval; "
      "OOS s1 %d cases) ==" % len(rows))
print("   %-8s %11s %12s   %11s %12s %9s"
      % ("arm", "in-set", "ship vs arm", "OOS", "ship vs arm", "transfer"))
for a in INSET_FILES:
    di = 100.0 * (si / INS[a][0] - 1.0)
    do = 100.0 * (so / OOS[a] - 1.0)
    tr = "%.0f %%" % (100.0 * do / di) if abs(di) > 1e-6 else "--"
    print("   %-8s %11.6f %11.4f%%   %11.6f %11.4f%% %9s"
          % (a, INS[a][0], di, OOS[a], do, tr))
print("   feasible in-set: %s"
      % {a: INS[a][2] for a in INSET_FILES if INS[a][2] != 100})

print("\n== LP k=2, priced from a dt measured THIS session ==")
dt_old = P.dt_by_n("results_L274_base_48c.json", "results_L276_k2.json")
dt_new = P.dt_by_n(INSET_FILES["ship"], INSET_FILES["lp2"])
q_new = P.quality_pct(INSET_FILES["ship"], INSET_FILES["lp2"])
print("   local runtime  ship %.2fs -> lp2 %.2fs  (+%.2fs)"
      % (INS["ship"][1], INS["lp2"][1], INS["lp2"][1] - INS["ship"][1]))
for lbl, dt in (("L276-era dt", dt_old), ("this session's dt", dt_new)):
    v = sorted(dt.values())
    print("   %-18s sum %+7.2fs  p50 %+.4fs  max %+.4fs"
          % (lbl, sum(v), v[len(v) // 2], v[-1]))
print("   in-set quality delta, this session: %+.4f %%" % q_new)

print("\n   %-24s %10s %10s" % ("dt divided by", "RF", "NET"))
fb = statistics.mean(dt_new.values())
for d in (1.00, 1.50, 1.91, 2.30, 2.87):
    rws = [dict(r, t=r["t"] * 0.8679) for r in P.load()]
    b = P.total(rws)
    sl = P.total(rws, lambda r: dt_new.get(r["n"], fb) / d)
    rf = 100.0 * (b - sl) / b
    print("   %-24.2f %+9.4f%% %+9.4f%%   %s"
          % (d, rf, q_new + rf, "GREEN" if q_new + rf > 0 else "RED"))
print("\n   defensible bracket d in [1.91, 2.87]; d=1.00 is the physically")
print("   wrong assumption that this box runs as fast as the grader.")
