"""L281: price the obvious escape -- "then let the bounding box grow".

Every oversized relocation would fit if the bbox rows were relaxed enough.  The
census recorded exactly how much: the chain excess on each axis.  `bbox_relax`
multiplies BOTH rows, so the area grows by relax^2, and area is priced by the
official cost at the same 0.5 weight as wire:

    cost = (1 + 0.5*(hgap + agap)) * exp(2*vrel),  agap = A/a_base - 1
    growing the box by factor R (area) adds  0.5 * (R-1) * A/a_base
    and  A/a_base = 1 + agap  is read straight off the anchor json.

Compare that against the entire first-order wire prize (l281_prize.py: +0.7994 %
for the best single unit per case).  Read-only; safe to run beside a scan.
"""
import json
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
DB = pickle.load(open(_DIR / "l281_cache.pkl", "rb"))["db"]

need = []
for k, v in DB.items():
    if k[0] != "census":
        continue
    ci = k[1]
    r = v["relo"]
    e = ANCH[ci]
    W0 = H0 = None
    for a, b in zip(r["exH"], r["exV"]):
        # cert stored exH = lH - W0 and exV = lV - H0; recover the ratios via
        # the stored W0/H0 is not kept, so use the per-candidate ratio form:
        need.append((ci, a, b))

# the census kept absolute excesses; recover the row lengths from the anchor
import math                                                        # noqa: E402
rows = []
for k, v in DB.items():
    if k[0] != "census":
        continue
    ci = k[1]
    pos = [tuple(p) for p in ANCH[ci]["positions"]]
    W0 = max(p[0] + p[2] for p in pos) - min(p[0] for p in pos)
    H0 = max(p[1] + p[3] for p in pos) - min(p[1] for p in pos)
    aob = 1.0 + ANCH[ci]["area_gap"]
    brk = 1.0 + 0.5 * (ANCH[ci]["hpwl_gap"] + ANCH[ci]["area_gap"])
    for a, b in zip(v["relo"]["exH"], v["relo"]["exV"]):
        relax = max(1.0 + a / W0, 1.0 + b / H0, 1.0)
        R = relax * relax                                  # area factor
        cost_pct = 100.0 * (0.5 * (R - 1.0) * aob) / brk
        rows.append((ci, relax, R, cost_pct))

rows.sort(key=lambda t: t[3])
n = len(rows)


def q(f):
    return rows[min(int(f * n), n - 1)]


print(f"== {n} oversized relocation candidates: what it costs to make them fit ==")
print(f"   {'quantile':<10}{'row relax':>11}{'area factor':>13}"
      f"{'cost of that growth':>22}")
for f, lbl in ((0.0, "min"), (0.25, "p25"), (0.5, "p50"), (0.75, "p75"),
               (0.99, "p99")):
    ci, relax, R, c = q(f)
    print(f"   {lbl:<10}{relax:>11.4f}{R:>13.4f}{c:>21.3f} %")
print(f"\n   the ENTIRE first-order wire prize (best unit per case, "
      f"unconstrained) is +0.7994 %")
cheap = sum(1 for r in rows if r[3] < 0.7994)
print(f"   candidates whose bbox growth costs less than that whole prize: "
      f"{cheap}/{n} = {100.0 * cheap / n:.1f} %")
