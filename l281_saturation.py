"""L281 side-measurement: how much slack does the anchor's own topology cell
have?  For each case, the block-level horizontal / vertical constraint graphs
built from the anchor's max-gap disjuncts have a longest node-weighted chain;
that chain is a LOWER BOUND on the bbox row it lives in.  slack = 1 - chain/row.

If slack is ~0 the topology cell is critical-path saturated: no rearrangement
that lengthens either critical chain can be placed without growing the bbox,
which is exactly what `bbox_relax=1.0` forbids.  No LP is involved.
"""
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import l281_reloc_probe as L                                       # noqa: E402

CASES = L.CASES

anchor = sys.argv[1] if len(sys.argv) > 1 else str(
    _DIR / "results_L274_base_48c.json")
aj = json.loads(open(anchor, "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
print(f"[anchor] {Path(anchor).name} total={aj['total_score']:.10f}")

rows = []
for ci in sorted(ANCH):
    P = [tuple(p) for p in ANCH[ci]["positions"]]
    n = CASES[ci]["n"]
    _u, unit_of, ukey, box, _m = L.unit_geo(ci, P)
    _pin, bb = L.pinned_keys(ci, P, _u, unit_of)
    EH, EV = L.base_graph(ci, P, unit_of, ukey, None)
    okH, lH = L.longest_chain(n, EH, [P[i][2] for i in range(n)])
    okV, lV = L.longest_chain(n, EV, [P[i][3] for i in range(n)])
    W0, H0 = bb[1] - bb[0], bb[3] - bb[2]
    sH, sV = 1.0 - lH / W0, 1.0 - lV / H0
    rows.append((ci, n, okH and okV, sH, sV))
    print(f"case {ci:3d} n={n:3d} acyclic={int(okH and okV)} "
          f"H chain {lH:10.4f}/{W0:10.4f} slack {100 * sH:7.4f}%   "
          f"V chain {lV:10.4f}/{H0:10.4f} slack {100 * sV:7.4f}%", flush=True)

sH = sorted(r[3] for r in rows)
sV = sorted(r[4] for r in rows)
m = len(rows)


def q(a, f):
    return a[min(int(f * len(a)), len(a) - 1)]


print(f"\n== {m} cases ==")
print(f"  H slack  min {100 * sH[0]:.4f}%  p25 {100 * q(sH, .25):.4f}%  "
      f"p50 {100 * q(sH, .5):.4f}%  p75 {100 * q(sH, .75):.4f}%  "
      f"max {100 * sH[-1]:.4f}%")
print(f"  V slack  min {100 * sV[0]:.4f}%  p25 {100 * q(sV, .25):.4f}%  "
      f"p50 {100 * q(sV, .5):.4f}%  p75 {100 * q(sV, .75):.4f}%  "
      f"max {100 * sV[-1]:.4f}%")
for thr in (1e-9, 1e-4, 1e-3, 1e-2):
    nb = sum(1 for r in rows if min(r[3], r[4]) <= thr)
    print(f"  cases with min(H,V) slack <= {100 * thr:8.5f}% : {nb}/{m}")
print(f"  cases whose anchor topology is acyclic: "
      f"{sum(1 for r in rows if r[2])}/{m}")
