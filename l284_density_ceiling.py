"""L284: how dense can THIS packer get at all, at any violation budget?

L283 restated the target as "raise utilisation while holding boundary and
grouping violations fixed", and priced our generator's exchange rate at 2.67 : 1
against.  But that framing quietly assumes the packer CAN raise utilisation --
that density is available and merely expensive.  Nobody has measured whether it
is available.

The same 4200 layouts answer it: 42 profiles x 100 in-set cases, already scored
in l283_cache.pkl, with positions in audit_cache_ship.pkl.  For each case take

    utilisation = sum(block area) / bbox area

over every feasible layout the pool ever produced, and compare

    selected      what the portfolio ships
    pool max      the densest layout this packer has EVER produced for that case
    label         the reference solution's own utilisation

Two very different worlds:
  (a) pool max is far below the label  -> density is not available at any
      violation budget; it is a packer CAPABILITY limit, and L283's exchange
      rate is a description of the edge of a small reachable set;
  (b) pool max is near the label       -> density IS available and the 2.67 : 1
      is a genuine price, so the target L283 stated is the right one.

Also reports, per case, the cost of the densest layout, so the question "is the
densest layout anywhere near competitive?" is answered directly rather than
inferred.
"""
import json
import math
import pickle
import statistics
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import m53_l3_probe as m53                                        # noqa: E402

CASES = m53.CASES
db = pickle.load(open(_DIR / "l283_cache.pkl", "rb"))
raw = pickle.load(open(_DIR / "audit_cache_ship.pkl", "rb"))["data"]
aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}

cases = sorted({a for a, _b in db})
profs = sorted({b for _a, b in db})


def util(pos):
    sa = sum(p[2] * p[3] for p in pos)
    W = max(p[0] + p[2] for p in pos) - min(p[0] for p in pos)
    H = max(p[1] + p[3] for p in pos) - min(p[1] for p in pos)
    return sa / (W * H)


rows = []
for ci in cases:
    n = CASES[ci]["n"]
    lab = [tuple(float(v) for v in CASES[ci]["tp"][i][:4]) for i in range(n)]
    ul = util(lab)
    pool = []
    for pi in profs:
        e = db.get((ci, pi))
        if not e or not e[3]:
            continue
        pool.append((util([tuple(p) for p in raw[(ci, pi)][0]]), e[4], e[2],
                     e[0], e[1], pi))
    if not pool:
        continue
    sel = min(pool, key=lambda t: t[1])
    dens = max(pool, key=lambda t: t[0])
    rows.append((ci, n, sel[0], dens[0], ul, sel[1], dens[1], sel[2], dens[2]))

W = {r[0]: math.exp(r[1] / 12.0) for r in rows}
ws = sum(W[r[0]] for r in rows)


def wm(f):
    return sum(W[r[0]] * f(r) for r in rows) / ws


print(f"== {len(rows)} in-set cases, {len(profs)} profiles each ==\n")
print(f"  utilisation, weighted exp(n/12)")
print(f"    selected (what we ship) : {100 * wm(lambda r: r[2]):.2f} %")
print(f"    pool max (densest this packer ever made) : "
      f"{100 * wm(lambda r: r[3]):.2f} %")
print(f"    label                   : {100 * wm(lambda r: r[4]):.2f} %")
gap_sel = wm(lambda r: r[4] - r[2])
gap_pool = wm(lambda r: r[4] - r[3])
print(f"\n    gap to label from the SHIPPED layout : {100 * gap_sel:+.2f} pp")
print(f"    gap to label from the POOL MAX       : {100 * gap_pool:+.2f} pp")
print(f"    -> the pool closes {100 * (1 - gap_pool / gap_sel):.1f} % of the "
      f"density gap to the label")

d = sorted(100 * (r[3] - r[2]) for r in rows)
print(f"\n  headroom the pool already holds over the shipped layout (pp):")
print(f"    p25 {d[len(d) // 4]:.2f}   p50 {d[len(d) // 2]:.2f}   "
      f"p75 {d[3 * len(d) // 4]:.2f}   max {d[-1]:.2f}")
nb = sum(1 for r in rows if r[3] - r[2] > 1e-9)
print(f"    cases where the pool has a denser layout than the one shipped: "
      f"{nb}/{len(rows)}")

print(f"\n  what the DENSEST layout costs (vs the one we ship):")
print(f"    cost   selected {wm(lambda r: r[5]):.6f}   densest "
      f"{wm(lambda r: r[6]):.6f}   "
      f"({100 * (wm(lambda r: r[6]) / wm(lambda r: r[5]) - 1):+.2f} %)")
print(f"    vrel   selected {wm(lambda r: r[7]):.5f}   densest "
      f"{wm(lambda r: r[8]):.5f}")
better = sum(1 for r in rows if r[6] < r[5] - 1e-12)
print(f"    cases where the densest layout is also the cheapest: "
      f"{better}/{len(rows)}")

# is utilisation even correlated with cost inside the pool?
cors = []
for ci in cases:
    pool = [(util([tuple(p) for p in raw[(ci, pi)][0]]), db[(ci, pi)][4])
            for pi in profs
            if (ci, pi) in db and db[(ci, pi)][3]]
    if len(pool) < 5:
        continue
    xs = [p[0] for p in pool]
    ys = [p[1] for p in pool]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        continue
    cors.append(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy))
cors.sort()
print(f"\n  within-case correlation between utilisation and COST "
      f"({len(cors)} cases):")
print(f"    p25 {cors[len(cors) // 4]:+.3f}   p50 {cors[len(cors) // 2]:+.3f}"
      f"   p75 {cors[3 * len(cors) // 4]:+.3f}")
print(f"    negative = denser is cheaper.  "
      f"{sum(1 for c in cors if c < 0)}/{len(cors)} cases negative")
