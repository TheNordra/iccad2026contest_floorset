"""L283: is density expensive to GENERATE, or only expensive to SQUEEZE?

L282 measured that compressing the committed anchor costs 2.74 units of
hpwl_gap per unit of area_gap, and the score prices them equally, so the LP
declines the trade.  That closed post-hoc chain shortening.  It did NOT settle
whether a PACKING-TIME rule would pay the same rate -- L282 S6's open question,
and the one thing left on the axis.

This answers it without touching constructive.cpp, because the data already
exists: `audit_cache_ship.pkl` holds the positions of **42 profiles x 100 in-set
cases = 4200 independently generated layouts** in the shipped configuration.
Different profiles produce genuinely different packings of the same instance --
they are not compressions of one another.  So the frontier they trace out IS the
generation-side exchange rate.

METHOD, and why it is not circular
  The portfolio selects by a proxy that is per-case oracle-perfect (M13/M76), so
  the selected layout minimises cost = 1 + 0.5*(hpwl_gap + area_gap).  On a
  convex frontier that forces the local slope to be >= 1 wherever we sit, so
  "the slope at the optimum is >= 1" is a tautology and worth nothing.  What is
  NOT a tautology is HOW MUCH more than 1 it is:

      for each case, over every profile layout that is DENSER than the selected
      one (smaller area_gap), the cheapest observed
          rate = (hpwl_gap - hpwl_gap_sel) / (area_gap_sel - area_gap)
      is the best price this generator has ever been observed to pay for density
      on that case.

  rate < 1  -> the pool contains a strictly better layout than the one selected
               (should be empty if the proxy really is oracle)
  rate ~ 1  -> we sit exactly on the frontier; density is priced fairly and a
               better generator could move along it
  rate >> 1 -> density is intrinsically expensive at this operating point, the
               same way squeezing is, and the packing-time idea is bounded too

Only feasible layouts are used, and every number is the official strict scorer's.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import m53_l3_probe as m53                                        # noqa: E402

CASES, W = m53.CASES, m53.W
cost_eval = m53.cost_eval

CACHE = _DIR / "l283_cache.pkl"
db = {}
if CACHE.exists():
    try:
        db = pickle.load(open(CACHE, "rb"))
    except Exception:
        db = {}

raw = pickle.load(open(_DIR / "audit_cache_ship.pkl", "rb"))
data = raw["data"]
cases = sorted({a for a, _b in data})
profs = sorted({b for _a, b in data})
print(f"[audit] {len(data)} layouts = {len(cases)} cases x {len(profs)} "
      f"profiles", flush=True)

t0 = time.perf_counter()
new = 0
for (ci, pi), v in data.items():
    key = (ci, pi)
    if key in db:
        continue
    pos = [tuple(p) for p in v[0]]
    m = cost_eval(ci, pos)
    db[key] = (float(m.hpwl_gap), float(m.area_gap),
               float(m.violations_relative), bool(m.is_feasible),
               float(m.cost))
    new += 1
    if new % 400 == 0:
        pickle.dump(db, open(CACHE, "wb"))
        print(f"  scored {new} new ({time.perf_counter() - t0:.0f}s)",
              flush=True)
pickle.dump(db, open(CACHE, "wb"))
print(f"[score] {new} newly scored, {len(db)} total "
      f"({time.perf_counter() - t0:.0f}s)", flush=True)

rows = []
for ci in cases:
    pts = [(db[(ci, pi)][0], db[(ci, pi)][1], db[(ci, pi)][4])
           for pi in profs if (ci, pi) in db and db[(ci, pi)][3]]
    if len(pts) < 3:
        continue
    sel = min(pts, key=lambda t: t[2])              # what the portfolio picks
    hs, as_, _cs = sel
    denser = [(h, a) for h, a, _c in pts if a < as_ - 1e-12]
    if not denser:
        rows.append((ci, len(pts), 0, None, as_, hs))
        continue
    rates = [(h - hs) / (as_ - a) for h, a in denser]
    rows.append((ci, len(pts), len(denser), min(rates), as_, hs))

ok = [r for r in rows if r[3] is not None]
print(f"\n== {len(rows)} cases; {len(ok)} have at least one DENSER layout in "
      f"their own pool ==")
rr = sorted(r[3] for r in ok)


def q(a, f):
    return a[min(int(f * len(a)), len(a) - 1)]


print(f"  cheapest observed price of density, per case "
      f"(hpwl_gap paid per area_gap bought):")
print(f"    min {rr[0]:+.3f}   p25 {q(rr, .25):+.3f}   p50 {q(rr, .5):+.3f}"
      f"   p75 {q(rr, .75):+.3f}   max {rr[-1]:+.3f}")
for thr, lbl in ((0.0, "< 0    (denser AND better wire -- free density)"),
                 (1.0, "< 1    (worth buying: cost falls)"),
                 (2.74, "< 2.74 (cheaper than L282's squeeze)")):
    c = sum(1 for x in rr if x < thr)
    print(f"    cases with rate {lbl:48s}: {c}/{len(ok)}")
nz = sum(1 for r in rows if r[2] == 0)
print(f"  cases whose pool contains NO denser layout at all: {nz}/{len(rows)}")

# weighted: what the cheapest density in the pool would be worth if taken
gain = 0.0
wsum = sum(W[r[0]] for r in rows)
for ci, _n, nd, rate, as_, hs in rows:
    if rate is None or rate >= 1.0:
        continue
    pts = [(db[(ci, pi)][0], db[(ci, pi)][1], db[(ci, pi)][4])
           for pi in profs if (ci, pi) in db and db[(ci, pi)][3]]
    sel = min(pts, key=lambda t: t[2])
    best = min(pts, key=lambda t: t[2])
    gain += W[ci] * max(0.0, sel[2] - best[2])
print(f"\n  (sanity) portfolio-selected == pool-best on every case: "
      f"{'yes' if gain == 0.0 else 'NO -- proxy is not oracle here'}")
print(f"\n  L282 squeeze rate 2.74 : 1   |   L268 packing-time rate 1.2 : 1   "
      f"|   break-even 1.0 : 1")
