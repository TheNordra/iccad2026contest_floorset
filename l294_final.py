"""L294 final: ICCAD_LP_GATE=0 on the in-set 100, priced two independent ways.

Four runs, a 2x2: {ship, ship_r2} x {gate0, gate0_r2}.  dt is the mean over all
four pairings, so neither side rests on a single wall measurement -- the wall
noise floor here is 1.3 % (ship) and 2.7 % (gate0).

PRICED TWICE, because the verdict now depends on the machine factor in a way
LP k=2's did not:
  (a) imported f, as `l293_frontier.py` does -- f = 3.17 (l172_depthmap.py:39,
      L157 5h: 2.71 WSL-to-grader x 1.17 Windows-to-WSL for the LP);
  (b) SAME-BOX RATIOS, where f cancels: express the LP's added time as a
      fraction of that case's local wall on THIS box, and apply the fraction to
      the grader's own measured per-case time.  The only external input is the
      grader's runtime vector, which we own.  This avoids L173 §4's withdrawn
      cross-box mixing entirely, and it reproduces LP k=2's published NET to
      0.014 pp, which is the control that says the method is sound.
"""
import json
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402

F, SCALE, GRADER_BETA, THRESH_S = 3.17, 0.8679, 52.0712, 64.1
SHIPS = ["l294_ship.json", "l294_ship_r2.json"]
G0S = ["l294_gate0.json", "l294_gate0_r2.json"]


def cases(f):
    return {r["test_id"]: r for r in json.load(open(DIR / f))["test_results"]}


# dt over all four pairings
ds = [P.dt_by_n(s, g) for s in SHIPS for g in G0S]
dt = {n: statistics.mean(d[n] for d in ds) for n in ds[0]}
q = statistics.mean(P.quality_pct(s, g) for s in SHIPS for g in G0S)

rows = [dict(x, t=x["t"] * SCALE) for x in P.load()]
base = P.total(rows)


def nfloor(dt_of):
    return sum(1 for r in rows
               if max(0.7, ((r["t"] + max(0.0, dt_of(r))) / r["med"]) ** 0.3)
               <= 0.7 + 1e-12)


print("== dt, all four pairings ==")
for lbl, d in zip(["ship/g0", "ship/g0_r2", "ship_r2/g0", "ship_r2/g0_r2"], ds):
    print("   %-14s sum %+7.2f s" % (lbl, sum(d.values())))
print("   %-14s sum %+7.2f s   p50 %+.3f s   max %+.3f s"
      % ("MEAN", sum(dt.values()),
         statistics.median(dt.values()), max(dt.values())))
n1 = P.dt_by_n(*SHIPS)
n2 = P.dt_by_n(*G0S)
print("   noise floor: ship pair %+.2f s, gate0 pair %+.2f s"
      % (sum(n1.values()), sum(n2.values())))

print("\n== (a) imported f ==")
print("   %-8s %10s %10s %10s %7s" % ("f", "RF", "NET", "grader s", "floor"))
for f in (1.00, 1.91, 2.71, 3.17):
    of = lambda r: dt.get(r["n"], 0.0) / f
    rf = 100.0 * (base - P.total(rows, of)) / base
    print("   %-8.2f %+9.4f%% %+9.4f%% %9.1fs %6d"
          % (f, rf, q + rf, GRADER_BETA * SCALE + sum(dt.values()) / f, nfloor(of)))
lo, hi = 0.05, 200.0
for _ in range(80):
    m = (lo + hi) / 2
    r = 100.0 * (base - P.total(rows, lambda x: dt.get(x["n"], 0.0) / m)) / base
    lo, hi = (lo, m) if q + r > 0 else (m, hi)
print("   break-even f = %.3f" % hi)

print("\n== (b) same-box ratios, f cancels ==")
tl = {}
for i in cases(SHIPS[0]):
    n = cases(SHIPS[0])[i]["block_count"]
    tl[n] = statistics.mean(cases(s)[i]["runtime_seconds"] for s in SHIPS)
frac = {n: dt[n] / tl[n] for n in dt}
of = lambda r: r["t"] * frac.get(r["n"], 0.0)
rf = 100.0 * (base - P.total(rows, of)) / base
gadd = sum(max(0.0, of(r)) for r in rows)
print("   LP share of local wall: p50 %.1f%%  max %.1f%%"
      % (100 * statistics.median(frac.values()), 100 * max(frac.values())))
print("   local +%.2f s -> grader +%.2f s   (implied f = %.2f)"
      % (sum(dt.values()), gadd, sum(dt.values()) / gadd))
print("   quality %+.4f%%   RF %+.4f%%   NET %+.4f%%   grader %.1f s   floor %d"
      % (q, rf, q + rf, GRADER_BETA * SCALE + gadd, nfloor(of)))
print("\n   headline: quality %+.4f%%, feasible 100/100, NET %+.4f%% .. %+.4f%%"
      % (q, q + rf,
         q + 100.0 * (base - P.total(rows, lambda r: dt.get(r["n"], 0.0) / F)) / base))
