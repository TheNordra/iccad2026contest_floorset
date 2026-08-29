"""L298: the composition, priced on a dt measured in ONE contiguous block.

WHY IT WAS RE-RUN.  G6 proved `mix` does bit-identical work to `gate0` on the 29
gated-off block counts and to `lp2` on the 71 gated-on ones.  Identical work must
cost the same -- and against the three-hour-old l294 baseline it read 1.27 s and
1.29 s cheaper on the two halves.  Consistent on both halves is a BIAS, not
scatter.  `both` was worse: on the 71, where it is bit-equal to `lp2`, it read
+7.93 s for the same work, and its repeats differed by 12.5 s of wall.

So: ship, mix, both, ship, gate0, ship -- one block, the baseline interleaved
between the arms rather than sitting hours away from them.  Each arm's dt is
differenced against the mean of all three ship runs, and the SAME-WORK identity
above becomes the gate on whether the new dt is trustworthy.
"""
import json
import math
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402

F, SCALE, GB, TH = 3.17, 0.8679, 52.0712, 64.1
SHIPS = ["l298_ship.json", "l298_ship2.json", "l298_ship3.json",
         "l298_ship4.json"]
ARMS = ["lp2", "gate0", "mix", "both", "both4"]
FILE = {"lp2": "l290_inset_lp2.json", "gate0": "l298_gate0.json",
        "mix": "l298_mix.json", "both": "l298_both.json", "both4": "l298_both4.json"}
W = lambda n: math.exp(n / 12.0)


def cs(f):
    return {r["test_id"]: r for r in json.load(open(DIR / f))["test_results"]}


t = (DIR / "optimizer_constructive.py").read_text(errors="replace")
i = t.index("_L196_LPGATE = {")
G = eval(t[i + len("_L196_LPGATE = "):t.index("}", i) + 1])
OFF = {n for n, v in G.items() if not v}
C = {a: cs(f) for a, f in FILE.items()}
C["ship"] = cs(SHIPS[0])
ids = sorted(C["ship"])


def dtv(f):
    ds = [P.dt_by_n(s, f) for s in SHIPS]
    return {n: statistics.mean(d[n] for d in ds) for n in ds[0]}


D = {a: dtv(f) for a, f in FILE.items()}
half = lambda d, off: sum(v for n, v in d.items() if (n in OFF) == off)

print("== G0 wall stability of the block ==")
for f in SHIPS + [FILE[a] for a in ARMS]:
    j = json.load(open(DIR / f))
    print("   %-22s %7.2f s   total %.9f   feasible %d"
          % (f, sum(x.get("runtime_seconds", 0.0) for x in j["test_results"]),
             j["total_score"],
             sum(1 for x in j["test_results"] if x["is_feasible"])))
w = [sum(x.get("runtime_seconds", 0.0) for x in cs(s).values()) for s in SHIPS]
print("   ship spread across the block: %.2f s (%.1f %% of %.1f)"
      % (max(w) - min(w), 100 * (max(w) - min(w)) / statistics.mean(w),
         statistics.mean(w)))

print("\n== G1 the same-work identity, now as a GATE on dt ==")
print("   mix and gate0 do identical work on the 29; mix and lp2 on the 71.")
print("   %-28s %10s %10s %9s" % ("", "on the 29", "on the 71", "total"))
for a in ARMS:
    print("   %-28s %9.2f s %9.2f s %8.2f s"
          % (a, half(D[a], True), half(D[a], False), sum(D[a].values())))
print("   %-28s %+9.2f s %+9.2f s"
      % ("mix - gate0 / mix - lp2 (want 0)",
         half(D["mix"], True) - half(D["gate0"], True),
         half(D["mix"], False) - half(D["lp2"], False)))
print("   %-28s %+9.2f s"
      % ("both - lp2 on the 71 (want 0)",
         half(D["both"], False) - half(D["lp2"], False)))

print("\n== G2 the decomposition of TIME, from the block ==")
p1 = half(D["gate0"], True)
p2on = half(D["lp2"], False)
p2off = half(D["both"], True) - p1
print("   1st pass on the 29 (the gate)      %7.2f s" % p1)
print("   2nd pass on the 71 (depth)         %7.2f s" % p2on)
print("   2nd pass on the 29 (the CROSS term)%7.2f s" % p2off)

print("\n== G3 final, in-set 100, official evaluator ==")
rows = [dict(x, t=x["t"] * SCALE) for x in P.load()]
base = P.total(rows)
tl = {C["ship"][i]["block_count"]:
      statistics.mean(cs(s)[i]["runtime_seconds"] for s in SHIPS) for i in ids}
print("   %-6s %12s %9s %8s %10s %10s %8s %5s"
      % ("arm", "total", "quality", "dt s", "NET@3.17", "NET ratio", "grader",
         "feas"))
print("   %-6s %12.9f %8.4f%% %8.2f %+9.4f%% %+9.4f%% %7.1fs %4d"
      % ("ship", json.load(open(DIR / SHIPS[0]))["total_score"], 0, 0, 0, 0,
         GB * SCALE, 100))
out = {}
for a in ARMS:
    dt = D[a]
    q = statistics.mean(P.quality_pct(s, FILE[a]) for s in SHIPS)
    rf = 100.0 * (base - P.total(rows, lambda r: dt.get(r["n"], 0.0) / F)) / base
    fr = {n: dt[n] / tl[n] for n in dt}
    of = lambda r: r["t"] * fr.get(r["n"], 0.0)
    rr = 100.0 * (base - P.total(rows, of)) / base
    gs = GB * SCALE + sum(max(0.0, of(r)) for r in rows)
    j = json.load(open(DIR / FILE[a]))
    out[a] = (q, q + rf, q + rr)
    print("   %-6s %12.9f %8.4f%% %8.2f %+9.4f%% %+9.4f%% %7.1fs %4d%s"
          % (a, j["total_score"], q, sum(dt.values()), q + rf, q + rr, gs,
             sum(1 for x in j["test_results"] if x["is_feasible"]),
             "" if gs < TH else "  <-- OVER"))
    lo, hi = 0.05, 400.0
    for _ in range(80):
        m = (lo + hi) / 2
        r = 100.0 * (base - P.total(rows, lambda x: dt.get(x["n"], 0.0) / m)) / base
        lo, hi = (lo, m) if q + r > 0 else (m, hi)
    out[a] = out[a] + (hi,)
print("\n   break-even f :  " + "   ".join("%s %.2f" % (a, out[a][3]) for a in ARMS))
print("   (measured f: 2.71 WSL->grader, 3.17 Windows->grader; the same-box")
print("    ratio column imports neither)")
