"""L302 final: mix's dt pinned, every arm on the same footing.

TWO CORRECTIONS ROLLED IN.

1. `k1_71` -- the noisiest unit (34 % over 5 observations), and the one I was
   asked to re-measure -- **cancels**. Every arm runs pass 1 on the 71 exactly
   as `ship` does (bit-for-bit, L296 G1), so it never enters any dt. It only
   appeared because whole-LP walls were being differenced across runs. Measured
   anyway, 5 more times, to demonstrate rather than assume the cancellation.

2. `min-of-N` is biased DOWNWARD with N, so an arm with 5 repeats cannot be
   compared against one with 1. The fix is the same as for quality: pool the
   observations by WORK UNIT, because which arm runs which unit is proven, and
   several arms observe the same unit.

       k1_29   gate0 x2, mix x5, mix3, mix4      9 observations
       p2_71   lp2 x2,   mix x5, both            8 observations
       k2_29   both                              1
       p23_71  mix3                              1
       p234_71 mix4                              1

   The three singletons are all far from the verdict boundary, so their thinner
   sampling changes no conclusion.
"""
import glob
import json
import math
import re
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402

BANDS = [(21, 50, 4.16, 4.67), (51, 80, 2.66, 3.08),
         (81, 100, 2.31, 2.73), (101, 120, 1.62, 2.13)]
fb = lambda n, hi: next((b if hi else a) for lo, up, a, b in BANDS if lo <= n <= up)
RX = re.compile(r"\[lptime\] n=(\d+) cpu=([\d.]+) wall=([\d.]+) passes=([\d.,]*)")


def parse(p):
    d = {}
    for n, c, w, ps in RX.findall(Path(p).read_text(errors="replace")):
        d[int(n)] = (float(w), [float(x) for x in ps.split(",") if x])
    return d


t = (DIR / "optimizer_constructive.py").read_text(errors="replace")
i = t.index("_L196_LPGATE = {")
G = eval(t[i + len("_L196_LPGATE = "):t.index("}", i) + 1])
OFF = {n for n, v in G.items() if not v}
LOGS = {"lp2": ["l301b_lp2.log", "l301b_lp2_r2.log"],
        "gate0": ["l301b_gate0.log", "l301b_gate0_r2.log"],
        "mix": sorted(glob.glob(str(DIR / "l302_mix_*.log"))) + ["l301b_mix.log"],
        "mix3": ["l301b_mix3.log"], "mix4": ["l301b_mix4.log"],
        "both": ["l301b_both.log"]}
R = {a: [parse(DIR / Path(f).name) for f in v] for a, v in LOGS.items()}

# marginal cost of the passes beyond the first, with that case's guard share
tail = lambda w, ps: w * (sum(ps[1:]) / sum(ps)) if sum(ps) > 0 else 0.0
UNIT = {"k1_29": [(a, 1, "whole") for a in ("gate0", "mix", "mix3", "mix4")],
        "k2_29": [("both", 1, "whole")],
        "p2_71": [(a, 0, "tail") for a in ("lp2", "mix", "both")],
        "p23_71": [("mix3", 0, "tail")],
        "p234_71": [("mix4", 0, "tail")]}
U, NOBS = {}, {}
for u, src in UNIT.items():
    m, k = {}, 0
    for a, off, how in src:
        for d in R[a]:
            k += 1
            for n, (w, ps) in d.items():
                if (n in OFF) == bool(off):
                    x = w if how == "whole" else tail(w, ps)
                    m[n] = min(m.get(n, 1e9), x)
    U[u], NOBS[u] = m, k

print("== G0 k1_71 cancels: demonstrated ==")
p1 = lambda d: sum(v[1][0] for n, v in d.items() if n not in OFF)
sp = [p1(parse(DIR / Path(f).name)) for f in
      sorted(glob.glob(str(DIR / "l302_ship_*.log"))) +
      ["l301b_ship.log", "l301b_ship_r2.log", "l301b_ship_r3.log"]]
mp = [p1(d) for d in R["mix"]]
print("   ship pass1-on-71 (%d runs): %s" % (len(sp), " ".join("%.2f" % x for x in sp)))
print("   mix  pass1-on-71 (%d runs): %s" % (len(mp), " ".join("%.2f" % x for x in mp)))
print("   ranges %.2f-%.2f vs %.2f-%.2f -- OVERLAPPING, and the work is"
      % (min(sp), max(sp), min(mp), max(mp)))
print("   bit-identical, so this 26 %-noisy term is in NO arm's dt.")

print("\n== G1 the work units, pooled ==")
for u in UNIT:
    print("   %-8s %2d observations   %6.2f s" % (u, NOBS[u], sum(U[u].values())))

COMPOSE = {"lp2": ["p2_71"], "gate0": ["k1_29"], "mix": ["k1_29", "p2_71"],
           "mix3": ["k1_29", "p23_71"], "mix4": ["k1_29", "p234_71"],
           "both": ["k2_29", "p2_71"]}
rows = [dict(x, t=x["t"] * 0.8679) for x in P.load()]
b0 = P.total(rows)
GB, TH = 52.0712 * 0.8679, 64.1
W = lambda n: math.exp(n / 12.0)
cs = lambda f: {r["test_id"]: r for r in json.load(open(DIR / f))["test_results"]}
sh = cs("l301b_ship.json")
ids = sorted(sh)
sw = sum(W(sh[i]["block_count"]) for i in ids)
ts = sum(W(sh[i]["block_count"]) * sh[i]["cost"] for i in ids) / sw
HAIR = 2.4056 / 2.5357                        # L300's measured Linux transfer

print("\n== G2 FINAL, pooled units, per-band f, both platforms ==")
print("   %-6s %13s %8s %7s | %9s %9s | %9s %9s | %7s"
      % ("arm", "total", "dt s", "on 29", "NET win lo", "win hi",
         "NET LNX lo", "LNX hi", "grader"))
print("   %-6s %13.9f %8.2f %7.2f | %+9.4f%% %+8.4f%% | %+9.4f%% %+8.4f%% | %6.1fs"
      % ("ship", 1.226325126, 0, 0, 0, 0, 0, 0, GB))
for a in ("lp2", "gate0", "mix", "mix3", "both", "mix4"):
    dt = {}
    for u in COMPOSE[a]:
        dt.update(U[u])
    C = cs("l301b_%s.json" % a)
    q = 100 * (ts - sum(W(sh[i]["block_count"]) * C[i]["cost"] for i in ids) / sw) / ts
    o = []
    for qq in (q, q * HAIR):
        for hi in (False, True):
            of = (lambda hi: lambda r: dt.get(r["n"], 0.0) / fb(r["n"], hi))(hi)
            o.append(qq + 100 * (b0 - P.total(rows, of)) / b0)
    gs = GB + sum(max(0.0, dt.get(r["n"], 0.0) / fb(r["n"], False)) for r in rows)
    print("   %-6s %13.9f %8.2f %7.2f | %+9.4f%% %+8.4f%% | %+9.4f%% %+8.4f%% | %6.1fs%s"
          % (a, json.load(open(DIR / ("l301b_%s.json" % a)))["total_score"],
             sum(dt.values()), sum(v for n, v in dt.items() if n in OFF),
             o[0], o[1], o[2], o[3], gs, "  OVER" if gs >= TH else ""))
print("\n   LNX = the Windows quality scaled by L300's measured 95 %% Linux transfer")
