"""L301 final: the deepen-on-the-71 arms, priced from per-WORK-UNIT min-of-N.

The arms are compositions of six distinct work units, and which unit an arm runs
on which band is proven bit-for-bit (L296 G1, re-verified here):

    k1_71  ship, gate0            k1_29  gate0, mix, mix3, mix4
    k2_71  lp2, mix, both         k2_29  both
    k3_71  mix3                   k4_71  mix4

So every unit has several INDEPENDENT observations, and an arm's LP cost is the
sum of its units. Two things follow that no single run gives:

  * the same-work identity is a free control on the clock -- and it fired:
    gate0 and ship disagreed 25.3 % on k1_71 while the three k2_71 observations
    agreed to 2.8 %;
  * min-of-N is applied PER UNIT rather than per arm, so an arm never inherits
    one bad run's inflation (CLAUDE.md: "量時間要用 min-of-N").
"""
import json
import math
import re
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402

BANDS = [(21, 50, 4.16, 4.67), (51, 80, 2.66, 3.08),
         (81, 100, 2.31, 2.73), (101, 120, 1.62, 2.13)]
fb = lambda n, hi: next((b if hi else a) for lo, up, a, b in BANDS if lo <= n <= up)
RX = re.compile(r"\[lptime\] n=(\d+) cpu=([\d.]+) wall=([\d.]+)")
lp = lambda t: {int(n): float(w) for n, _c, w in
                RX.findall((DIR / ("l301b_%s.log" % t)).read_text(errors="replace"))}

t = (DIR / "optimizer_constructive.py").read_text(errors="replace")
i = t.index("_L196_LPGATE = {")
G = eval(t[i + len("_L196_LPGATE = "):t.index("}", i) + 1])
OFF = {n for n, v in G.items() if not v}

RUNS = ["ship", "ship_r2", "ship_r3", "lp2", "lp2_r2", "gate0", "gate0_r2",
        "mix", "mix3", "mix4", "both"]
T = {r: lp(r) for r in RUNS}
# which run observes which unit
OBS = {"k1_71": [("ship", 0), ("ship_r2", 0), ("ship_r3", 0), ("gate0", 0),
                 ("gate0_r2", 0)],
       "k1_29": [("gate0", 1), ("gate0_r2", 1), ("mix", 1), ("mix3", 1),
                 ("mix4", 1)],
       "k2_71": [("lp2", 0), ("lp2_r2", 0), ("mix", 0), ("both", 0)],
       "k2_29": [("both", 1)],
       "k3_71": [("mix3", 0)],
       "k4_71": [("mix4", 0)]}

print("== G0 every observation of every work unit ==")
U = {}
for u, obs in OBS.items():
    tot = {}
    for r, off in obs:
        tot[r] = sum(v for n, v in T[r].items() if (n in OFF) == bool(off))
    lo, hi = min(tot.values()), max(tot.values())
    print("   %-6s %s" % (u, "  ".join("%s %.2fs" % (r, v) for r, v in tot.items())))
    print("          spread %.1f %%   -> min-of-%d = %.2f s"
          % (100 * (hi - lo) / lo, len(tot), lo))
    best = min(tot, key=tot.get)
    U[u] = {n: v for n, v in T[best].items()
            if (n in OFF) == bool(dict(obs)[best])}

# per-case min across every run that observes the unit
for u, obs in OBS.items():
    m = {}
    for r, off in obs:
        for n, v in T[r].items():
            if (n in OFF) == bool(off):
                m[n] = min(m.get(n, 1e9), v)
    U[u] = m
print("   (the table below uses the PER-CASE min across those runs)")

COMPOSE = {"lp2": ["k2_71"], "gate0": ["k1_71", "k1_29"],
           "mix": ["k2_71", "k1_29"], "mix3": ["k3_71", "k1_29"],
           "mix4": ["k4_71", "k1_29"], "both": ["k2_71", "k2_29"]}
SHIP = U["k1_71"]

rows = [dict(x, t=x["t"] * 0.8679) for x in P.load()]
b0 = P.total(rows)
GB, TH = 52.0712 * 0.8679, 64.1
W = lambda n: math.exp(n / 12.0)
cs = lambda f: {r["test_id"]: r for r in json.load(open(DIR / f))["test_results"]}
sh = cs("l301b_ship.json")
ids = sorted(sh)
sw = sum(W(sh[i]["block_count"]) for i in ids)
ts = sum(W(sh[i]["block_count"]) * sh[i]["cost"] for i in ids) / sw

print("\n== G1 priced from per-unit min-of-N, per-band f ==")
print("   %-6s %13s %9s %8s %7s | %9s %8s | %9s %8s"
      % ("arm", "total", "quality", "dt s", "on 29", "NET@f_lo", "grader",
         "NET@f_hi", "grader"))
print("   %-6s %13.9f %+8.4f%% %8.2f %7.2f | %+8.4f%% %7.1fs | %+8.4f%% %7.1fs"
      % ("ship", json.load(open(DIR / "l301b_ship.json"))["total_score"],
         0, 0, 0, 0, GB, 0, GB))
for a in ("lp2", "gate0", "mix", "mix3", "both", "mix4"):
    w = {}
    for u in COMPOSE[a]:
        w.update(U[u])
    dt = {n: w.get(n, 0.0) - SHIP.get(n, 0.0) for n in set(w) | set(SHIP)}
    C = cs("l301b_%s.json" % a)
    q = 100 * (ts - sum(W(sh[i]["block_count"]) * C[i]["cost"] for i in ids) / sw) / ts
    o = []
    for hi in (False, True):
        of = (lambda hi: lambda r: dt.get(r["n"], 0.0) / fb(r["n"], hi))(hi)
        o += [q + 100 * (b0 - P.total(rows, of)) / b0,
              GB + sum(max(0.0, of(r)) for r in rows)]
    print("   %-6s %13.9f %+8.4f%% %8.2f %7.2f | %+8.4f%% %7.1fs | %+8.4f%% %7.1fs%s"
          % (a, json.load(open(DIR / ("l301b_%s.json" % a)))["total_score"], q,
             sum(dt.values()), sum(v for n, v in dt.items() if n in OFF),
             o[0], o[1], o[2], o[3], "" if o[1] < TH else "  OVER"))

print("\n== G2 the deepening steps on the 71 ==")
u = {k: sum(v.values()) for k, v in U.items()}
for a, b, lbl in (("k1_71", "k2_71", "k=1 -> k=2"), ("k2_71", "k3_71", "k=2 -> k=3"),
                  ("k3_71", "k4_71", "k=3 -> k=4")):
    print("   %-11s +%5.2f s of LP on the 71" % (lbl, u[b] - u[a]))
print("   the gate's own 1st pass on the 29: +%.2f s for +2.2282 pp = %.4f pp/s"
      % (u["k1_29"], 2.2282 / u["k1_29"]))
