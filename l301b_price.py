"""L301b: the deepen-on-the-71 arms, priced from the LP's OWN clock.

WHY NOT WALL DIFFERENCING.  The L301 block's CONTROL failed: `mix` does
bit-identical work in both blocks (total 1.195229398 in each, every same-work
gate passing) yet its dt read 35.36 s there against 22.34 s in the L298 block,
and `mix3` read CHEAPER than `mix` while doing strictly more passes. The box was
loaded. `ICCAD_LP_TIMING=1` measures the LP inside the process instead --
L159 built it for exactly this.
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
ARMS = ["ship", "gate0lp", "mix", "mix3", "mix4"]
LOG = {"ship": "l301b_ship.log", "mix": "l301b_mix.log",
       "mix3": "l301b_mix3.log", "mix4": "l301b_mix4.log"}
RES = {"ship": "l301b_ship.json", "mix": "l301b_mix.json",
       "mix3": "l301b_mix3.json", "mix4": "l301b_mix4.json"}
RX = re.compile(r"\[lptime\] n=(\d+) cpu=([\d.]+) wall=([\d.]+)")


def lp(tag):
    """{n: wall seconds the LP spent on that case}, plus the cpu/wall ratio."""
    txt = Path(DIR / LOG[tag]).read_text(errors="replace")
    w, c = {}, {}
    for n, cpu, wall in RX.findall(txt):
        w[int(n)] = float(wall)
        c[int(n)] = float(cpu)
    return w, c


L = {t: lp(t) for t in LOG}
print("== G0 the instrument is sound ==")
for t in LOG:
    w, c = L[t]
    tot = sum(w.values())
    print("   %-5s %3d cases in the LP   LP wall %7.2f s   cpu/wall %.3f"
          % (t, len(w), tot, sum(c.values()) / tot))
print("   cpu/wall ~ 1 => the LP is single-threaded and nothing else stole it")

print("\n== G1 the LP's own cost, and the added seconds vs shipped ==")
base = L["ship"][0]
print("   %-5s %10s %10s %10s %10s"
      % ("arm", "LP wall", "dt vs ship", "dt on the 29", "dt on the 71"))
t = (DIR / "optimizer_constructive.py").read_text(errors="replace")
i = t.index("_L196_LPGATE = {")
G = eval(t[i + len("_L196_LPGATE = "):t.index("}", i) + 1])
OFF = {n for n, v in G.items() if not v}
DT = {}
for a in ("mix", "mix3", "mix4"):
    w = L[a][0]
    dt = {n: w[n] - base.get(n, 0.0) for n in w}
    DT[a] = dt
    print("   %-5s %9.2fs %9.2fs %11.2fs %11.2fs"
          % (a, sum(w.values()), sum(dt.values()),
             sum(v for n, v in dt.items() if n in OFF),
             sum(v for n, v in dt.items() if n not in OFF)))

print("\n== G2 priced, per-band f ==")
rows = [dict(x, t=x["t"] * 0.8679) for x in P.load()]
b0 = P.total(rows)
GB, TH = 52.0712 * 0.8679, 64.1
W = lambda n: math.exp(n / 12.0)
cs = lambda f: {r["test_id"]: r for r in json.load(open(DIR / f))["test_results"]}
sh = cs(RES["ship"])
ids = sorted(sh)
sw = sum(W(sh[i]["block_count"]) for i in ids)
ts = sum(W(sh[i]["block_count"]) * sh[i]["cost"] for i in ids) / sw
print("   %-5s %13s %9s %8s | %9s %8s | %9s %8s"
      % ("arm", "total", "quality", "dt s", "NET@f_lo", "grader", "NET@f_hi", "grader"))
Q = {}
for a in ("mix", "mix3", "mix4"):
    C = cs(RES[a])
    q = 100 * (ts - sum(W(sh[i]["block_count"]) * C[i]["cost"] for i in ids) / sw) / ts
    Q[a] = q
    o = []
    for hi in (False, True):
        of = (lambda hi: lambda r: DT[a].get(r["n"], 0.0) / fb(r["n"], hi))(hi)
        o += [q + 100 * (b0 - P.total(rows, of)) / b0,
              GB + sum(max(0.0, of(r)) for r in rows)]
    print("   %-5s %13.9f %+8.4f%% %8.2f | %+8.4f%% %7.1fs | %+8.4f%% %7.1fs%s"
          % (a, json.load(open(DIR / RES[a]))["total_score"], q,
             sum(DT[a].values()), o[0], o[1], o[2], o[3],
             "" if o[1] < TH else "  OVER"))

print("\n== G3 the marginal step, on the 71 ==")
prev = ("mix", Q["mix"], sum(DT["mix"].values()))
for a in ("mix3", "mix4"):
    d = sum(DT[a].values())
    print("   %-4s -> %-4s  %+7.4f pp for %+6.2f s = %+.4f pp/s"
          % (prev[0], a, Q[a] - prev[1], d - prev[2],
             (Q[a] - prev[1]) / (d - prev[2])))
    prev = (a, Q[a], d)
print("   reference: the gate's own 1st pass on the 29 bought +0.1693 pp/s")
