"""L302: mix's dt, pinned with a SELF-CONTAINED estimator.

k1_71 -- the noisiest work unit (34 % over 5 observations) and the one the user
asked to re-measure -- turns out never to enter any arm's dt at all. Every arm
runs pass 1 on the 71 exactly as `ship` does (bit-for-bit, L296 G1), so it
cancels. It only appeared because whole-LP walls were being differenced across
runs. Remove the differencing and it is gone:

    gate0 dt = LP wall on the 29                       (ship spends 0 there)
    mix   dt = LP wall on the 29 + pass 2+ on the 71   (pass 1 cancels)

Both terms come from ONE process in ONE run. The pass times exclude the
per-pass guard (`_proxy_metrics` + `hard_ok`), which is real marginal cost, so
the 71-side is reported as a BRACKET: pass time alone (lower) and pass time
scaled up by that case's wall/pass ratio (upper).
"""
import glob
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
RX = re.compile(r"\[lptime\] n=(\d+) cpu=([\d.]+) wall=([\d.]+) passes=([\d.,]*)")


def parse(path):
    d = {}
    for n, c, w, ps in RX.findall(Path(path).read_text(errors="replace")):
        d[int(n)] = (float(w), [float(x) for x in ps.split(",") if x])
    return d


t = (DIR / "optimizer_constructive.py").read_text(errors="replace")
i = t.index("_L196_LPGATE = {")
G = eval(t[i + len("_L196_LPGATE = "):t.index("}", i) + 1])
OFF = {n for n, v in G.items() if not v}

MIX = [parse(p) for p in sorted(glob.glob(str(DIR / "l302_mix_*.log")))] + \
      [parse(DIR / "l301b_mix.log")]
SHIP = [parse(p) for p in sorted(glob.glob(str(DIR / "l302_ship_*.log")))] + \
       [parse(DIR / ("l301b_%s.log" % s)) for s in ("ship", "ship_r2", "ship_r3")]

print("== G0 k1_71 cancels -- demonstrated, not assumed ==")
p1 = lambda d: sum(v[1][0] for n, v in d.items() if n not in OFF)
sp, mp = [p1(d) for d in SHIP], [p1(d) for d in MIX]
print("   ship  pass 1 on the 71 : " + " ".join("%.2f" % x for x in sp))
print("   mix   pass 1 on the 71 : " + " ".join("%.2f" % x for x in mp))
print("   ship  %.2f-%.2f s   mix %.2f-%.2f s   the ranges OVERLAP,"
      % (min(sp), max(sp), min(mp), max(mp)))
print("   and the work is bit-identical, so this term is not part of any dt.")
print("   spread within ship alone: %.1f %%; within mix alone: %.1f %%"
      % (100 * (max(sp) - min(sp)) / min(sp), 100 * (max(mp) - min(mp)) / min(mp)))

print("\n== G1 the two terms that ARE mix's dt, %d repeats ==" % len(MIX))
t29 = [sum(v[0] for n, v in d.items() if n in OFF) for d in MIX]
t71lo = [sum(sum(v[1][1:]) for n, v in d.items() if n not in OFF) for d in MIX]
t71hi = [sum(v[0] * (sum(v[1][1:]) / sum(v[1])) for n, v in d.items()
             if n not in OFF and sum(v[1]) > 0) for d in MIX]
for lbl, v in (("LP wall on the 29 ", t29), ("pass 2 on the 71  ", t71lo),
               ("  + its guard share", t71hi)):
    print("   %s : %s   min %.2f  spread %.1f %%"
          % (lbl, " ".join("%5.2f" % x for x in v), min(v),
             100 * (max(v) - min(v)) / min(v)))

print("\n== G2 mix's dt, per-case min over the repeats ==")


def permin(sel):
    m = {}
    for d in MIX:
        for n, v in d.items():
            x = sel(n, v)
            if x is not None:
                m[n] = min(m.get(n, 1e9), x)
    return m


dt_lo = permin(lambda n, v: v[0] if n in OFF else sum(v[1][1:]))
dt_hi = permin(lambda n, v: v[0] if n in OFF else
               v[0] * (sum(v[1][1:]) / sum(v[1])) if sum(v[1]) > 0 else 0.0)
print("   lower  %.2f s   upper  %.2f s   (the bracket is the per-pass guard)"
      % (sum(dt_lo.values()), sum(dt_hi.values())))
print("   of which on the 29: %.2f s (%.0f %%)"
      % (sum(v for n, v in dt_lo.items() if n in OFF),
         100 * sum(v for n, v in dt_lo.items() if n in OFF) / sum(dt_lo.values())))

rows = [dict(x, t=x["t"] * 0.8679) for x in P.load()]
b0 = P.total(rows)
GB = 52.0712 * 0.8679
cs = lambda f: {r["test_id"]: r for r in json.load(open(DIR / f))["test_results"]}
sh, mx = cs("l301b_ship.json"), cs("l302_mix_1.json")
W = lambda n: math.exp(n / 12.0)
ids = sorted(sh)
sw = sum(W(sh[i]["block_count"]) for i in ids)
ts = sum(W(sh[i]["block_count"]) * sh[i]["cost"] for i in ids) / sw
q = 100 * (ts - sum(W(sh[i]["block_count"]) * mx[i]["cost"] for i in ids) / sw) / ts
ql = q * (2.4056 / 2.5357)          # the Linux haircut measured in L300

print("\n== G3 mix, priced on the pinned dt ==")
print("   %-22s %9s %9s %9s %8s"
      % ("dt / quality", "dt s", "NET@f_lo", "NET@f_hi", "grader"))
for dlbl, dt in (("lower", dt_lo), ("upper", dt_hi)):
    for qlbl, qq in (("Windows", q), ("LINUX  ", ql)):
        o = []
        for hi in (False, True):
            of = (lambda hi: lambda r: dt.get(r["n"], 0.0) / fb(r["n"], hi))(hi)
            o += [qq + 100 * (b0 - P.total(rows, of)) / b0]
        gs = GB + sum(max(0.0, dt.get(r["n"], 0.0) / fb(r["n"], False)) for r in rows)
        print("   dt %-5s / q %-9s %9.2f %+8.4f%% %+8.4f%% %7.1fs"
              % (dlbl, qlbl, sum(dt.values()), o[0], o[1], gs))
print("\n   quality: Windows %+.4f %%, Linux %+.4f %% (L300 measured 95 %% transfer)"
      % (q, ql))
