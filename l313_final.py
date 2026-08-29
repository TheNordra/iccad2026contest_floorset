"""L313 -- the candidates, priced with f MEASURED (L308) and applied only to the LP.

The chain, every link a same-session or same-phase ratio:

  grader_ship(n) = grader_beta(n) * poolratio(n)          + LP_ship(n)  / f(n)
  grader_arm(n)  = grader_ship(n)                         + dLP(n)      / f(n)

  poolratio(n) = l285_lp_off(n) / l285_betacfg(n)   -- both LP-free, same session,
                 so it is the pool phase's own ratio and f cancels inside it.
  LP_*(n)      = ICCAD_LP_TIMING, measured directly (cpu/wall 1.01-1.04 => the
                 LP is single-threaded, which is why a single-thread f is the
                 right constant for it).
  f(n)         = L308's per-band single-thread ratio.  ONLY the LP uses it, and
                 the LP is the only term that cannot be expressed as a ratio.

`gate0` and `gate0+k=2` change nothing before the LP, so their pool term is the
shipped one exactly -- the middle term of the chain vanishes for them.
"""
import json, math, pickle, statistics, sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402
import l299_project2 as J                                         # noqa: E402
import l309_lptime as L                                           # noqa: E402

RANK = J.LB
BETA = {r["block_count"]: r for r in json.load(
    open(DIR / "beta_2026-08-16" / "beta_evaluation_results.json"))["test_results"]}
MED = {x["n"]: x["med"] for x in P.load()}
WGT = {x["n"]: x["w"] for x in P.load()}
SER = pickle.load(open(DIR / "l307_serial.pkl", "rb"))
NS = sorted(BETA)


def band_f(mode):
    out = {}
    for a, b in [(21, 50), (51, 80), (81, 100), (101, 120)]:
        k = [n for n in SER if a <= n <= b]
        num = sum((max(SER[n]["M"], SER[n]["C"]) if mode == "lo"
                   else SER[n]["M"] + SER[n]["C"]) + SER[n]["S"] for n in k)
        den = sum(BETA[n]["runtime_seconds"] for n in k)
        for n in k:
            out[n] = num / den
    return out


def wall(js):
    return {r["block_count"]: r.get("runtime_seconds", 0.0)
            for r in json.load(open(DIR / js))["test_results"]}


LPOFF, BCFG = wall("l285_lp_off.json"), wall("l285_betacfg.json")
POOLR = {n: LPOFF[n] / BCFG[n] for n in NS}
LP = {tag: {n: w for n, (c, w) in L.load(DIR / ("l312_%s.log" % tag)).items()}
      for tag in ("ship", "k2", "g0", "g0k2")}


def grader(tag, F):
    g = {}
    for n in NS:
        g[n] = BETA[n]["runtime_seconds"] * POOLR[n] + LP[tag].get(n, 0.0) / F[n]
    return g


def rf_factor(g):
    num = den = 0.0
    fl = 0
    for n in NS:
        rf = max(0.7, (g[n] / MED[n]) ** 0.3)
        rf0 = max(0.7, (BETA[n]["runtime_seconds"] / MED[n]) ** 0.3)
        fl += rf <= 0.7 + 1e-12
        num += WGT[n] * rf / rf0
        den += WGT[n]
    return num / den, fl, sum(g.values())


R = J.graded()
gh0, ga0, ph0 = J.factors(J.M73, "l285_lp_on.json")
QA = {"ship": (1.0, 1.0, 1.0)}
for tag, js, bs in (("k2", "l290_inset_lp2.json", "l294_ship.json"),
                    ("g0", "l294_gate0.json", "l294_ship.json"),
                    ("g0k2", "l297_g0k2.json", "l297_ship2.json")):
    QA[tag] = J.factors(bs, js)

print("== f, measured (L308) ==")
for m in ("lo", "hi"):
    F = band_f(m)
    print("   %s: 21-50 %.2f | 51-80 %.2f | 81-100 %.2f | 101-120 %.2f   (global %.2f)"
          % (m, F[30], F[60], F[90], F[110],
             sum((max(SER[n]["M"], SER[n]["C"]) if m == "lo" else SER[n]["M"] + SER[n]["C"])
                 + SER[n]["S"] for n in NS) / sum(BETA[n]["runtime_seconds"] for n in NS)))
print()
print("== LP cost, measured directly (ICCAD_LP_TIMING) ==")
for tag in ("ship", "k2", "g0", "g0k2"):
    print("   %-5s LP total %6.2f s local   (%d cases run the LP)"
          % (tag, sum(LP[tag].values()), len(LP[tag])))
print()
print("   %-8s %-4s %9s %9s %9s | %10s %10s %9s"
      % ("arm", "f", "grader s", "on floor", "RF", "A total", "B total", "vs rank1"))
for m in ("lo", "hi"):
    F = band_f(m)
    base = grader("ship", F)
    rf_ship, fl_s, tot_s = rf_factor(base)
    for tag in ("ship", "k2", "g0", "g0k2"):
        g = grader(tag, F)
        rf, fl, tot = rf_factor(g)
        gh, ga, ph = QA[tag]
        a = J.total(R, gh, ga, ph, J.DQ_FLAT) * rf
        b = J.total(R, gh0 * gh, ga0 * ga, ph0 * ph) * J.CODE * rf
        best = min(a, b)
        print("   %-8s %-4s %8.1fs %6d/100 %+8.4f%% | %10.6f %10.6f %+8.3f%% %s"
              % (tag, m, tot, fl, 100 * (rf - 1), a, b, 100 * (RANK[1] / best - 1),
                 "<== beats rank 1" if max(a, b) < RANK[1] else ""))
    print()
print("   (RF is relative to the beta runtime vector, the same convention as l276.)")

print()
print("== what multiplier on the measured f each arm needs to break even vs the shipped package ==")
Flo, Fhi = band_f("lo"), band_f("hi")
for tag in ("k2", "g0", "g0k2"):
    for m, F0 in (("lo", Flo), ("hi", Fhi)):
        lo_, hi_ = 0.05, 50.0
        gh, ga, ph = QA[tag]
        for _ in range(60):
            mid = (lo_ + hi_) / 2
            F = {n: F0[n] * mid for n in F0}
            rf, _fl, _t = rf_factor(grader(tag, F))
            rf0, _, _ = rf_factor(grader("ship", F))
            a = J.total(R, gh, ga, ph, J.DQ_FLAT) * rf
            a0 = J.total(R, 1, 1, 1, J.DQ_FLAT) * rf0
            if a < a0:
                hi_ = mid
            else:
                lo_ = mid
        print("   %-5s vs shipped, f_%s: break-even at %.2fx the measured f  (i.e. heavy-band f = %.2f)"
              % (tag, m, hi_, F0[110] * hi_))
