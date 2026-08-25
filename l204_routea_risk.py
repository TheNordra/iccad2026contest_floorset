"""L204 - the bet's DOWNSIDE has never been drawn. This draws it.

Every table in this ledger prices route A at two points: 1.00 ("neutral") and
0.68 ("the bet"). Both are non-negative outcomes. But the only DIRECT
measurement anyone has of route A is that it costs 2.9x on this box's 16
physical cores, and the ledger separately records it sinking every case to SA
fallback at 48 cores. `ra > 1` is not a hypothetical branch -- it is the only
branch that has ever been observed. Nothing in the ledger says what happens
there, so the package is carrying an unpriced tail.

Two things this file establishes and the handoff table cannot:

  1. THE NEUTRAL COLUMN IS NOT A BRANCH OF THE BET -- IT IS AN ALTERNATIVE
     PACKAGE. `_shape_lp_on()` and route A fire on the same >=40-core gate, so
     "route A neutral" is exactly what shipping with route A OFF produces, and
     that is a code default we can set with certainty. So the choice is not
     bet-vs-nothing; it is

         A  route A ON   -> ra unknown, outcome anywhere on the curve below
         B  route A OFF  -> ra = 1.00 exactly, 0.9149, rank 4, NO tail

     Option B's number is the same 0.9149 the handoff calls "the downside" --
     but as a guarantee rather than as a floor that depends on route A being
     merely useless rather than harmful.

  2. WHERE THE CURVE CROSSES. Two thresholds matter: the ra at which A stops
     beating B (trivially ra = 1), and the ra at which A stops beating BETA --
     our own previously graded submission. Below that second crossing we would
     be shipping something worse than what we already scored.

  <python> l204_routea_risk.py
"""
import json
import math
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
THR = 0.7 ** (1 / 0.3)
BETA = 0.9265861161320369
Q_POOL_FULL = 0.3976 + 2.6588
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333),
         (6, 0.9552271810705998)]


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t - 1e-9) + 1


def ins(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


def oos(fn):
    return {r["test_id"]: r for r in json.load(open(DIR / fn))["test_results"]}


wm, _ = ins("_l181_m73.json")
wo, _ = ins("_l181_cur.json")
wk, _ = ins("_l189_k1.json")
ROWS = M.rows_new()
W = sum(r["w"] for r in ROWS)
BETA_NUM = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in ROWS) / W
POOL, DT, MED, ROW = {}, {}, {}, {}
for r in ROWS:
    n = r["n"]
    if not wm.get(n):
        continue
    k = r["t"] / wm[n]
    POOL[n] = wo[n] * k
    DT[n] = max(0.0, (wk[n] - wo[n]) * k)
    MED[n] = r["med"]
    ROW[n] = r
NS = sorted(POOL)
A = {s: (oos("l194_{}_fulloff.json".format(s)),
         oos("l192_{}_full.json".format(s))) for s in ("s1", "s2")}


def time_gate(s):
    return {n: 1 if POOL[n] + DT[n] <= THR * MED[n] * s else 0 for n in NS}


def rf_at(g, ra):
    num = sum(ROW[n]["w"] * ROW[n]["q"]
              * max(0.7, ((POOL[n] * ra + (DT[n] if g.get(n, 0) else 0.0))
                          / ROW[n]["med"]) ** 0.3) for n in NS)
    num += sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in ROWS if r["n"] not in POOL)
    return 100 * (BETA_NUM - num / W) / BETA_NUM


def qual_pern(g, sample):
    off, on = A[sample]
    ids = sorted(set(off) & set(on))
    so, sn, cnt = {}, {}, {}
    for i in ids:
        n = off[i]["n"]
        so[n] = so.get(n, 0.0) + off[i]["cost"]
        sn[n] = sn.get(n, 0.0) + on[i]["cost"]
        cnt[n] = cnt.get(n, 0) + 1
    ns = [n for n in cnt if n in ROW]
    sw = sum(ROW[n]["w"] for n in ns)
    qo = sum(ROW[n]["w"] * so[n] / cnt[n] for n in ns) / sw
    qg = sum(ROW[n]["w"] * (sn[n] if g.get(n, 0) else so[n]) / cnt[n]
             for n in ns) / sw
    return 100 * (qo - qg) / qo


def graded(g, ra, phi=1.0):
    q = sum(qual_pern(g, s) for s in ("s1", "s2")) / 2
    return BETA * (1 - (q + Q_POOL_FULL + rf_at(g, 1.0 - phi + phi * ra))
                   / 100.0)


def bisect(f, lo, hi, tol=1e-4):
    """smallest ra in [lo,hi] with f(ra) True, or None."""
    if not f(hi):
        return None
    if f(lo):
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) else (mid, hi)
    return hi


def main():
    print(__doc__)
    ship = time_gate(1.2)
    lpoff = {n: 0 for n in NS}

    print("=" * 78)
    print("THE CURVE -- shipped gate (s=1.2, 63 on), route A multiplier ra")
    print("  ra < 1 route A saves wall   ra > 1 route A COSTS wall")
    print()
    print("{:>7}{:>12}{:>11}{:>7}   {}".format(
        "ra", "NET", "graded", "rank", "vs the alternatives"))
    print("-" * 78)
    b_graded = graded(ship, 1.0)
    for ra in (0.50, 0.68, 0.80, 0.90, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50,
               2.00, 2.90):
        gr = graded(ship, ra)
        net = 100 * (1 - gr / BETA)
        note = []
        note.append("beats beta" if gr < BETA else "WORSE THAN BETA")
        if ra > 1.0:
            note.append("worse than route-A-OFF ({:+.3f}pp)"
                        .format(100 * (b_graded - gr) / BETA))
        mark = "  <-- route A OFF ships exactly here" if ra == 1.00 else ""
        print("{:>7.2f}{:>+11.3f}%{:>11.5f}{:>7}   {}{}"
              .format(ra, net, gr, rank_of(gr), "; ".join(note), mark))
    print("-" * 78)

    # ---- how much of the wall does route A actually touch? -----------------
    # ⚠️ CORRECTION TO THE ROW ABOVE. `ra` there multiplies the WHOLE pool
    # wall, which over-states the risk: route A replaces the profile-execution
    # phase, but the per-case wall also contains the serial proxy/post-process
    # tail that runs on the main thread and that route A cannot touch. M47
    # measured that tail at 29% of the scored wall on n>100. So the honest
    # model is t = POOL*(1-phi + phi*ra) + dt*g with phi ~ 0.7, not phi = 1.
    # Drawing my own number's sensitivity rather than quoting the alarming end
    # of it. MEASURED: the break-even moves 1.052 -> 1.103 across phi in
    # [1.0, 0.5] -- from "5% of wall" to "10% of wall". I expected phi to
    # roughly double the margin; it does not, and the reason is worth keeping:
    # phi rescales the perturbation but not the 1.26pp margin over beta, and
    # RF's t^0.3 means the margin is spent at 0.3% of score per 1% of wall
    # either way. The conclusion is insensitive to phi.
    print()
    print("=" * 78)
    print("SENSITIVITY -- phi = the fraction of the wall route A actually touches")
    print("  phi = 1.00 is the row above (route A owns the whole case wall).")
    print("  M47 measured the serial proxy tail at 29% of n>100 wall => "
          "phi ~ 0.71.")
    print()
    print("{:>7}{:>14}{:>16}{:>14}".format(
        "phi", "beta lost at", "rank 3 holds to", "graded @ra=1.35"))
    print("-" * 78)
    for phi in (1.00, 0.85, 0.71, 0.60, 0.50):
        xb = bisect(lambda ra: graded(ship, ra, phi) >= BETA, 1.0, 12.0)
        x3 = bisect(lambda ra: rank_of(graded(ship, ra, phi)) > 3, 0.3, 3.0)
        print("{:>7.2f}{:>14}{:>16}{:>14.5f}"
              .format(phi,
                      "ra = {:.3f}".format(xb) if xb else "never in [1,12]",
                      "ra <= {:.3f}".format(x3) if x3 else "-",
                      graded(ship, 1.35, phi)))
    print("-" * 78)
    print("  Even at phi = 0.50 the package is worse than beta once route A")
    print("  costs ~10% of the wall it owns. The asymmetry does not come from")
    print("  phi; it comes from the margin over beta being only 1.26pp while")
    print("  RF enters as t^0.3.")

    # ---- the crossings -----------------------------------------------------
    print()
    print("THE CROSSINGS  (phi = 1.00, the conservative end)")
    x_beta = bisect(lambda ra: graded(ship, ra) >= BETA, 1.0, 6.0)
    x_r4 = bisect(lambda ra: rank_of(graded(ship, ra)) > 4, 1.0, 6.0)
    x_r2 = bisect(lambda ra: rank_of(graded(ship, ra)) > 2, 0.3, 3.0)
    x_r3 = bisect(lambda ra: rank_of(graded(ship, ra)) > 3, 0.3, 3.0)
    print("  rank 2 holds while ra <= {:.3f}".format(x_r2) if x_r2 else
          "  rank 2 never reached")
    print("  rank 3 holds while ra <= {:.3f}".format(x_r3) if x_r3 else
          "  rank 3 never reached")
    print("  we stop beating beta at ra = {:.3f}".format(x_beta) if x_beta
          else "  beta is never lost in [1, 6]")
    print("  i.e. route A may cost up to {:.0f}% more wall before this package"
          .format(100 * (x_beta - 1)) if x_beta else "")
    print("       is worse than the one we already graded.")
    print()
    print("  MEASURED, for scale: route A costs 2.9x on this box's 16 physical")
    print("  cores. If the grader's 48 cores behave like 16 rather than like")
    print("  the projection, ra = 2.9 lands at graded {:.5f}, rank {}."
          .format(graded(ship, 2.9), rank_of(graded(ship, 2.9))))

    # ---- expected value over the bet ---------------------------------------
    print()
    print("=" * 78)
    print("EXPECTED OUTCOME vs P(route A delivers 0.68)")
    print("  Option B (route A OFF) is a CERTAINTY at {:.5f}, rank {}."
          .format(b_graded, rank_of(b_graded)))
    print("  Option A only pays if route A actually helps; the row below shows")
    print("  the mixture where the loss branch is the MEASURED 2.9x, not 1.0.")
    print()
    print("{:>7}{:>12}{:>12}{:>12}".format(
        "p", "E[graded] (loss=1.0)", "(loss=1.35)", "(loss=2.9)"))
    print("-" * 78)
    g68 = graded(ship, 0.68)
    for p in (0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0):
        row = [p * g68 + (1 - p) * graded(ship, L) for L in (1.0, 1.35, 2.9)]
        print("{:>7.1f}{:>20.5f}{:>12.5f}{:>12.5f}"
              .format(p, row[0], row[1], row[2]))
    print("-" * 78)
    print("  Compare every cell against Option B's certain {:.5f}."
          .format(b_graded))
    print("  Cells above it are bets that lose money in expectation.")

    # ---- and what if we also drop the LP -----------------------------------
    print()
    print("For reference, the same package with the LP gate closed entirely")
    print("  (LP off, route A off): graded {:.5f}, rank {}"
          .format(graded(lpoff, 1.0), rank_of(graded(lpoff, 1.0))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
