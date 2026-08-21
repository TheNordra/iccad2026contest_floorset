"""OFFLINE (never shipped): price the L154 band-catch ON TOP OF L147's own cost.

🚨 THE BASE MATTERS AND THE OBVIOUS ONE IS WRONG. Pricing the retry against the
raw beta timings reads "RF cost +0.0000%, free" -- because the beta run's n=117
case sits at t=1.130s against a median of 7.248s, i.e. 1.077s of headroom before
RF leaves the 0.7 floor, and the measured retry there is 0.935s. But the beta run
was the PRE-L147 package. L147 already spends slack on every LP case (RF
-0.9726%, 1/100 -> 9/100 cases off the floor), so the catch's retry has to be
stacked on L147's own per-case dt, not on the beta timings. Same trap as
HANDOFF_2026-08-20 §4.4, one level up: the right base is the arm we actually
ship, not the last thing that was graded.

Retry cost, min-of-3, arms interleaved, one case at a time (l154_price.txt):

    case 10  n= 31   CATCH off 2.4278s -> on 2.4555s    +0.028s   (an actual retry)
    case 21  n= 42   CATCH off 2.2760s -> on 2.3745s    +0.098s   (an actual retry)
    case 96  n=117   LP off 5.0100s -> band 5.9452s     +0.935s   (what a big-n
    case 92  n=113   LP off 3.6873s -> band 3.9150s     +0.228s    rejection pays)

The last two are the shipped band's own LP cost at that size, which is exactly
the program the retry solves.
"""
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import l146_rf_price as L                                           # noqa: E402
from l147_price import per_case_min                                 # noqa: E402

rows = L.load()
ns = sorted(r["n"] for r in rows)


def near(t):
    return min(ns, key=lambda n: abs(n - t))


ctrl, _ = per_case_min([f"t{i}_ctrl" for i in (1, 2, 3)])
arm, _ = per_case_min([f"t{i}_r15g" for i in (1, 2, 3)])
DT147 = {n: arm[n] - ctrl[n] for n in ctrl}

base = L._total(rows, lambda r: 1.0)


def rf(dt):
    num = den = 0.0
    for r in rows:
        t = r["t"] + max(0.0, dt.get(r["n"], 0.0))
        num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        den += r["w"]
    return 100 * (base - num / den) / base


print(__doc__)
sp = sorted(DT147.values())
print(f"L147 per-case dt: p50 {st.median(sp):+.3f}s  p90 "
      f"{sp[int(0.9 * len(sp))]:+.3f}s  max {sp[-1]:+.3f}s")
rf147 = rf(DT147)
print(f"L147 alone: RF {rf147:+.4f}%   (handover records -0.9726%)\n")

SCEN = {
    "in-set windows, as measured (n=31, n=42 reject)":
        {near(31): 0.028, near(42): 0.098},
    "+ one big-n rejection (linux case 96 / OOS n=116, n=119)":
        {near(31): 0.028, near(42): 0.098, near(117): 0.935},
    "+ two big-n rejections (pessimistic)":
        {near(31): 0.028, near(42): 0.098, near(117): 0.935, near(113): 0.228},
}
QUAL = {"in-set windows, as measured (n=31, n=42 reject)": 0.0009,
        "+ one big-n rejection (linux case 96 / OOS n=116, n=119)": 0.0356,
        "+ two big-n rejections (pessimistic)": 0.0532}

print(f"{'scenario':<58}{'RF total':>10}{'RF incr':>10}{'quality':>10}{'NET incr':>10}")
for label, extra in SCEN.items():
    stacked = dict(DT147)
    for n, v in extra.items():
        stacked[n] = stacked.get(n, 0.0) + v
    r_tot = rf(stacked)
    incr = r_tot - rf147
    q = QUAL[label]
    print(f"{label:<58}{r_tot:>+9.4f}%{incr:>+9.4f}%{q:>+9.4f}%{q + incr:>+9.4f}%")

print("\nhow much headroom is left on the case that pays the most (n=117),"
      "\nAFTER L147 has already spent its share:")
r117 = [r for r in rows if r["n"] == near(117)][0]
t147 = r117["t"] + DT147.get(near(117), 0.0)
head = L.THR * r117["med"] - t147
print(f"   beta t {r117['t']:.3f}s  + L147 {DT147.get(near(117), 0.0):+.3f}s "
      f"= {t147:.3f}s;  floor leaves at {L.THR * r117['med']:.3f}s"
      f"  =>  headroom {head:+.3f}s, retry needs 0.935s "
      f"({100 * 0.935 / head:.0f}% of it)" if head > 0 else
      f"   beta t {r117['t']:.3f}s + L147 {DT147.get(near(117), 0.0):+.3f}s "
      f"= {t147:.3f}s is ALREADY past the floor threshold "
      f"{L.THR * r117['med']:.3f}s")


# ---------------------------------------------------------------------------
# SELF-CONSISTENT PER-EVENT PRICING. Pairing "OOS quality" with "beta RF" is the
# project's convention, but here it hides the thing that decides the verdict:
# the gain and the cost are the SAME EVENT on the SAME CASE. A big-n case
# rejects -> the catch recovers the shipped band's cost on it AND pays that
# case's band-LP time. So price both on one beta row and read the net directly.
#
# Recovered cost per event, measured (raw, not weighted):
#   linux case 96  n=117  1.215357 -> 1.186644   -0.028713
#   OOS s1 case 220 n=116 1.465193 -> 1.434681   -0.030512
#   OOS s2 case 233 n=119 1.527081 -> 1.492752   -0.034328
#   in-set case 10  n= 31 1.243075 -> 1.149935   -0.093140
#   in-set case 21  n= 42 1.410762 -> 1.355328   -0.055434
print("\n\n=== per-event NET, both sides on the same beta case, "
      "stacked on L147 ===")
print(f"{'event':<44}{'dq':>9}{'dt':>8}{'quality':>10}{'RF':>10}{'NET':>10}")


def net_event(n_target, dq, dt_extra):
    n = near(n_target)
    stacked = dict(DT147)
    stacked[n] = stacked.get(n, 0.0) + dt_extra
    num = den = 0.0
    num0 = den0 = 0.0
    for r in rows:
        t0 = r["t"] + max(0.0, DT147.get(r["n"], 0.0))
        t1 = r["t"] + max(0.0, stacked.get(r["n"], 0.0))
        q1 = r["q"] - (dq if r["n"] == n else 0.0)
        num0 += r["w"] * r["q"] * max(0.7, (t0 / r["med"]) ** 0.3)
        den0 += r["w"]
        num += r["w"] * q1 * max(0.7, (t1 / r["med"]) ** 0.3)
        den += r["w"]
    a, b = num0 / den0, num / den
    # split: quality alone (dt unchanged), RF alone (q unchanged)
    numq = sum(r["w"] * (r["q"] - (dq if r["n"] == n else 0.0))
               * max(0.7, ((r["t"] + max(0.0, DT147.get(r["n"], 0.0)))
                           / r["med"]) ** 0.3) for r in rows) / den0
    return (100 * (a - numq) / a, 100 * (numq - b) / a * -1 + 0, 100 * (a - b) / a)


for label, n, dq, dt_x in (
        ("big-n reject rescued (linux 96 / OOS 220,233)", 117, 0.030, 0.935),
        ("mid-n reject rescued (n=113)", 113, 0.030, 0.228),
        ("in-set case 10 (n=31)", 31, 0.093140, 0.028),
        ("in-set case 21 (n=42)", 42, 0.055434, 0.098),
):
    q, _r, netv = net_event(n, dq, dt_x)
    print(f"{label:<44}{-dq:>+9.3f}{dt_x:>+8.3f}{q:>+9.4f}%{netv - q:>+9.4f}%"
          f"{netv:>+9.4f}%")
