"""OFFLINE (never shipped): decompose the Windows/Linux gap on the 48-core lane.

The shape LP is degenerate, so the two scipy/HiGHS builds land on different
optima of the same program (L119). This asks the question that decides how the
L147 gain gets REPORTED: is the Linux shortfall a systematic loss, or a handful
of degenerate ties falling the other way -- in which case the sign is a coin
flip per case and the grader's own draw is neither of ours.

    python l153_xplat.py
"""
import json
from pathlib import Path


def load(p):
    j = json.loads(Path(p).read_text(encoding="utf-8"))
    return j, {r["test_id"]: r for r in j["test_results"]}


def gap(name, win_p, lin_p):
    wj, w = load(win_p)
    lj, l = load(lin_p)
    ids = sorted(set(w) & set(l))
    d = [(l[i]["cost"] - w[i]["cost"], i) for i in ids]
    mv = sorted([x for x in d if abs(x[0]) > 1e-9], key=lambda x: -abs(x[0]))
    worse = [x for x in mv if x[0] > 0]
    better = [x for x in mv if x[0] < 0]
    print(f"\n== {name}")
    print(f"   windows {wj['total_score']!r}   linux {lj['total_score']!r}   "
          f"linux-windows {lj['total_score'] - wj['total_score']:+.9f}")
    print(f"   movers {len(mv)}/{len(ids)}   linux worse on {len(worse)}, "
          f"better on {len(better)}")
    print(f"   sum of mover deltas {sum(x[0] for x in mv):+.6f}  "
          f"(worse {sum(x[0] for x in worse):+.6f}, "
          f"better {sum(x[0] for x in better):+.6f})")
    for dd, i in mv:
        n = w[i]["block_count"]
        print(f"     case {i:3d} n={n:3d}: win {w[i]['cost']:.6f} -> "
              f"lin {l[i]['cost']:.6f}  {dd:+.6f}")
    return mv


print(__doc__)
mv_ctrl = gap("SHIPPED BAND (already uploaded) -- the cross-platform variance "
              "L147 did NOT introduce",
              "results_L137_48c_cap4.json", "results_L153_linux_ctrl.json")
mv_arm = gap("L147 ARM", "results_L147_on_L137.json", "results_L153_linux_arm.json")
gap("PRE-LP (LP entirely off) -- the control for 'is it the LP at all'",
    "results_L153_lpoff_L137.json", "results_L153_linux_lpoff.json")

print("\n== the gain, both platforms (ctrl -> arm, in-set 100)")
for plat, c, a in (("windows", "results_L137_48c_cap4.json",
                    "results_L147_on_L137.json"),
                   ("linux", "results_L153_linux_ctrl.json",
                    "results_L153_linux_arm.json")):
    cj, _ = load(c)
    aj, _ = load(a)
    print(f"   {plat:8s} {cj['total_score']!r} -> {aj['total_score']!r}   "
          f"{100 * (1 - aj['total_score'] / cj['total_score']):+.4f}%")

print("\n== per-case gain agreement (ctrl->arm delta, windows vs linux)")
_, wc = load("results_L137_48c_cap4.json")
_, wa = load("results_L147_on_L137.json")
_, lc = load("results_L153_linux_ctrl.json")
_, la = load("results_L153_linux_arm.json")
ids = sorted(set(wc) & set(lc))
agree = sum(1 for i in ids
            if abs((wa[i]["cost"] - wc[i]["cost"])
                   - (la[i]["cost"] - lc[i]["cost"])) < 1e-9)
wgain = [wc[i]["cost"] - wa[i]["cost"] for i in ids]
lgain = [lc[i]["cost"] - la[i]["cost"] for i in ids]
print(f"   identical per-case gain on {agree}/{len(ids)} cases")
print(f"   cases improved by the cut: windows "
      f"{sum(1 for g in wgain if g > 1e-9)}, linux {sum(1 for g in lgain if g > 1e-9)}")
print(f"   cases hurt by the cut:     windows "
      f"{sum(1 for g in wgain if g < -1e-9)}, linux {sum(1 for g in lgain if g < -1e-9)}")


# ---------------------------------------------------------------------------
# WEIGHTED attribution. total_score = sum(cost_i * e^{n_i/12}) / sum(e^{n_j/12}),
# so an unweighted mover list is the wrong story: HANDOFF_2026-08-20 §4.1 is the
# same trap (specs[0:16] is 0.0126% of the weighted score). Case 9 at n=30
# carries weight e^{-7.5} = 5.5e-4 against n=120's 1.0.
import math

print("\n== WEIGHTED attribution of the linux-windows gap on the L147 arm")
_, wa2 = load("results_L147_on_L137.json")
_, la2 = load("results_L153_linux_arm.json")
ids2 = sorted(set(wa2) & set(la2))
maxn = max(wa2[i]["block_count"] for i in ids2)
W = {i: math.exp((wa2[i]["block_count"] - maxn) / 12) for i in ids2}
SW = sum(W.values())
contrib = sorted(((la2[i]["cost"] - wa2[i]["cost"]) * W[i] / SW, i)
                 for i in ids2 if abs(la2[i]["cost"] - wa2[i]["cost"]) > 1e-9)
contrib.sort(key=lambda x: -abs(x[0]))
tot = sum(c for c, _ in contrib)
print(f"   sum of weighted contributions {tot:+.9f}  "
      f"(observed total gap {sum(la2[i]['cost'] * W[i] for i in ids2) / SW - sum(wa2[i]['cost'] * W[i] for i in ids2) / SW:+.9f})")
for c, i in contrib:
    n = wa2[i]["block_count"]
    print(f"     case {i:3d} n={n:3d}  w={W[i]:.4f}  raw "
          f"{la2[i]['cost'] - wa2[i]['cost']:+.6f}  ->  weighted {c:+.9f}"
          f"   ({100 * c / tot:5.1f}% of the gap)")
