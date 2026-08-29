"""L258 step 3 -- the wall-free gated set: quality of the profiles we can afford.

Under a MAX-bound profile phase (the grader is: the research handoff's own
d_max 3.204s > sum/48 2.501s at n=120), running the shrink on profile p costs
NOTHING unless p's shrunk time exceeds the pool's current max-setter. So the
affordable gated set is exactly {p : t_on(p) <= max_off}.

This prices that set's quality against the unconstrained greedy, and re-runs a
greedy restricted to affordable profiles.
"""
import math
import pickle
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
from l257_twin import build            # noqa: E402

C = pickle.load(open(DIR / "l257_cache.pkl", "rb"))
B = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
T = pickle.load(open(DIR / "l258_times.pkl", "rb"))
mo, mn = T["off"], T["on"]

cases = []
for k in sorted(C, key=lambda k: -C[k]["n"]):
    if k not in B:
        continue
    e = C[k]
    base = {i: dict(area=r["area"], hpwl=r["hpwl"], vrel=r["vrel"])
            for i, r in B[k]["recs"].items() if i in e["basecost"]}
    if len(base) < 2:
        continue
    cases.append(dict(n=e["n"], sumA=e["sumA"], base=base,
                      basecost=e["basecost"], new=e["recs"]))

base_cost, _ = build(cases, [])
max_off = max(mo.values())
max_p = max(mo, key=lambda p: mo[p])
safe = sorted([p for p in mn if mn[p] <= max_off + 1e-9])
unsafe = sorted([p for p in mn if mn[p] > max_off + 1e-9])

print("[l258] pool max-setter prof {} at {:.3f}s".format(max_p, max_off))
print("  affordable (t_on <= max_off): {}".format(safe))
print("  would raise the wall:         {}".format(unsafe))
print()

full, _ = build(cases, sorted(mn))
sf, _ = build(cases, safe)
print("  base                          {:.6f}".format(base_cost))
print("  all 8 greedy parents          {:.6f}   {:+.4f}%".format(
    full, 100 * (full - base_cost) / base_cost))
print("  AFFORDABLE subset only        {:.6f}   {:+.4f}%".format(
    sf, 100 * (sf - base_cost) / base_cost))
print()

# added work, for the sum-bound view
add = sum(mn[p] - mo[p] for p in safe)
tot = sum(mo.values())
print("  affordable set: max-bound wall x1.0000 (by construction)")
print("                  sum-bound  wall x{:.4f}  (+{:.2f}%)".format(
    1 + add / tot, 100 * add / tot))
q = -100 * (sf - base_cost) / base_cost
for lab, pct in (("max-bound", 0.0), ("sum-bound", 100 * add / tot)):
    print("    {:10s} quality +{:.3f} pp - wall {:.3f} pp  => NET {:+.3f} pp".format(
        lab, q, 0.151 * pct, q - 0.151 * pct))

# greedy restricted to affordable profiles
print()
print("  greedy restricted to affordable profiles:")
chosen, cur = [], base_cost
for k in range(1, len(safe) + 1):
    bc, bi = None, None
    for i in safe:
        if i in chosen:
            continue
        c, _ = build(cases, chosen + [i])
        if bc is None or c < bc:
            bc, bi = c, i
    if bi is None or not (bc < cur - 1e-12):
        break
    chosen.append(bi)
    cur = bc
    a2 = sum(mn[p] - mo[p] for p in chosen)
    q2 = -100 * (cur - base_cost) / base_cost
    print("   K={} add {:3d}  {:.6f}  quality +{:.3f} pp   sum-bound NET {:+.3f} pp".format(
        k, bi, cur, q2, q2 - 0.151 * 100 * a2 / tot))


# ---------------------------------------------------------------------------
# TRANSFER. M76 measured in-sample source-set selection transferring at ~5%, and
# the gated set above was picked greedily on these same 40 cases. Split-half is
# free here and is the cheapest honest check available without capturing s2.
print()
print("  split-half transfer (pick the set on A, score it on B, and vice versa):")
A = [c for i, c in enumerate(cases) if i % 2 == 0]
Bh = [c for i, c in enumerate(cases) if i % 2 == 1]


def greedy(cs, pool, kmax=4):
    ch, cur = [], build(cs, [])[0]
    for _ in range(kmax):
        bc, bi = None, None
        for i in pool:
            if i in ch:
                continue
            c, _ = build(cs, ch + [i])
            if bc is None or c < bc:
                bc, bi = c, i
        if bi is None or not (bc < cur - 1e-12):
            break
        ch.append(bi)
        cur = bc
    return ch


for nm, tr, te in (("A->B", A, Bh), ("B->A", Bh, A)):
    ch = greedy(tr, safe)
    b_tr = build(tr, [])[0]
    b_te = build(te, [])[0]
    q_tr = -100 * (build(tr, ch)[0] - b_tr) / b_tr
    q_te = -100 * (build(te, ch)[0] - b_te) / b_te
    # what the best possible set on the TEST half would have given
    ch_te = greedy(te, safe)
    q_best = -100 * (build(te, ch_te)[0] - b_te) / b_te
    print("    {}  chose {}  train +{:.3f} pp -> TEST +{:.3f} pp"
          "   (test-optimal +{:.3f} pp, transfer {:.0f}%)".format(
              nm, ch, q_tr, q_te, q_best,
              100 * q_te / q_best if q_best > 1e-9 else 0.0))
