"""L266 -- does the frame-ladder choice transfer? Split-half on s1.

The `tuned` rung values (1.06/1.09/1.12/1.16) were chosen AFTER measuring
s_min ~ 1.11 on this same 40-case sample. L258 measured exactly that pattern --
a constant picked on the scoring sample -- transferring at 0%, in both directions.
So before spending an s2 capture, ask the cheap version.

Two tests, both free from the saved per-case costs:

  A  SELECTION transfer: pick the best ladder on half A, score it on half B.
     Weak (only 4 discrete ladders) but it is the exact shape L258 failed.
  B  STABILITY of the per-case winner: if the same ladder wins on both halves by
     a similar margin, the choice is not sample-specific.

Rows in l264_dense40.pkl and l265_tuned40.pkl were produced from the same key
list (sorted by -n, limit 40), so they join by index -- asserted on n.
"""
import math
import pickle
from pathlib import Path

DIR = Path(__file__).parent
A = pickle.load(open(DIR / "l264_dense40.pkl", "rb"))["rows"]
B = pickle.load(open(DIR / "l265_tuned40.pkl", "rb"))["rows"]
assert len(A) == len(B), (len(A), len(B))
for x, y in zip(A, B):
    assert x["n"] == y["n"], (x["n"], y["n"])
    assert abs(x["base"] - y["base"]) < 1e-12, "base differs -> not the same case"

CASES = []
for x, y in zip(A, B):
    CASES.append(dict(n=x["n"], base=x["base"], dense=x["dense"],
                      coarse=x["coarse"], tuned=y["tuned"]))
LAD = ["base", "dense", "coarse", "tuned"]


def wcost(cs, arm):
    sw = sum(math.exp(c["n"] / 12.0) for c in cs)
    return sum(math.exp(c["n"] / 12.0) * c[arm] for c in cs) / sw


def gain(cs, arm):
    b = wcost(cs, "base")
    return 100.0 * (b - wcost(cs, arm)) / b       # +pp = better


print("[l266] {} cases".format(len(CASES)))
print()
print("  full sample:")
for a in LAD:
    print("    {:8s} {:+.4f} pp".format(a, gain(CASES, a)))

splits = {
    "alternating": ([c for i, c in enumerate(CASES) if i % 2 == 0],
                    [c for i, c in enumerate(CASES) if i % 2 == 1]),
    "by size": (CASES[:len(CASES) // 2], CASES[len(CASES) // 2:]),
}

for nm, (H1, H2) in splits.items():
    print()
    print("  split: {}   ({} / {} cases)".format(nm, len(H1), len(H2)))
    for tr, te, lab in ((H1, H2, "1->2"), (H2, H1, "2->1")):
        best = max(LAD, key=lambda a: gain(tr, a))
        g_tr = gain(tr, best)
        g_te = gain(te, best)
        best_te = max(LAD, key=lambda a: gain(te, a))
        g_opt = gain(te, best_te)
        xfer = 100.0 * g_te / g_opt if g_opt > 1e-9 else float("nan")
        print("    {}  train picks {:7s} (+{:.3f} pp)  ->  TEST +{:.3f} pp"
              "   | test-optimal {:7s} +{:.3f} pp   transfer {:.0f}%".format(
                  lab, best, g_tr, g_te, best_te, g_opt, xfer))
    print("    per-ladder gain on each half:")
    for a in LAD:
        if a == "base":
            continue
        print("      {:8s} H1 {:+.3f} pp   H2 {:+.3f} pp".format(
            a, gain(H1, a), gain(H2, a)))

print()
print("  reading: 'tuned' is only safe if it is the winner on BOTH halves and its")
print("  gain has the same sign and rough magnitude on each. A ladder that wins")
print("  big on one half and loses on the other is L258 repeating.")
