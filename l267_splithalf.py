"""L267/L268 -- free transfer check. RUN THIS BEFORE ANY s2 CAPTURE.

L266's method, applied to the new arms. Two split schemes over the same 40 rows;
gain in pp vs base, positive = better.

The headline to distrust is the SELECTION transfer number: L266 read 100% while
the candidate that actually mattered was flipping sign, because the selection
picked a different (genuinely stable) arm. Read the per-arm per-half table.

An arm with no fitted constants should not be able to flip -- that is the whole
argument for the adaptive search, and this is the test of it.

  <python> l267_splithalf.py [pkl]
"""
import math
import pickle
import sys
from pathlib import Path

DIR = Path(__file__).parent
PKL = DIR / (sys.argv[1] if len(sys.argv) > 1 else "l267_q40.pkl")
D = pickle.load(open(PKL, "rb"))
ROWS, ARMS = D["rows"], [a for a in D["arms"] if a != "ship"]
LAD = ["base"] + ARMS


def wcost(cs, arm):
    sw = sum(math.exp(c["n"] / 12.0) for c in cs)
    return sum(math.exp(c["n"] / 12.0) * c[arm] for c in cs) / sw


def gain(cs, arm):
    b = wcost(cs, "base")
    return 100.0 * (b - wcost(cs, arm)) / b


print("[l267sh] {}  {} cases  arms {}".format(PKL.name, len(ROWS), ARMS))
print()
print("  full sample:")
for a in LAD:
    print("    {:9s} {:+.4f} pp".format(a, gain(ROWS, a)))

splits = {
    "alternating": ([c for i, c in enumerate(ROWS) if i % 2 == 0],
                    [c for i, c in enumerate(ROWS) if i % 2 == 1]),
    "by size": (ROWS[:len(ROWS) // 2], ROWS[len(ROWS) // 2:]),
}

for nm, (H1, H2) in splits.items():
    print()
    print("  split: {}   ({} / {} cases)".format(nm, len(H1), len(H2)))
    for tr, te, lab in ((H1, H2, "1->2"), (H2, H1, "2->1")):
        best = max(LAD, key=lambda a: gain(tr, a))
        best_te = max(LAD, key=lambda a: gain(te, a))
        g_opt = gain(te, best_te)
        xfer = 100.0 * gain(te, best) / g_opt if g_opt > 1e-9 else float("nan")
        print("    {}  train picks {:9s} (+{:.3f} pp)  ->  TEST {:+.3f} pp"
              "   | test-optimal {:9s} +{:.3f} pp   transfer {:.0f}%".format(
                  lab, best, gain(tr, best), gain(te, best), best_te, g_opt, xfer))
    print("    per-arm gain on each half   <- THIS is the table that decides it:")
    for a in ARMS:
        g1, g2 = gain(H1, a), gain(H2, a)
        flag = "  <-- FLIPS" if g1 * g2 < 0 else ""
        print("      {:9s} H1 {:+.3f} pp   H2 {:+.3f} pp{}".format(a, g1, g2, flag))

print()
print("  per-case detail (weight exp(n/12), sorted heaviest first):")
print("    {:>4s} {:>10s}".format("n", "base") +
      "".join("{:>12s}".format(a) for a in ARMS))
for r in ROWS:
    print("    {:4d} {:10.6f}".format(r["n"], r["base"]) +
          "".join("{:+11.3f}%".format(100 * (r[a] / r["base"] - 1)) for a in ARMS))
