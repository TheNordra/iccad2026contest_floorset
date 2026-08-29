"""How much of the real 240-case OOS weighted score do the screened slices cover?

The A/B harness weights each case by exp(n/12) and m77._specs() returns cases
sorted by n ASCENDING.  So specs[0:64] is the SMALLEST-n tail.  This prints the
weight mass of each slice used in the screen, which is what any "+0.15% on 64
cases" claim has to be read against.  Read-only.
"""
import math
import os
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

import m77_oos_probe as m77                                         # noqa: E402

specs = m77._specs("s1")
ns = [s[3] for s in specs]
w = [math.exp(n / 12.0) for n in ns]
W = sum(w)
print(f"240 OOS s1 cases: n from {min(ns)} to {max(ns)}")
print(f"weight exp(n/12): min {min(w):.3g}  max {max(w):.3g}  "
      f"ratio {max(w) / min(w):.3g}")
print()
for lo, hi, label in [(0, 16, "specs[0:16]   (their 16-case A/B)"),
                      (0, 32, "specs[0:32]   (their 32-case A/B)"),
                      (0, 64, "specs[0:64]   (their 64-case A/B)"),
                      (64, 128, "specs[64:128] (their disjoint slice)"),
                      (128, 192, "specs[128:192]"),
                      (192, 240, "specs[192:240] (largest 48)"),
                      (0, 128, "specs[0:128]  (everything they screened)")]:
    m = sum(w[lo:hi])
    print(f"  {label:38s} n={ns[lo]:>3}..{ns[hi - 1]:>3}  "
          f"weight mass {100 * m / W:8.4f}% of the 240-case score")
print()
# which single case dominates
top = sorted(range(240), key=lambda i: -w[i])[:5]
print("top-5 cases by weight:")
for i in top:
    print(f"  idx {i:>3}  n={ns[i]:>3}  {100 * w[i] / W:6.2f}% of total weight  "
          f"{specs[i][0]}")
print()
print(f"the single heaviest case is {100 * w[top[0]] / W:.2f}% of the 240-case "
      f"weighted score; all of specs[0:128] together is "
      f"{100 * sum(w[0:128]) / W:.4f}%")
