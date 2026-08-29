"""L144 VERIFY2 - what fraction of the s1 weighted score did the colleague's
band scan actually cover? Read-only; touches no binary."""
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
n = [s[3] for s in specs]
w = [math.exp(v / 12.0) for v in n]
W = sum(w)
print(f"s1: {len(specs)} cases, n {min(n)}..{max(n)}, sum(exp(n/12))={W:.1f}")

bands = [(0, 48), (78, 16), (118, 16), (158, 16), (200, 20)]
covered = set()
for off, c in bands:
    idx = range(off, min(off + c, len(specs)))
    covered |= set(idx)
    ww = sum(w[i] for i in idx)
    print(f"  band off={off:>3} c={c:>2}  n={n[off]}..{n[min(off+c,len(specs))-1]:>3}"
          f"  weight={ww:>10.1f}  = {100*ww/W:>5.2f}% of s1")
cw = sum(w[i] for i in covered)
print(f"\nCOVERED {len(covered)}/{len(specs)} cases = {100*cw/W:.2f}% of the "
      f"s1 weighted score")
print(f"UNCOVERED {len(specs)-len(covered)} cases = {100*(W-cw)/W:.2f}%")

gaps = []
i = 0
while i < len(specs):
    if i in covered:
        i += 1
        continue
    j = i
    while j < len(specs) and j not in covered:
        j += 1
    gaps.append((i, j))
    i = j
print("\nuncovered runs (idx range, n range, weight share):")
for i, j in gaps:
    ww = sum(w[k] for k in range(i, j))
    print(f"  [{i:>3},{j:>3})  n={n[i]}..{n[j-1]:>3}  {100*ww/W:>6.2f}%")

print("\ntop-10 heaviest s1 cases (idx, case, n, share of total weight):")
order = sorted(range(len(specs)), key=lambda i: -w[i])[:10]
for i in order:
    print(f"  idx={i:>3} {specs[i][0]:>36} n={n[i]:>4} "
          f"{100*w[i]/W:>5.2f}%  covered={'Y' if i in covered else 'N'}")
