"""L144 VERIFY2 - aggregate the band A/Bs by their TRUE share of the s1=240
weighted score (weight = exp(n/12)).

The colleague's report reads the bands as an unweighted list ("negative in 4/5
bands"). The score is not unweighted: the top 20 cases carry 37.6% and the
bottom 48 carry 0.08%. This weights each band by what it is actually worth.

Bands marked SRC=them are transcribed from their report; SRC=me are my own runs.
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
w = [math.exp(s[3] / 12.0) for s in specs]
n = [s[3] for s in specs]
W = sum(w)

# (offset, cases, delta_pct_improvement, source)   delta > 0 == better
BANDS = [
    (0,   48, -0.280, "them"),
    (78,  16, -0.386, "them"),
    (118, 16, -0.206, "them"),
    (158, 16, +0.000, "me (reproduced exact)"),
    (180, 20, -1.262, "me (NEW - band they never ran)"),
    (200, 20, -0.195, "me (reproduced exact)"),
    (220, 20, +0.119, "me (NEW - band they never ran)"),
]

print(f"s1: {len(specs)} cases, sum(exp(n/12)) = {W:.1f}\n")
print(f"{'band':>16} {'n range':>12} {'share':>8} {'delta':>9}  source")
num = den = 0.0
cov = set()
for off, c, d, src in BANDS:
    idx = list(range(off, min(off + c, len(specs))))
    cov |= set(idx)
    ww = sum(w[i] for i in idx)
    sh = 100 * ww / W
    num += ww * d
    den += ww
    print(f"  off={off:<4} c={c:<3} {n[idx[0]]:>4}..{n[idx[-1]]:<4} "
          f"{sh:>7.2f}% {d:>+8.3f}%  {src}")

cw = sum(w[i] for i in cov)
print(f"\nmeasured coverage: {len(cov)}/{len(specs)} cases = "
      f"{100*cw/W:.2f}% of the s1 weighted score")
print(f"WEIGHT-WEIGHTED AGGREGATE over what was measured: {num/den:+.4f}%")

them = [b for b in BANDS if b[3] == "them"]
tn = td = 0.0
tc = set()
for off, c, d, _ in them + [(158, 16, 0.0, ""), (200, 20, -0.195, "")]:
    idx = list(range(off, min(off + c, len(specs))))
    tc |= set(idx)
    ww = sum(w[i] for i in idx)
    tn += ww * d
    td += ww
print(f"\nsame arithmetic on ONLY the 5 bands they ran: "
      f"coverage {100*sum(w[i] for i in tc)/W:.2f}%, aggregate {tn/td:+.4f}%")
print("  (their prose reads these as '-0.195% to -0.386% in every live band')")

print("\nthe two bands they never ran, and what they are worth:")
for off, c in ((180, 20), (220, 20)):
    idx = list(range(off, min(off + c, len(specs))))
    print(f"  off={off} n={n[idx[0]]}..{n[idx[-1]]}  "
          f"{100*sum(w[i] for i in idx)/W:>6.2f}% of s1")
