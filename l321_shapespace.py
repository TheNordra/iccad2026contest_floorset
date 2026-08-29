"""L321: the label's shape space is DISCRETE and tiny -- verify it.

L320 found: every label coordinate is an integer, every label block's area equals
its target EXACTLY (not within the 1 % tolerance the contest allows), block aspect
ratio maxes out at exactly 3.00, utilisation p50 is 96.9 %, and 0/19 sampled labels
are guillotine/slicing.

If w*h = A exactly with w,h positive integers and 1/3 <= w/h <= 3, then each block's
shape is drawn from the DIVISOR PAIRS of A inside the aspect band -- a handful of
options, not a continuum. The contest lets us use any real w,h within 1 % of A; the
label used a far smaller space. That asymmetry is free information.
"""
import glob
import sys
from collections import Counter

import torch

LAB = sorted(glob.glob("LiteTensorDataTest/config_*/litelabel_1.pth"))
DAT = sorted(glob.glob("LiteTensorDataTest/config_*/litedata_1.pth"))


def case(i):
    metrics, poly = torch.load(LAB[i], weights_only=False)[0]
    d = torch.load(DAT[i], weights_only=False)[0][0]
    n = int((d[:, 0] > 0).sum())
    rs = []
    for k in range(n):
        p = poly[k]
        v = p[p[:, 0] != -1]
        x0, y0 = v.min(dim=0).values.tolist()
        x1, y1 = v.max(dim=0).values.tolist()
        rs.append((x0, y0, x1 - x0, y1 - y0))
    return d[:n], rs, metrics


def divpairs(A, lo=1.0 / 3, hi=3.0):
    A = int(round(A))
    out = []
    w = 1
    while w * w <= A:
        if A % w == 0:
            for a, b in ((w, A // w), (A // w, w)):
                if lo - 1e-12 <= a / b <= hi + 1e-12:
                    out.append((a, b))
        w += 1
    return sorted(set(out))


N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
exact = tot = 0
nopt = []
inband = 0
prime_like = 0
free_ratio = []
for i in range(N):
    d, rs, metrics = case(i)
    for k, (x, y, w, h) in enumerate(rs):
        A = float(d[k, 0])
        tot += 1
        exact += (abs(w * h - A) < 1e-9)
        opts = divpairs(A)
        nopt.append(len(opts))
        if not opts:
            prime_like += 1
        inband += ((int(w), int(h)) in opts)
        # how much smaller is the label's space than the contest's?
        # contest: any real (w,h) with |wh - A| <= 1% -> a continuum
        free_ratio.append(len(opts))

q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print("== L321 the label's shape space, %d cases, %d blocks ==" % (N, tot))
print("   area EXACTLY equals target        : %d/%d (%.1f %%)" % (exact, tot, 100 * exact / tot))
print("   label (w,h) is an integer divisor")
print("   pair of A inside aspect [1/3, 3]  : %d/%d (%.1f %%)" % (inband, tot, 100 * inband / tot))
print("   number of such pairs per block    : p10 %d  p50 %d  p90 %d  max %d  mean %.1f"
      % (q(nopt, .1), q(nopt, .5), q(nopt, .9), max(nopt), sum(nopt) / len(nopt)))
print("   blocks with NO legal pair         : %d" % prime_like)
print("   => the generator picked from a MEAN OF %.1f discrete shapes per block;" % (sum(nopt) / len(nopt)))
print("      we are searching a continuum with a 1 %% area band.")
c = Counter(nopt)
print("   distribution of options/block     : %s"
      % dict(sorted(c.items())[:10]))
