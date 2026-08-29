"""L323: do OUR layouts carry the label's structural invariant?

L322: 7050/7050 label blocks are bottom-supported (y=0 or abutting something below);
92.9 % are also left-supported. 100.0 % is an invariant, not a tendency -- the
generator finishes with a vertical compaction.

If our own layouts are NOT bottom-supported, every unsupported block has dead space
underneath it, and that dead space is area_gap we are paying for.
"""
import json
import sys


def ov(a, b):
    return (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
            and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])


def stats(rs, eps=1e-9):
    n = len(rs)
    down = left = 0
    slack = []
    for k, r in enumerate(rs):
        x, y, w, h = r
        if y <= eps:
            down += 1
            gap = 0.0
        else:
            below = [o[1] + o[3] for j, o in enumerate(rs) if j != k
                     and o[0] < x + w - eps and x < o[0] + o[2] - eps
                     and o[1] + o[3] <= y + eps]
            top = max(below) if below else 0.0
            gap = y - top
            if gap <= eps:
                down += 1
        slack.append(gap)
        if x <= eps or any(abs(o[0] + o[2] - x) <= eps for j, o in enumerate(rs) if j != k
                           and o[1] < y + h - eps and y < o[1] + o[3] - eps):
            left += 1
    return down, left, n, slack


j = json.load(open(sys.argv[1]))
tD = tL = tN = 0
wasted = []
per = []
for t in j["test_results"]:
    p = t.get("positions")
    if not p:
        continue
    rs = [tuple(map(float, r)) for r in p]
    d, l, n, slack = stats(rs)
    tD += d; tL += l; tN += n
    per.append(d / n)
    W = max(r[0] + r[2] for r in rs) - min(r[0] for r in rs)
    H = max(r[1] + r[3] for r in rs) - min(r[1] for r in rs)
    used = sum(r[2] * r[3] for r in rs)
    # area that could be reclaimed if every block slid down onto its support
    reclaim = sum(s * r[2] for s, r in zip(slack, rs))
    wasted.append((used / (W * H), reclaim / (W * H)))
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print("== L323 %s : %d blocks ==" % (sys.argv[1], tN))
print("   bottom-supported : %5d/%d  (%.1f %%)      LABEL: 100.0 %%" % (tD, tN, 100 * tD / tN))
print("   left-supported   : %5d/%d  (%.1f %%)      LABEL:  92.9 %%" % (tL, tN, 100 * tL / tN))
print("   per-case bottom-supported fraction: p10 %.3f  p50 %.3f  p90 %.3f"
      % (q(per, .1), q(per, .5), q(per, .9)))
u = [a for a, _b in wasted]; r = [b for _a, b in wasted]
print("   utilisation      : p50 %.4f                LABEL p50: 0.9693" % q(u, .5))
print("   area under unsupported blocks (as %% of bbox): p50 %.2f %%  p90 %.2f %%"
      % (100 * q(r, .5), 100 * q(r, .9)))
