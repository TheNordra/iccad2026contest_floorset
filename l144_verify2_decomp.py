"""L144 VERIFY2 - how concentrated is the reported -0.948%?

Per-case table transcribed verbatim from my own re-run of
  l144v3_ab.py --sample s1 --cases 16 --profile 0 --modes 0,1,2,3 --trace --percase
(cost_m0, cost_m1, n). Pure arithmetic, no solver.
"""
import math

# (case, n, cost_mode0, cost_mode1)
ROWS = [
    ("worker_0/layouts_2352/L104", 23, 3.0781, 3.0781),
    ("worker_0/layouts_4592/L102", 22, 2.3334, 2.3334),
    ("worker_1/layouts_1008/L64",  21, 2.8580, 2.8580),
    ("worker_1/layouts_2464/L94",  24, 1.8983, 1.8983),
    ("worker_3/layouts_6720/L2",   23, 1.7708, 1.7708),
    ("worker_3/layouts_6832/L72",  27, 3.0914, 3.0914),
    ("worker_4/layouts_8624/L92",  24, 2.2938, 2.2938),
    ("worker_4/layouts_9744/L46",  21, 2.0369, 2.0369),
    ("worker_6/layouts_4480/L89",  27, 1.9440, 1.9440),
    ("worker_7/layouts_2688/L50",  28, 1.4905, 1.4905),
    ("worker_7/layouts_5040/L67",  26, 1.7445, 1.9726),
    ("worker_7/layouts_8176/L77",  26, 2.4927, 2.4927),
    ("worker_8/layouts_6496/L67",  25, 1.7120, 1.7120),
    ("worker_8/layouts_7728/L102", 25, 2.8440, 2.8440),
    ("worker_9/layouts_7392/L80",  22, 2.2159, 2.3255),
    ("worker_9/layouts_8512/L82",  28, 2.4424, 2.4424),
]

W = sum(math.exp(n / 12.0) for _, n, _, _ in ROWS)
base = sum(math.exp(n / 12.0) * c0 for _, n, c0, _ in ROWS) / W
new = sum(math.exp(n / 12.0) * c1 for _, n, _, c1 in ROWS) / W
print(f"weighted mode0 = {base:.6f}   mode1 = {new:.6f}   "
      f"delta = {100*(base-new)/base:+.3f}%   (report: 2.250520 / 2.271854 / -0.948%)")

print("\nwho carries the delta:")
tot = 0.0
contrib = []
for ck, n, c0, c1 in ROWS:
    w = math.exp(n / 12.0)
    d = w * (c1 - c0)
    tot += d
    if abs(d) > 1e-9:
        contrib.append((d, ck, n, c0, c1))
for d, ck, n, c0, c1 in sorted(contrib, key=lambda r: -abs(r[0])):
    print(f"  {ck:>28} n={n:>3} w={math.exp(n/12.0):>7.3f} "
          f"{c0:.4f} -> {c1:.4f}  weighted d={d:>+8.4f}  "
          f"= {100*d/tot:>5.1f}% of the total regression")
print(f"  cases that moved: {len(contrib)}/{len(ROWS)}")

print("\nleave-one-out (drop the single worst case, recompute):")
for _, drop, _, _, _ in sorted(contrib, key=lambda r: -abs(r[0])):
    keep = [r for r in ROWS if r[0] != drop]
    w2 = sum(math.exp(n / 12.0) for _, n, _, _ in keep)
    b2 = sum(math.exp(n / 12.0) * c0 for _, n, c0, _ in keep) / w2
    n2 = sum(math.exp(n / 12.0) * c1 for _, n, _, c1 in keep) / w2
    print(f"  drop {drop:>28} -> delta {100*(b2-n2)/b2:+.3f}%")

print("\nthis 16-case slice as a share of the s1=240 weighted score:")
print(f"  sum(exp(n/12)) over these 16 = {W:.1f} vs s1 total 997738.3 "
      f"= {100*W/997738.3:.4f}%")
