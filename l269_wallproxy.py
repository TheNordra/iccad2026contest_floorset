"""L269 -- the CORRECTED deterministic wall proxy, and why the first one was wrong.

🚨 `l267_quality.py`'s `<arm>_pkmax` (max over profiles of pack count) is NOT a
proxy for a max-bound wall, and reading it as one understated `adapt` by 4x.
Measured here: the max-by-PACKS profile is prof 33, which is the 49th SLOWEST of
51 profiles (0.301 s against the max-setter's 0.794 s). A profile is slow because
its packs are expensive -- big REFINE band, costly knobs -- not because it does
many of them. Taking a max over the wrong axis picks a cheap profile that happens
to loop a lot.

The right deterministic proxy weights each profile's own MEASURED shipped time by
that profile's own pack ratio, and only then takes the max:

    est_t[p] = t_ship[p] * (packs_arm[p] / packs_ship[p])      <- per profile
    ratio    = max_p est_t[p] / max_p t_ship[p]

On `adapt` this reads 1.2310x against the stopwatch's 1.2417x (sum 1.1390x vs
1.1070x) -- i.e. the two instruments AGREE once the proxy is specified correctly,
and the disagreement was never evidence about the stopwatch.

The pack ratios come from the 40-case quality capture, so this proxy is far less
sample-limited than the 3-case stopwatch, and it is deterministic: it cannot move
between runs the way the max-setter's identity does (prof 100 / 93 / 7 / 40 / 3
across five captures).

  <python> l269_wallproxy.py <quality.pkl> <wall.pkl>
"""
import math
import pickle
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent
QP = DIR / (sys.argv[1] if len(sys.argv) > 1 else "l267_q40.pkl")
WP = DIR / (sys.argv[2] if len(sys.argv) > 2 else "l267_wall.pkl")

Q = pickle.load(open(QP, "rb"))
W = pickle.load(open(WP, "rb"))
per, arms = Q["per"], Q["arms"]
if "ship" not in arms:
    print("!! the quality capture needs a 'ship' arm to give per-profile pack baselines")
    sys.exit(1)
ship_t = W["ship"]
Wt = lambda n: math.exp(n / 12.0)

pk = {a: {} for a in arms}
for d in per:
    for a in arms:
        k = a + "_pk"
        if k in d:
            pk[a][d["prof"]] = pk[a].get(d["prof"], 0.0) + Wt(d["n"]) * d[k]

common = sorted(set(pk["ship"]) & set(ship_t))
mx = max(ship_t[p] for p in common)
tot = sum(ship_t[p] for p in common)
p_time = max(common, key=lambda p: ship_t[p])
p_pack = max(common, key=lambda p: pk["ship"][p])

print("[l269wp] {}  x  {}   {} profiles".format(QP.name, WP.name, len(common)))
print("  shipped max-setter BY TIME  = prof {} at {:.3f}s".format(p_time, mx))
print("  shipped max-setter BY PACKS = prof {} at {:.3f}s (time rank {} of {})".format(
    p_pack, ship_t[p_pack],
    sorted(common, key=lambda p: -ship_t[p]).index(p_pack) + 1, len(common)))
print("  => 'max pack count' is the wrong axis; see the module docstring.")
print()
print("  {:10s} {:>11s} {:>11s} {:>13s} {:>13s}".format(
    "arm", "est x max", "est x sum", "NET max-bound", "NET sum-bound"))
for a in arms:
    if a == "ship":
        continue
    est = {p: ship_t[p] * pk[a][p] / pk["ship"][p] for p in common if pk["ship"][p] > 0}
    rm, rs = max(est.values()) / mx, sum(est.values()) / tot
    # L248's conversion, as used by L257/L264: 0.151 pp of NET per 1% of heavy-band wall
    print("  {:10s} {:11.4f} {:11.4f} {:+12.3f}pp {:+12.3f}pp".format(
        a, rm, rs, -0.151 * 100 * (rm - 1.0), -0.151 * 100 * (rs - 1.0)))

print()
print("  cross-check against the stopwatch, where both exist:")
for a in arms:
    if a == "ship" or a not in W:
        continue
    est = {p: ship_t[p] * pk[a][p] / pk["ship"][p] for p in common if pk["ship"][p] > 0}
    sw = {p: W[a][p] for p in common if p in W[a]}
    if not sw:
        continue
    d = [W[a][p] / ship_t[p] - pk[a][p] / pk["ship"][p]
         for p in common if p in W[a] and pk["ship"][p] > 0]
    print("    {:10s} est max {:.4f}x   stopwatch max {:.4f}x   per-profile diff "
          "median {:+.3f} sd {:.3f}".format(
              a, max(est.values()) / mx, max(sw.values()) / mx,
              st.median(d), st.pstdev(d)))
