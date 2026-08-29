"""L252 step 0a -- is s_min a real cliff edge, or just the ladder's granularity?

l252_frames.py reports s_min = the tightest frame in the CANDIDATE LIST that packs.
That is an UPPER BOUND on the true cliff: if scale 1.05 fails and 1.10 succeeds, the
true edge is somewhere in (1.05, 1.10] and the ladder cannot see it. This prints, per
case, the largest FAILING s below s_min, so the unresolved interval is visible.
"""
import math
import pickle
from pathlib import Path

C = pickle.load(open(Path(__file__).parent / "l252_cache.pkl", "rb"))
RH = 1.0
try:
    import sys
    sys.argv = ["x"]
except Exception:
    pass

rows = []
for key, e in sorted(C.items(), key=lambda kv: -kv[1]["n"]):
    idxs = sorted(e["recs"])
    if len(idxs) < 2:
        continue
    met = [e["recs"][i] for i in idxs]
    A_hat = 1.035 * max(e["sumA"], 1e-9)
    hmin = min(m["hpwl"] for m in met) or 1.0
    prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
            for m in met]
    k = min(range(len(idxs)), key=lambda j: prox[j])
    w = met[k]
    tot = w["tot"] or e["sumA"]
    s_of = {i: math.sqrt(max(ww * hh, 1e-18) / max(tot, 1e-18))
            for i, ww, hh in w["frames"]}
    oks = [i for i, (ok, _s) in w["tries"].items() if ok]
    bad = [i for i, (ok, _s) in w["tries"].items() if not ok]
    if not oks:
        continue
    s_min = min(s_of[i] for i in oks)
    below = [s_of[i] for i in bad if s_of[i] < s_min - 1e-12]
    s_fail = max(below) if below else None
    s_low = min(s_of.values())
    rows.append((e["n"], s_low, s_fail, s_min, len(bad), len(w["frames"])))

print("{:>5s} {:>8s} {:>9s} {:>8s} {:>9s} {:>6s}".format(
    "n", "s_low", "s_failmax", "s_min", "unres", "nfail"))
unres = []
for n, s_low, s_fail, s_min, nb, nf in rows:
    g = (s_min - s_fail) if s_fail is not None else None
    unres.append(g if g is not None else 0.0)
    print("{:5d} {:8.4f} {:>9s} {:8.4f} {:>9s} {:6d}".format(
        n, s_low,
        "{:.4f}".format(s_fail) if s_fail is not None else "-",
        s_min,
        "{:.4f}".format(g) if g is not None else "none", nb))

nb0 = sum(1 for r in rows if r[2] is None)
print()
print("cases                                  {}".format(len(rows)))
print("cases where the tightest frame packed  {}  (no failure below s_min ->"
      " s_min is NOT a cliff, the ladder just never offered anything tighter)"
      .format(nb0))
print("cases with an unresolved interval      {}".format(len(rows) - nb0))
if len(rows) - nb0:
    g = [x for x in unres if x > 0]
    g.sort()
    print("unresolved width  min/med/max        {:.4f} / {:.4f} / {:.4f}".format(
        g[0], g[len(g) // 2], g[-1]))
