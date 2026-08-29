"""L281 corpus check (L275 rule (a) / L280 rule (c)): is the critical-chain
saturation that blocks relocation a property of the GRADED corpus only, or of
the mechanism?

The certificate's binding term is the longest node-weighted chain in the
block-level horizontal / vertical constraint graph versus the bbox row.  That
quantity needs only POSITIONS, so it can be measured on the OOS heavy band
(l252_cache.pkl) with no dataset metadata at all -- and on the in-set 100 the
same way, so the two are apples to apples.

Every block is its own node here (no cluster / frozen distinction), on BOTH
corpora, precisely so the comparison is like for like.
"""
import json
import math
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import l281_reloc_probe as L                                       # noqa: E402

RH = 1.4


def slack_of(pos):
    """(H slack, V slack) for a placement given as [(x, y, w, h), ...]."""
    n = len(pos)
    EH, EV = [], []
    for i in range(n):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = pos[j]
            g = (xj - (xi + wi), xi - (xj + wj),
                 yj - (yi + hi), yi - (yj + hj))
            k = 0
            for t in (1, 2, 3):
                if g[t] > g[k]:
                    k = t
            if k == 0:
                EH.append((i, j))
            elif k == 1:
                EH.append((j, i))
            elif k == 2:
                EV.append((i, j))
            else:
                EV.append((j, i))
    okH, lH = L.longest_chain(n, EH, [p[2] for p in pos])
    okV, lV = L.longest_chain(n, EV, [p[3] for p in pos])
    W0 = max(p[0] + p[2] for p in pos) - min(p[0] for p in pos)
    H0 = max(p[1] + p[3] for p in pos) - min(p[1] for p in pos)
    if not okH or not okV:
        return None
    return 1.0 - lH / W0, 1.0 - lV / H0


def summarise(tag, rows):
    if not rows:
        print(f"{tag}: no rows")
        return
    sH = sorted(r[0] for r in rows)
    sV = sorted(r[1] for r in rows)
    mn = sorted(min(r[0], r[1]) for r in rows)

    def q(a, f):
        return 100.0 * a[min(int(f * len(a)), len(a) - 1)]

    print(f"\n== {tag}  ({len(rows)} placements) ==")
    print(f"  H slack   p25 {q(sH, .25):7.4f}%  p50 {q(sH, .5):7.4f}%  "
          f"p75 {q(sH, .75):7.4f}%")
    print(f"  V slack   p25 {q(sV, .25):7.4f}%  p50 {q(sV, .5):7.4f}%  "
          f"p75 {q(sV, .75):7.4f}%")
    print(f"  min(H,V)  p25 {q(mn, .25):7.4f}%  p50 {q(mn, .5):7.4f}%  "
          f"p75 {q(mn, .75):7.4f}%")
    for thr in (1e-9, 1e-3, 1e-2):
        nb = sum(1 for v in mn if v <= thr)
        print(f"  saturated (min slack <= {100 * thr:7.4f}%): "
              f"{nb}/{len(mn)} = {100.0 * nb / len(mn):.1f} %")


# -- in-set 100, the graded shape -------------------------------------------
aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
inset = []
for t in aj["test_results"]:
    r = slack_of([tuple(p) for p in t["positions"]])
    if r:
        inset.append(r)
summarise("IN-SET 100 (graded shape, shipped anchor)", inset)
heavy = []
for t in aj["test_results"]:
    if t["block_count"] >= 101:
        r = slack_of([tuple(p) for p in t["positions"]])
        if r:
            heavy.append(r)
summarise("IN-SET heavy 20 (n >= 101)", heavy)

# -- OOS heavy band, proxy-selected layout, same construction ---------------
C = pickle.load(open(_DIR / "l252_cache.pkl", "rb"))
for sample in ("s1", "s2"):
    keys = sorted([k for k in C if k[0] == sample], key=lambda k: -C[k]["n"])
    rows = []
    for key in keys:
        e = C[key]
        idxs = sorted(e["recs"])
        met = [e["recs"][i] for i in idxs]
        A_hat = 1.035 * max(e["sumA"], 1e-9)
        hmin = min(m["hpwl"] for m in met) or 1.0
        prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin)
                * math.exp(2.0 * m["vrel"]) for m in met]
        win = idxs[min(range(len(idxs)), key=lambda t: prox[t])]
        r = slack_of([tuple(p) for p in e["recs"][win]["pos"]])
        if r:
            rows.append(r)
    summarise(f"OOS heavy 40, sample {sample} (proxy-selected, n>=101)", rows)
