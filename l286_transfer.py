"""L286: does the in-set gain since M73/M74 show up on OOS?  The first OOS
measurement of the whole PACKAGE rather than of one mechanism.

Two caches share all 80 OOS case keys, so the comparison needs no placer runs:
  l252_cache.pkl          the shipped ladder's per-profile OOS layouts (PRE-LP)
  m67_oos_cache_c48.pkl   the M74-era wrapper's OOS costs (POST-LP, 41 profiles)

🚨 TWO SILENT FAILURES THIS SCRIPT EXISTS TO AVOID.

(1) PRE-LP vs POST-LP.  Comparing l252's records straight against m67's costs
    reads as a 2.1-2.6 % regression that is entirely the missing shape LP.  The
    LP is re-applied here before the comparison.  (Recomputing the l252 base with
    m67._cost reproduces L275's published 1.5116 / 1.4868 to six figures, which
    is what establishes that L275's row is the pre-LP one.)

(2) `import m67_oos_probe` STRIPS ICCAD_* FROM THE ENVIRONMENT, and
    `_shape_lp_maybe` never raises by design.  Setting ICCAD_SHAPE_LP before the
    import therefore produces a run where the LP is off, every case is returned
    unchanged, and the table prints happily with "LP moved 0/40".  The env is set
    AFTER the imports and the flag is ASSERTED live before the loop.
"""
import math
import os
import pickle
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import torch                                                      # noqa: E402
import m67_oos_probe as m67                                       # noqa: E402
import m77_oos_probe as m77                                       # noqa: E402
import optimizer_constructive as oc                               # noqa: E402

# --- MUST be after the imports above (they clear ICCAD_*) --------------------
os.environ["ICCAD_SHAPE_LP"] = "1"
os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
assert oc._shape_lp_on(), ("shape LP is off -- every case would be returned "
                           "unchanged and the table would still print")

RH = 1.4
l252 = pickle.load(open(DIR / "l252_cache.pkl", "rb"))
m67c = pickle.load(open(DIR / "m67_oos_cache_c48.pkl", "rb"))["cases"]


def pick(recs, sumA):
    """The deployed _RH=1.4 proxy, identical to l271_quality.pick."""
    idxs = sorted(recs)
    met = [recs[i] for i in idxs]
    A = 1.035 * max(sumA, 1e-9)
    hmin = min(m["hpwl"] for m in met) or 1.0
    pr = [(m["area"] / A + RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
          for m in met]
    return idxs[min(range(len(idxs)), key=lambda t: pr[t])]


print(f"{'':4}{'pre-LP':>12}{'post-LP':>12}{'M74-era':>12}"
      f"{'now vs M74':>13}{'LP moved':>10}")
for samp in ("s1", "s2"):
    spec = {ck: (fk, L, n) for ck, fk, L, n in m77._specs(samp)}
    loaded, a, b, c, den, nlp = {}, 0.0, 0.0, 0.0, 0.0, 0
    for ck in [k[1] for k in l252 if k[0] == samp]:
        e = l252[(samp, ck)]
        fk, L, n = spec[ck]
        if fk not in loaded:
            loaded.clear()
            loaded[fk] = torch.load(m67._path_of(fk))
        lay = m67._load_case(loaded[fk], L)
        lay["base"], _ = m67._baseline_official(lay)
        pos = e["recs"][pick(e["recs"], e["sumA"])]["pos"]
        margs = (lay["at"], lay["b2b"], lay["p2b"], lay["pins"], lay["cons"],
                 lay["n"])
        try:
            p2 = oc._shape_lp_maybe(pos, lay["n"], lay["at"], lay["b2b"],
                                    lay["p2b"], lay["pins"], lay["cons"], margs)
        except Exception:
            p2 = pos
        if p2 is not pos:
            nlp += 1
        w = math.exp(n / 12.0)
        den += w
        a += w * float(m67._cost(pos, lay).cost)
        b += w * float(m67._cost(p2, lay).cost)
        c += w * m67c[ck]["cost"]
    print(f"{samp:4}{a / den:>12.6f}{b / den:>12.6f}{c / den:>12.6f}"
          f"{100 * ((b / den) / (c / den) - 1):>12.2f}%{nlp:>7}/40")

print("\n  in-set 48c reference: M74 1.293461 -> now 1.226325 = -5.19 %")
print("  read this as 'the in-set gain is not visible on the OOS heavy band',")
print("  NOT as a transfer coefficient -- see L286 S4 for the four limits, and")
print("  L275 for why this band is the wrong predictor of the graded corpus.")
