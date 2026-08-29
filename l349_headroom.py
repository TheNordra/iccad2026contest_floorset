"""L349 -- where is the prize, on the FINAL corpus, with the corrected baseline?

Every "where is the headroom" statement in this project was made either on the in-set
(HANDOFF sec.2: hpwl_gap 0.2316 contributes 0.1158, 2.5x the area term) or on the graded
corpus with the stale D-arm baseline (L343/L347). L348 fixed the baseline (RF-SAFE, DQ
-5.769 %, projection 0.871174, gap +1.461 %) and measured something that changes the
framing: the hidden corpus is **easier than validation on geometry by 1.83 %** and harder
only on violations.

So redo the decomposition properly, on the graded corpus, with the corrected DQ, and ask
the only question that matters for choosing a research line:

    for each axis, what RELATIVE improvement is needed to close the +1.461 % gap?

and pair it with what the ledger has already measured about each axis's resistance.

TWO ESTIMATES, deliberately. The graded corpus's own per-case (h, a) are the BETA
package's. RF-SAFE changed the h:a mix. So the same decomposition is computed twice:
  (i)  graded corpus's own beta-era h/a, scaled by DQ
  (ii) RF-SAFE's validation h/a, rescaled by L348's MEASURED per-band geometry ratio
If the two agree, the answer is robust to which one you believe.

Offline analysis. No solver runs, nothing on the shipping path.
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
from l296_project import RANK1, graded  # noqa: E402

DQ = -0.05769                      # L348: RF-SAFE gain -6.199 % x transfer 0.931
BANDS = ((21, 50), (51, 80), (81, 100), (101, 120))


def band_of(n):
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return (lo, hi)
    return BANDS[-1]


def load_val(path):
    out = {}
    for t in json.load(open(path))["test_results"]:
        n = t.get("block_count")
        if n is None or t.get("hpwl_gap") is None:
            continue
        out[n] = dict(n=n, w=math.exp(n / 12.0),
                      h=max(0.0, t["hpwl_gap"]), a=max(0.0, t["area_gap"]),
                      v=t["violations_relative"])
    return out


def main():
    R = graded()

    # ---- (ii) rebuild the graded corpus's geometry from RF-SAFE's own numbers,
    #      rescaled by the per-band geometry ratio L348 measured -----------------
    V0 = load_val(DIR / "results_L160_m73_local.json")     # beta pkg, validation
    V1 = load_val(DIR / "l313_win48_rfsafe.json")          # RF-SAFE,  validation
    # per-band corpus ratios hidden/validation, measured with the SAME code (beta pkg).
    # These are properties of the corpora, so they transfer to any arm to first order.
    gh = {}
    for lo, hi in BANDS:
        sh = [r for r in R if lo <= r["n"] <= hi]
        sv = [V0[k] for k in V0 if lo <= k <= hi]
        if not sh or not sv:
            continue

        def wavg(s, f):
            return sum(x["w"] * f(x) for x in s) / sum(x["w"] for x in s)
        gh[(lo, hi)] = (wavg(sh, lambda x: x["h"]) / max(wavg(sv, lambda x: x["h"]), 1e-9),
                        wavg(sh, lambda x: x["a"]) / max(wavg(sv, lambda x: x["a"]), 1e-9),
                        wavg(sh, lambda x: x["v"]) / max(wavg(sv, lambda x: x["v"]), 1e-9))

    def parts(r, src):
        """(h, a, v) for this case under the chosen source, and the DQ that goes with it.

        beta   : the graded corpus's own beta-era numbers; DQ converts beta -> RF-SAFE.
        rfsafe : RF-SAFE's OWN validation numbers transported to the hidden corpus by the
                 measured per-band ratios. The improvement is already inside them, so
                 applying DQ as well would double-count it -- DQ = 0 here.
        """
        if src == "beta":
            return r["h"], r["a"], r["v"], DQ
        rb, ra, rv = gh[band_of(r["n"])]
        m = V1.get(r["n"])
        if not m:
            return r["h"], r["a"], r["v"], DQ
        return m["h"] * rb, m["a"] * ra, m["v"] * rv, 0.0

    def q(r, fh=1.0, fa=1.0, fv=1.0, src="beta"):
        h, a, v, dq = parts(r, src)
        return (1 + 0.5 * (h * fh + a * fa)) * math.exp(2 * v * fv) * (1 + dq)

    def tot(fh=1.0, fa=1.0, fv=1.0, src="beta"):
        return (sum(x["w"] * q(x, fh, fa, fv, src) * x["rf"] for x in R)
                / sum(x["w"] for x in R))

    print("== L349: where is the prize, on the FINAL corpus, corrected baseline ==")
    for src in ("beta", "rfsafe"):
        T0 = tot(src=src)
        need = 100 * (T0 / RANK1 - 1)
        W = sum(x["w"] for x in R)
        wh = sum(x["w"] * parts(x, src)[0] for x in R) / W
        wa = sum(x["w"] * parts(x, src)[1] for x in R) / W
        wv = sum(x["w"] * parts(x, src)[2] for x in R) / W
        print()
        print("-" * 76)
        print("SOURCE = %s   projection %.6f   gap to rank-1 %+.3f %%"
              % ("graded corpus's own beta-era h/a, DQ applied" if src == "beta"
                 else "RF-SAFE's own numbers transported by measured band ratios, DQ=0",
                 T0, need))
        print("-" * 76)
        print("   weighted hpwl_gap %.4f   area_gap %.4f   vrel %.4f" % (wh, wa, wv))
        print("   contribution to the score:  hpwl %.4f   area %.4f   violations %.4f"
              % (0.5 * wh, 0.5 * wa, math.exp(2 * wv) - 1))
        print()
        print("   %-26s %10s %10s %12s"
              % ("axis driven to ZERO", "score", "worth", "vs gap"))
        for lab, kw in (("hpwl_gap -> 0", dict(fh=0.0)),
                        ("area_gap -> 0", dict(fa=0.0)),
                        ("violations -> 0", dict(fv=0.0)),
                        ("hpwl + area -> 0", dict(fh=0.0, fa=0.0)),
                        ("everything -> 0", dict(fh=0.0, fa=0.0, fv=0.0))):
            t = tot(src=src, **kw)
            print("   %-26s %10.6f %+9.3f%% %11s"
                  % (lab, t, 100 * (t / T0 - 1),
                     "BEATS rank1" if t < RANK1 else "%+.3f%% short"
                     % (100 * (t / RANK1 - 1))))
        print()
        print("   RELATIVE improvement needed on each axis ALONE to close %+.3f %%:"
              % need)
        for lab, key in (("hpwl_gap", "fh"), ("area_gap", "fa"), ("violations", "fv")):
            lo, hi = 0.0, 1.0
            got = None
            for _ in range(60):
                mid = (lo + hi) / 2
                if tot(src=src, **{key: mid}) <= RANK1:
                    got = mid
                    lo = mid
                else:
                    hi = mid
            if got is None or tot(src=src, **{key: 0.0}) > RANK1:
                print("   %-14s  IMPOSSIBLE -- zeroing it still leaves %+.3f %%"
                      % (lab, 100 * (tot(src=src, **{key: 0.0}) / RANK1 - 1)))
            else:
                print("   %-14s  cut by **%.1f %%**  (factor %.3f)"
                      % (lab, 100 * (1 - got), got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
