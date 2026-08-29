"""L348 -- measure the quantity every graded projection rests on.

THE METHOD, from L347: when an inference is load-bearing, go and measure the quantity it
depends on rather than being more careful with the same data.

WHAT IS STILL INFERRED. `l296_project.project()` applies a scalar `DQ_SHIP = -0.0497` to
every graded case: "93 % transfer of the in-set -5.34 % since M73". Every number this
session has produced sits on top of it -- L343's baseline, L347's recount, the "+2.321 %
behind rank 1". It is a transfer ASSUMPTION with two parts, and one of them is measurable
right now with no new runs:

  (a) CORPUS DIFFICULTY -- how much harder is the hidden set than validation, for the
      SAME code?  MEASURABLE: the beta package ran on both.
        beta package on hidden   = beta_evaluation_results.json, raw (RF-free) 1.32066494
        beta package on val @48c = results_L160_m73_local.json,   RF forced 1.0, 1.29554782
      Both are  sum w*(1+0.5(h+a))*exp(2*vrel) / sum w. Directly comparable.

  (b) IMPROVEMENT TRANSFER -- how much of what we gained since M73 shows up there. NOT
      directly measurable (we cannot run the new code on the hidden set). But (a) tells us
      whether the two corpora differ UNIFORMLY or by band, and our gains since M73 are
      measurable per band on validation. If the gains land in bands where the corpora
      differ most, a scalar transfer is biased and the bias is computable.

GATE. Each input must reproduce its own published/recorded total from the per-case rows
before any of it is used.

Offline analysis. No solver runs, no training, nothing on the shipping path.
"""
import json
import math
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
from l296_project import DQ_SHIP, RANK1, graded  # noqa: E402

HIDDEN = DIR / "beta_2026-08-16" / "beta_evaluation_results.json"
VAL_M73 = DIR / "results_L160_m73_local.json"      # beta package, validation, 48 cores
VAL_NOW = DIR / "l313_win48_rfsafe.json"           # current RF-SAFE, validation, 48 cores
RAW_HIDDEN = 1.3206649447461245                    # published beta raw (RF-free)
TOT_M73_48 = 1.295547821428148                     # recorded M73 @48c validation anchor
TOT_NOW_48 = 1.2152391322773048                    # recorded RF-SAFE @48c validation


def rows(path):
    out = {}
    for t in json.load(open(path))["test_results"]:
        n = t.get("block_count")
        if n is None or t.get("hpwl_gap") is None:
            continue
        q = (1 + 0.5 * (max(0.0, t["hpwl_gap"]) + max(0.0, t["area_gap"]))) \
            * math.exp(2.0 * t["violations_relative"])
        out[n] = dict(n=n, w=math.exp(n / 12.0), q=q,
                      g=0.5 * (max(0.0, t["hpwl_gap"]) + max(0.0, t["area_gap"])),
                      v=t["violations_relative"])
    return out


def wtot(d):
    return (sum(r["w"] * r["q"] for r in d.values())
            / sum(r["w"] for r in d.values()))


BANDS = ((21, 50), (51, 80), (81, 100), (101, 120))


def band_of(n):
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return (lo, hi)
    return None


def bw(d, lo, hi, f=lambda r: r["q"]):
    sel = [r for r in d.values() if lo <= r["n"] <= hi]
    if not sel:
        return float("nan")
    return sum(r["w"] * f(r) for r in sel) / sum(r["w"] for r in sel)


def main():
    H, V0, V1 = rows(HIDDEN), rows(VAL_M73), rows(VAL_NOW)
    print("== L348: measuring what DQ_SHIP assumes ==")
    print()
    print("GATE -- each input must reproduce its own recorded total")
    for lab, d, ref in (("beta pkg on HIDDEN (raw)", H, RAW_HIDDEN),
                        ("beta pkg on VAL @48c", V0, TOT_M73_48),
                        ("RF-SAFE on VAL @48c", V1, TOT_NOW_48)):
        t = wtot(d)
        ok = abs(t - ref) < 1e-6
        print("   %-26s %d cases  %.9f  vs recorded %.9f   %s"
              % (lab, len(d), t, ref, "PASS" if ok else "*** FAIL ***"))
        if not ok:
            print("   -> input does not reproduce its anchor; stopping.")
            return 1
    print()

    # ---- (a) corpus difficulty, MEASURED ----------------------------------
    th, tv = wtot(H), wtot(V0)
    print("=" * 78)
    print("(a) CORPUS DIFFICULTY -- same code, two corpora, both RF-free")
    print("=" * 78)
    print("   hidden %.9f / validation %.9f  =  **%+.3f %%**"
          % (th, tv, 100 * (th / tv - 1)))
    print()
    print("   %12s %8s %10s %10s %9s | %9s %9s"
          % ("band", "w-share", "val", "hidden", "harder", "val vrel", "hid vrel"))
    for lo, hi in BANDS:
        a, b = bw(V0, lo, hi), bw(H, lo, hi)
        ws = sum(r["w"] for r in H.values() if lo <= r["n"] <= hi) \
            / sum(r["w"] for r in H.values())
        print("   %12s %7.1f%% %10.4f %10.4f %+8.2f%% | %9.4f %9.4f"
              % ("%d-%d" % (lo, hi), 100 * ws, a, b, 100 * (b / a - 1),
                 bw(V0, lo, hi, lambda r: r["v"]), bw(H, lo, hi, lambda r: r["v"])))
    print()
    print("   geometry only (drop exp(2*vrel)):  hidden %.4f / val %.4f = %+.3f %%"
          % (bw(H, 21, 120, lambda r: 1 + r["g"]), bw(V0, 21, 120, lambda r: 1 + r["g"]),
             100 * (bw(H, 21, 120, lambda r: 1 + r["g"])
                    / bw(V0, 21, 120, lambda r: 1 + r["g"]) - 1)))
    print("   => the corpora differ almost entirely on VIOLATIONS, not geometry.")
    print()

    # ---- (b) where our gains since M73 actually sit ------------------------
    print("=" * 78)
    print("(b) OUR GAIN SINCE M73, per band, measured on validation @48c")
    print("=" * 78)
    print("   in-set total %.9f -> %.9f  = %+.3f %%"
          % (tv, wtot(V1), 100 * (wtot(V1) / tv - 1)))
    print()
    print("   %12s %8s %10s %10s %9s %11s"
          % ("band", "w-share", "M73", "RF-SAFE", "gain", "corpus diff"))
    gains, diffs, wts = [], [], []
    for lo, hi in BANDS:
        a, b = bw(V0, lo, hi), bw(V1, lo, hi)
        d = 100 * (bw(H, lo, hi) / bw(V0, lo, hi) - 1)
        g = 100 * (b / a - 1)
        ws = sum(r["w"] for r in H.values() if lo <= r["n"] <= hi) \
            / sum(r["w"] for r in H.values())
        gains.append(g); diffs.append(d); wts.append(ws)
        print("   %12s %7.1f%% %10.4f %10.4f %+8.3f%% %+10.2f%%"
              % ("%d-%d" % (lo, hi), 100 * ws, a, b, g, d))
    print()

    # ---- the cross term ----------------------------------------------------
    print("=" * 78)
    print("IS A SCALAR DQ_SHIP DEFENSIBLE?")
    print("=" * 78)
    scal = sum(w * g for w, g in zip(wts, gains))
    print("   weighted mean gain (= what a scalar uses)   %+.3f %%" % scal)
    print("   gain spread across bands                     %.3f pp (min %+.3f, max %+.3f)"
          % (max(gains) - min(gains), min(gains), max(gains)))
    print("   corpus-difficulty spread across bands        %.2f pp (min %+.2f, max %+.2f)"
          % (max(diffs) - min(diffs), min(diffs), max(diffs)))
    print()
    print("   DQ_SHIP in l296_project                      %+.3f %%" % (100 * DQ_SHIP))
    print("   in-set gain since M73 (measured here)        %+.3f %%"
          % (100 * (wtot(V1) / tv - 1)))
    print("   implied transfer coefficient it assumes      %.3f"
          % (DQ_SHIP / (wtot(V1) / tv - 1)))
    print()

    # ---- what a band-wise DQ does to the projection ------------------------
    R = graded()
    dq_band = {}
    for (lo, hi), g in zip(BANDS, gains):
        dq_band[(lo, hi)] = (g / 100.0) * (DQ_SHIP / (wtot(V1) / tv - 1))

    def total(dq):
        num = den = 0.0
        for r in R:
            d = dq if isinstance(dq, float) else dq[band_of(r["n"])]
            q = (1 + 0.5 * (r["h"] + r["a"])) * math.exp(2.0 * r["v"]) * (1 + d)
            num += r["w"] * q * r["rf"]
            den += r["w"]
        return num / den
    t_scalar, t_band = total(DQ_SHIP), total(dq_band)
    print("=" * 78)
    print("PROJECTION: scalar DQ vs band-wise DQ (same transfer coefficient)")
    print("=" * 78)
    print("   scalar     %.6f   gap to rank-1 %+.3f %%"
          % (t_scalar, 100 * (t_scalar / RANK1 - 1)))
    print("   band-wise  %.6f   gap to rank-1 %+.3f %%   (moves %+.4f pp)"
          % (t_band, 100 * (t_band / RANK1 - 1), 100 * (t_band - t_scalar) / t_scalar))
    print()
    print("   band DQ used:", {("%d-%d" % k): round(100 * v, 3) for k, v in dq_band.items()})
    print()

    # ---- DQ_SHIP is STALE: it models D, not RF-SAFE ------------------------
    coef = DQ_SHIP / -0.0534               # the transfer coefficient it was built with
    gain_now = wtot(V1) / tv - 1
    dq_now = coef * gain_now
    print("=" * 78)
    print("DQ_SHIP IS STALE -- it was built from a -5.34 % in-set gain")
    print("=" * 78)
    print("   -5.34 %% is D's in-set gain over M73 (1.226325 / 1.295548 - 1 = %+.3f %%)."
          % (100 * (1.226325126 / TOT_M73_48 - 1)))
    print("   RF-SAFE's in-set gain over M73, measured above: %+.3f %%" % (100 * gain_now))
    print("   transfer coefficient DQ_SHIP encodes: %.3f  (docstring says 0.93)" % coef)
    print()
    for lab, dq in (("DQ_SHIP as-is (= the D arm)", DQ_SHIP),
                    ("same coefficient, RF-SAFE gain", dq_now)):
        t = total(dq)
        print("   %-32s DQ %+7.3f %%  ->  %.6f   gap to rank-1 %+.3f %%"
              % (lab, 100 * dq, t, 100 * (t / RANK1 - 1)))
    print()
    print("   SHIP_DECISION puts D+RF-SAFE at 0.86726-0.86994 by an independent route;")
    print("   this lands at %.5f, i.e. the two methods agree to within %.2f pp."
          % (total(dq_now), 100 * abs(total(dq_now) / 0.8686 - 1)))
    print()
    print("   => every graded projection this session quoted (+2.321 %% behind rank 1)")
    print("      was the **D** arm. For what we would actually ship it is %+.3f %%."
          % (100 * (total(dq_now) / RANK1 - 1)))

    # ---- recount the violation prize against the corrected gap -------------
    T0 = total(dq_now)
    need = 100 * (T0 / RANK1 - 1)
    try:
        import pickle
        law = pickle.load(open(DIR / "l347_nslaw.pkl", "rb"))
    except Exception:
        law = None
    if law:
        for r in R:
            r["k"] = 1
            if r["V"] > 0 and law.get(r["n"]):
                h = law[r["n"]]
                ks = [k for k in range(1, 16) if min(h) <= k * r["NS"] <= max(h)]
                if ks:
                    med = statistics.median(sorted(h))
                    r["k"] = min(ks, key=lambda k: abs(k * r["NS"] - med))

        def q1(r, dV=0):
            V = max(0, r["k"] * r["V"] + dV)
            return ((1 + 0.5 * (r["h"] + r["a"]))
                    * math.exp(2.0 * V / (r["k"] * r["NS"])) * (1 + dq_now))

        def tot1(mod=None):
            return (sum(r["w"] * (mod(r) if mod else q1(r)) * r["rf"] for r in R)
                    / sum(r["w"] for r in R))
        base = tot1()
        for r in R:
            r["s"] = (100 * (1 - tot1(lambda x, t=r: q1(x, -1 if x is t else 0)) / base)
                      if r["V"] > 0 else 0.0)
        hit = sorted([r for r in R if r["V"] > 0], key=lambda r: -r["s"])

        def joint(k):
            top = {id(x) for x in hit[:k]}
            return 100 * (1 - tot1(lambda x: q1(x, -1 if id(x) in top else 0)) / base)
        kneed = next((k for k in range(1, len(hit) + 1) if joint(k) >= need), None)
        print()
        print("   violation recount against the CORRECTED gap of %+.3f %%: **%s violations**"
              % (need, kneed))
        print("   (L343 said 2; L347 corrected to 7 against the stale +2.32 %% gap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
