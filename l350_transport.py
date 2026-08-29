"""L350 -- resolve L349's 2 pp bracket by testing the transport method's assumption.

THE BRACKET. Two constructions of "RF-SAFE on the hidden corpus" disagree by 2 pp and
straddle rank 1:

  (i)  graded corpus's own numbers x scalar DQ        0.871177   +1.461 % behind
  (ii) RF-SAFE's own numbers transported per-component
       by L348's measured per-band corpus ratios      0.853973   -0.543 % ahead

WHERE THE 2 pp COMES FROM -- worked out first, because it is not what I assumed. It is
NOT mainly the transfer coefficient (0.931 vs 1.0; that is worth only ~0.43 pp). It is
that the two constructions apply the improvement at different LEVELS:

  scalar DQ  : multiplies the whole per-case q = (1+G)*exp(2v) by (1+DQ)
  transport  : shrinks h, a and v INDIVIDUALLY by their measured factors

Those differ whenever the corpus's component MIX differs from validation's -- and it does:
the hidden corpus carries vrel 0.0425 against validation's 0.0241 (+76.5 %), while
RF-SAFE's single biggest relative win is exactly violations (-41.9 %). So component-wise
transport predicts a LARGER total improvement there (-7.6 %) than the scalar allows
(-6.2 %). That is a real mix effect a scalar cannot express, not an arithmetic slip.

=> the bracket reduces to ONE question: **does a mechanism's per-component relative effect
   survive a corpus change?** If yes, (ii) is the better construction and we may already be
   ahead. If the effect shrinks on a harder corpus, (i) stands.

THE TEST. Seven mechanisms were measured OFF and ON, each on BOTH OOS samples s1 and s2.
For each mechanism and each component x in {h, a, v}:

      rho_OFF = x_s2(off) / x_s1(off)          the corpus ratio measured on one arm
      rho_ON  = x_s2(on)  / x_s1(on)           the same ratio measured on the other

Transport assumes the corpus ratio is ARM-INDEPENDENT, i.e. rho_OFF == rho_ON. Any
systematic gap between them is exactly the error the bracket needs.

Equivalently and more directly: does the mechanism's relative effect hold across corpora?

      eff_s1 = x_s1(on)/x_s1(off)   vs   eff_s2 = x_s2(on)/x_s2(off)

HONEST LIMIT, stated up front: s1 and s2 are both drawn from floorset_lite and differ in
difficulty by only ~0.13 %, so this test has good power on ARM-INDEPENDENCE but weak power
on how the ratio behaves under a LARGE corpus shift. It bounds the error; it does not
extrapolate it.

Offline analysis. No solver runs, nothing on the shipping path.
"""
import json
import math
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent

# (label, s1-off, s1-on, s2-off, s2-on)
PAIRS = [
    ("l151 lp-gate", "l151_oos_s1_off.json", "l151_oos_s1_on.json",
     "l151_oos_s2_off.json", "l151_oos_s2_on.json"),
    ("l186 twins", "l186_s1_notwins.json", "l186_s1_twins.json",
     "l186_s2_notwins.json", "l186_s2_twins.json"),
    ("l192 thin-pool", "l192_s1_full.json", "l192_s1_thin.json",
     "l192_s2_full.json", "l192_s2_thin.json"),
    ("l213 refine-k8", "l213_s1_base.json", "l213_s1_k8.json",
     "l213_s2_base.json", "l213_s2_k8.json"),
    ("l223 r2/k8r2", "l223_s1_r2.json", "l223_s1_k8r2.json",
     "l223_s2_r2.json", "l223_s2_k8r2.json"),
    ("l243 devex", "l243_s1_base.json", "l243_s1_devex.json",
     "l243_s2_base.json", "l243_s2_devex.json"),
]


def comp(path):
    """weighted (hpwl_gap, area_gap, vrel, q) over a run."""
    tr = [t for t in json.load(open(DIR / path))["test_results"]
          if t.get("hpwl_gap") is not None]
    # OOS runs use n/vrel; in-set runs use block_count/violations_relative
    nk = "n" if "n" in tr[0] else "block_count"
    vk = "vrel" if "vrel" in tr[0] else "violations_relative"
    W = sum(math.exp(t[nk] / 12.0) for t in tr)

    def w(f):
        return sum(math.exp(t[nk] / 12.0) * f(t) for t in tr) / W
    h = w(lambda t: max(0.0, t["hpwl_gap"]))
    a = w(lambda t: max(0.0, t["area_gap"]))
    v = w(lambda t: t[vk])
    q = w(lambda t: (1 + 0.5 * (max(0.0, t["hpwl_gap"]) + max(0.0, t["area_gap"])))
          * math.exp(2.0 * t[vk]))
    return h, a, v, q


def main():
    print("== L350: does a mechanism's per-component effect survive a corpus change? ==")
    print("   (this is the single assumption the 2 pp bracket reduces to)")
    print()
    print("   %-16s %-6s %10s %10s %10s %10s"
          % ("mechanism", "comp", "eff on s1", "eff on s2", "drift", "|drift|"))
    rows = []
    for lab, f1o, f1n, f2o, f2n in PAIRS:
        try:
            o1, n1, o2, n2 = comp(f1o), comp(f1n), comp(f2o), comp(f2n)
        except FileNotFoundError as e:
            print("   %-16s (missing %s)" % (lab, e.filename))
            continue
        for i, cn in enumerate(("hpwl", "area", "vrel", "q")):
            if o1[i] <= 0 or o2[i] <= 0:
                continue
            e1, e2 = n1[i] / o1[i], n2[i] / o2[i]
            d = 100 * (e2 / e1 - 1)
            rows.append((lab, cn, e1, e2, d))
            print("   %-16s %-6s %10.4f %10.4f %+9.3f%% %9.3f"
                  % (lab, cn, e1, e2, d, abs(d)))
        print()

    print("=" * 78)
    print("HOW ARM-DEPENDENT IS THE CORPUS RATIO?")
    print("=" * 78)
    for cn in ("hpwl", "area", "vrel", "q"):
        ds = [r[4] for r in rows if r[1] == cn]
        if not ds:
            continue
        print("   %-6s  n=%d   mean drift %+7.3f %%   median %+7.3f %%   max|drift| %6.3f %%"
              % (cn, len(ds), statistics.mean(ds), statistics.median(ds),
                 max(abs(d) for d in ds)))
    allq = [r[4] for r in rows if r[1] == "q"]
    print()
    print("   Reading: 'drift' is how much a mechanism's relative effect changes between")
    print("   two corpora. drift ~ 0 => the corpus ratio is arm-independent => transport")
    print("   is sound => construction (ii). drift systematically < 0 on the harder corpus")
    print("   => effects shrink => construction (i).")
    print()
    if allq:
        mean_q = statistics.mean(allq)
        print("   measured mean drift on q: %+.3f %%   over %d mechanisms"
              % (mean_q, len(allq)))
        print("   s1->s2 difficulty gap is only ~0.13 %%, so this bounds arm-dependence")
        print("   but does NOT extrapolate to the validation->hidden shift (~+1.94 %%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
