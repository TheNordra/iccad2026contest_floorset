"""L309 - median-drift sensitivity for the Final round (answers HackMD review B).

The reviewer's point: `runtime_factor`'s denominator is a CROSS-SUBMISSION
median, recomputed every round (`iccad2026_evaluate.py:552`), so every RF-priced
decision in this tree (`_M49_REFINE_BAND` 2/2, `_L196_LPGATE` 71, the L211 pool
drop) is anchored on a number that will not be the same at Final. Nothing in the
tree handled that. This does.

WHAT IS NEW HERE, over l146_rf_price.py

  1. l146 prices a mechanism as a runtime MULTIPLIER applied to the beta rows.
     This projects MEASURED per-case arms instead, via a per-case bridge, so the
     machine-speed factor f cancels case by case instead of being assumed.
  2. Both published median vintages are used (2026-08-16 and the 2026-08-23
     re-publish), so the drift is MEASURED, not hypothesised.
  3. Two independent identity tests, not one (see VALIDATION).

THE BRIDGE.  The beta hidden set and the local in-set share test_id ->
block_count exactly (100/100, checked at run time below), so for every arm A:

    q_i^proj = q_i^beta * (q_i^A,local / q_i^betacfg,local)
    t_i^proj = t_i^beta * (t_i^A,local / t_i^betacfg,local)

Both ratios are same-machine, same-case, so the machine-speed factor f divides
out per case rather than being calibrated once and applied everywhere. That
matters: L308 measured f at 2.38-2.84 and it is band-dependent, so a single
global f would inject up to a 1.19x error exactly on the heavy cases that carry
the weight.

🔑 f AND m ARE CONFOUNDED.  RF_i depends on t and M only through

    R_i = t_i^local / (f * m * M_i)

so a global median scale m and a machine-speed error f enter as the PRODUCT.
One axis covers both. That is why the table below runs to m = 0.8 and 1.3 even
though the reviewer only expects the medians to move: an f that is 20% off
looks exactly like an m that is 20% off, and cannot be told apart from inside.

VALIDATION -- and what it does NOT cover

    arm = betacfg, m = 1, medians 2026-08-16  ->  0.9245183669982832 (published)
    arm = betacfg, m = 1, medians 2026-08-23  ->  0.9265861161320369 (published)
    plus the discrete check: exactly ONE case off the RF floor on 08-16 medians
    (test_id 66, n=87 -- the single case that makes cwRF 0.70004 and not 0.70)

🚨 Be precise about what that proves. For the CONTROL arm the bridge collapses
to the identity (q^proj = q^beta * 1, t^proj = t^beta * 1), so these tests
validate the SCORING PIPELINE -- weights, the RF formula, the median vectors,
the published totals, on two independent vintages -- and say nothing at all
about the bridge. l146_rf_price.py's own docstring warns about exactly this
shape of self-congratulation; it applies here too.

What the bridge rests on instead is `l285_betacfg.json` standing in for the
submitted beta package. It does not stand in perfectly: that arm is "M73-LIKE",
rebuilt from the shipped code's kill switches, so it still carries the L131/L136
correctness fixes the real beta submission did not have. Its local quality
(1.2599) is therefore BETTER than the beta package's true quality, which makes
every arm's projected quality gain over beta too SMALL.

Consequence, and it is the thing to read this file for:

  * absolute totals and the implied rank column are CONSERVATIVE. This file
    projects D to ~0.900 / rank 4 where HANDOFF_2026-08-29 projects 0.87511 /
    rank 2. The gap is the quality leg, not the RF leg, and this file is the
    pessimistic end of it. Do not re-plan the ranking off this column.
  * the NET-vs-D table is ROBUST, because every arm shares the same bridge
    denominator, so the bias cancels in the difference. That column is the
    answer to the reviewer's question and is what the decision rule uses.

Run:  <python> l309_median_sensitivity.py
"""
import argparse
import csv
import json
import math
import os
import statistics as st
from pathlib import Path

_DIR = Path(__file__).parent

# C: no hardcoded home directories. Repo copies first, then the historical
# Downloads location, then $ICCAD_MEDIAN_CSV. Same fix as l146_rf_price.py.
_MED_16 = ("C_median_runtimes_beta_hidden.csv",
           "beta_2026-08-16/C_median_runtimes_beta_hidden.csv")
_MED_23 = ("beta_2026-08-23/C_median_runtimes_beta_hidden_update.csv",)

PUBLISHED = {"2026-08-16": 0.9245183669982832,
             "2026-08-23": 0.9265861161320369}

# 2026-08-23 leaderboard (C_beta_leaderboard_update_20260823.csv)
RANKS = [("rank 1", 0.8586322662042342), ("rank 2", 0.888187391),
         ("rank 3", 0.8993286931994098), ("rank 4", 0.9265861161320369)]


def _find(cands, env=None):
    if env and os.environ.get(env):
        return Path(os.environ[env])
    for c in cands:
        p = _DIR / c
        if p.exists():
            return p
    home = Path.home() / "Downloads" / Path(cands[0]).name
    if home.exists():
        return home
    raise SystemExit(f"missing median csv; tried {cands} and {home}")


def medians(which):
    p = _find(_MED_16 if which == "2026-08-16" else _MED_23,
              "ICCAD_MEDIAN_CSV" if which == "2026-08-16" else None)
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            keys = list(row)
            out[int(row[keys[0]])] = float(row["median_runtime_s"])
    return out, p


def results(name):
    return {r["test_id"]: r for r in json.load(open(_DIR / name))["test_results"]}


def repeats(*names):
    """Per-CASE min runtime across repeated runs of the same arm.

    L296: min-of-N belongs on the WORK UNIT, not the arm -- taking the min of
    arm totals lets one noisy case decide, and the noisiest unit is usually the
    baseline. Costs must be bit-identical across repeats; if they are not, the
    runs are not the same arm and the assert says so. That check is free and it
    is also a determinism gate.
    """
    runs = [results(n) for n in names]
    base = runs[0]
    for k, r in enumerate(runs[1:], 1):
        bad = [i for i in base if r[i]["cost"] != base[i]["cost"]]
        assert not bad, f"{names[k]} differs from {names[0]} in cost on {bad[:5]}"
    return {i: dict(base[i],
                    runtime_seconds=min(r[i]["runtime_seconds"] for r in runs))
            for i in base}


def compose(parts):
    """Per-case pick-by-band. Exact, not an approximation: L238 verified each
    REFINE kill switch moves only its own band, and this run re-checks it."""
    base = results(parts["base"])
    out = dict(base)
    for fname, lo, hi in parts["bands"]:
        arm = results(fname)
        for i, r in arm.items():
            if lo < r["block_count"] <= hi:
                out[i] = r
    return out


def project(arm_local, beta, ref_local):
    """beta-grader rows re-scaled by the same-machine arm/reference ratios."""
    rows = []
    for i, b in beta.items():
        a, ref = arm_local[i], ref_local[i]
        rows.append(dict(
            i=i, n=b["block_count"], w=math.exp(b["block_count"] / 12.0),
            q=b["cost"] * (a["cost"] / ref["cost"]),
            t=b["runtime_seconds"] * (a["runtime_seconds"] / ref["runtime_seconds"]),
        ))
    return rows


def score(rows, M, m):
    num = den = qw = 0.0
    off = 0
    for r in rows:
        rf = max(0.7, (r["t"] / (m * M[r["i"]])) ** 0.3)
        if rf > 0.7:
            off += 1
        num += r["w"] * r["q"] * rf
        den += r["w"]
        qw += r["w"] * r["q"]
    return num / den, num / qw, off          # total, cost-weighted RF, off-floor


def rank_of(total):
    better = sum(1 for _, t in RANKS if t < total)
    return better + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintage", default="2026-08-23",
                    choices=["2026-08-16", "2026-08-23"],
                    help="which published median vector to scale (default: the "
                         "latest, i.e. the best estimate of Final)")
    a = ap.parse_args()

    beta = results("beta_2026-08-16/beta_evaluation_results.json")
    # per-case min over every repeat we have of each arm (see repeats())
    ref = repeats("l285_betacfg.json", "l285_betacfg_r2.json")
    D = repeats("results_L237_post.json", "l285_ship_r2.json", "l285_lp_on.json",
                "l302_ship_1.json", "l302_ship_2.json")
    MIX = repeats("l302_mix_1.json", "l302_mix_2.json", "l302_mix_3.json",
                  "l302_mix_4.json", "l303_mixpkg_c48.json")

    # ---- corpus check: this is the assumption the whole bridge rests on -----
    assert set(beta) == set(D) == set(ref), "test_id sets differ"
    bad = [i for i in beta if beta[i]["block_count"] != D[i]["block_count"]]
    assert not bad, f"block_count mismatch on {bad[:5]}"
    print(f"corpus check: test_id -> block_count identical on "
          f"{len(beta)}/{len(beta)} cases (beta hidden vs local in-set)\n")

    # ---- band-disjointness check for the composed REFINE arm ---------------
    H, Mi = results("results_L238_refine4.json"), results("results_L238_refinemid6.json")
    mh = [i for i in D if H[i]["cost"] != D[i]["cost"]]
    mm = [i for i in D if Mi[i]["cost"] != D[i]["cost"]]
    assert all(D[i]["block_count"] > 100 for i in mh), "heavy knob left its band"
    assert all(60 < D[i]["block_count"] <= 100 for i in mm), "mid knob left its band"
    assert not (set(mh) & set(mm)), "REFINE knobs overlap"
    print(f"band check: heavy moves {len(mh)} cases (all n>100), mid moves "
          f"{len(mm)} (all 60<n<=100), overlap 0 -> composition is exact\n")

    arms = {
        "D (shipped)": D,
        "mix (L303 gate100+k2)": MIX,
        "REFINE restored (4/6)": compose({
            "base": "results_L237_post.json",
            "bands": [("results_L238_refine4.json", 100, 10**9),
                      ("results_L238_refinemid6.json", 60, 100)]}),
        "LP gate 100 (ungated)": results("results_L238_gateoff.json"),
        "LP off (floor ref)": results("results_L238_lpoff.json"),
        "beta config (control)": ref,
    }
    proj = {k: project(v, beta, ref) for k, v in arms.items()}

    # ---- VALIDATION: two published numbers, two median vectors -------------
    print("VALIDATION -- scoring pipeline only (for the control arm the bridge")
    print("is an identity, so this does NOT validate the bridge; see docstring)")
    ok = True
    for vint in ("2026-08-16", "2026-08-23"):
        M, path = medians(vint)
        got, cw, off = score(proj["beta config (control)"], M, 1.0)
        exp = PUBLISHED[vint]
        rel = abs(got - exp) / exp
        flag = "PASS" if rel < 5e-6 else "FAIL"
        ok &= rel < 5e-6
        print(f"  {vint}  got {got:.12f}  published {exp:.12f}  rel {rel:.1e}  {flag}")
        print(f"           cwRF {cw:.7f}   cases off the RF floor: {off}")
    # discrete check: on the 08-16 medians exactly one case leaves the floor,
    # and it is test_id 66. A continuous total can be right by luck; this cannot.
    M16, _ = medians("2026-08-16")
    offs = [r["i"] for r in proj["beta config (control)"]
            if (r["t"] / M16[r["i"]]) ** 0.3 > 0.7]
    hit = offs == [66]
    ok &= hit
    print(f"  discrete: cases off floor on 08-16 medians = {offs} "
          f"(expect [66], n=87)  {'PASS' if hit else 'FAIL'}")
    if not ok:
        raise SystemExit("\nVALIDATION FAILED -- do not read the table below.")

    # ---- measured drift between the two published vintages -----------------
    M16, _ = medians("2026-08-16")
    M23, _ = medians("2026-08-23")
    rat = sorted(M23[i] / M16[i] for i in M16)
    print(f"\nMEASURED drift 08-16 -> 08-23 (one real round of median churn):")
    print(f"  per-case M23/M16   p10 {rat[9]:.3f}  p50 {st.median(rat):.3f}  "
          f"p90 {rat[89]:.3f}   min {rat[0]:.3f}  max {rat[-1]:.3f}")
    print(f"  -> the medians moved DOWN ~{100*(1-st.median(rat)):.0f}% in one week, "
          f"and NOT uniformly (spread {rat[-1]/rat[0]:.2f}x).")

    # ---- the sensitivity table ---------------------------------------------
    M, path = medians(a.vintage)
    print(f"\nSENSITIVITY -- medians = {a.vintage}, scaled by m "
          f"(m<1 = medians shrink = less budget)")
    print("  m also absorbs machine-speed error: R = t_local/(f*m*M).\n")
    order = ["D (shipped)", "mix (L303 gate100+k2)", "REFINE restored (4/6)",
             "LP gate 100 (ungated)", "LP off (floor ref)"]
    hdr = f"{'m':>6} | " + " | ".join(f"{k.split(' (')[0]:>21}" for k in order)
    print(hdr)
    print("-" * len(hdr))
    # 0.841 and 1.004 are L308's measured f = 2.38 / 2.84 expressed in m
    grid = [0.742, 0.8, 0.841, 0.9, 1.0, 1.004, 1.1, 1.2, 1.3]
    tab = {}
    for m in grid:
        cells = []
        for k in order:
            tot, cw, off = score(proj[k], M, m)
            tab[(m, k)] = (tot, cw, off)
            cells.append(f"{tot:.5f} r{rank_of(tot)} {off:>3}off")
        print(f"{m:>6.3f} | " + " | ".join(f"{c:>21}" for c in cells))

    print(f"\n  cell = projected total / implied rank on the 08-23 board / "
          f"cases off the RF floor")

    # ---- NET vs D, the decision currency ------------------------------------
    print(f"\nNET vs D  (positive = better than shipping D; % of weighted total)")
    hdr2 = f"{'m':>6} | " + " | ".join(f"{k.split(' (')[0]:>21}" for k in order[1:])
    print(hdr2)
    print("-" * len(hdr2))
    for m in grid:
        base = tab[(m, "D (shipped)")][0]
        cells = [f"{100*(base-tab[(m,k)][0])/base:>+20.3f}%" for k in order[1:]]
        print(f"{m:>6.3f} | " + " | ".join(cells))

    # ---- crossover search ---------------------------------------------------
    print(f"\nCROSSOVER -- the m at which each arm would overtake D")
    cross = {}
    for k in order[1:]:
        lo, hi, found = 0.2, 6.0, None
        for _ in range(80):
            mid = (lo + hi) / 2
            b = score(proj["D (shipped)"], M, mid)[0]
            v = score(proj[k], M, mid)[0]
            if v < b:
                found, hi = mid, mid
            else:
                lo = mid
        cross[k] = found if (found and 0.2 < found < 5.9) else None
        if cross[k]:
            print(f"  {k:24s} overtakes D at m >= {cross[k]:.2f}")
        else:
            print(f"  {k:24s} never overtakes D on m in [0.2, 6.0]")

    # ---- f <-> m: the two lines' axes are the same axis --------------------
    f1 = (sum(r["runtime_seconds"] for r in ref.values())
          / sum(r["runtime_seconds"] for r in beta.values()))
    print(f"""
RECONCILIATION WITH THE f-BASED PRICING (L296-L308, the mix line)
  That line prices in f = local_seconds / grader_seconds, holding the published
  medians fixed. This file prices in m = a scale on the medians, holding the
  MEASURED f fixed. They are the same axis: RF depends on t and M only through

      R_i = t_i^local / (f * m * M_i)

  so only the product f*m is identifiable. Neither line can separate "the
  grader is faster than we think" from "the medians came in wider".

  This file's m = 1 corresponds to the measured aggregate f = {f1:.2f}
  (beta-config local {sum(r['runtime_seconds'] for r in ref.values()):.1f} s
   / beta grader {sum(r['runtime_seconds'] for r in beta.values()):.1f} s).
  So:  f = {f1:.2f} * m      and      m = f / {f1:.2f}""")
    for k in order[1:]:
        c = cross.get(k)
        if c:
            print(f"    {k.split(' (')[0]:22s} breaks even at m {c:.2f}  ==  f {f1 * c:.2f}")
        else:
            print(f"    {k.split(' (')[0]:22s} never breaks even")
    print(f"""  L308 measured f = 2.38-2.84  ==  m = {2.38 / f1:.2f}-{2.84 / f1:.2f}
  The mix line's quoted break-even f = 1.56  ==  m = {1.56 / f1:.2f}

  KEY: the two lines DISAGREE about where mix breaks even, and the disagreement
  is not about arithmetic -- it is about which corpus the RF bill is computed on.
  See the mix row above: this file bills mix on the beta hidden set's published
  per-case medians; the f-based line bills it on a machine-speed factor with the
  medians held at their published values. Read the mix row and the f column
  together, not either alone.""")

    # ---- exposure: how much of the total each arm puts at the mercy of m ----
    print(f"\nEXPOSURE -- swing in projected total across m in [0.742, 1.3]")
    swing = {}
    for k in order:
        vals = [tab[(m, k)][0] for m in grid]
        swing[k] = max(vals) - min(vals)
        print(f"  {k:24s} {min(vals):.5f} .. {max(vals):.5f}   swing {swing[k]:.5f}")
    worst = max((v, k) for k, v in swing.items() if k != "D (shipped)")
    print(f"  -> D is {worst[0] / swing['D (shipped)']:.1f}x less exposed to median "
          f"drift than {worst[1].split(' (')[0]}")

    # ---- the decision rule (pre-registered, per project discipline) ---------
    print(f"""
DECISION RULE
  m is NOT observable before the deadline: the Final medians are cross-team and
  are published only after evaluation. So this cannot be a "wait and see" rule;
  it has to be decided on the prior. Two facts decide it:

    1. The only round of median churn we can actually observe moved medians
       DOWN (p50 {st.median(rat):.3f}), not up. The reviewer's premise -- Final medians
       grow because the top teams run 110 s, so the 2.69% quality we paid for
       RF is wasted -- is the OPPOSITE of the one measurement available.
       It may still happen; it just is not the base case.
    2. m and the machine-speed factor f are confounded, and L308 puts f at
       2.38-2.84 (a 1.19x band on its own). So even a correct guess about the
       medians leaves the same size of uncertainty in the same variable.

  RULE: ship D.
    * REFINE restored never overtakes D at any m -- close it, unconditionally.
    * LP gate 100 overtakes D only at m >= {cross.get('LP gate 100 (ungated)') or float('nan'):.2f}, i.e. only if the Final
      medians come in at least as generous as the 08-23 vector. That is a bet
      AGAINST the only observed move, for at most {100 * (tab[(1.3, 'D (shipped)')][0] - tab[(1.3, 'LP gate 100 (ungated)')][0]) / tab[(1.3, 'D (shipped)')][0]:.1f}% upside, against
      {abs(100 * (tab[(0.742, 'D (shipped)')][0] - tab[(0.742, 'LP gate 100 (ungated)')][0]) / tab[(0.742, 'D (shipped)')][0]):.1f}% downside if the medians move as they moved last time.
    * D is also the flattest arm ({worst[0] / swing['D (shipped)']:.1f}x less exposed), which is worth more than
      its point estimate: it is the arm whose ranking does not depend on a
      number we cannot see.

  WHAT WOULD CHANGE IT: a published Final median vector, or a mechanism whose
  gain is quality-only at equal wall clock. Nothing time-for-quality clears
  this bar while m is unknown.""")


if __name__ == "__main__":
    raise SystemExit(main())
