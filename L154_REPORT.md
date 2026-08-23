# L154 — the band-catch: rescue a rejected case with the shipped band

Implements and prices the one open proposal from `L153_REPORT.md` §4.

**Verdict: it works exactly as designed, it can never lose quality, and it halves
the cross-platform spread — but the mean is +0.0009% to +0.15%, which is 2× to
300× BELOW the 0.30% bar. Ship it as insurance if the teammate wants a second
hunk; do not delay the merge for it.**

One knob, `ICCAD_SHAPE_LP_CATCH=1`, default OFF. One file,
`optimizer_constructive.py`, 3 hunks, all inside the `_shape_lp` LP block —
below L137's additions, same as L147's.

---

## 1. The mechanism

A tangent-arm case that `hard_ok` refuses currently falls all the way back to the
**pre-LP** layout. It therefore loses the whole *shipped* LP gain on that case,
not just the tangent increment — L147's report flagged this as a risk and L153
measured it happening: Linux rejects case 96 (n=117), the shipped band holds it
at 1.186644, and rejection returns 1.215357.

The shipped band's own layout is a legal second tier. So:

    tier 1   the requested rows (tangent)     -- as today
    tier 2   the shipped band                 -- NEW, only if tier 1 was refused
    tier 0   pre-LP                           -- only if both were refused

The accept guard is not duplicated: the chain body is factored into `_chain(kw)`
and called twice with different rows. A second tier adjudicated by a different
rule would be a different keep rule wearing the same name.

`ICCAD_SHAPE_LP_STATS` gained a third field, the tier that kept the case, so a
band-catch run is distinguishable from a run where the tangent rows just got
luckier.

## 2. Correctness gates

| gate | result |
|---|---|
| **CATCH off ≡ L147**, in-set 48c, official eval | **bit-identical**: total `1.1966792860111928`, **100/100 cost, 100/100 positions** |
| **CATCH off ≡ L147**, OOS s1, the `m77_oos_probe` driver | **bit-identical**: **240/240 cost, 240/240 positions**, weighted cost `1.434419892` = L151's figure |
| **CATCH off ≡ L147**, Linux 48c, packaged | total `1.201017738792057` — the same digits L153 LANE 4 produced |
| **tier-2 lands on the shipped band bit-for-bit** | all 5 tier-2 cases across both platforms: **cost AND positions equal** to the shipped-band run |
| `l113_ship_gate --cores 48` + 5 flags, `--anchor results_L154_catchon.json` | **ALL PASS**, total `1.1966681429319186`, **cost-equal 100/100, positions-equal 100/100**, route A peak 31/queue 32 |
| feasibility | 100/100, 100/100, 240/240, 240/240 |
| **cases made worse** | **0 out of 680 case-runs** (13 moved, 13 better) |

The refactor being bit-identical through **two independent drivers** is what
lets the L151 `l151_oos_s2_on.json` arm be reused as the s2 OFF side instead of
re-run — `HANDOFF_2026-08-20` §4.4, closed rather than assumed.

## 3. Quality — measured on four corpora

| corpus | OFF (L147) | ON (L154) | gain | moved | worse |
|---|---|---|---|---|---|
| in-set 100, **windows** | 1.196679286011 | **1.196668142932** | **+0.0009%** | 2 | 0 |
| in-set 100, **linux** | 1.201017738792 | **1.199218207747** | **+0.1498%** | 3 | 0 |
| **OOS s1** 240 | 1.434420 | **1.433909** | **+0.0356%** | 4 | 0 |
| **OOS s2** 240 (disjoint) | 1.437675 | **1.436911** | **+0.0532%** | 4 | 0 |

🔑 The Linux number was **predicted from the L153 data before the run**
(1.1992182077469893) and measured at `1.1992182077469895`. The estimate method
in L153 §4 is sound.

### The tier census says what the mechanism is actually doing

    in-set  CATCH off: tier0=2 tier1=98         rejected n=[31, 42]
    in-set  CATCH on : tier0=0 tier1=98 tier2=2 caught   n=[31, 42]
    OOS s1  CATCH off: tier0=6 tier1=234        rejected n=[34, 49, 51, 64, 67, 116]
    OOS s1  CATCH on : tier0=2 tier1=234 tier2=4 caught  n=[49, 64, 67, 116]; still rejected n=[34, 51]
    OOS s2  CATCH on : tier0=0 tier1=236 tier2=4 caught  n=[48, 77, 89, 119]
    linux   CATCH on : tier0=0 tier1=97  tier2=3 caught  n=[31, 42, 117]

Rejection rate **2.0–2.5%** of cases; the catch rescues **67–100%** of them.

⚠️ **The mean is carried by one case every time, and weighting decides which.**
On OOS s1 case 220 (n=116) is **94.5%** of the gain; on s2 case 233 (n=119) is
**91.2%**; on Linux case 96 (n=117) is essentially all of it. On the Windows
in-set both rejections were n=31 and n=42, whose weights are `e^{-7.4}` and
`e^{-6.5}` — which is exactly why that corpus reads +0.0009% and not +0.15%.
Same trap as `HANDOFF_2026-08-20` §4.1: this mechanism's value is entirely a
function of whether a **high-weight** case happens to reject.

## 4. 🚨 The price, and the base that inverts it

The retry is one shipped-band LP solve, paid only on rejecting cases. Measured
min-of-3, arms interleaved, one case at a time (`l154_price.txt`):

    case 10  n= 31   CATCH off 2.4278s -> on 2.4555s   +0.028s   (a real retry)
    case 21  n= 42   CATCH off 2.2760s -> on 2.3745s   +0.098s   (a real retry)
    case 96  n=117   LP off 5.0100s -> band 5.9452s    +0.935s   (what a big-n
    case 92  n=113   LP off 3.6873s -> band 3.9150s    +0.228s    rejection pays)

⚠️ Single-run timings are useless here and say so loudly: the two in-set runs
read L154 as **0.37 s/case FASTER** than L147, which is impossible for
strictly-added work. A 2-case effect cannot be seen through a 2.8% p50 / 8.9%
max whole-run spread.

🚨 **Priced against the raw beta timings the retry reads "+0.0000%, free" — and
that is wrong.** The beta run was the PRE-L147 package. The n=117 beta case sits
at t=1.130 s against a median of 7.248 s, so it has 1.077 s of headroom before RF
leaves the 0.7 floor, and 0.935 s fits. But L147 already spends 0.213 s of that
case's slack:

    beta t 1.130s + L147 0.213s = 1.343s;  the floor is left at 2.207s
    => headroom 0.864s, and the retry needs 0.935s = 108% of it

So stacked on the arm we would actually ship:

| rejection pattern | RF total | RF incremental |
|---|---|---|
| small/mid-n only (n=31, n=42) | −0.9726% | **+0.0000%** |
| one big-n (n=117) | −1.0305% | **−0.0560%** |

`rf(DT147)` reproduces the handover's −0.9726% exactly, which is what validates
the reimplementation.

### Per event, both sides on the same case

Pairing OOS-weighted quality against beta RF is the project convention, but here
it hides the verdict, because the gain and the cost are **the same event on the
same case**. Priced together on one beta row, stacked on L147:

| event | quality | RF | **NET** |
|---|---|---|---|
| big-n rescue (n=117 — linux 96, OOS 220/233) | +0.1401% | −0.0560% | **+0.0841%** |
| mid-n rescue (n=113) | +0.1004% | +0.0000% | **+0.1004%** |
| small rescue (n=31) | +0.0003% | +0.0000% | +0.0003% |
| small rescue (n=42) | +0.0005% | +0.0000% | +0.0005% |

Break-even: the recovery would have to be **below 0.012 of cost** for a big-n
rescue to go NET negative. Observed recoveries are 0.0287 / 0.0305 / 0.0343 —
2.4× to 2.9× the break-even. Positive, but not by a structural margin: the RF
cost lands on the one case with the least slack, by construction.

## 5. What it is actually for: the spread, not the mean

| | windows | linux | **spread** |
|---|---|---|---|
| L147 vs shipped band | +2.4852% | +2.1712% | **0.3140 pp** |
| **L154 vs shipped band** | **+2.4861%** | **+2.3178%** | **0.1683 pp** |

**The cross-platform spread halves (−46%).** That is the whole case for this
mechanism. L153 had to report L147's in-set gain as a range **+2.11%~+2.55%**
because one high-weight case's LP acceptance is a platform draw; with the catch
the range is **+2.32%~+2.49%**, and the downside of any future rejection is
bounded at "no worse than what is already deployed" instead of "back to pre-LP".

## 6. Recommendation

**Ship-worthy, sub-bar, zero-risk-to-quality.** Concretely:

* it **cannot** lose quality against L147 — measured 0 worse in 680 case-runs,
  and structurally, since every case either keeps tier 1 or takes the layout
  that already ships;
* the only cost channel is RF, worth −0.056% and only when a big case rejects,
  against +0.14% of quality on the same case;
* **it does not clear the 0.30% bar on the mean** and should not be presented as
  if it does. Its value is the tail.

⚠️ It costs the teammate a **second hunk** in the file they are merging. L147's
"one file, 13 hunks, trial-applied clean" property becomes "one file, 16 hunks".
That is a merge-cost judgement, not a technical one, and it is theirs.

If it ships, flip `_catch` to default ON the way L137's defaults were flipped;
the `ICCAD_SHAPE_LP` master switch (cores ≥40, fail-closed) still gates the whole
lane, so nothing fires below 40 detected cores.

## 7. Reproduce

```bash
cd /c/ICCAD_ml/ship_final && bash l154_inset.sh
```
```bash
cd /c/ICCAD_ml/ship_final && bash l154_oos.sh
```
```bash
cd /c/ICCAD_ml/ship_final && bash l154_price.sh && PYTHONIOENCODING=utf-8 "C:/Users/.01/anaconda3/envs/floorset/python.exe" l154_price.py
```
```bash
wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l154_wsl.sh
```

Artefacts: `results_L154_catch{off,on}.json`, `results_L154_linux_{off,on}.json`,
`l154_oos_s{1,2}_{off,on}.json`, `l154_price.txt`, the `l154_*stats*.txt` tier
files. Tools added: `l154_price.py` (the stacked pricer), `l154_oos_cmp.py`.
`build_submission/cadc1075*` was re-staged twice for the gates and restored
byte-identical to the tracked 08-17 stage both times.
