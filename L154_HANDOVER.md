# L154 handover — the band-catch, ready if you want it

**One file changed: `optimizer_constructive.py`. No C++, no ELF rebuild.**
`bin/constructive_linux` is untouched, so `_binary_matches_source()` passes
exactly as it does today. This sits on top of L147 and changes nothing about it.

> **Read this first: it does NOT clear the 0.30% bar.** Mean gain is +0.0009% to
> +0.15% depending on corpus. What it buys is the *tail* — it cannot lose
> quality, and it halves the Windows/Linux spread that `L153_REPORT.md` forced us
> to report as a range. Ship it as insurance if you are happy carrying a second
> hunk block; **do not delay the L147 merge for it.**

## What to set

    ICCAD_SHAPE_LP_CATCH=1

on top of L147's four. Default is OFF, and with it OFF the code is
**bit-identical to L147** — proved through two different drivers, not assumed
(Gate A below). If it ships as a default, flip `_catch` in `_shape_lp` the way
L137's defaults were flipped; the existing `ICCAD_SHAPE_LP` master switch
(cores ≥40, fail-closed) still gates the whole lane.

## The patch

`git diff 4509645 -- optimizer_constructive.py` — **3 hunks**, all inside the
`_shape_lp` LP block, below L137's additions in the same region L147 touches.
Your merge cost goes from "one file, 13 hunks" to "one file, 16 hunks".

## What it does

A tangent-arm case that `hard_ok` refuses currently falls all the way back to the
**pre-LP** layout — so it loses the whole *shipped* LP gain on that case, not
just the tangent increment. L153 caught that happening: Linux rejects case 96
(n=117), the shipped band holds it at 1.186644, rejection returns 1.215357.

    tier 1   the requested rows (tangent)     -- as today
    tier 2   the shipped band                 -- NEW, only if tier 1 was refused
    tier 0   pre-LP                           -- only if both were refused

The accept guard is **not** duplicated. The chain body is factored into
`_chain(kw)` and called twice with different rows — a second tier adjudicated by
a different rule would be a different keep rule wearing the same name.

`ICCAD_SHAPE_LP_STATS` gained a third field, the tier that kept the case, so a
band-catch run is distinguishable from a run where the tangent rows just got
luckier. Lines are now `n kept tier`.

## What it is worth

| corpus | OFF (L147) | ON (L154) | gain | moved | worse |
|---|---|---|---|---|---|
| in-set 100, **windows** 48c | 1.196679286011 | **1.196668142932** | **+0.0009%** | 2 | **0** |
| in-set 100, **linux** 48c | 1.201017738792 | **1.199218207747** | **+0.1498%** | 3 | **0** |
| **OOS s1** 240 | 1.434420 | **1.433909** | **+0.0356%** | 4 | **0** |
| **OOS s2** 240 (disjoint) | 1.437675 | **1.436911** | **+0.0532%** | 4 | **0** |

**0 cases worse in 680 case-runs**, all feasible. Structurally it cannot lose:
every case either keeps tier 1 or takes the layout that already ships.

### The real reason it exists — the spread, not the mean

| | windows | linux | **spread** |
|---|---|---|---|
| L147 vs shipped band | +2.4852% | +2.1712% | **0.3140 pp** |
| **L154 vs shipped band** | **+2.4861%** | **+2.3178%** | **0.1683 pp** |

L153 had to hand you **+2.11%~+2.55%** because one high-weight case's LP
acceptance is a platform draw. With the catch that becomes **+2.32%~+2.49%**,
and the downside of any future rejection is bounded at "no worse than what is
already deployed" instead of "back to pre-LP".

### 🚨 Where the mean actually comes from — do not misread this

It is **one case every time**, and `e^{n/12}` decides which. On OOS s1 case 220
(n=116) is **94.5%** of the gain; on s2 case 233 (n=119) is **91.2%**; on Linux
case 96 (n=117) is essentially all of it. The Windows in-set reads +0.0009% only
because both its rejections were n=31 and n=42, whose weights are `e^{-7.4}` and
`e^{-6.5}`. Same trap as `HANDOFF_2026-08-20` §4.1 — weight before you call
anything a mover.

Rejection rate is **2.0–2.5%** of cases; the catch rescues **67–100%** of them.

## 🚨 The price, and the base that inverts it

Priced against the raw beta timings the retry reads **"+0.0000%, free" — and
that is wrong**, because the beta run was the PRE-L147 package. The n=117 beta
case sits at t=1.130 s with 1.077 s of headroom before RF leaves the 0.7 floor,
and the retry needs 0.935 s, so it "fits". But L147 already spends 0.213 s of
that case's slack:

    beta t 1.130s + L147 0.213s = 1.343s;  the floor is left at 2.207s
    => headroom 0.864s, and the retry needs 0.935s = 108% of it

Stacked on the arm you would actually ship:

| rejection pattern | RF total | RF incremental |
|---|---|---|
| small/mid-n only (n=31, n=42) | −0.9726% | **+0.0000%** |
| one big-n (n=117) | −1.0305% | **−0.0560%** |

`l154_price.py` reproduces L147's own −0.9726% from `DT147`, which is what
validates the reimplementation.

**Per event, both sides on the same case** (the honest framing here, because the
gain and the cost are the same event on the same case):

| event | quality | RF | **NET** |
|---|---|---|---|
| big-n rescue (n=117 — linux 96, OOS 220/233) | +0.1401% | −0.0560% | **+0.0841%** |
| mid-n rescue (n=113) | +0.1004% | +0.0000% | **+0.1004%** |
| small rescue (n=31 / n=42) | +0.0003% / +0.0005% | +0.0000% | +0.0003% / +0.0005% |

Break-even: the recovery would have to drop **below 0.012 of cost** for a big-n
rescue to go NET negative. Observed recoveries are 0.0287 / 0.0305 / 0.0343 —
2.4× to 2.9× clear. Positive, but not by a structural margin: by construction
the RF cost lands on the case with the least slack.

⚠️ **Single-run timings are useless at this size and say so loudly.** The two
in-set runs read L154 as **0.37 s/case FASTER** than L147, which is impossible
for strictly-added work — a 2-case effect cannot be seen through a 2.8% p50 /
8.9% max whole-run spread. Only the per-case min-of-3 numbers are timing
evidence: n=31 +0.028 s, n=42 +0.098 s, n=117 +0.935 s.

## How to reproduce

Each block is self-contained. Windows in-set is ~25 min, OOS ~2 h, Linux ~15 min.

**In-set A/B (Gate A + Gate B):**

```bash
cd /c/ICCAD_ml/ship_final && bash l154_inset.sh
```

**OOS, both samples:**

```bash
cd /c/ICCAD_ml/ship_final && bash l154_oos.sh
```

**Per-case retry cost (min-of-3, arms interleaved) then the stacked pricer:**

```bash
cd /c/ICCAD_ml/ship_final && bash l154_price.sh && PYTHONIOENCODING=utf-8 "C:/Users/.01/anaconda3/envs/floorset/python.exe" l154_price.py
```

**Linux 48c A/B on the packaged tar:**

```bash
wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l154_wsl.sh
```

**Ship gate** (the packaged path must reproduce the measured total; the
anchor is the in-set CATCH-on result committed alongside this):

```bash
cd /c/ICCAD_ml/ship_final && "C:/Users/.01/anaconda3/envs/floorset/python.exe" l113_ship_gate.py --cores 48 --anchor results_L154_catchon.json --env ICCAD_SHAPE_LP_R=1.5 --env ICCAD_SHAPE_LP_G=1.10 --env ICCAD_SHAPE_LP_TOL=0.006 --env ICCAD_SHAPE_LP_PRICE=1.0 --env ICCAD_SHAPE_LP_CATCH=1
```

⚠️ Two inputs are **not in git** and have to be on the box:
`C:/Users/.01/Downloads/C_median_runtimes_beta_hidden.csv` (the RF pricer's
medians, the same file `l146_rf_price.py` uses), and `l151_oos_s1_on.json` /
`l151_oos_s2_on.json` if you want `l154_oos.sh` to skip re-running the s2 OFF
arm. Without the latter, run s2 OFF fresh — it costs another ~35 min.

## What "it worked" looks like

* **Gate A — CATCH off is a no-op.** `results_L154_catchoff.json` must be
  bit-identical to `results_L147_on_L137.json`: 100/100 cost *and* 100/100
  positions. It is also bit-identical through the OOS driver (240/240 both), and
  on Linux it reproduces L153's `1.201017738792057` to the last digit. That
  second driver is the point — `HANDOFF_2026-08-20` §4.4 is about a base
  measured on one path and an arm on another.
* **Tier-2 cases land on the shipped band bit-for-bit** — cost *and* positions,
  all 5 such cases across both platforms. If they do not, the second tier is not
  running the program you think it is.
* **The tier census** (third field of the stats file) should read:

      in-set  off: tier0=2 tier1=98          rejected n=[31, 42]
      in-set  on : tier0=0 tier1=98 tier2=2   caught  n=[31, 42]
      linux   on : tier0=0 tier1=97 tier2=3   caught  n=[31, 42, 117]
      OOS s1  on : tier0=2 tier1=234 tier2=4  caught  n=[49, 64, 67, 116]
      OOS s2  on : tier0=0 tier1=236 tier2=4  caught  n=[48, 77, 89, 119]

  `tier0 > 0` with CATCH on is normal — 2 of s1's 6 rejections are refused by
  `hard_ok` at tier 2 as well, which is the guard doing its job.
* **A drop in tier1** would mean the tangent arm itself regressed; the catch
  never touches tier 1.

Full evidence and the measurement log: `L154_REPORT.md`. The Linux verify this
came out of: `L153_REPORT.md`.
