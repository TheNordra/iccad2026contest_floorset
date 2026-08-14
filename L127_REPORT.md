# L127 — twin-screening the knobs that are already in the shipping binary

Follow-on to L124's reusable method ("a mechanism's RED may only be a RED for its
deployment form") and to L125's correction to it ("…and the ON side has to be
time-neutral"). **Zero risk by construction**: every knob here is already
compiled into `constructive.exe`, gated off, so nothing was rebuilt, no md5 moved
and no offline cache was invalidated.

---

## 1. The pre-screen L125 said to run first

L125 cost a day and would have been killed in two minutes by pricing the eligible
source set before writing the C++. So this round starts with a cheap pre-screen:
in-set heavy band (n≥101, 20 cases) × 6 shipped recipes, shipping exe,
`l125_beam_probe.py flag --flag NAME --val V`.

| knob | changed | solo better/worse | dt p50 / p90 | 6-recipe twin oracle | decision |
|---|---|---|---|---|---|
| `CLUSTER_BND_PERMUTE` | 1/120 | 0 / 1 | 0.95 / 1.00 | +0.0000% (0/20) | drop |
| **`CLUSTER_ORD=1`** | **7/120** | **5 / 2** | 1.00 / 1.25 | **+0.2041% (2/20)** | **screen** |
| **`CLUSTER_ORD=2`** | 93/120 | 14 / 79 | 1.00 / 1.27 | **+0.1349% (2/20)** | **screen** |
| `BFS_NORM` | 100/120 | 17 / 83 | 1.11 / 1.43 | +0.0000% (0/20) | drop |
| `FREE_CLUSTER_BND` | 0/120 | 0 / 0 | 1.00 / 1.02 | +0.0000% (0/20) | drop |

Not pre-screened, with reasons: `ANCHORED_BND_REPACK` and `CLUSTER_BND_CORNER`
(M75 re-measured them at 1/3500 and 0/3500 live); `HPWL_SAFE_CLUSTER_SLIDE` (M75:
every mover is n≤55, and `exp(n/12)` makes the heavy band 28× the weight);
`REFRAME` (it re-runs the whole pipeline, so it is ~2× by construction and fails
L125's time-neutrality precondition before quality is even asked about).

`FREE_CLUSTER_BND`'s 0/120 is a real reading, not an empty antecedent: 3 of the 6
recipes (22, 25, 26) do set `ICCAD_FREE_CLUSTER`, which is its gate.

### 🔑 Liveness is not a proxy for value — they were anti-correlated here

`BFS_NORM` changes **100 of 120** outputs and is worth **exactly zero**.
`CLUSTER_ORD=1` changes **7** and pre-screened highest. A pre-screen that only
asked "does the flag do anything" would have ranked these backwards. It has to
measure the **per-case twin oracle**, which is what M75 meant by "liveness cannot
be read from portfolio output" — restated from the other side: liveness cannot be
read as *worth*, either.

## 2. `CLUSTER_ORD=1`: ceiling exactly 0.0000%

Full twin screen, OOS s2, 80 cases, 48-core pool of 51 profiles, per-case proxy
arbitration (`l127_ord1_curve_s2.log`):

```
  K=0 (today, pool unchanged)   1.509085
  K=1 … K=16                    +0.0000%   (identical totals)
  K=all (51)                    +0.0000%
  ON twins that ever win: 0 profiles
```

**Verified not to be a silent no-op** — that reading has the same shape as a flag
that never reached the binary, which is the failure this project keeps recording.
Direct comparison of the cached captures:

| screen | (case,profile) pairs differing ON vs OFF | cases affected |
|---|---|---|
| `l127_ord1_cache.pkl` | **120 / 4076 (2.9%)** | 9 / 80 |
| `l125_beam_cache.pkl` (control) | 2603 / 4076 (63.9%) | 80 / 80 |

So the flag genuinely changes the layout on 9 OOS cases and **not one of those
layouts ever wins arbitration against the existing pool**. This is M75's "live
antecedent, absorbed effect" pattern, now measured out-of-sample against the real
51-profile pool rather than in-sample against a handful of recipes.

## 3. `CLUSTER_ORD=2`: clears the bar in-sample, and 15–25% of it transfers

This one looked alive. Same screen, s2 (`l127_ord2_curve_s2.log`):

```
  K=4   +0.2211%      K=8   +0.3365%      K=12/all   +0.4023%
  ON twins that ever win: 11 profiles
```

K=8 is **over the 0.30% bar**. It is also live beyond doubt — 2651/4030 pairs
(65.8%) differ on 53/80 cases.

**But every one of the 11 winners wins exactly one case.** L124's winners
repeated; L126's top winner took 3. A tally of all-ones says the K set is fitting
individual cases rather than a reproducible property of a profile, which predicts
a bad transfer. So both halves were built and the K set was picked on one sample
and scored on the other (`--order-sample`, added for this):

| | K=4 | K=8 | K=12 | K=all |
|---|---|---|---|---|
| **in-sample** (K picked on s2, scored on s2) | +0.2211% | **+0.3365%** | +0.4023% | +0.4023% |
| K picked on **s1**, scored on **s2** | +0.0832% | **+0.0849%** | +0.0849% | +0.4023% |
| K picked on **s2**, scored on **s1** | +0.0235% | **+0.0328%** | +0.0406% | +0.2196% |

**Transfer is 15–25%**, against L124's measured 80–83%. The deployable value is
+0.03% to +0.08% — an order of magnitude under the bar, before any RF cost is
even charged. **RED.**

### 🔑 The tally's shape predicts the transfer, and it is free

`--order-sample` costs a second 15-minute build. The **win-count distribution is
already on screen after the first one**, and it called this correctly:

| mechanism | top win counts | transfer |
|---|---|---|
| L124 MIB bucketing | repeated winners | **80–83%** |
| L126 anch_cross | 3, 2, 2, 2, … | (ceiling failed first) |
| L127 `CLUSTER_ORD=2` | 1, 1, 1, 1, … (all 11) | **15–25%** |

**A tally of all-ones means the K-curve is noise-fitting**, and the in-sample
number should be discounted by ~4× before deciding anything. Read the tally
before believing the curve.

## 4. Where this leaves the twin-screen line

Three mechanisms have now been through the same machine on the same OOS sample:

| mechanism | ceiling (s2, K=all) | verdict | why |
|---|---|---|---|
| L124 MIB bucketing | — (K=8 +0.4712% → +0.4697% measured) | **shipped** | time-neutral, in-set/held-out asymmetry |
| L125 beam | **+0.8017%** | RED | unaffordable — the value sits on the 35/51 profiles that *are* the wall |
| L126 anch_cross | +0.2860% | RED | ceiling below the 0.30% bar |
| L127 `CLUSTER_ORD=1` | +0.0000% | RED | live but wholly absorbed by the pool |
| L127 `CLUSTER_ORD=2` | +0.4023% | RED | clears the bar in-sample; **15–25% transfers** |

**Four distinct failure modes, and together they bracket the axis.** A twin has to
clear *all* of:

1. **quality ceiling** ≥ the bar — L126 failed here;
2. **time-neutrality** — L125 failed here, with the best ceiling ever measured;
3. **not already reachable** by the 51-profile pool — `CLUSTER_ORD=1` failed here;
4. **cross-sample transfer** — `CLUSTER_ORD=2` failed here.

L124 remains the only mechanism that passed all four, and its own report already
said why: a structural in-set / held-out asymmetry in MIB violations that no other
structural axis shares (boundary 34.1 vs 33.0%, cluster members 28.8 vs 27.2%,
mixed clusters 53.8 vs 56.7%). That was not luck of the draw — it was the one
place the two corpora genuinely differ.

**The screen now has a cheap gate for each**, in the order they cost least:
liveness + dt + 6-recipe twin oracle (2 min, `l125_beam_probe.py flag`) → eligible
source set (45 min, `l125_beam_price.py afford`, only if the ON side is slower) →
ceiling and tally shape (15 min, `curve`) → cross-sample transfer (a second 15 min,
`--order-sample`). Run them in that order and most candidates die before the
expensive step.

## 4. Artefacts

| file | what |
|---|---|
| `l125_beam_probe.py flag` | the general pre-screen: liveness + solo A/B + twin oracle + dt ratio for any gated-off knob, ~2 min |
| `l124_r3_scale.py --on-val` | the screen now handles non-boolean knobs (`CLUSTER_ORD=2`) |
| `l124_r3_scale.py --order-sample` | pick K on one sample, score on the other — the transfer test |
| `l127_ord1_cache.pkl` | `CLUSTER_ORD=1` s2 capture, 80 × 51 × 2 |
| `l127_ord2_cache.pkl` | `CLUSTER_ORD=2`, **both** s1 and s2 (160 records) |
| `l127_ord{1,2}_curve_s2.log` | the K-curves above |

## 6. Honest range

- the pre-screen is in-set heavy band × 6 recipes; a knob whose value is entirely
  in the mid/small bands would read as dead there. That is deliberate — `exp(n/12)`
  makes n=120 worth 28× n=80 — but it is a restriction, not a proof of nothing
- `HPWL_SAFE_CLUSTER_SLIDE` and `REFRAME` were reasoned out, not measured; the
  reasons are M75's own measurement and L125's precondition respectively
- transfer was measured on one mechanism (`CLUSTER_ORD=2`) plus L124's recorded
  80–83%. The "tally of all-ones predicts bad transfer" rule has **two** data
  points behind it and should be treated as a heuristic worth the two minutes it
  takes to read, not an established law
