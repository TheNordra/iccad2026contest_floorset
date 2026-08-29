# L271 / L272 — the wire half of the ordering trade

`L267_L269_REPORT.md` §2.5 left one thing open: the density `nosize` buys is real
and the best ever measured on this packer, and the only reason it cannot be
collected is that reordering costs wire. This report attacks that half. It finds a
form that collects it on s1 — and then loses it on s2.

**Verdict: RED on s2.** The mechanism is real, its exchange rate is the best ever
measured here, it has **no fitted constant**, and it was **positive on all four
s1 split-halves** — and it still collapsed from **−0.6932 %** to **−0.0142 %** on
the disjoint sample. It is not shippable.

**The most useful thing in this report is that sentence.** The standing heuristic
— *no fitted constant ⇒ L266's failure mode cannot apply ⇒ split-half is enough* —
is **false**, and this is the counterexample. What varies across samples need not
be a fitted constant; here it is which corpus the mechanism's **side-effects** land
in. §4.1 has the decomposition: the **area transfers almost perfectly**
(−0.0109 → −0.0100); the wire cost and the violation sign do not.

`ICCAD_L271=6` — at the frame the trial loop first succeeds on, re-pack **that
same frame** in the density order and keep the better by `layout_score`.

| | s1 | s2 |
|---|---|---|
| gross | −0.4587 % | −0.1836 % |
| **deployable** | **−0.6932 %** | **−0.0142 %** |
| exchange ratio | 0.01 | 0.53 |
| isolated (same profile) | −0.8417 % (393/213) | **+0.1988 %** (323/220) |
| cases better / worse | 11 / 3 | 4 / 6 |
| wall (max-setter) | 1.022–1.030× | — |

On s1 alone it read NET **+0.24 … +0.36 pp** against a 0.30 % bar. That number is
retained below only to show what it looked like before the gate that killed it.

**Four supporting findings, each of which was needed to get there.**

1. **The guide helps only when the retry frame DIFFERS.** At a tighter frame it
   moves the exchange rate 1.05 → 0.12; at the same frame it *hurts* (+0.058 % vs
   −0.459 %), because there the guide **is** the incumbent and pulling toward it
   collapses the very difference the retry exists to create.
2. **The retry's PACK bill is flat because it replaces a trial, not adds one** — a
   successful retry consumes one of `max_trials`, so the loop stops one frame
   earlier. But the stopwatch still reads 1.022–1.030×: the density-order pack it
   substitutes in is dearer than the aspect trial it displaced. **Flat attempts,
   not flat seconds** — the same trap L267 fell into, confirmed a second time.
3. **The shape LP compounds with this mechanism and substitutes for `nosize`.**
   Gross → deployable: `l271sng` −0.459 % → **−0.693 %**; `nosize` +0.163 % →
   **+0.218 %**. The area `nosize` buys is area the LP would have taken anyway;
   the area `l271sng` buys is area the LP cannot reach.
4. **L272 — feeding the L137 hint into the wire term — is RED**, and the reason
   is worth keeping: it made hpwl *and* area worse, so the hint is not accurate
   enough to be a per-edge wire target.

Two corrections to the predecessor report are in §1. Nothing shipped;
`constructive.cpp` (md5 `e2c7b2f418ef2b70b6bff99f7adfbd37`) is untouched.

---

## 0. Instruments and gates

`constructive_l271/272/273.exe`, all built by `l271_patch.py` from the pristine
shipping source (md5-guarded, 8 patches + 5 wire sites); `l273` is the superset.

    ICCAD_L252=1     stderr instrumentation only
    ICCAD_L268=4     the "nosize" density order (independent re-implementation)
    ICCAD_L271=1     retry tightest-K FAILED frames, density order, WITH guide
    ICCAD_L271=2     ... shipped order, WITH guide      (isolates the order)
    ICCAD_L271=4     ... density order, NO guide        (isolates the guide)
    ICCAD_L271=5     retry the SAME frame, density order, WITH guide
    ICCAD_L271=6     ... SAME frame, density order, NO guide     <- the candidate
    ICCAD_L271_K / _REF
    ICCAD_L272=1     the L137 hint feeds the WIRE term for unplaced neighbours

| gate | arm A (flags off) | arm B (flag on) |
|---|---|---|
| `l271.exe` + `ICCAD_L252` | **102/102 PASS** | **102/102 PASS** |
| `l271.exe` + `ICCAD_L271=1` | 102/102 PASS | 13/102 differ |
| `l272.exe` + `ICCAD_L252` | **102/102 PASS** | **102/102 PASS** |
| `l272.exe` + `ICCAD_L272` | 102/102 PASS | **102/102 differ** |
| `l273.exe` + `ICCAD_L271=5` | 102/102 PASS | 29/102 differ |

### 0.1 🚨 Two instrument bugs found here, both of the "plausible table" kind

**(a) `l252_identity.py` is value-blind.** It sets every flag to the string
`"1"`. This binary accepts `ICCAD_L268` **only when the value is 4**. Gating
`--flags ICCAD_L268` therefore reports **arm B PASS 102/102** — which for a
mechanism reads as "silent no-op" — when the flag was simply never set to a value
the binary recognises.

`l271_liveness.py` was written for this: it runs the arm with its **real** values
and requires every byte-identical pair to carry a traceable reason.

**(b) The reason taxonomy has to be ordered correctly.** Its first version tested
"no frame failed → EMPTY" before "retry packed but lost → LOST". Modes 5/6 retry
the *same* frame and need no failed one, so that ordering filed real LOSTs as an
empty antecedent and **understated the blast radius by 2×**. Fixed; the two
mechanisms read very differently:

| | mode 1 (tighter frame) | mode 5 (same frame) |
|---|---|---|
| output differs from stock | 13/102 | **29/102** |
| retry packed / failed to pack | 27 / 18 | **97 / 5** |
| EMPTY — no failed frame to retry | **57** | 0 (n/a) |

🔑 **The 57 EMPTY rows predicted mode 1's portfolio result before it was
measured**: on 56 % of profile-runs the ladder's tightest rung already packs, so
there is nothing tighter to retry. That is the whole reason the tighter-frame
family stays small, and the reason mode 5/6 exists.

### 0.2 A free cross-check worth more than either gate

`ICCAD_L268=4` was re-implemented here from pristine source, independently of the
other session's `constructive_l270.exe`. Both land on **+0.1626 %**, hpwl
**0.3332**, area **0.1972**, vrel **0.0884** — four quantities, four decimals, two
separately written patches. Every run also carries a `ship` arm that reproduced
the cached baseline at **+0.0000 %, 0 layouts changed, 40/40**.

---

## 1. Two corrections to `L267_L269_REPORT.md`

### 1.1 §2.5's "~−1.9 %" understates the prize

| method | own vrel | ship vrel |
|---|---|---|
| exchange-rate model on weighted-mean gaps | −2.047 % | −2.438 % |
| **per-case recomputation from `l268_q40.pkl`** | **−2.019 %** | **−2.507 %** |
| L255's prize curve at a 5.00 % bbox shrink | — | ≈ −2.45 % |

⇒ the honest bracket is **−2.0 … −2.5 %**. The likely provenance is §2.4's own
"+1.90 %" with the sign flipped and applied to a different arm.

### 1.2 The whole family was being scored PRE-LP — and it mattered, in both directions

`l267_quality.py` scores the **raw binary stdout** the spy captures and discards
`m67._solve_one`'s return, which is the wrapper's own portfolio pick with
`_shape_lp_maybe` applied — the deployable layout. Verified in source.

Recording both costs nothing (the solve already ran). On the 40-case band:

    the shape LP is worth -1.9215% on the shipped arm by itself
    ship gaps   pre-LP  hpwl 0.2924  area 0.2300   ->  post-LP  0.2788 / 0.2020

| arm | gross | **deployable** | |
|---|---|---|---|
| `nosize` | +0.1626 % | **+0.2175 %** | LP **substitutes** — it already had that area |
| `l271s` | +0.0577 % | −0.0910 % | LP compounds |
| **`l271sng`** | **−0.4587 %** | **−0.6932 %** | LP **compounds** — 1.5× the gross gain |

🔑 **Whether the LP substitutes or compounds is per-mechanism, and it is not
guessable.** The default assumption — "the LP already takes the area, so any
density mechanism is discounted" — is right for `nosize` and exactly wrong for
`l271sng`. Any density claim in this family should quote the deployable column.

---

## 2. What the wire term loses, and the substitute that does NOT work

`constructive.cpp:1128` (and four more sites):

    if (done[nb.first]){ ... }        // placed — order-dependent
    else if (use_prev){ ... }         // guide  — order-dependent
    else continue;                    // ← an unplaced neighbour contributes NOTHING

An item committed early has few placed neighbours, so most of its wire is
invisible and it is scored on `area` alone. Compound cluster items are the
connectivity anchors and lead the shipped order, which is why demoting them costs
wire.

### 2.1 `HINT_MODE=2` — right instinct, wrong term

| arm | portfolio | hpwl | area | vrel |
|---|---|---|---|---|
| ship | +0.0000 % | 0.2924 | 0.2300 | 0.0893 |
| ship_h2 | +0.0697 % | 0.2879 | 0.2281 | 0.0910 |
| nosize | +0.1626 % | 0.3332 | 0.1972 | 0.0884 |
| nosize_h2 | **−0.0977 %** | 0.3409 | 0.1968 | **0.0859** |

    HINT_MODE=2 under the SHIPPED order   +0.0697 pp   (harmful)
    under the DENSITY order               -0.2603 pp   (helpful)
    INTERACTION                           -0.3300 pp

The interaction has the predicted sign and the **wrong mechanism**: `nosize_h2`'s
hpwl is *worse* than `nosize`'s (0.3409 vs 0.3332); the entire gain is vrel.

The reason is structural: **`estimate_anchors()` runs exactly once, before
packing, when only preplaced blocks are placed** — the classic anchor is *already*
order-independent. Mode 2 swaps one order-independent term for another, both at
`ANCHOR_W = 0.10`, while the damaged term is `wire` at `ww·WIRE_MULT`
(ww = 50/70/150). Wrong term by construction.

⚠️ The corpus's only prior (L137 commit `990004a`) reads **0.601 pp worse**,
in-set 100 @48c with REFINE=4; here it is +0.070 pp on the OOS heavy band with
REFINE=2. Same sign, an order of magnitude apart — a direction, not a magnitude.

### 2.2 L272 — the hint in the *right* term. Still RED.

Zero extra packs, live on 102/102 pairs.

| arm | portfolio | Δhpwl | Δarea | isolated |
|---|---|---|---|---|
| `l272` shipped order | **+1.3338 %** | +0.0075 | **+0.0321** | +1.26 % (912/1128) |
| `nosize_l272` density order | +0.6673 % | +0.0466 | −0.0172 | +1.03 % (933/1107) |

🔑 **hpwl got worse.** A pure scaling failure — the wire term swamping `area`
because it suddenly has far more contributors — would inflate area *and improve*
hpwl. Area inflated (0.2300 → 0.2621) **and hpwl still degraded**. Optimising the
*estimated* wire made the *actual* wire worse.

⇒ **The GORDIAN hint is not accurate enough to be a per-edge wire target.** Good
enough as a weak global anchor (L137 ships it at 0.10), not as a stand-in for a
neighbour's position. Independent support for L128, and it closes the zero-cost
version of the repair: what a wire-blind placement needs is a *layout*, not an
estimate.

---

## 3. L271 — retry, and where the guide does and does not belong

### 3.1 A tighter frame: the guide is what makes it affordable

Retry the tightest frame that **failed**, in the density order, with the
successful layout scaled into it as `prev_pos`.

| arm | portfolio | deployable | Δhpwl | Δarea | **ratio** | packs |
|---|---|---|---|---|---|---|
| `nosize` density order everywhere | +0.1626 % | +0.2175 % | +0.0409 | −0.0328 | **1.25** | 0.922× |
| `l271ship` shipped order + guide | −0.0037 % | — | −0.0001 | +0.0000 | — | 1.050× |
| `l271ng` density order, **no guide** | −0.0869 % | — | +0.0063 | −0.0060 | **1.05** | 1.050× |
| `l271` density order + guide | −0.1348 % | −0.0502 % | **+0.0007** | −0.0054 | **0.12** | 1.050× |
| `l271nr` (no REFINE on retry) | −0.0638 % | −0.0109 % | | | | 1.050× |
| `l271k` (K = 8) | −0.0736 % | −0.0610 % | | | | 1.268× |

🔑 The guide moves the exchange rate **1.05 → 0.12**. `l271ship` changed only
**8/2040** layouts, so the density order unlocks the frame and the guide makes it
affordable — two separable halves, both necessary.

**But the family is closed on size**: deployable −0.05 %, and it is **exactly
+0.000 pp on the heaviest 20** because the antecedent is empty on 56 % of
profile-runs. `l271k` widens it and costs 1.268× packs for nothing.

### 3.2 🏆 The same frame: `area_gap` is the layout's bbox, not the frame

The insight that unlocks it: **`area_gap` is measured on the layout's bounding
box, and the frame is only an upper bound.** So a denser pack inside the *very
same* frame still buys area — and the antecedent is then never empty.

| arm | portfolio | **deployable** | Δhpwl | Δarea | **ratio** | resid | packs (max) |
|---|---|---|---|---|---|---|---|
| `nosize` | +0.1626 % | +0.2175 % | +0.0409 | −0.0328 | 1.25 | +0.022 | 0.922× |
| `l271s` same frame **+ guide** | +0.0577 % | −0.0910 % | +0.0042 | −0.0060 | 0.70 | +0.010 | 1.001× |
| **`l271sng` same frame, NO guide** | **−0.4587 %** | **−0.6932 %** | **+0.0001** | **−0.0120** | **0.01** | +0.031 | **1.000×** |

Isolated: **606/2040 layouts changed (29.7 %), −0.8417 %, 393 better / 213 worse.**
The exchange model reproduces it to 0.031 pp ⇒ geometry, not selection.

**Why the guide reverses sign here.** At the same frame the guide *is* the
incumbent layout `c1`, so pulling the retry toward it re-anchors it to exactly the
layout it exists to differ from: area 0.2240 vs 0.2180, and hpwl worse too. At a
*tighter* frame the guide adds information without collapsing the difference.
⇒ **Seed a foreign guide when the retry frame differs; never when it does not.**

**Why the pack bill is flat.** A successful retry consumes one of
`max_trials`(=4), so it **replaces** a frame trial rather than adding one. Pack
bill 1.000× max-setter, 0.996× on total work. ⚠️ That is attempts, not seconds —
the stopwatch disagrees, see §4.

### 3.2.1 The retry's own REFINE passes are the mechanism, not polish

`ICCAD_L271_REF=0` skips them. It is the obvious cost reduction and it is a bad
trade:

| arm | gross | **deployable** | hpwl | area | packs (max) | isolated |
|---|---|---|---|---|---|---|
| `l271sng` | −0.4587 % | **−0.6932 %** | 0.2924 | 0.2180 | 1.000× | −0.842 % (393/213) |
| `l271sngnr` no REFINE | −0.1355 % | −0.0828 % | 0.2941 | 0.2280 | **0.903×** | −0.030 % (235/178) |

It saves 10 % of the packs and gives up **88 %** of the deployable gain — and the
area barely moves (0.2280 vs the shipped 0.2300). ⇒ The density does not come from
the density-ordered *pack*; it comes from the guided passes that follow it. The
raw re-pack only supplies a different starting point.

### 3.3 Transfer — positive on 4/4 halves, in both views

Gain in pp vs base, positive = better:

| view | arm | alt H1 | alt H2 | **by-size H1 (heaviest 20)** | by-size H2 |
|---|---|---|---|---|---|
| gross | `l271sng` | **+0.432** | **+0.484** | **+0.608** | **+0.244** |
| gross | `l271s` | −0.291 | +0.165 ← flips | −0.075 | −0.032 |
| gross | `nosize` | −0.262 | −0.067 | −0.744 | +0.675 ← flips |
| deployable | `l271sng` | **+0.301** | **+1.065** | **+0.986** | **+0.264** |
| deployable | `l271s` | −0.535 | +0.686 ← flips | +0.340 | −0.274 ← flips |
| deployable | `nosize` | −0.667 | +0.209 ← flips | −0.462 | +0.140 ← flips |

🔑 `l271sng` is the only arm that never flips, and it is **strongest on the
heaviest 20** — the half `exp(n/12)` actually weights. That is the property
`tuned` (L266) and `l271` both lacked.

### 3.4 ⚠️ But the gain is concentrated, and that is the weakest thing about it

Per case, on the deployable column:

    11 cases better / 3 worse / 26 unchanged
    top-1 case = 28 % of the gain, top-3 = 61 %, top-5 = 86 %
    best  single case  -5.84 % (n=119)     worst single case  +3.96 % (n=113)
    heaviest 20  -0.9864 %  (7 better / 1 worse)
    lightest 20  -0.2638 %  (4 better / 2 worse)

Two-thirds of the gain rides on three cases. That is the same shape the other
session flagged for `l269p2` (top-3 = 55 %), and it is the reason the s2 capture
in §4.1 matters more here than the split-half does: a 40-case split-half cannot
distinguish "thin but real" from "three lucky cases", and the halves share the
same three.

The mitigating facts: the mechanism is **downside-protected by construction**
(`layout_score` keeps the incumbent unless the retry beats it), the losing side is
3 cases against 11, and the isolated per-profile view — 606 layouts, 393 better /
213 worse — is far too broad to be three cases.

---

## 4. Wall, and the NET

Same batch, min-of-2, all 51 profiles × 3 heavy cases, box otherwise idle. Only
ratios are claimed. The grader's profile phase is max-bound at n=120.

| arm | × max-setter | × total work | wall cost, max-bound |
|---|---|---|---|
| `l271s` | 0.9956 | 1.0026 | +0.066 pp |
| `l271` | 1.0112 | 1.0040 | −0.168 pp |
| `nosize` | 1.0187 | 0.9983 | −0.283 pp |
| **`l271sng`** | **1.0224** | **1.0041** | **−0.338 pp** |

A second, independent capture — **min-of-4, two arms only**, same box, same idle
conditions — puts it at **1.0298×** (−0.450 pp). Both runs found the same shipped
max-setter (prof 2, 0.786 s), so within this pair the identity was stable.

⇒ **Two independent estimates, 1.0224× and 1.0298×.** The cost is ~2–3 % and it is
**real, not noise** — my first instinct was to call it noise because the
max-setter's identity had moved across earlier captures (prof 40 → 3 → 2), but
that variation was across different arm sets and binaries; with the arm set held
fixed and reps doubled the ratio reproduced.

🔑 **And note the pack column lied again, in the same direction as L267.**
`l271sng` reads **1.000×** on packs and **1.0224×** on the stopwatch: the retry
replaces a frame trial one-for-one, but a density-order pack at the incumbent
frame is more expensive than the aspect trial it displaced. Attempts are not
seconds — this is the second independent confirmation in two reports.

**NET**, using L248's 0.151 pp per 1 % of heavy-band wall:

| quality used | quality | wall | **NET** |
|---|---|---|---|
| **deployable (post-LP)**, wall 1.0224× | **+0.6932 pp** | −0.338 pp | **+0.355 pp** |
| **deployable (post-LP)**, wall 1.0298× | **+0.6932 pp** | −0.450 pp | **+0.243 pp** |
| gross (pre-LP), wall 1.0224× | +0.4587 pp | −0.338 pp | +0.121 pp |

⇒ **NET +0.24 … +0.36 pp against a 0.30 % ship bar — it straddles it.** This is a
candidate, not a result, and the honest statement is that the two wall captures
disagree by more than the margin does.

The gross row is kept to make one dependence explicit: **priced pre-LP, as
everything else in this family has been, this mechanism reads +0.12 pp and would
have been discarded.** The post-LP column is not a nicety here; it is the
difference between a rejected arm and a candidate.

### 4.1 🚨 s2 — the gate that killed it, and exactly what failed to transfer

Same 40-case protocol, disjoint sample, **constants frozen** (there are none):

| | s1 | s2 |
|---|---|---|
| gross | −0.4587 % | −0.1836 % |
| **deployable** | **−0.6932 %** | **−0.0142 %** |
| Δarea (deployable) | −0.0109 | **−0.0100** |
| Δhpwl (deployable) | **+0.0001** | **+0.0053** |
| Δvrel (deployable) | **−0.0013** | **+0.0008** |
| exchange ratio | 0.01 | 0.53 |
| isolated, same profile | −0.8417 % (393 / 213) | **+0.1988 %** (323 / 220) |
| cases better / worse | 11 / 3 | 4 / 6 |

🔑 **The density transfers; the side-effects do not.** `Δarea` reproduces almost
exactly (−0.0109 → −0.0100) — the mechanism really does pack denser, on both
samples. What moves is everything it costs to get there: the wire goes from free
(+0.0001) to real (+0.0053), and **vrel flips sign** (−0.0013 → +0.0008). The
violation term alone is worth **+0.180 %** on s2 through `exp(2·vrel)`, which is
more than the whole remaining area credit.

The exchange rate is *still* below 1.0 on s2 (0.53), so it remains a geometrically
paying trade — it is simply too small to survive its own violation noise. And the
isolated per-profile view flips sign outright, which is the cleanest statement
that this is not selection luck being unmasked but the mechanism itself behaving
differently on a different corpus.

⚠️ **`nosize` moves too** (+0.1626 % → −0.0286 % gross; ratio 1.25 → 1.08). The
whole density family is sample-dependent, which is worth knowing before anyone
quotes an s1 density number as a property of the packer.

### 4.2 The methodological finding, which outlives the mechanism

`l271sng` had every property the project uses to certify a candidate short of a
disjoint sample:

* **no fitted constant** — it is a mode, not a value, so there is nothing for a
  sample to over-fit;
* **positive on 4/4 split-halves**, in both the gross and deployable views;
* **strongest on the heaviest 20**, the half `exp(n/12)` weights;
* isolated per-profile delta over 606 layouts, 393 better / 213 worse — far too
  broad to be a handful of lucky cases;
* exchange-model residual 0.031 pp, i.e. geometry rather than selection.

It still failed s2. ⇒ **"No fitted constant" does not license skipping s2.**
L266's lesson was about over-fitting a *value*; this is a different failure with
the same symptom — the mechanism's by-products (wire, violations) are corpus
dependent even when its main effect is not. The only instrument that caught it
was the disjoint sample.

The one prior signal that pointed the right way was §3.4's concentration warning
(top-3 = 61 % of the gain). **Concentration was a better predictor of transfer
failure than split-half was.**

## 5. Honest limits

1. **Heavy band only** — 40 cases, n ≥ 101, on each of s1 and s2. The deployed
   score is 100 cases over three bands, so even the s1 number was never a
   deployed-score claim.
2. `l271sng` has **no fitted constant** (it is a mode, not a value). That was
   taken as the reason its 4/4 split-half made it credible; §4.2 is the record of
   why that inference was wrong.
3. **The retry consumes a trial slot**, so it does not merely add a candidate — it
   trades one aspect trial for one density-order trial at the same frame. That is
   a behavioural change, and it is why the pack bill reads 1.000×.
4. **Pack count is not wall.** L267 measured 1.063× packs against 1.2417× wall;
   here 1.000× against 1.022–1.030×. §4 is the number, not the pack column.
5. Moot given §4.1, but recorded: this would have been a `constructive.cpp`
   change, which forces a Linux ELF rebuild, and the shipping tree is frozen at
   `build_submission.D`.
6. **The s2 capture is one disjoint sample, not two.** The other session's
   `l269p2` cleared 8/8 halves across *both* corpora; `l271sng` was measured on
   s1 (4/4) and then failed s2 outright, so no third sample was needed.

## 6. Files

```
l271_patch.py        pristine constructive.cpp -> constructive_l271.cpp (md5-guarded)
constructive_l271.exe  L252 + L268=4 + L271 modes 1/2/4
constructive_l272.exe  + L272            constructive_l273.exe  + L271 modes 5/6
l271_liveness.py     liveness with an ORDERED reason taxonomy
l271_quality.py      arms + the free post-LP deployable column
l271_exchange.py     the exchange-rate predictor (reproduces every arm to <=0.031 pp)
l271_wall.py         same-batch min-of-N wall, my arms

l271_h2.pkl  l271_q40.pkl  l272_q40.pkl  l271_lp40.pkl  l273_q40.pkl  l271_wall.pkl
```

Nothing owned by the other session was modified.
