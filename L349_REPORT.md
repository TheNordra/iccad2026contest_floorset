# L349 — where the prize is, on the final corpus: HPWL, and it is the axis we have barely touched

**HPWL is simultaneously the largest remaining contributor and the least-worked axis.**
Since M73 we have taken **−38.7 % of area_gap** and **−41.9 % of vrel**, and only
**−10.8 % of hpwl_gap**. On the hidden corpus hpwl contributes more to the score than
either of the other two.

**And both of the other two are at measured ceilings** — violations closed by measurement
this session (L342/L345/L347), area at L284's density ceiling with the only route past it
(the B\*-tree manifold) dead on runtime *and* violations.

Tool `l349_headroom.py`, output `l349_out.txt`. No shipping change.

---

## 1. The decomposition, measured on the final corpus

| | hpwl_gap | area_gap | vrel | → hpwl | → area | → violations |
|---|---|---|---|---|---|---|
| beta pkg / validation | 0.2705 | 0.1970 | 0.0241 | 0.1353 | 0.0985 | 0.0494 |
| **RF-SAFE / validation** | 0.2414 | 0.1209 | 0.0140 | 0.1207 | 0.0604 | 0.0284 |
| beta pkg / **HIDDEN** | 0.2174 | 0.2051 | 0.0425 | 0.1087 | 0.1025 | 0.0888 |

The two comparisons that matter:

```
RF-SAFE vs beta, same corpus :  hpwl -10.8 %   area -38.7 %   vrel -41.9 %
HIDDEN vs validation, same code:  hpwl -19.6 %   area  +4.1 %   vrel +76.5 %
```

* **Our progress since M73 is almost entirely area and violations.** hpwl moved 10.8 %,
  roughly a quarter of what the other two moved.
* **The hidden corpus is kinder on hpwl** (−19.6 %) and much harsher on violations
  (+76.5 %) — L348's finding, per component.

## 2. What each axis would have to give

Against the corrected gap of **+1.461 %** (L348), using the graded corpus's own numbers:

| axis alone | relative cut needed | zeroing it is worth |
|---|---|---|
| hpwl_gap | **16.0 %** | −9.01 % |
| area_gap | **17.0 %** | −8.48 % |
| violations | **16.6 %** | −8.29 % |

**The three are essentially interchangeable at the margin** — the contributions are
0.1087 : 0.1025 : 0.0888. So the choice of axis is not decided by prize size. It is
decided by **which axis still has room**, and that is the question the ledger answers.

## 3. Which axis still has room

| axis | already taken | ceiling status |
|---|---|---|
| violations | −41.9 % | **closed by measurement this session.** Selection oracle-perfect (+0.0124 %); pool's violation floor 5.9 % below what we pick; `paid/δ*` = 2.30; and the favourable `N_soft` band that would flip it **does not exist** (0 of 201 264 heavy training layouts) |
| area | −38.7 % | **at L284's density ceiling** (85.4 % vs label 96.6 %), and cost-vs-density is U-shaped with the minimum *at* the shipped point. The one route past it — the B\*-tree manifold, which reaches 0.9455 — is dead on runtime (27–56×, L340) **and** on violations (+2.6 to +3.8, L342) |
| **hpwl** | **−10.8 %** | see below |

## 4. HPWL: the levers that are closed, and the one diagnosis that keeps recurring

Closed, each measured:

* **coordinates** — L276: exact HPWL optimisation on the fixed topology recovers only
  **0.9–1.3 %** of hpwl_gap (against 3.8–4.6 % of area_gap). L128: deeper LP buys area,
  not wirelength.
* **weights** — L296 §7: raising the HPWL weight in `layout_score` made the final HPWL
  *worse*. That is the **fourth** independent time an HPWL knob has been moved and
  returned worse HPWL (L276, L280 ×3, L296 §7).
* **connectivity grouping** — L280: mutual-top-1 binding, RED on both corpora, hpwl
  0.2484 → 0.2860.
* **deferred commitment (beam)** — L125: highest quality ceiling ever measured on the twin
  screen (**+0.8017 %**), and dead. ⚠️ *I checked whether it could be re-opened under the
  RF-floor accounting. It cannot, and not for the reason I guessed:* the kill is the pool's
  **bimodal runtime distribution**. At 48 cores the wall is the max-setter, so a twin costs
  the dRF of its slowest member — 13 profiles cost 0.0000 %, one costs 0.0267 %, then a
  cliff to 11.45 % with **nothing in between**. Only 14/51 profiles can absorb a 2×
  slowdown, and only **one** of those is in the ≥40-core tiers where the beam's value
  lives. Restricted to affordable sources the ceiling collapses +0.8017 % → **+0.0956 %**.
  M41/M42/M45 already pruned this pool by runtime, so what survives is either far below the
  wall or *is* the wall. That is structural, not a stale constant.

**The diagnosis underneath all of them is the same and has never been contradicted:**
hpwl is set by the *topology*, the topology is committed by a greedy whose step score is
in practice a pure area minimiser (`hw·hpwl` carries **0.08 %** of `layout_score`), and
**pre-refinement HPWL is anti-predictive of post-refinement HPWL** (L296 §7's own
explanation for why raising the weight backfires).

That last clause is the sharpest unexploited statement in the ledger. It says the failures
are not "the hpwl weight is wrong" but **"the hpwl signal is being read at the wrong point
in the pipeline"** — `layout_score` runs before compaction, `hpwl_push` and the shape LP.
Every attempt so far has re-weighted a signal measured at the wrong time rather than moving
where it is measured.

⚠️ **I have not verified that selecting on post-refinement HPWL is untried**, and it is not
obviously affordable (it implies refining each frame candidate before choosing). It is
named here as *the* candidate the measurements point at, not as an open lever.

## 5. 🚨 Before any of that: we do not currently know whether there is a gap

Two constructions of "RF-SAFE on the hidden corpus" disagree by 2 pp and **bracket rank 1**:

| construction | projection | vs rank 1 |
|---|---|---|
| graded corpus's own numbers × `DQ` (L348, cross-validates with `SHIP_DECISION` to 0.30 pp) | 0.871177 | **+1.461 % behind** |
| RF-SAFE's own numbers transported by L348's measured per-band corpus ratios | 0.853973 | **−0.543 % ahead** |

The second assumes the corpus penalty is purely multiplicative and that our gain transfers
at 100 %; both are optimistic, and the first has independent corroboration, so the prior
favours "behind". **But the disagreement is resolvable by measurement**, and the same way
as everything else this session: 52 OOS 240-case runs with per-case gaps exist at 48 cores,
so the transport method can be *validated on a third corpus* — predict an arm's OOS numbers
from its validation numbers using ratios measured on a different arm, and compare with its
actual OOS result. The prediction error is the error bar the bracket needs.

**That is worth more than any of the three axes**, because if the optimistic construction is
right the correct action is "upload RF-SAFE and stop", not "find another 16 %".

## 6. The answer, in one line

> **HPWL** — it is the biggest contributor on the corpus that will be graded, and the only
> axis we have not already pushed to a measured ceiling. But every known lever on it is
> closed, so it needs a structurally new mechanism, not another knob — and the ledger's own
> rule for finding one (M71: look for a structural decision that was never parameterised)
> points at *when* the hpwl signal is read, not *how heavily* it is weighted.
>
> **First, though, resolve the 2 pp bracket in §5.** It decides whether there is a gap to
> close at all.

## 7. 🚨 CLOSED — §4's nominated candidate has been tried, twice

Verified after this report was first written. "Select the frame on post-refinement HPWL"
is **not** open:

**(a) It was tried directly, and the record is in the source.** `constructive.cpp:2005-2016`:

> *Compaction ... Applied to the single chosen frame, not per-frame: feeding the imperfect
> `layout_score` proxy every frame's compacted variant lets it overfit and pick a low-proxy
> / high-true-cost outline (same failure mode as trying too many frames). ... A csc-pool
> variant (compact+push every frame finalist, pick by `csc` — the "M17" experiment) was
> also tried and is a **DEAD END**: `csc`'s fixed `hw` weight mis-ranks across different
> outlines (it trades cluster fragments for boundary violations; single-base
> **1.5197 -> 1.5293**) and as a portfolio profile its oracle-min gain is **+0.008 %
> (1 case)**. Cross-frame selection needs the wrapper's shapely proxy, which already does
> exactly this across profiles.*

So both scorings were tried — `layout_score` on compacted variants (overfits) and `csc`
(mis-ranks across outlines) — and both failed.

**(b) It already exists one layer up, and that layer is oracle-perfect.** Verified:
`_proxy_metrics` is called on `pos = f.result()`, the **complete post-processed** output of
each profile (`optimizer_constructive.py:2752`) — compaction, `hpwl_push`, refinement and
the shape LP have all run. So the portfolio's shapely proxy *is* post-refinement
cross-candidate selection, and L345 measured it at **99.95 % efficiency, 39/40 cases the
proxy IS the oracle**.

**What genuinely remains unmeasured, and it is narrow.** Within a profile the frame is
still chosen pre-refinement, so the portfolio can only arbitrate among the 51
(profile, its-own-chosen-frame) pairs — it cannot recover a frame a profile rejected. The
oracle over all (profile x frame) pairs scored post-refinement has never been measured.
Two independent bounds say it is small: M17's **+0.008 %**, and L252's frame-axis ceiling
of **+1.50 %**, itself an upper bound on the lever ("loosening the frame trades area for
violations").

⇒ **§6's answer needs amending: hpwl is still the biggest contributor and the least-worked
axis, but it now has no identified open lever.** Coordinates (L276), weights (L296 §7),
grouping (L280), deferred commitment (L125), and selection timing (M17 + the portfolio
layer) are all closed. That is worth recording so the next session does not re-derive it.
