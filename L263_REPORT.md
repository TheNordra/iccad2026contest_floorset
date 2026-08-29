# L263 — eviction produces no score. The DENSE LADDER does, and that was not the plan.

**Two verdicts.**

1. **Eviction (L262) fails the quality gate**, the same way L256 did: portfolio
   **+0.4474 %** against its own control, and isolated it wins 75 / loses 91.
2. **The dense frame ladder alone is worth −1.6808 %** of true cost on the same
   12 heavy cases — a substantial, simple, *unexpected* lever that has nothing to
   do with eviction and needs pricing.

Nothing shipped.

---

## 1. Why three arms

L262 only bites on a dense ladder (the shipped `1.00, 1.05, 1.10, 1.20` has no
rung near `s_min ≈ 1.11`). Comparing it to the shipped baseline would have
conflated the ladder with the eviction, so:

    base    shipped ladder, no eviction    (l252_cache.pkl, already captured)
    dense   26-rung ladder, no eviction    <- isolates the LADDER
    evict   26-rung ladder + ICCAD_L262=1  <- isolates the EVICTION

## 2. Result (12 heavy cases, s1, official strict scorer)

| arm | weighted true cost | vs base | feasible |
|---|---|---|---|
| base (shipped ladder) | 1.508991 | — | — |
| **dense** | **1.483628** | **−1.6808 %** | 12/12 |
| evict | 1.490266 | −1.2409 % | 12/12 |

    eviction alone (evict vs dense)      +0.4474%   <- WORSE

    gaps at the picked layout
      dense   hpwl 0.2936   area 0.1920   vrel 0.0883
      evict   hpwl 0.2901   area 0.1968   vrel 0.0904

Isolated, same profile before/after (free of the `hmin` re-ranking that made
L256's portfolio A/B meaningless):

    layouts eviction actually changed    166 / 612
    weighted true cost   1.698518 -> 1.689683   -0.5202%
    better 75   worse 91

🚨 **Read those two lines together.** The weighted mean improves but the count
goes the wrong way — a few large wins against many small losses. That is not a
mechanism you can ship: it is a lottery whose expectation happens to be positive
on this sample, and at portfolio level the proxy does not collect it (+0.4474 %).

**Eviction is closed on the same gate that closed L256.** The geometry argument
(L259–L261) was right — `s_min` really does fall (L262: util 81.6 → 82.5 %) — and
it still does not become score. Density was never the whole story: `area` improves
(0.1920 → is already the dense arm's) but hpwl and vrel both pay.

## 3. The finding that was not the plan

**`ICCAD_FRAME_SCALES` = 26 rungs instead of 4 is worth −1.68 % of true cost**,
feasible 12/12, with no code change at all — it is an existing shipped knob.

Per case: −9.03, −4.58, −4.40, −2.93, −1.48, −1.15, −1.08, −0.99, −0.08, 0, 0,
**+4.70**.

This is L252's ladder-grain component being realised: L252 measured the ladder
grain at **−2.38 % of area** (26/40 cases) and put the whole frame axis at
**+1.50 % of quality as an upper bound**. The measured −1.68 % sits slightly
*above* that bound, which is a flag, not a triumph — see §4.

## 4. Why this is not yet a result

* **12 cases.** L252's ladder numbers were on 40. One case moved **+4.70 %**;
  with a sample this small the weighted mean is not safe.
* **It exceeds L252's own upper bound** (+1.68 % measured vs +1.50 % bounded).
  Either the bound's assumption (hpwl unchanged) cuts the other way here, or the
  sample is flattering it. That has to be reconciled before the number is used.
* **Wall is unpriced and it is the obvious cost.** 26 rungs means the trial loop
  walks many *failing* frames before its first success, and L254 measured those
  failures as full pack attempts. `max_trials` counts successes, so the failures
  are pure overhead. This is very likely where the −1.68 % goes.
* **No OOS, no s2, no selection check** beyond the portfolio number itself.

## 5. What to do next, in order

1. **Re-run the dense arm on all 40 cases** (`l263_quality.py --limit 40`, drop
   the evict arm) to see whether −1.68 % survives.
2. **Price its wall** — `l258_maxsetter.py` already does exactly this, and the
   question is sharp: does a dense ladder make some profile the new max-setter?
   A coarser dense ladder (e.g. 1.04→1.16 step 0.02, 7 rungs) would buy most of
   the grain at a fraction of the failed-pack cost, and is the obvious first
   variant to try.
3. Only if both survive: OOS s1 + s2, then the twin/gate machinery
   (`l257_twin.py`, `l258_gate.py`) which is already built and needs no changes.

**Do not** spend more on eviction. Its geometry is proven and its score is not
there, twice over.

## 6. Files

```
l263_quality.py     the three-arm gate -> l263_quality.pkl
l263_quality.log
```
