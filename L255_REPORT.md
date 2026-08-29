# L255 — the fragmentation band is thin. Relocation is worth ~+1.2 %, then a wall.

**Verdict: L254's "the edge is soft" is true and narrow.** The greedy's ability to
place area decays *smoothly* as the frame tightens — it is a slope, not a step —
so a repair pass that relocates the last couple of blocks buys **−2.50 % of area
≈ +1.22 % of quality** and then stops. The residual cliff to the label is still
**+13.27 % of area**.

This prices the mechanism L254 named, and the price is small.

---

## 1. The metric

L254 burned one metric that was bounded by construction, so this one is chosen to
not be: **`placed_frac` = placed area / total block area at the moment the pack
dies.** It is free to be anything in [0, 1] at any frame scale.

    placed_frac ~ 1.0   the greedy got essentially all the area in and is stuck on
                        the last block or two -> a repair pass could cross it
    placed_frac  low    large fractions of the design cannot be placed at all
                        -> a repair pass is the wrong tool

Same instrument as L254 (`constructive_l254.exe`, identity-gated 102/102), same
dense 26-rung ladder, same proxy-winning profile per case, but reading **every**
failed frame below the first success instead of only the edge.

## 2. The curve (OOS s1, n ≥ 101, 30 cases with sub-cliff frames)

| frame scale | frame allows | **p50 placed_frac** | p10 | frames |
|---|---|---|---|---|
| 1.00 | 100.0 % | **0.7907** | 0.20 | 46 |
| 1.02 | 96.1 % | **0.8543** | 0.20 | 50 |
| 1.04 | 92.5 % | 0.8786 | 0.32 | 66 |
| 1.05 | 90.7 % | **0.8907** | 0.32 | 74 |
| 1.06 | 89.0 % | 0.9042 | 0.61 | 74 |
| 1.08 | 85.7 % | 0.9454 | 0.80 | 63 |
| 1.10 | 82.6 % | 0.9598 | 0.18 | 23 |
| 1.11 | 81.2 % | **0.9780** | 0.15 | 10 |

🔑 **At the label's own density (s = 1.017, 96.6 %), our greedy places about
85 % of the area** — and the label proves 100 % is packable there. The gap is
real, but it is 15 % of the design, not "the last two blocks".

**The band where L254's story holds is thin.** `placed_frac ≥ 0.98` survives only
down to about s = 1.08; by s = 1.05 the greedy is leaving ~11 % of the area
unplaced, and by s = 1.02, ~15 %.

## 3. The price of relocation

Only **15 of 30** cases ever had a sub-cliff frame that still placed ≥ 98 % of the
area. Weighted over those:

| | s | util |
|---|---|---|
| `s_min` what the packer actually reaches | 1.0967 | 83.1 % |
| `s_floor` tightest frame still ≥ 98 % placed | **1.0829** | **85.3 %** |
| the label | 1.0174 | 96.6 % |

    PRIZE for perfectly relocating the last 2% of area   -2.50 % of area
    residual cliff to the label even then                +13.27 % of area

Converted on L251's method (`QF = 1 + 0.5·(hpwl_gap + area_gap)`, base 1.2511,
area_gap 0.2256):

    area -2.50%  ->  area_gap 0.1950  ->  QF 1.2358  ->  +1.22 % of quality

### 3.1 The prize scales with how much you are willing to re-place

The curve gives the whole trade directly, and it is the useful form for planning:

| re-place this much of the area | reachable util | area | quality |
|---|---|---|---|
| 2 % (a repair pass) | 85.3 % | −2.50 % | **+1.22 %** |
| ~10 % (s ≈ 1.06) | 89.0 % | −6.6 % | **+3.23 %** |
| ~15 % (s ≈ 1.02) | 96.1 % | — | approaches the label |

⇒ **The +2.2 % needed for rank 1 sits at roughly "re-place 10 % of the blocks".**
That is not a repair pass bolted onto the greedy; that is a local-search packer —
which is M27's domain, and which L129 already attempted from scratch (reaching
1.745 against the shipped 1.237, winning 2/64 cases, unpriced at 2.3 s/case).

## 4. What this settles

L252 → L253 → L254 → L255 now read as one closed argument:

    the topology is already right           L253   6.8% edit distance, no gradient
    we cannot pack it densely               L252   81.3% vs 96.6%
    not because the space is missing        L254   >=10x free area at the jam
    but the soft band is only ~2% deep      L255   +1.22%, then a wall

**"Add relocation to the greedy" is now priced at +1.22 % and is not the answer.**
The answer, if there is one, is a packer that re-places ~10 % of the design — and
the ledger's one attempt at that class is far behind.

## 5. Honest limits

* `--thresh 0.98` is a choice, not a measurement. §3.1 shows the sensitivity
  explicitly rather than hiding it; every row there is read off the same curve.
* 15/30 cases contribute to the `s_floor` figure — the other 15 never placed
  ≥ 98 % at any sub-cliff frame, i.e. for them the repair story never applies at
  all. That makes +1.22 % an **upper** bound on the repair mechanism.
* p10 of `placed_frac` is wild (0.15–0.80) — case-to-case variance is large and
  the medians should not be read as typical of every case.
* Sample s1 only; the profile measured is the proxy winner, not all 51.
* Nothing here is priced for runtime.

## 6. Files

```
l255_floor.py     the sub-cliff sweep and the curve
l255_floor.log
l255_floor.pkl    {rows, curve} -- per-case floors and the raw scale->placed_frac
```
