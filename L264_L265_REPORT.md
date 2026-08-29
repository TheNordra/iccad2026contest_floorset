# L264 / L265 — the frame ladder, priced. `tuned` is the first positive, affordable candidate.

**Verdict: re-placing the four shipped ladder rungs — same count, no code change —
is worth −0.4593 % of true cost at a wall indistinguishable from shipped.
NET +0.28 pp (max-bound) to +0.50 pp (sum-bound).**

🚨 **But the rung values were chosen after looking at this sample**, which is
exactly the failure mode L258 measured at 0 % transfer. This is a candidate, not a
result, until it is re-scored on s2 with the constants frozen.

Nothing shipped.

---

## 1. The four ladders

    ship     1.00, 1.05, 1.10, 1.20        (the shipped default, 44/55 profiles)
    dense    1.00 -> 1.25 step 0.01        (26 rungs)
    coarse   1.04 -> 1.16 step 0.02        ( 7 rungs)
    tuned    1.06, 1.09, 1.12, 1.16        ( 4 rungs -- SAME COUNT as shipped)

`tuned` exists because two of the shipped four are dead weight: `s_min ≈ 1.11`
(L252, L262), so **1.00 always fails** and **1.20 is far looser than anything the
selector ever picks**. Frames are tried in area order until `max_trials`
*successes*, so a ladder with the same rung count costs the same attempts.

## 2. Quality (OOS s1, n ≥ 101, 40 cases, official strict scorer)

| arm | weighted true cost | vs base | better/worse/same | feasible |
|---|---|---|---|---|
| base | 1.511619 | — | — | — |
| dense | 1.500362 | **−0.7447 %** | 21 / 11 / 8 | 40/40 |
| coarse | 1.506189 | −0.3592 % | 21 / 12 / 7 | 40/40 |
| **tuned** | **1.504677** | **−0.4593 %** | — | 40/40 |

⚠️ The dense arm read **−1.68 %** on 12 cases and **−0.74 %** on 40. The 12-case
number was sample luck, and it was flagged as unsafe when it was reported.

Gaps at the picked layout (L251 reference: hpwl 0.2766 / area 0.2256 / vrel 0.0857):

    dense    hpwl 0.2955   area 0.1994   vrel 0.0909
    coarse   hpwl 0.2935   area 0.2226   vrel 0.0888
    tuned    hpwl 0.2837   area 0.2297   vrel 0.0889

`tuned` gets its gain mostly from **hpwl** (0.2766 → 0.2837 is worse, but far less
worse than dense's 0.2955) while barely touching area. The tighter ladders buy
area and pay it back in wire — L251's coupling, measured.

### 2.1 Where the gain lives

On the 16 cases where dense wins > 1 %, a coarse ladder already captures **86 %**
of it (−3.093 % vs −2.651 %). The aggregate only reads 48 % because **9 cases
where dense loses > 1 %** drag it down — and on two of those, coarse *wins* where
dense loses (n=113: dense +2.58 % vs coarse −2.27 %).

⇒ **The gain is cheap to capture; the losses come from offering the selector more
over-tight frames and having it occasionally pick a worse one.** That is M78's
"adding candidates is harmful by default", at the frame level instead of the
origin level.

## 3. Wall (all four arms, same batch, min-of-2, 51 profiles × 3 heavy cases)

    ship: max-setter prof 7 at 0.808s, total work 32.32s

| arm | × max-setter | × total work |
|---|---|---|
| dense | **1.4767** | 1.0427 |
| coarse | 1.0161 | 0.9861 |
| **tuned** | **1.0121** | **0.9972** |

**Dense makes the max-setter 1.48×.** Failed frames are not cheap: L254 measured
that a failing pack has already placed ~91 % of blocks, and `max_trials` counts
only successes, so every rung below `s_min` is paid in full.

⚠️ The max-setter's *identity* moved between two runs (prof 93 → prof 7), so the
±1–2 % readings for `coarse` and `tuned` are inside measurement noise. The honest
statement is **tuned's wall is indistinguishable from shipped**, not that it is
1.2 % slower.

## 4. NET (L248's conversion: 0.151 pp per 1 % of heavy-band wall)

| arm | quality | max-bound cost | **NET max-bound** | **NET sum-bound** |
|---|---|---|---|---|
| dense | +0.745 pp | −7.20 pp | **−6.46 pp** | +0.10 pp |
| coarse | +0.359 pp | −0.24 pp | +0.12 pp | +0.57 pp |
| **tuned** | **+0.459 pp** | **−0.18 pp** | **+0.28 pp** | **+0.50 pp** |

The grader is **max-bound** at n=120 (the research handoff's own `d_max 3.204 s >
sum/48 2.501 s`), so the max-bound column is the operative one: **tuned +0.28 pp.**

## 5. 🚨 Why this is not yet a result

1. **The rung values are fitted.** I chose 1.06/1.09/1.12/1.16 *after* measuring
   `s_min ≈ 1.11` on this sample. L258 measured exactly this pattern — constants
   picked on the scoring sample — transferring at **0 %**, twice, in both
   directions. This is the single thing that decides whether tuned is real.
2. **Heavy band only.** 40 cases at n ≥ 101, sample s1. The ship bar is OOS NET
   ≥ 0.30 % and the deployed score covers all 100 cases across three bands.
3. **No s2.** The project's own rule for anything with a fitted constant.
4. The wall is 3 cases × min-of-2 on a box whose absolute timings are worthless;
   only the ratios are claimed, and two of them are inside noise.

## 5b. 🚨 SPLIT-HALF (L266) — `tuned` does not transfer. It is closed.

Run before spending an s2 capture, free from the saved per-case costs. Two split
schemes, gain in pp vs base (positive = better):

| ladder | alternating H1 | alternating H2 | by-size H1 (heaviest 20) | by-size H2 |
|---|---|---|---|---|
| dense | **+0.350** | **+1.121** | **+0.826** | **+0.628** |
| coarse | +0.025 | +0.678 | **−0.188** | +1.148 |
| **tuned** | **−0.187** | **+1.076** | **−0.024** | **+1.155** |

🚨 **`tuned` flips sign in BOTH split schemes.** Its full-sample +0.459 pp is one
half carrying it entirely. And the by-size split is worse than a coin flip would
be: tuned is **≈0 on the heaviest 20 cases** and all of its gain sits in the
lighter half — which is exactly the half that carries least weight under
`exp(n/12)`.

`coarse` flips too (−0.188 / +1.148).

**Only `dense` is stable**: positive on all four halves (+0.350, +1.121, +0.826,
+0.628). And dense is the one we cannot afford — max-setter ×1.48, NET −6.46 pp.

⇒ **The frame-ladder axis closes on a clean dilemma: the only quality-robust
ladder is unaffordable, and every affordable ladder is sample-specific.** L252's
+1.50 % upper bound on the frame axis stands as the final word, and the reason it
was never collectable is now measured rather than argued.

⚠️ Method note worth keeping: the selection test ("train picks X, score X on the
test half") reads **100 % transfer** on the alternating split — because it picks
`dense`, which is genuinely stable. A transfer number can look perfect while the
candidate you actually care about is the one flipping sign. **Read the per-ladder
per-half table, not the transfer headline.**

## 6. Next step, and it is a single unambiguous one

**Re-score `tuned` on s2 with the rung values frozen** — no re-fitting, no
re-tuning. If it holds, it is a shippable one-line change to `_PROFILES`
(`ICCAD_FRAME_SCALES`) with no C++ modification at all. If it collapses the way
L258's gated set did, the frame axis is closed for good and L252's +1.50 % upper
bound stands as the final word on it.

A useful intermediate: split-half on s1 (pick rungs on half, score on the other),
which is free and which caught L258.

## 7. Files

```
l263_quality.py    the arm framework (base / dense / coarse / tuned / evict)
l264_dense40.pkl   dense + coarse, 40 cases
l265_tuned40.pkl   tuned, 40 cases
l264_wall.py       four-arm same-batch timing -> l264_wall.pkl
l264_wall.log l265_wall.log l264_dense40.log l265_tuned40.log
```
