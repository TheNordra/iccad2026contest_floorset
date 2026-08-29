# L346 / L347 — the prize band does not exist. It was a reduced fraction, and the violation line closes.

**L343's headline was wrong.** It said two soft-constraint violations close the rank-1
gap, and that on `N_soft ≤ 14` cases we hold a licence to make the layout 87.8 % worse in
geometry to buy one. Both numbers came from `N_soft` values recovered by inverting
`violations_relative` — and **those are reduced fractions.** A full scan of the training
set proves it: across **1 008 000 layouts, 201 264 of them with `n ≥ 101`, not one has
`N_soft < 41`.**

Corrected: **7 violations to close the 1.32 % gap, not 2**; the corrected licence on the
graded heavy band is δ\* ≈ 0.0438, which sits **inside the band L345 already measured** —
so L345's blind spot closes, in the unfavourable direction, and its verdict
(`paid / δ* = 2.30`, the trade does not pay) applies to the prize cases directly.

> ⚠️ **L348 update:** the `+2.321 %` gap used throughout below is the **D** arm --
> `l296_project`'s `DQ_SHIP` was built from D's in-set gain, not RF-SAFE's. For what we
> would actually ship the gap is **+1.461 %**, and the violation recount becomes **8**,
> not 17. See `L348_REPORT.md`.

Tools `l346_corpus.py`, `l347_recount.py`; outputs `l346_scan.txt`, `l347_out.txt`.
No shipping change.

---

## 1. The gate that made this possible

`N_soft` = boundary blocks + Σ(MIB−1) + Σ(cluster−1) is computable **from `constraints`
alone**. Checked against the evaluator's own `max_possible_violations`:

```
validation cases checked : 100
mismatches               : 0        *** PASS 100/100 ***
```

So `N_soft` is an **input** quantity — no label, no packer run, available inside `solve()`.
(That remains true and remains useful, whatever happens to the rest of this report.)

## 2. The scan — the antecedent does not exist anywhere

L345 could not test the `N_soft ≤ 33` heavy band because no runnable corpus contained one.
The plan was to build that corpus from the 1 M training shards. The scan says it cannot be
built:

```
9000 shards, 1 008 000 layouts, 201 264 with n >= 101   (188 s)
N_soft over the heavy layouts:  min 41   p50 65   max 90
layouts with n >= 101 AND N_soft <= 33 :  0   (0.00 %)
```

| corpus | heavy `n ≥ 101` | `N_soft` min / p50 / max |
|---|---|---|
| validation 100 | 20 | 43 / 61 / 67 |
| OOS s1 (`l252`) | 40 | 59 / 68 / 81 |
| **training set (1 M)** | **201 264** | **41 / 65 / 90** |
| graded, *as recovered by L296* | 19 | 14 / 52 / 65 ← **impossible** |

Three independent corpora agree on the law; the fourth is the one that was *inferred*
rather than measured. The hidden set is drawn from the same generator with noise on shapes
and placements, **not** on connectivity or constraint generation (Q&A A23/A24), so it
obeys the same law. **The outlier is the inference, not the corpus.**

## 3. Disambiguating the fractions

`Fraction(v).limit_denominator()` returns the reduced pair, so the truth is `(k·V₀, k·NS₀)`
for the integer `k` that puts `k·NS₀` inside the observed range at that `n`. That is a hard
constraint and it pins most cases:

```
uniquely pinned 15/19 heavy cases;  k = 1 IMPOSSIBLE on 6 of them
```

| case | n | L296/L343 read | feasible k | **truth** |
|---|---|---|---|---|
| 95 | 116 | 1 / 18 | 3, 4 | **4 / 72** |
| 90 | 111 | 1 / 14 | 4, 5, 6 | **5 / 70** |
| 98 | 119 | 1 / 28 | 2, 3 | **2 / 56** |
| 96 | 117 | 1 / 33 | 2 | **2 / 66** |
| 88 | 109 | 2 / 29 | 2 | **4 / 58** |
| 84 | 105 | 1 / 26 | 2, 3 | **2 / 52** |

Over all 100 cases: **22 of 88 violated cases had `k = 1` impossible**, and the total
violation count is **189, not 152**.

## 4. The corrected prize

`vrel` itself is exact, so **the total violation mass is untouched** (−8.29 % if driven to
zero, and the projection is still 0.878564). Only its decomposition into *count × per-unit*
moves — and the decomposition is exactly what L343 quoted.

| case | n | L343 said | **truth** | overstated by |
|---|---|---|---|---|
| 95 | 116 | 0.6294 % | **0.1640 %** | 3.8× |
| 90 | 111 | 0.5454 % | **0.1154 %** | 4.7× |
| 98 | 119 | 0.5429 % | **0.2763 %** | 2.0× |
| 96 | 117 | 0.3589 % | **0.1821 %** | 2.0× |
| 97 | 118 | 0.2608 % | 0.2608 % | — (k = 1) |
| 88 | 109 | 0.2313 % | **0.1177 %** | 2.0× |

Exact joint removal, best-first: `k=1` 0.2763 · `k=2` 0.5371 · `k=3` 0.7192 · `k=5` 1.0441
· `k=10` 1.7421 · `k=20` 2.5550.

> **L343: "2 violations close the 1.32 % gap; 5 close 2.32 %."**
> **Corrected: 7 close 1.32 %; 17 close 2.32 %.**

⚖️ **One part of the correction runs the *other* way and is worth keeping.** L343's honest
negative said the top prize cases are `1 → 0` fixes, "the hardest kind". They are not — we
commit **4, 5, 2, 2, 3, 4** violations on them. The individual fixes are `5 → 4`, `4 → 3`,
ordinary rather than heroic. **The prize per unit shrank; the difficulty per unit shrank
too.**

## 5. The corrected licence, and L345's blind spot closing

| `N_soft` band (corrected) | cases | δ\* | as % of that case's G |
|---|---|---|---|
| 1–33 | 21 | 0.0902 | 41.7 % ← light cases only |
| 34–49 | 33 | 0.0617 | 26.7 % |
| 50–59 | 27 | 0.0465 | 23.9 % |
| 60–69 | 5 | 0.0387 | 19.8 % |
| 70+ | 2 | 0.0354 | 14.9 % |

**Graded heavy band, corrected: `N_soft` 49–72, δ\* median 0.0438.** L345 measured
`N_soft` 59–81 and found `paid / δ*` median **2.30**. The prize cases now sit **inside**
that band.

⇒ **L345's blind spot is closed, and it closes against the line.** The residual mismatch
is the 49–58 sub-band, where δ\* is at most 1.39× larger than at 68 — which moves the
ratio from 2.30 to ≈1.65, still above 1. **The violation trade does not pay anywhere on
the graded corpus, and there is no longer anywhere we cannot measure.**

## 6. 🔑 The lesson

**L296 wrote the caveat. L343 repeated the caveat. L343 then built its headline on the
lower bound anyway.** Writing "these are lower bounds" next to a number does not stop the
number from being used as a point estimate two sections later.

What broke the tie was not more care with the same data — it was **going and measuring the
quantity that the inference depended on**. The `N_soft`-vs-`n` law was sitting in the
training set the whole time, costs 188 seconds to extract, and turns "1/14, could be 2/28,
who knows" into "k ∈ {4,5,6}, and k=1 is impossible".

Same family as `[[aggregate-is-not-its-decomposition]]`: the aggregate (`vrel`, exact) was
right, and every number derived from splitting it into count × per-unit was wrong.

## 7. Where the violation line stands now

* **Selection**: oracle-perfect, +0.0124 % (L345, reproducing L250/L251).
* **Generation**: the pool's violation floor is 5.9 % below what we pick; forcing it costs
  +2.14 %; `paid / δ*` = 2.30.
* **The favourable band that would have flipped this**: does not exist — 0 of 201 264 heavy
  training layouts, 0 of 20 validation, 0 of 40 OOS s1.
* ⇒ **The violation axis is closed on the graded corpus by measurement, not by
  extrapolation.** L343's probe 2 (`BP_WEIGHT × (exp(2/N_soft) − 1)`) loses its motivation:
  the licence it was meant to exploit is 2–5× smaller than L343 computed, and uniform over
  a much narrower `N_soft` range (49–72, a 1.5× spread, not 3.7×).

## 8. Reproduce

```bash
cd ship_final
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l346_corpus.py gate   # must PASS 100/100
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l346_corpus.py scan   # 188 s
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l346_corpus.py pick
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l347_recount.py       # 155 s first run
```
