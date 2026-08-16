# L132 — coverage, and why the in-set gate cannot measure this candidate

L130 items 2 and 3. Submission untouched throughout.

**Two results, one of them uncomfortable.** Coverage can be raised 64 → 78 by one
existing env knob. And the in-set portfolio gate turns out to be a **one-case
instrument** for this candidate, with a noise floor at roughly the bar itself —
which retroactively means no gate difference measured this session, including
L130's, was resolvable.

## 1. The oscillation is real, and worth zero coverage

`legalise` has two give-up paths: a pinned unit with no predecessor to flip
(`A_no_pred`), or the repair loop exhausting `rounds` (`B_rounds`).
`l132_coverage_probe.py` classifies every failure on the in-set 100:

    ok           64
    B_rounds     36        <- ALL of them
    A_no_pred     0

Several fail after 40 rounds having touched **one distinct pair**. The loop flips
a pair x→y, y-compaction fails on that same pair, and it flips it back, forever.
Raising `rounds` 40 → 400 changes the outcome on **exactly zero** cases.

`LEGAL_LOCK` (`L129_LEGAL_LOCK=1`) flips each pair at most once and otherwise
takes the next binding predecessor. With it, 35 of the 36 failures reclassify to
`C_moves_out` — the repair runs out of predecessors and the pin *still* overruns.

    full 100, both arms, LEGAL_LOCK on:  byte-identical to LEGAL_LOCK off
    base    1.712144 / 64 coverage
    GORDIAN 1.625964 / 63 coverage

🔑 **The oscillation was a symptom, not the cause.** It is a genuine bug — the
loop now terminates honestly instead of burning 40 rounds, and the diagnosis is
only legible because of it — but it buys nothing. Kept, default OFF.

Every one of the 36 failing cases has preplaced units. The real blocker is a
pinned unit whose longest-path lower bound exceeds where the pin must sit, and
flipping relations cannot fix it.

## 2. DENSITY buys coverage, monotonically, and it is already a knob

`spread` sizes the free-unit box at `sqrt(total_area / DENSITY)`. Sweeping it
against first-pass `legalise` success:

| DENSITY | 0.98 | 0.90 | **0.80** | 0.70 | 0.60 | 0.50 | ≤0.40 |
|---|---|---|---|---|---|---|---|
| legalised | 54 | 58 | **64** | 68 | 72 | 76 | **78** |

Clean and monotone, saturating at 78. Note the direction: a **bigger** box gives
**more** coverage. More room means fewer pairs overlap, fewer relations are
forced, chains are shorter, and a pin is less likely to be overrun.

## 3. 🚨 But the gate does not follow coverage, or solo cost, or anything

> As in L130, every gate number here is `--dt 0` — the quality contribution with
> runtime assumed free. Measured, this candidate costs 30–38% of total score
> (L133 §2). The noise-floor argument below is unaffected, since all seven
> DENSITY settings share the assumption.

Full 100, baseline arm with exact abutment, gate at 48c (bar 0.05%):

| DENSITY | coverage | weighted solo | **gate** | wins |
|---|---|---|---|---|
| 0.20 | 78 | 1.795200 | +0.001% | 2 |
| 0.30 | 78 | 1.806057 | +0.016% | 3 |
| **0.40** | 78 | 1.831823 | **+0.045%** | 3 |
| 0.50 | 76 | 1.770798 | +0.001% | 2 |
| 0.60 | 72 | 1.770230 | +0.002% | 2 |
| 0.70 | 68 | 1.723938 | +0.002% | 2 |
| 0.80 | 64 | 1.712144 | +0.011% | 3 |

The peak at 0.40 is not a mechanism, and here is the proof rather than the
suspicion:

* **DENSITY 0.30 and 0.40 cover the IDENTICAL 77 feasible cases** (symmetric
  difference is empty) — same instances, same count;
* their gates differ by **2.8×** (+0.016% vs +0.045%);
* the entire difference is **case 67's cost**: 1.28200 vs 1.21514;
* at 0.20, same coverage of 78, case 67 costs 1.44799 and the gate collapses to
  **+0.001%** — a 45× swing across three settings with equal coverage.

🔑 **The in-set gate for this candidate is one case wearing a percentage sign.**
The candidate sits at solo ~1.7–1.8 against a portfolio at 1.2935, so it
contributes only where it beats the portfolio outright — 2 to 3 cases — and one
heavy case dominates that. Anything that perturbs case 67 moves the gate more
than any mechanism does.

**Noise floor ≈ ±0.04%, and the bar is 0.05%.** The instrument cannot resolve the
bar it is being compared against.

## 4. What this does to the rest of the session's conclusions

It invalidates the *direction* of several gate comparisons, and that has to be
said plainly:

| | gate | what I claimed | what is true |
|---|---|---|---|
| v6 + abut | +0.011% | "better than GORDIAN" | inside the noise floor |
| GORDIAN + abut | +0.003% | "GORDIAN is worse at the gate" | **not resolvable** |
| base + D040 | +0.045% | — | one case, not a mechanism |
| GORDIAN + D040 | +0.000% | — | resolvable only as "not better" |

L130 §4 read "a −5.1% solo win moves the gate the wrong way" and explained it via
case 67. The explanation was right about the mechanism and **wrong to treat the
+0.011% vs +0.003% difference as signal at all**. Both are noise.

What survives unambiguously, because none of it depends on the gate:

* the alternation's solo quality: **−5.1% weighted cost, all three deficit terms,
  40/58 cases** (L130 §3);
* the abutment fix: **+0.0758% on the shipped result, officially measured**
  (L131) — this one is a correctness fix on a frozen artefact, not a gate reading;
* coverage 64 → 78 from DENSITY (§2 above);
* solo cost and the gate are close to **anti-correlated** here: the best solo
  (1.6260) gates at +0.003%, the worst (1.8318) at +0.045%. M77's opening
  argument, in the direction nobody expected.

## 5. What would actually settle it

Only OOS. `m77_oos_probe.py` (240 cases × 35 profiles, bar **NET +0.30%**) is the
instrument with enough cases for the per-case noise to average out. Its bar is
**6× the best in-set number ever produced by this line**, so the honest reading
is that the line is not close, and that no further in-set tuning can tell anyone
whether it is getting closer.

**Do not tune DENSITY on the in-set gate.** 0.40 looks 4× better than 0.80 and
the difference is one case's cost. If DENSITY is worth anything it has to show up
OOS, and it should be picked on **coverage** (which is smooth and monotone) rather
than on the gate (which is not).

## 6. Reproduce

```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l132_coverage_probe.py --rounds 40
```
```bash
L129_LEGAL_LOCK=1 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l132_coverage_probe.py --rounds 40
```
```bash
L129_EXACT_ABUT=1 L129_DENSITY=0.40 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l129_global_placer.py run --out results_L132_d040.json
```
