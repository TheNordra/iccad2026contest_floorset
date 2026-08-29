> # 🚨 SUPERSEDED IN PART BY L347 — read `L346_L347_REPORT.md` first.
>
> The `N_soft` values this report is built on come from `Fraction(vrel).limit_denominator()`
> and are **reduced fractions**. L346 scanned all 1 008 000 training layouts and measured
> that **no heavy layout anywhere has `N_soft < 41`** (min 41 / p50 65 / max 90; validation
> and OOS s1 agree). So every `N_soft` below 41 quoted below is wrong, and with it:
>
> * §1 "**2 violations close the 1.32 % gap**" -> the truth is **7** (17 for 2.32 %).
> * §2's 87.8 % licence band (`N_soft` 9-14) **is empty**; the graded heavy band is
>   `N_soft` 49-72 with delta* median 0.0438.
> * §5's honest negative ("four of the top five are 1 -> 0 fixes") is also wrong, in the
>   FAVOURABLE direction: we commit 4, 5, 2, 2, 3, 4 violations on them.
>
> **What still stands:** the delta* formula itself and its exact numeric verification; the
> `w`/`rf` cancellation; the targeting-oracle ratios; and the correction to L296 sec.3.

# L343 — L296 × L342: the break-even price of one violation, on the corpus we will actually be graded on

**Two violations close the rank-1 gap.** Not two percent — two individual soft-constraint
violations, on two specific heavy cases. And the licence to buy them is far larger than
anyone has assumed: on a small-`N_soft` case you may make the layout **87.8 % worse in
geometry** to remove one violation and still break even.

**But the honest negative comes with it: we are not making disproportionately many
violations on those cases.** On four of the top five we already commit exactly **one**.
The concentration is structural — `exp(n/12)` times the `1/N_soft` divisor — not a defect
we are producing. So the prize is real, large, and sits on the hardest kind of fix: 1 → 0.

Pure analysis, no solver runs, no shipping path touched. Tool `l343_stack.py`, output
`l343_out.txt`.

---

## 0. Why this is now worth doing

Q&A **A21**: the final is scored on the **same hidden testcases as beta**. That promotes
L296's reverse-solve — `Fraction(vrel).limit_denominator()` recovers `V` and `N_soft`
exactly, 88/88 — from *a characterisation of the beta corpus* to **the per-case data of
the corpus we will be graded on**. L342 then established the axis: the violation term,
not geometry, is where L340's SA lost 30–60× what it won.

⚠️ **A21 improves the ruler, not the reach.** We still cannot run on these cases. Knowing
their statistics only helps if it changes what we ship for cases *like* them.

## 1. The concentration, exactly

Baseline is `l296_project`'s graded projection (its `DQ_SHIP = −0.0497` models roughly the
**D** arm: 0.878564, +2.321 % behind rank 1). `SHIP_DECISION` puts **D+RF-SAFE at
1.00–1.32 %** behind.

| rank | case | n | `N_soft` | `V` | this one | exact joint |
|---|---|---|---|---|---|---|
| 1 | 95 | 116 | 18 | **1** | 0.6294 % | 0.6294 % |
| 2 | 90 | 111 | 14 | **1** | 0.5454 % | **1.1747 %** |
| 3 | 98 | 119 | 28 | **1** | 0.5429 % | 1.7176 % |
| 4 | 96 | 117 | 33 | **1** | 0.3589 % | 2.0765 % |
| 5 | 97 | 118 | 53 | 3 | 0.2608 % | **2.3373 %** |

**2 violations close the 1.32 % gap to rank 1; 5 close 2.32 %.** All 88 violated cases,
first-order, total 5.03 %.

## 2. 🔑 The break-even price — new, and exactly verified

Per-case cost is `w·(1+G)·exp(2V/N_soft)·rf` with `G = 0.5(hpwl_gap + area_gap)`. Removing
one violation and paying `δ` of geometry breaks even when `(1+G+δ) = (1+G)·exp(2/N_soft)`:

> **δ\* = (1 + G)·(exp(2/N_soft) − 1)**

**`w` and `rf` cancel.** Both terms carry them, so the trade *within* a case depends only
on `N_soft` and that case's own geometry term. Weight decides *which* cases to spend
effort on; it does not change the exchange rate once you are there.

Verified numerically on case 95: remove one violation → 0.873034, then pay δ\* = 0.1451 of
geometry → **0.878564 = T0 exactly, residual +0.00e+00**.

| `N_soft` | cases | `G` median | δ\* | **as % of that case's own G** |
|---|---|---|---|---|
| 9–14 | 7 | 0.2338 | 0.2052 | **87.8 %** |
| 15–24 | 13 | 0.2190 | 0.1266 | **57.8 %** |
| 25–34 | 20 | 0.2156 | 0.0889 | 41.2 % |
| 35–49 | 26 | 0.2342 | 0.0608 | 26.0 % |
| 50–65 | 22 | 0.1947 | 0.0458 | 23.5 % |

**A 3.7× spread, entirely from `N_soft`, which is countable from `constraints` with no
label.** Our packer's boundary penalty (`BP_WEIGHT`, and `layout_score`'s 150000 : 6500
split) is a **per-profile constant** — identical on an `N_soft = 14` case and an
`N_soft = 65` one. This is a live misalignment, not a hypothesis.

🔑 It also supplies a candidate explanation for a ledger entry: **`BP_WEIGHT` is recorded
as "雙向封卷" — swept 30000→1M with no change and 10000→300 with 0 wins.** Those were
*global* sweeps. If the right value differs by 3.7× across cases, a global sweep averages
over the spread and correctly finds no interior optimum. Same shape as M80's lesson
("單獨死不代表聯合死"), one level up: **a flat global sweep is what a case-dependent
optimum looks like from a case-independent knob.**

## 3. 🚨 Correction to L296 §3

| change | total |
|---|---|
| geometry −1 % | −0.1748 % |
| violations −1 % | −0.0874 % |
| violations −10 % | −0.8697 % |

**Per equal relative change, geometry is 2.00× more valuable than violations** — the
opposite of L296 §3's annotation *"10 % of violations −0.870 % ← 5× more per relative
point"*. That row compares **10 %** of violations against **1 %** of geometry; the 4.97
ratio is the 10× built into the comparison, not a per-point rate.

**L296's verdict flips are unaffected** — they came from `project(g, phi)`, which is
correct arithmetic; only the annotation comparing the rows is wrong. What survives, and is
the real point, is **concentration**: violations are not worth more per point, they are
worth far more *per targeted unit*.

## 4. The targeting oracle — what aim alone is worth

Both columns remove the **same number** of violations.

| k | best-first | uniform | **ratio** |
|---|---|---|---|
| 1 | 0.6294 % | 0.0575 % | **10.9×** |
| 2 | 1.1747 % | 0.1150 % | **10.2×** |
| 5 | 2.3373 % | 0.2871 % | **8.1×** |
| 25 | 4.4846 % | 1.4254 % | 3.2× |

And **aim is free**: `w = exp(n/12)` from `block_count`, `N_soft` from `constraints`. Both
are `solve()` inputs. No label, no fitting, no learned table — the same shape as L294's
ungate (which transferred at 100–111 % precisely because it fitted nothing).

## 5. 🚨 The honest negative

The obvious follow-on hypothesis — *"we violate more where it hurts most"* — **is false.**

* `r(N_soft, vrel) = −0.314` looks like support, but **`vrel = V/N_soft`, so that
  correlation is negative by arithmetic alone.** It is not evidence of anything. (Caught
  in this probe's own first draft.)
* The real test is `r(N_soft, V) = **+0.480**` — our violation *count* does rise with the
  number of soft constraints. And on `N_soft ≤ 24` cases we average **V = 1.10** against
  an overall mean of 1.73.

⇒ **The concentration is structural** — `exp(n/12)` × the `1/N_soft` divisor — **not a
defect we are producing.** On four of the top five prize cases we already commit exactly
one violation. The prize is a **1 → 0** fix, which is the hardest kind, and it is
consistent with L277 finding only 12/81 removable on the graded shape.

`N_soft` is still worth having: `r(N_soft, n) = +0.602`, so it carries information n does
not (`N_soft` p50 37, range 9–65; `N_soft/n` p50 0.568).

## 6. What is actually open now

Everything above prices a fix we do not have. The routes that were tried all measured ~0,
and each was tried **under a constraint L343 now shows was never necessary**:

| | what it did | why L343 re-opens the question |
|---|---|---|
| L277 | after-the-fact boundary snap → **+0.0012 %** | a snap is a *repair*: it may not make geometry worse. δ\* says we could have paid **up to 87.8 %** of the geometry term |
| L279 | 23/59 boundary violations sit on **preplaced** blocks | those are unfixable by placement, but they are a *subset* — the top-5 prize cases have not been checked against it |
| L281 / chain-saturation | after-the-fact topology repair | same: repair, not construction |

**The un-attacked question, now sharply specified:** *on a case with `N_soft ≤ 24`, if the
packer is permitted to spend up to δ\* of geometry, can the last boundary violation be
removed at all?* Nobody has run that experiment, because nobody knew the licence was that
large.

**Two probes, in order, neither needing training:**

1. **Is it a generation gap or a selection gap?** The proxy already contains the correct
   per-case `exp(2·vrel)` and is oracle-perfect on selection (M13/M76/M77). So compute,
   from the existing `audit_cache_ship.pkl` / `m77_oos_audit.pkl`, the **per-case min-`V`
   across the 51-profile pool** versus the `V` of the selected candidate. If the pool never
   produces fewer, the gap is generation and no re-weighting of selection can reach it.
2. **Case-adaptive boundary penalty.** Scale `BP_WEIGHT` by `(exp(2/N_soft) − 1)` — exact,
   label-free, unfitted. Isolated `constructive_l343.cpp` / `.exe` only; the shipping tree
   and its ELF are not touched, so no cache is invalidated (M80 discipline).

⚠️ **Neither ships by the 08-30 freeze**, and neither should be attempted to. Anything
touching `constructive.cpp` requires rebuilding `bin/constructive_linux`, re-running the
five Linux lanes and re-staging.

## 7. Reproduce

```bash
cd ship_final
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l343_stack.py
```

Seconds. Reads only `l296_project.graded()` (the beta results json + the published
per-case median csv). The δ\* verification line must print residual `+0.00e+00`.
