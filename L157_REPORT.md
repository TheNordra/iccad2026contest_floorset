# L157 — selective LP depth: spend k=2 only where the RF floor pays for it

**Verdict: real, deployable, and ON the bar. NET lands in +0.231% to +0.411%
depending on how much per-case discrimination the gate keeps, against a 0.30%
bar. Unlike L154 this is not clearly sub-bar — it is genuinely undecided, and
the OOS transfer of k=2's quality is the one measurement that would settle it.**

No new evaluation was run. Every input was already committed: the k=2 arm
(`results_L148_lp2.json`), its min-of-3 timings (`results_L149_t{1,2,3}_*`), the
beta medians, and L155's row census. The question turned out to be answerable
by combining artefacts rather than by measuring again.

---

## 1. The idea, and why it was worth testing

`HANDOFF_2026-08-20` §3 killed LP depth as an all-or-nothing: k=2 buys
**+0.5967%** of quality for **+23.18s**, which prices at RF −1.06% ⇒ NET
−0.4593%. That pricing treats the runtime budget as one global pool.

**It is not one pool.** Measured per case against the published beta medians:

    slack (how much slower a case may get before leaving the RF floor)
      min 0.96x | p10 1.24x | p50 1.74x | p90 2.63x | max 3.91x
      cases already past the edge: 1/100

So some cases have 2.6× of headroom and others have none. **Every mechanism this
project shelved on runtime grounds was priced as if they all had the average.**

The measured incremental cost of a second LP pass (L149, min-of-3, on the eval
path) is p50 **0.165s**, p90 0.483s, max 1.667s, sum **23.23s** — which
reproduces the handoff's +23.18s and validates the reconstruction.

This is M62's "break-even" line, which `CLAUDE.md` still lists as **GREEN but
never started**; its stated prerequisite was "a pre-build time predictor +
machine-speed calibration", and L146's measured medians are that calibration.

## 2. The deployable gate

Nothing here needs an oracle. At runtime, for each case:

    spend the second LP pass iff   t_case + dt_lp2  <=  0.3046 * M-hat(n)

* `t_case` — the case's own elapsed time. **Observed**, on the grader's machine.
* `dt_lp2` — the cost of the second pass. **Observed**: the first pass just ran.
* `0.3046` — `0.7^(1/0.3)`, the point where `max(0.7, R^0.3)` leaves the floor.
* `M-hat(n)` — the only estimated quantity. Fitted from the beta medians:

      M-hat(n) = 0.0196 * n^1.168      R^2 = 0.907
      residual M / M-hat:  p10 0.80x  p50 0.98x  p90 1.20x  (51% p10->p90 spread)

Substituting `M-hat` for the true median moves the selection from 71 cases to
75, agreeing on 68 and over-spending on 7. **The predictor's error costs
−0.0201% of RF** rather than the true gate's exact zero — the fit is good enough
that it is not the binding problem.

## 3. What it is worth — the number that decides it

Applying the gate to both sides (selection derived on the beta rows, quality
scored on the in-set 100 against the committed k=2 arm):

| final medians vs beta | 1.20× | 1.10× | **1.00×** | 0.90× | 0.80× |
|---|---|---|---|---|---|
| cases selected | 91 | 83 | **75** | 58 | 33 |
| quality | +0.4029% | +0.3501% | **+0.2917%** | +0.2335% | +0.0900% |
| RF | −0.0081% | −0.0277% | **−0.0607%** | −0.1127% | −0.0303% |
| **NET** | +0.3948% | +0.3224% | **+0.2310%** | +0.1207% | +0.0598% |

Two bracketing estimates, because the two corpora are not the same cases:

* **+0.411%** — per-case affordability on the beta rows (n≤60: 33/40,
  60<n≤100: 24/40, **n>100: 15/20**), quality credited to the in-set cases with
  the most slack in each band. **This is the more faithful bracket**, because
  the deployed gate IS per-case: it reads the case's own elapsed time.
* **+0.231%** — the table above, where the gate is coarsened to a set of `n` and
  every in-set case with a selected `n` is credited. This is what the mechanism
  degrades to if per-case timing turns out not to discriminate on the grader.

**Centre of the range ≈ +0.32% against a 0.30% bar.** The honest reading is that
this is undecided, not that it passes.

🔑 The measured second-pass cost is **cheaper than the first pass on the heavy
band** — which is why n>100 affordability is 15/20 here and only 10/20 with
L155's standalone-pass proxy. `_shape_lp` builds the LP case once and reuses it
across `iters`, so the second pass pays the solve but not the build. That is a
real property of the deployed path, and it is what puts the heavy band — where
81% of the weight sits — inside the budget.

## 4. ⚠️ Three things that would each have inverted the answer

* **The cost proxy.** L155's standalone LP-pass timing (p50 0.119s) is *not* the
  eval-path increment (p50 0.165s overall, and *cheaper* on the heavy band
  because the case build is not repeated). Using the proxy reads the heavy band
  at 10/20 affordable where the truth is 15/20 — it moves the headline bracket
  from +0.335% to +0.411%. **Price the arm you would ship, measured the way it
  would run.**
* **A correlation that looked fatal and was not.** `corr(per-case k=2 gain, case
  runtime) = +0.608`, which reads as "the gain lives in the slow cases and the
  slack lives in the fast ones, so selection destroys the gain". It does not:
  that correlation is almost entirely an **n effect** (big cases are both slow
  and high-gain). Controlling for band, slack-ranked selection captured *more*
  than a random share of the band. ⇒ A correlation pooled across bands says
  nothing about within-band selection.
* **In-sample greedy.** Choosing the `n` values greedily by gain-per-second
  instead of by the principled free-or-not rule peaks at **NET +0.2846%** on the
  in-set 100 — 11 values picked from 100 by in-sample search, which is precisely
  the noise-fitting mode M76 measured at ~5% transfer and L127 at 15–25%. The
  principled gate is worth less in sample and is the only one worth quoting.

## 5. What would settle it

The remaining unknown is the **OOS transfer of k=2's quality**. L147 transferred
at 86% (in-set +2.5881% → OOS +2.2416%). At that rate the brackets become
**+0.199% / +0.353%**, centre +0.276% — still straddling. So the transfer rate
does not settle it by assumption either; it has to be measured.

⇒ **The ~2h OOS run is the deciding measurement**, because nothing else
distinguishes "ship a second knob" from "record another sub-bar mechanism". It
is the same `l140_oos_soft_audit.py` driver L151/L154 used, two samples, one
extra arm (`ICCAD_SHAPE_LP_ITERS=2`), then the same selection applied offline.

The mechanism has one genuinely good property worth recording regardless: it
**degrades gracefully**. If the final medians come in smaller than beta's, fewer
cases qualify and the gate simply does less — NET +0.12% at 0.90×, +0.06% at
0.80×, never strongly negative. That is the opposite of the all-or-nothing k=2,
which is −0.90% at the measured medians.

## 6. What survives regardless

🔑 **The RF floor is per-case and heterogeneous, and the whole ledger was priced
as if it were one global budget.** That is a lens, not a mechanism, and it
applies to every shelved runtime-bound result. The clearest other instance is
**L125 §4.4**, whose kill reads:

> per case 44–47 of 51 profiles have 15% of headroom, but the source must be
> affordable on **every** case at once, and the weighted intersection is 16

— eligibility 16 under a global constraint, versus 44–47 per case, with the
quality ceiling running from +0.1398% at 16 sources to +0.8017% unconstrained.
`_pool_indices(n)` is already per-case, so that intersection was never
structurally required; it was required because per-case slack was not knowable
before L146. **Re-deriving L125's eligibility per case is an offline re-pricing,
needs no new runs, and is the one place the same unlock might be worth more than
it was here.**

## 7. Reproduce

All inputs are committed; the analysis is `l157_selective_depth.py`.

```bash
cd /c/ICCAD_ml/ship_final && PYTHONIOENCODING=utf-8 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l157_selective_depth.py
```

⚠️ `C:/Users/.01/Downloads/C_median_runtimes_beta_hidden.csv` is **not in git**
and must be on the box — it is the same file `l146_rf_price.py` needs.

⚠️ Nothing in `optimizer_constructive.py` was changed. The gate described in §2
is **not implemented**; this report prices it, it does not build it.
