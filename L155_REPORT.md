# L155 — the LP cost census, and the bar the last open line has to clear

`HANDOFF_2026-08-20.md` §5.2 is the only line still open: **k=2 is +0.5967% of
real quality blocked purely by +23.18s.** This report does not attempt the
speedup. It establishes **how much speedup is needed**, **where the time
actually goes**, and **which of §5.2's premises survive contact**.

**Headline: the bar is f\* = 1.75×–2.0×, and every partial speedup prices at
exactly 0.0000%. This line cannot be banked in pieces — it is all-or-nothing.**

Tool: `l155_lp_rows.py census`. Artefacts `l155_all_b8.txt` (100 cases),
`l155_heavy.txt` (n≥100, five prune_B), `l155_exact2.txt` (n≥110, the gate).

---

## 1. 🚨 The strategic fact: partial progress is worth zero

Pricing an LP speedup *on its own* against the measured beta medians
(`l146_rf_price.price_seconds`, negative dt):

| LP speedup | 1.10× | 1.25× | 1.50× | 2.00× | 3.00× |
|---|---|---|---|---|---|
| **NET** | **+0.0000%** | **+0.0000%** | **+0.0000%** | **+0.0000%** | **+0.0000%** |

Not "small" — **exactly zero, at every factor.** We sit on the RF floor
(cost-weighted RF 0.7000400), where `max(0.7, R^0.3)` has derivative 0, so time
we give back is time nobody pays us for. This is the same floor arithmetic L154
used in the other direction, and it has a hard consequence:

> **A faster LP is not a deliverable. It is only a key to the k=2 door.** There
> is no intermediate state worth shipping, no partial credit, and no reason to
> spend a merge slot on a speedup that does not come with the depth.

## 2. The bar: f\* = 1.75×–2.0×

k=2's quality is fixed at +0.5967%; only RF moves. Pricing it at LP speedup `f`,
with the per-case dt profile this census measured:

| f | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | 2.50 | 3.00 | ∞ |
|---|---|---|---|---|---|---|---|---|
| NET (our dt profile) | −0.898% | −0.312% | +0.016% | +0.228% | **+0.382%** | +0.477% | +0.542% | +0.597% |
| NET (calibrated to the handoff's measured −1.056% RF) | −0.459% | −0.022% | +0.232% | **+0.394%** | +0.451% | +0.534% | +0.583% | +0.597% |

The two rows differ only in how the same ~23s is distributed across cases; our
standalone-pass profile is the more pessimistic. **Plan against f\* = 2.0×.**

Below **1.5× the line is NET negative** — worth stating plainly, because a
mechanism that lands at 1.3× is not "progress toward" anything, it is a loss.

## 3. Where the time actually goes — 37% is not the solver

100 cases, shipped `ICCAD_SHAPE_LP_B=8`, one pass, min-of-3:

    total LP wall            22.85 s
      scipy sparse assembly   0.71 s   ( 3%)   <- this is all `t_build` times
      linprog                13.73 s   (60%)
      everything else         8.41 s   (37%)   <- Python

🔑 **`t_build` has been measuring the wrong thing.** It brackets only
`sparse.csr_matrix(...)` at `optimizer_constructive.py:2760`, *after* the
`add_ub`/`add_eq` loops have already filled the triplet arrays. The Python row
construction — 477,644 `add_ub` calls — is inside the unmeasured 37%, along with
`solve_pruned`'s verification loop over 350,185 dropped terms.

⇒ **~97% of LP cost is proportional to row count.** A row removed is paid for
twice, in construction and in solve. That is the single most useful fact here
and it makes the 2× target reachable in principle: rows −50% ≈ wall −50%.

(Corrects `[[l100-lp-speed-levers-closed]]`'s "solve 佔 92%": that predates the
exact prune's verify loop and the L147 tangent rows.)

## 4. Composition — two stale premises corrected

Post-prune, 100 cases at B=8, **477,644 rows**:

| origin | rows | share |
|---|---|---|
| **hpwl** | 283,050 | **59.3%** |
| separation | 72,726 | 15.2% |
| **area_tangent** | 71,270 | **14.9%** |
| envelope | 45,984 | 9.6% |
| boundary_eq | 4,264 | 0.9% |
| bbox | 350 | 0.1% |

* §5.2's "**HPWL at ~80% of all LP rows**" is the **pre**-prune matrix. Post-prune
  it is 59.3% — still the plurality, so §5.2's instinct was right even though its
  number was not. On the heavy band alone it is lower (43–46% at B=2).
* `lp_pass`'s own comment — "after pruning, **separation is the MAJORITY of the
  remaining rows (56–73% on the heavy cases)**" — is now **wrong**: separation is
  **15.2%**. That comment predates L147, whose tangent cuts added a whole
  14.9% band that did not exist when it was written. Anyone planning from it
  would attack a sixth of the matrix believing it was two thirds.
* **`area_tangent` is L147's own footprint and nobody has priced it.** 71,270
  rows: `steps+1 = 10` tangents per reshapeable unit, of which one or two can
  bind at the optimum.

## 5. `prune_B` is a pure speed knob — and it is nearly exhausted

§5.2 warns that pruning harder "probably means giving up exactness". **It does
not, and this is structural, not empirical.** `solve_pruned` solves a provable
lower bound, checks every dropped term's assumed sign, forces offenders back,
and falls back to the unpruned build after `max_rounds`. Its own docstring says
prune_B is "only a HEURISTIC for which terms to try dropping — a wrong guess
costs a round, never correctness."

Measured on n≥110 (53% of the score's weight), B=2 vs B=8, min-of-3:

    same objective, different vertex : 10 (case, B) pairs -- degeneracy
    objective MOVED                  : none

**So `ICCAD_SHAPE_LP_B` is already a free, exact speed knob and needs no new
code.** The catch is that it is spent:

| prune_B (n≥100) | 2 | 4 | **8 (shipped)** | 16 | 32 |
|---|---|---|---|---|---|
| rows (summed over rounds) | 166,076 | 170,670 | 190,280 | 201,757 | 285,515 |
| builds (repair rounds) | 52 | 45 | 37 | 28 | 28 |
| **wall** | **10.80 s** | 11.53 s | 11.44 s | 13.20 s | 22.44 s |
| vs shipped | **1.06×** | 0.99× | 1.00× | 0.87× | 0.51× |

B=2 wins **1.06×** — real and exact, but 3% of the way to f\*. The reason it is
not more: pruning harder removes solve rows *and* adds verification work, and
the verifier is a pure-Python loop over the dropped terms. **The two effects
almost cancel.** That points at the first lever rather than closing the line.

⚠️ **B=32 is 2× SLOWER than shipped.** Anyone reaching for "prune less to be
safer" would pay double for a guarantee they already had.

## 6. ⚠️ Methodology: degeneracy is not inexactness

The first version of the gate flagged **74** (case, B) pairs as exactness
failures. They were things like `1.5881561605419323 → 1.5881561605419328` —
**3e-16 relative** — with a different layout hash. The gate had OR-ed "objective
moved" with "layout differs", and this LP is massively degenerate (L119: Windows
and Linux scipy land on different optima of the *same* program). Separating the
two turns 74 failures into 10 vertex changes and zero defects.

One real caveat survives: `lp_pass` freezes units when a cluster breaks and
retries, so a different degenerate vertex can produce a different **freeze set**
and hence a genuinely different program. A moved objective is therefore not by
itself proof of inexactness — but anything above ~1e-6 must have that path ruled
out before its speed number is used. (One cell did move for real: case 81 at
B=32, +0.162%. B=32 is not a candidate.)

## 7. The route, in the order the census implies

Budget to hit 2×: 22.85 s → 11.4 s.

| # | lever | rows attacked | why it is exact | why this order |
|---|---|---|---|---|
| **A** | **vectorise the prune verifier** — replace `solve_pruned`'s Python loop over 350,185 dropped terms with one sparse matvec `A_drop @ x` | none directly | same arithmetic test, just batched | unlocks B: today B=2's 35% row cut nets only 6% because the verifier eats it. Everything downstream gets cheaper rounds. |
| **B** | **lazy envelope rows** — 4 per block (`:2646-2662`); only bbox-frontier blocks can bind | 9.6% | **provable without a verify round**: a block whose slack to the frontier exceeds its max movement (`bounds` are ±rho·dim, displacement bounded) *cannot* bind | cheapest correct win; no rounds added |
| **C** | **lazy area_tangent rows** — 10 per reshapeable unit (`:2680-2694`), 1–2 bind | 14.9% | textbook cutting plane: every tangent is a valid cut, add only violated ones, terminate when none is violated | needs A first, since it trades rows for rounds |
| **D** | **push `prune_B` below 2** | up to 59.3% | already proven (§5) | only worth trying once A has made rounds cheap |

Composite estimate: rows −35~40%, Python overhead roughly halved ⇒ **~1.8×**.
That is *at* the edge of f\*, which is the honest summary of this line: it is a
coin flip with a crisp gate, not a safe bet.

## 8. Recommendation

**Run A → B → C → D with a measurement after each, and kill the line the moment
the cumulative measured speedup cannot reach 1.75×.** The handoff's hard stop of
**08-23** is the right one; nothing else in the ledger is open, so the
opportunity cost of trying is low, but the probability of success is not high.

🚨 **And the honest ordering of the whole week: this line is worth at most
+0.45~0.60% and quite possibly 0, while L147 — merged and shipped — is worth
+1.17~1.27% and is already measured and Linux-verified. If there is any doubt
about the merge landing before 08-28, that is where the attention belongs, not
here.**

## 9. Reproduce

```bash
cd /c/ICCAD_ml/ship_final && "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l155_lp_rows.py census --b 8 --reps 3
```
```bash
cd /c/ICCAD_ml/ship_final && "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l155_lp_rows.py census --b 2,4,8,16,32 --minn 100 --reps 3
```
```bash
cd /c/ICCAD_ml/ship_final && "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l155_lp_rows.py census --b 2,8 --minn 110 --reps 3
```

The census runs the real `_shape_lp` LP on the layouts the shipped portfolio
actually hands it — pre-LP positions read from `results_L153_lpoff_L137.json` —
so unlike `l134_lp_price.py` (which measured l129's global placer) these numbers
transfer to the deployed path. No shipping code was changed.
