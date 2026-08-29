# L253 — the topology is already reached. The gap is density, not arrangement.

**Verdict: §5 item 3 is closed. Our layouts are 93.2 % topologically compatible
with the label, the label is not an outlier relative to our own pool, and moving
closer to it makes us WORSE.** Imitation is not a coherent target. Combined with
L252 the two findings state one thing: **same topology, different density.**

No shipping change. Offline oracle probe (permitted under the 2026-08-05 ruling,
same standing as L250/L251/L252).

---

## 1. Why L128 did not already answer this

L128 is quoted as "a topology cannot be transplanted", and it is true — but it
varied **shapes on the label's own arrangement** (`blend` of aspect ratios,
label → ours, at constant area), and found a cliff: at blend 0.02 the LP is
already worse than shipped (1.3510 vs 1.2368) and at 0.30 it is infeasible on
98/100.

It never measured the distance between **our** arrangement and the label's. That
is a different question, and it is the one that decides whether *imitation* — as
opposed to *transplant* — is even worth defining.

## 2. Metric

Topology is the project's own definition, the max-gap pair relation
`build_and_solve` derives at `optimizer_constructive.py:3346-3357`:

    g0 = x_j - (x_i + w_i)  i LEFT  j        pick = argmax(g)
    g1 = x_i - (x_j + w_j)  j LEFT  i
    g2 = y_j - (y_i + h_i)  i BELOW j
    g3 = y_i - (y_j + h_j)  j BELOW i

Two distances over all n(n-1)/2 pairs:

* **`d_pick`** — fraction of pairs whose `argmax` differs. The project's own
  notion, but **noisy**: a pair separated on *both* axes flips its argmax under an
  arbitrarily small move with no reordering at all. Reported for completeness
  (0.2370) and not used for any conclusion.
* **`d_hard`** — fraction of pairs whose feasible relation **sets are disjoint**,
  i.e. no relation satisfies both layouts, so the pair genuinely has to be
  reordered. A lower bound on the number of edits. This is the honest distance.

## 3. Result (OOS s1, n ≥ 101, 40 cases, weighted exp(n/12), 51 candidates each)

| | value |
|---|---|
| `d_pick` shipped vs label | 0.2370 |
| **`d_hard` shipped vs label** | **0.0679** |
| `d_min` closest of the 51 to the label | 0.0459 |
| `d_max` furthest of the 51 | 0.1342 |
| `d_int` the pool's own **mean** pairwise distance | 0.0756 |
| `d_nn` **median** pool member's nearest neighbour | 0.0307 |
| `d_nn_max` the **most isolated** pool member | 0.0815 |

🔑 **Only 6.8 % of block pairs have to be reordered to match ground truth.** At
n = 120 that is ~485 of 7 140 pairs — our layouts are **93.2 % topologically
compatible with the label already**.

### 3.1 Is the label exotic? No.

The first cut of this compared `d_min` (a **min**) against `d_int` (a **mean**),
which is not apples-to-apples and flattered the conclusion. Corrected — compare
`d_min(label)` to each pool member's **own nearest neighbour**:

    d_min(label)                     0.0459
    median member's nearest nb       0.0307    -> the label is 1.50x further out
    most isolated member's nearest   0.0815    -> but well inside the pool's range

    cases where the label is NOT more isolated than our own loneliest candidate:
                                     39/40

So the label sits somewhat further out than a typical pool member, and **not
further out than candidates we already generate ourselves.** Its topology is
unremarkable.

## 4. The gradient — there isn't a usable one

The "is there a monotone path" half of §5 item 3:

| | |
|---|---|
| Spearman(`d_hard`, true cost) across the 51, per case | **+0.230**, positive in 32/40 |
| rank of the label-**closest** profile in true cost | **21.8 / 51** (≈ random) |
| cost of the proxy pick (what we ship) | **1.511619** |
| cost of the label-**closest** profile | **1.719179** ← **13.7 % WORSE** |
| cost of the true-best profile | 1.511432 |
| `d_hard` of the true-best profile | 0.0697 (**further** than the pick's 0.0679) |
| the label itself | 1.245233 |

There is a weak positive trend, but it is not usable: the candidate closest to
the label costs **13.7 % more** than the one we ship, ranks in the middle of the
pack, and the *best* candidate is **further** from the label than our pick is.
**Walking toward the label's topology walks away from quality.**

🔑 **Harness cross-check**: the last three rows reproduce `L250` exactly — proxy
pick 1.511619, oracle over pool 1.511432, label 1.245233. Same numbers from an
independently written path.

## 5. What this means, joined with L252

    L253   topology distance to the label       6.8%   -- already essentially there
    L252   utilisation ceiling of this packer   81.3%  vs the label's 96.6%
    L250   generation loss                      17.6%

**We are at the label's arrangement and cannot pack it.** That is why imitation
fails and why L128's transplant failed for the same underlying reason from the
other side: a 96.6 %-dense layout is a rigid interlock with ~3.4 % total slack, so
any layout that is *nearly* the label's but realised at 81 % density is not
almost-as-good — it is worse than a layout designed for 81 % density in the first
place. That is the cliff, restated in topology space.

⇒ **The remaining 17.6 % is a legaliser/packer property, not a search property.**
Every search-side axis now has a bound: ordering +0.005 % (M26), seed +0.001 %
(M68), shape +0.099 % (M79), selection +0.0124 % (L250), frame +1.50 % (L252),
topology — this file — **no usable gradient at all**.

## 6. Honest limits

* Sample **s1 only**. s1 is a training corpus for *ML* candidates, which is why
  `m77`'s note insists on s2 for those — irrelevant here, since this measures a
  classical packer's geometry against labels it never sees. Still one sample.
* `d_hard` is a **lower bound** on edits (disjoint relation sets). The true number
  of moves needed to convert one layout into the other is larger; the conclusion
  only uses the bound in the direction it is valid (small ⇒ close).
* The gradient test is correlational across 51 fixed candidates, not a walk. It
  rules out an *easily discoverable* monotone path; it does not prove no path
  exists in the full space.

## 7. Files

```
l253_editdist.py     the metric, the pool-spread comparison, the gradient test
l253_editdist.log
l252_cache.pkl       reused -- 40 cases x 51 profiles of positions, free
```
