# L235/L236/L237 — the LP's Python half, and why "a faster LP is worth exactly 0.0000%" stopped being true

`L155_REPORT.md` §1 closed this line with a sentence that was correct when it was
written and is now the most expensive stale premise in the ledger:

> *"We sit on the RF floor, where `max(0.7, R^0.3)` has derivative 0, so time we
> give back is time nobody pays us for. There is no partial credit on this line
> and nothing intermediate worth shipping."*

Two things have changed underneath it. We spent the floor ourselves — cwRF
0.70004 at beta, 0.72757 before L223, 0.70523 after — and, far more
importantly, **the package now carries `_L196_LPGATE`**. A faster LP no longer
returns time to nobody. It moves the line at which the gate can afford to turn
the LP **on**.

    the LP fully on, in set     +4.80 %      (l230 arms A vs C, weighted)
    what the gate can afford    +1.57 %      (G7, 71 of 100 block counts)
    -------------------------------------------------------------------
    quality standing behind the LP's runtime   3.24 pp

---

## 0. TL;DR

| | |
|---|---|
| whole-LP speedup, 100 cases, min-of-3 | **1.170×** |
| its Python half alone | **1.553×** |
| identity gate | **PASS on all 100 cases** — objective, layout hash, rows-by-origin, kept/dropped, call count, `hard_ok` |
| whole-package gate | **100/100 identical** on cost *and* positions (`results_L237_base` vs `results_L237_post`) |
| worth, gate unchanged | **+0.129 … +0.338 pp**, positive at every pool ratio in the honest interval |
| worth, gate re-widened on top | rejected — see §5 |

The change is Python-only: the same rows in the same order with the same floats,
so HiGHS receives an identical program. That makes the gate **equality**, not a
quality measurement, and it needs no out-of-sample runs at all.

## 1. Where the time is — phases, not cProfile

cProfile charges per-call overhead to the caller, and on a function called 682k
times per census run that *is* most of what it reports. Real timers around whole
phases (21 cases n ≥ 100, min-of-3, `l235_timers.py`):

| phase | before | after | share after |
|---|---|---|---|
| **solve** (HiGHS) | 21.72 s | 21.41 s | **75.2 %** |
| hpwl term construction | 3.74 | 4.02 | 14.1 % |
| separation row build | **3.55** | **0.14** | 0.5 % |
| separation reduce + emit | 1.24 | 1.06 | 3.7 % |
| scipy sparse assembly | 0.88 | 0.86 | 3.0 % |
| prologue | 0.61 | 0.62 | 2.2 % |
| boundary + envelope + tangent | 0.34 | 0.35 | 1.2 % |
| **total** | 32.08 | 28.45 | |

**The ceiling of this whole line is 1.48×** — that is what the LP would be if
every line of Python in it took zero time. 1.170× of it is banked.

## 2. The one structural fact that paid

L155's census concluded *"~97 % of LP cost is proportional to row count"*, and
that is true of the **solver**. It is not true of the builder, and following it
would have sent the work to the wrong place: **separation is 13–15 % of the rows
but its generator walked every one of the n(n−1)/2 pairs** — 7 140 at n = 120 —
building a dict and a `terms` list for each, and then **~89 % of them were
thrown away by the transitive reduction** (25 568 rows survive from ~233 000
candidate pairs).

So the rewrite is not "make the loop faster". It is:

* enumerate the pairs, compute the four gaps and take the arg-max in **numpy**;
* build coefficients **only for the rows the reduction keeps**.

3.55 s → 0.14 s, a **26×** on that phase, and it is the single largest item in
the whole change.

## 3. What "identical" had to mean, and the three rules that got it

The LP is massively degenerate — L119 has Windows and Linux landing on different
optima of the *same* program — so "the objective matched" would not have been a
gate. The gate is that HiGHS receives byte-identical inputs, which every patch
was written to guarantee:

1. **Float accumulation order is never changed.** `slack` was built as
   `0.0; += pb0; [+= hs_i]; += pb0; [+= hs_j]`; the flat expressions that
   replace it associate left-to-right in that same order, and `0.0 + pb0` is
   exactly `pb0`. `wsc * sgn * coef` is `(wsc*sgn)*coef`, which is what hoisting
   `wss` computes.
2. **Row emission order is never changed.** A row's index is `len(bub)` at the
   moment it is added, so inlining `add_ub` means reproducing its appends in
   sequence, not merely emitting the same set.
3. **Tie-breaks are never changed.** `max(cands, key=lambda t: t[0])` returns the
   **first** maximal element; `np.argmax` also returns the first maximum, and
   `np.triu_indices(n, 1)` enumerates in exactly the order the double loop did.

One branch turned out to be unreachable and was removed rather than translated:
the mask's final `if not r["terms"] and r["rhs"] >= 0.0` pass. A separation row
exists only for a pair whose two units **differ**, so at most one of `(ul, ur)`
is `None` and `terms` is never empty. That deleted a whole extra pass over every
candidate pair.

## 4. The full patch list

`l235_patch.py`, 15 patches, each asserted to match exactly once:

| # | what | why it is free |
|---|---|---|
| 1–4 | `_sep_reduction_mask` takes parallel arrays | caller never materialises dicts |
| 5 | `add_ub`: hoisted bound methods | `a.append(x), b.append(y), c.append(z)` allocated a throwaway 3-tuple per triplet |
| 6 | `dsize` → two per-axis dicts + precomputed hpwl slack | 141k calls to index a 2-tuple |
| 7–8 | `add_hpwl_rows` unrolled; rows emitted straight into the triplets | removes 2 list concatenations, a list comprehension and 2 calls per kept term |
| 9 | **separation: numpy + lazy terms** | §2 |
| 10 | telemetry off the arrays | — |
| 11–13 | `lin` as one tuple allocation; `prune_B is not None` hoisted; `wsc*sgn` hoisted; `tuple(lin)` copy dropped | 682k calls |
| 14–15 | the two hpwl edge loops: `w * hw_scale` was evaluated twice per edge | — |

## 5. What it is worth, and why the gate was NOT re-widened

Scored by re-running the gate optimisation at the cheaper `dt_lp`, on the L234
package, across the honest pool-ratio interval:

**The speedup alone (gate unchanged at 71 on):**

| pool ratio | NET before | NET after | Δ | graded |
|---|---|---|---|---|
| 0.72 | +5.447 % | +5.577 % | **+0.129 pp** | 0.87491 |
| **0.7682 (measured)** | +5.053 % | **+5.224 %** | **+0.171 pp** | **0.87818** |
| 0.80 | +4.639 % | +4.873 % | +0.234 pp | 0.88144 |
| 0.82 | +4.376 % | +4.656 % | +0.280 pp | 0.88344 |

Positive everywhere, and **largest exactly where the rest of the projection is
worst** — when the pool is slower more cases sit above the floor, and there the
wall the LP gives back is wall someone was charging us for. That is the right
shape for a hedge.

🚨 **Widening the gate on top of it is NOT robust and was rejected.** The first
scoring pass read +0.234 pp for "speedup + widening" and that number is real but
mis-attributed: it is ~+0.17 pp of speedup and ~+0.06 pp of widening. Scored
properly — both sides at the same `(rb, f)` — the widening reads:

| candidate | rb=0.72 | rb=0.7682 | rb=0.82 |
|---|---|---|---|
| s=1.2 (+6 block counts) | +0.29 pp | +0.06 pp | **−0.11 pp** |
| s=1.25 (+10) | +0.29 | +0.02 | **−0.20** |
| s=1.3 (+11) | +0.27 | −0.00 | **−0.29** |

Every candidate is negative at the pessimistic end of an interval we are inside.
L230 rejected the handoff's gate table for exactly this shape; the same rule
applies to our own.

## 6. What is left, and why it stops here

| phase | share | why it is not worth taking |
|---|---|---|
| hpwl construction | 14.1 % | `solve_pruned` iterates each dropped term's `lin` **in order** to verify its assumed sign, so the tuple has to exist and be ordered. What remains is the tuple, the fold and the record — all three are required. |
| sep reduce + emit | 3.7 % | the transitive reduction is Python big-int bitmask work; numpy has no shift-accumulate over 120-bit masks |
| sparse assembly | 3.0 % | scipy's own conversion of the triplets |
| prologue | 2.2 % | `decompose` / `reshapeable` / `_aggregate_pairwise_edges` |
| bnd+env+tangent | 1.2 % | O(n) already |

Everything above 4 % is now one phase, and that phase's remaining cost is
required by the exactness contract. Two further ideas were priced and dropped:
caching term construction across repair rounds (43 % of builds are repairs, but
the fold/emit still has to re-run, so ≈1.04×) and vectorising the edge-loop
filter (≈57 % of edges survive it, so ≈1.02×). **Together ≈+0.05 pp against a
rewrite of the most intricate code in the package.**

`prune_B` was re-swept while this was running (21 cases n ≥ 100, min-of-3):
B=4 is **1.055×** faster than the shipped B=8, exactly the "already spent" L155
recorded — and its exactness gate flagged one case at 9.1e-3, which is the
freeze-set path that would have to be ruled out first. Not taken.

## 7. Files

```
l235_patch.py       the 15 patches; --inplace applies them to the shipping tree
l235_lpbench.py     prof | ab  -- the identity gate and the A/B timing
l235_timers.py      builds optimizer_l235t.py, the phase-timed probe
l235_ab_all.out     the 100-case identity gate, PASS
l236_gate.py        prices the speedup by re-optimising the gate
l237_ship.sh        anchor -> patch -> whole-portfolio bit identity -> stage
optimizer_l235lp.py the probe copy the A/B passed on (never shipped)
```
