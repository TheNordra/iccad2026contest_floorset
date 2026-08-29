# L276 — the hpwl deficit is topology, measured on the graded shape

L275 corrected the target: on the corpus that gets graded, **hpwl is 56 % of the
headroom** (−10.41 % if zeroed) against area's 31 %. This asks whether any of it is
reachable without changing which blocks are adjacent to which.

**It is not. ~99 % of `hpwl_gap` survives an exact minimisation of the official
objective inside our own topology.**

    graded shape (in-set 100, 48 cores, weighted exp(n/12))

      LP depth   hpwl_gap            area_gap
      k = 1      0.2484              0.1355
      k = 2      0.2462  ( 0.9 % removed)   0.1303  ( 3.8 % removed)
      k = 4      0.2452  ( 1.3 % removed)   0.1292  ( 4.6 % removed)

The constraint-graph LP minimises **relative hpwl and area improvement under the
official 0.5/0.5 split** (`optimizer_constructive.py:3836-3862`; L171 repaired the
hpwl weight, which had been 1.24× too low). So it is free to trade between the two
terms — and given that freedom it takes **four times more area than wire**.

🔑 That asymmetry is the finding. It is not "the LP ignores hpwl"; the LP weights
hpwl exactly as the scorer does and still cannot move it. Within a fixed
constraint graph you *can* squeeze the bounding box — compaction is a global
operation available to every block at once. You cannot bring a connected pair
closer if other blocks lie between them in the ordering: no coordinate assignment
reaches it, only a topology change does.

⇒ **The wire deficit is not in the coordinates. It is in the adjacency.**

---

## 1. Why this closes the hpwl line rather than opening it

The corollary is that every hpwl mechanism must change topology, and the ledger
has already closed every topology route — now re-read with L275's corpus
correction in mind:

| route | status | note |
|---|---|---|
| packer rewrite (M27) | closed | greedy already on the (area, HPWL) frontier |
| from-scratch global placer (L129) | closed | 1.745 against the shipped 1.237 |
| unit-pair topology flips (M64) | closed | 529 flips, **0 movers**; 86.8 % LP-infeasible |
| ruin-and-recreate / eviction (L256, L262) | closed | L259: at the jam the largest block has **0** legal positions |
| big-first commitment order (L268, L271) | **closed on the graded shape by L275** | +1.86 % / +0.19 % in-set |
| imitating the label's topology (L253) | closed | nearest candidate costs **+13.7 %** |
| connectivity orderings (`WIRE_ORDER` / `_TIEBREAK` / `_BFS` / `BFS_PIN`) | closed | ≤ 0.063 %, and measured **in-set**, i.e. on the right corpus already |

And the two attempts made this session to give the existing greedy better wire
information both **degraded hpwl**, which is the same wall from the other side:

* **L272** — the L137 hint feeding the wire term for unplaced neighbours:
  hpwl 0.2924 → 0.2999 (OOS), area also worse.
* **`ICCAD_GUIDE_MED=1`** — a connectivity-weighted L1-median as an extra
  candidate origin, on the graded shape: **+0.4982 %**, hpwl 0.2484 → **0.2538**,
  78 movers, 31 better / 47 worse.

Both are instances of M78's *"adding candidates is harmful by default"*: the
greedy scores an origin with `bbox_area_with`, which is short-sighted, so a
wire-optimal origin buys local wire and loses more globally.

⇒ **Better wire information does not help, because the greedy is not
information-limited — it is commitment-limited.** By the time an item is placed,
the blocks it should sit next to are already fixed somewhere else.

## 2. The LP depth axis, priced exactly — and it stays shut

Same runs, priced against the **current** medians:

| arm | quality | RF | **NET** | dt p50 / max |
|---|---|---|---|---|
| LP k = 2 | +0.3075 % | **−1.2611 %** | **−0.9536 %** | +0.018 s / +0.745 s |
| LP k = 4 | +0.3983 % | −3.4854 % | **−3.0871 %** | +0.139 s / +1.225 s |

Deeper LP *is* quality-positive on the graded shape — and it cannot pay for
itself. This corroborates the shipped `_L157_DEPTH` being flat at 1 for every n,
which is where the wrapper already stands.

### 2.1 🚨 The pricer was reading stale medians, and it flips this decision

`l146_rf_price.py` hardcodes `C_median_runtimes_beta_hidden.csv` — the medians as
published **2026-08-19**. They were republished on **2026-08-23** and every one of
the 100 came down:

    new/old ratio   min 0.4837   p50 0.7418   max 0.9428
    p50 median      2.585 s  ->  2.060 s

Lower medians raise `t/M`, so RF leaves its 0.7 floor sooner and every slowdown
costs more. Priced on the stale file the same two arms read **−0.19 %** and
**−0.61 %** of RF instead of −1.26 % and −3.49 % — understating the bill by
**6.6×**, and turning LP k = 2 from **NET −0.95 %** into **NET +0.12 %**, i.e. from
rejected into a shippable-looking candidate.

`l276_price.py` reads the 2026-08-23 file, prices **added seconds** rather than a
multiplier (l146's own docstring is right that a dt distribution is not a ratio —
the expensive cases are the big-n ones and they have the least slack), and prints
the stale-median number alongside so the gap stays visible.

## 3. What would have to be true for the hpwl line to reopen

Stated so it is falsifiable rather than discouraging. One of:

1. **A topology-changing move the ledger has not tried.** Every closed route above
   is either a unit-PAIR exchange (M64), repair-after-commitment (L256/L259/L262),
   or whole-placer replacement (M27/L129). ⚠️ **Corrected 2026-08-27:** M64 was
   already *multi-pair* — its own docstring says a target is a UNIT pair and "ALL
   block pairs spanning the two units get their separation row REPLACED". So the
   untried thing is not "multi-pair" but **RELOCATION**: move one unit to a
   different position in the ordering, flipping every pair involving it at once.
   See `L280_GROUPING_RED.md` §5 for why M64's 86.8 % LP-infeasibility may be
   self-inflicted by its own over-constrained move semantics rather than a property
   of the instance. L129's memory names "full GORDIAN alternation" as the same
   unfinished work.
2. **A commitment rule that places connected blocks together**, rather than a
   scoring rule that prefers connected positions. The distinction is §1's finding:
   information at placement time is not the binding constraint.
3. **Evidence that the graded corpus is unlike the in-set** after all. Everything
   here rests on L275's corpus argument; if the hidden set turned out to resemble
   OOS, the area line reopens and this one changes weight.

⚠️ Any of them must be measured on **both** corpora from the first arm — L275's
rule. Two counterexamples in two days is enough.

## 4. Honest limits

1. The LP-depth arms are `k = 2, 4`. `k = 12` was not run; the trend (0.9 → 1.3 %
   of hpwl_gap) and the L267_L269 §2.4 figure (7.5 % at depth 12, on OOS) both say
   deeper does not change the conclusion, but it was not measured here.
2. The RF pricing population is the **beta hidden** set (real runtimes, real
   medians, real per-case quality) with the locally measured `dt` mapped on by
   `block_count`. That mapping is the approximation; it is legitimate because both
   corpora are 100 cases over the same n-range, and `dt` is a property of the
   mechanism at a given n, but it is not the hidden set's own `dt`.
3. Local eval forces `RF = 1.0`, so the quality columns are RF-free by
   construction and the RF term is supplied entirely by the pricer.

## 5. Files

```
l276_lpdepth.sh          the k=2 / k=4 driver
results_L276_k{2,4}.json
l276_price.py            exact RF pricing on the 2026-08-23 medians, from a
                         measured per-case dt vector; prints the stale-median
                         number alongside
l276_lpdepth.log
```

Nothing was shipped and `constructive.cpp` was not modified.
