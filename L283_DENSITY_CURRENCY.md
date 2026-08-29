> # 🚨 CORRECTION (2026-08-27, same day) — §1-§3 measured a superseded configuration
>
> The 4200 layouts come from `audit_cache_ship.pkl`, which is **stale**: REFINE
> band 6/4 against the shipped 2/2, no shape LP, and its pool-best reproduces the
> shipped per-case cost on only **4/100** cases (weighted 1.292646 vs 1.226325).
> It is an M74-era pool.
>
> **Invalidated:** the frontier rates in §2 (median 0.977), the 2.67 : 1 exchange
> rate in §3, the boundary 67.3 % / grouping 32.7 % split in §3.1, and the "25 of
> 100 cases" population. They describe the M74-era placer.
>
> **Still valid:** §4 in full — the label's operating point is recomputed from the
> dataset itself, and our own utilisation there is taken from the shipped anchor
> `results_L274_base_48c.json`. (Note §4 quotes 84.5 % unweighted; the exp(n/12)
> weighted figure is 85.16 %.) §3.2's enrichment test also uses the shipped
> anchor and stands.
>
> **Directionally unaffected but unquantified:** that density is paid for in soft
> violations rather than in wire. The mechanism was reproduced from two
> independent directions (L279 and this report), but its *price* has not been
> measured on the shipped configuration.

# L283 — yes, a shorter chain can be had without compressing. The bill just moves to violations.

L282 closed post-hoc chain shortening and left one question open: its 2.74 : 1
exchange rate was measured by *squeezing a committed layout*, so it might have
been an artefact of squeezing rather than a property of density. If a shorter
chain could be **generated** rather than squeezed into, the packing-time idea
would still be alive.

**It can. Wire is not the obstacle — the generation-side price of density is
0.977 : 1, i.e. break-even, against 2.74 : 1 for squeezing. And it changes
nothing, because density is paid for in a third currency: soft violations, at
2.67 : 1 against.**

    the 25 in-set cases whose own profile pool contains a layout that is
    BOTH denser AND better-wired than the one the portfolio selected

      quality bracket gained (area + hpwl)   +2.5564 %
      violation multiplier paid              +6.8343 %
      NET cost                               **+4.0602 %**   (worse)
      ratio paid : gained                    **2.67 : 1**

`vrel` rose in **25 of 25**. Not one of them is free.

---

## 1. Method — no new placer, because the data already existed

`audit_cache_ship.pkl` holds the positions of **42 profiles × 100 in-set cases =
4200 layouts** in the shipped configuration. Different profiles are genuinely
different packings of the same instance, not compressions of one another, so the
frontier they trace out *is* the generation-side exchange rate. All 4200 were
re-scored with the official strict scorer (322 s).

**Why this is not circular.** The portfolio selects by a proxy that is per-case
oracle-perfect (M13/M76) and minimises `1 + 0.5·(hpwl_gap + area_gap)`, so on a
convex frontier the local slope at the selected point is ≥ 1 by construction —
measuring "the slope is ≥ 1" would prove nothing. What is not tautological is
*how much* more than 1, measured on the denser side only:

    for each case, over every pool layout DENSER than the selected one,
      rate = (hpwl_gap - hpwl_gap_sel) / (area_gap_sel - area_gap)

(The sanity check that the portfolio-selected layout is the pool's cost-minimum
on every case passes, confirming the proxy is behaving as the ledger claims.)

## 2. Wire is not the obstacle

    84 of 100 cases have at least one denser layout in their own pool
    16 of 100 pools contain no denser layout at all

    cheapest observed hpwl_gap paid per area_gap bought
      min −275.4   p25 −0.115   **p50 +0.977**   p75 +2.460   max +35.4

      rate < 0     (denser AND better wire)        25 / 84
      rate < 1     (cost falls if taken)           43 / 84
      rate < 2.74  (cheaper than L282's squeeze)   67 / 84

**Median 0.977 is break-even, and 25 cases are outright negative.** So L282's
2.74 : 1 really was an artefact of compressing an already-committed layout, and
L282 §6's open question is answered in the affirmative: **a shorter chain is
available without paying wire for it.**

## 3. And it is still not worth taking, because violations pay instead

For the 25 cases where density is free in *both* quality terms, the layout is
nonetheless rejected — correctly — because `Cost = (1 + 0.5·(hpwl_gap +
area_gap))·exp(2·vrel)` and the third factor moves:

      d_vrel  p25 +0.03030   **p50 +0.04651**   p75 +0.05660
      exp(2·d_vrel) at the median = **1.0975**

      quality bracket gained  +2.5564 %
      violation cost paid     +6.8343 %      ratio **2.67 : 1**

🔑 **Compare the two numbers this project now has for buying density:**

| route | currency | rate against |
|---|---|---|
| L282 — squeeze the committed layout | **hpwl** | 2.74 : 1 |
| L268 — big-first commitment order (packing time) | **hpwl** | 1.2 : 1 |
| **L283 — generate a denser layout** | **violations** | **2.67 : 1** |

The currency changed and the price did not. That is the finding.

### 3.1 Which soft constraint pays

    soft-violation counts over the same 25 cases

      type        selected   denser   delta   share of the increase
      boundary          27       60     +33          67.3 %
      grouping           5       21     +16          32.7 %
      mib                0        0      +0           0.0 %

Packing denser pulls blocks off the bbox edges they are required to touch, and
splits clusters. MIB contributes nothing, which is consistent with L278's
finding that in-set MIB is 0 by construction.

This is **L279 from the other side.** L279 concluded "preplaced boundary
violations *are* the density deficit under another name" by showing the
violations could only be fixed by closing the area gap. L283 measures the same
identity in the opposite direction — closing the area gap re-opens the
violations — and puts a price on it.

### 3.2 The mechanism is *not* that the chain is made of pinned blocks

The obvious explanation would be that the critical chain is long because it is
built from boundary-constrained blocks that cannot move. It is not:

      blocks on the binding critical chain          1278 / 7050  (18.1 %)
      blocks carrying a boundary constraint         2403 / 7050  (34.1 %)
      chain blocks that are also boundary-constrained  460 / 1278 = 36.0 %
      against a 34.1 % base rate  ->  enrichment **1.06×**

Essentially no enrichment. The chain is not a queue of pinned blocks; the whole
packing is loose, and the looseness is what buys soft-constraint satisfaction.

## 4. The reference solution is on the same trade — and gets a better rate

Recomputed from the dataset rather than quoted:

      utilisation   label **97.1 %**   ours 84.5 %
      vrel          label **0.05110**  ours 0.01889   (the label is 2.7× worse)
      label cost = exp(2·vrel), both gaps zero by construction = 1.10761

**The label buys 12.6 pp of extra density with +0.03222 of vrel.** It is on
exactly the trade L283 measures — it is not avoiding it, it is taking it. The
difference is the rate: the label converts that violation budget into *both*
gaps going to zero, whereas our own pool's denser layouts pay **more** vrel
(+0.04651) for **far less** quality (+2.56 % of bracket).

⇒ The trade is not intrinsically bad. **Our generator's version of it is.**

## 5. Verdict — the axis closes, and now for the right reason

The chain-shortening idea is now bounded from both ends:

* **post-hoc** (L282): the unit cannot reach its destination, and where it can,
  wire costs 2.74 : 1;
* **at generation** (L283): wire is free, but soft violations cost 2.67 : 1.

⇒ **A packing-time chain-shortening rule is not blocked by wire, as L282's
reading suggested — it is blocked by boundary and grouping constraints.** Any
such rule must be measured against `vrel`, not against `area_gap`, and the
sub-question L282 posed ("is the wire penalty intrinsic to density?") is
answered: **no, and it does not help.**

### 5.1 What would actually have to be true

Not "shorten the chain". The falsifiable target is now sharper:

> **a mechanism that raises utilisation while holding boundary and grouping
> violations fixed.**

The label proves such layouts exist (97.1 % utilisation is achievable on these
instances). Our 42-profile pool proves our generator cannot produce them: every
denser layout it has ever generated pays for the density in violations, at a
rate 2.67 : 1 worse than the score will accept. That is a statement about the
*packer*, which puts it back in M27/L129 — a different placer — exactly where
L281 §10.1 and the previous handoff already pointed.

⚠️ Do not read this as "violations are the new lever". L277 measured the
violation axis directly and found only 12/81 removable with a post-hoc snap
worth +0.0012 %. The point here is not that violations are cheap to fix; it is
that they are what density *costs*, so the density and violation prizes are one
prize counted twice — L279's rule, now with a number.

## 6. Honest limits

1. The pool is **our own 42 profiles**. It bounds what this generator reaches,
   not what is reachable. §4 is the counterexample and it is the label's, not a
   candidate we can build.
2. The 25 "free density" cases are the ones where a strictly-dominating layout
   exists in both quality terms. The rate table in §2 covers all 84 cases with
   any denser layout; the §3 pricing covers only those 25.
3. In-set 100 only. Not re-run on OOS — and per L281 §8 the in-set is the harder
   corpus for anything needing slack, so an OOS reading would be more
   favourable and would not transfer.
4. Everything is the official strict scorer at RF = 1.0; no runtime term is
   involved, and nothing here is a shippable mechanism.

## 7. Files

```
l283_generate_vs_squeeze.py   re-score all 4200 pool layouts, frontier rates
l283_cache.pkl                4200 scored layouts (hpwl_gap, area_gap, vrel, cost)
l283.log
```

`constructive.cpp`, `optimizer_constructive.py` and `build_submission.D/` were
not touched.
