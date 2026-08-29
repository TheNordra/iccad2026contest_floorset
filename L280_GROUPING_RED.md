# L280 — commit connected blocks together. RED, and it completes a pattern.

L276 diagnosed the wire deficit as **commitment-limited, not information-limited**:
by the time the greedy places an item, the blocks it should sit beside are already
fixed elsewhere, and no scoring rule can undo that. The corollary was that only
*committing them together* could help — and a corpus re-audit confirmed the axis
had **no verdict on any corpus**: `constructive.cpp:1772-1789` builds compound
items from `cluster_map[blocks[i].cluster]` and from nothing else. Connectivity has
only ever re-ORDERED items or re-scored candidate positions, never formed one.

**Built, gated, measured on the graded shape: +4.6163 %. Decisively RED — and it
made `hpwl` worse, which is the term it exists to fix.**

---

## 1. The mechanism, and why it has no fitted constant

Two movable, non-cluster blocks become one compound item iff they are each other's
**heaviest b2b neighbour** — mutual top-1. That is a matching, so every block joins
at most one synthetic item. No threshold, no K, no weight: L266's over-fitting mode
cannot apply.

Inserted between the cluster loop (which sets `used[]`) and the singles loop (which
skips `used[]`), so a paired block is emitted once, as part of its compound item.

| gate | | |
|---|---|---|
| `constructive_l280.exe`, flag off | **102/102 PASS** | bit-identical to shipped |
| flag on | **0/102** | live on every profile |

## 2. The antecedent is large, so the null is informative

Per L278's rule — count the instances before reading a delta:

    in-set 100:  1276 mutual top-1 pairs, 100/100 cases carry at least one,
                 covering 36.2 % of all blocks   (heavy band: 422 pairs, 38.2 %)

This is nothing like the MIB twins' zero. If the mechanism did anything, this
corpus would show it.

## 3. Result

    in-set 100, official strict scorer, 48 cores, weighted exp(n/12)

      base    1.226325    hpwl 0.2484   area 0.1355
      l280    1.282936    hpwl 0.2860   area 0.1798    +4.6163 %
      heavy 20                                          +4.6954 %
      movers 100/100   (26 better / 74 worse)   feasible 100/100

Both terms degrade, and hpwl degrades by more than 15 % of its own value.

**Why.** A rigid compound item is a bigger, less flexible object with a frozen
internal arrangement. The pair's own edge gets shorter — but the pair now has to be
placed as a unit, so its position is a compromise between two blocks' neighbourhoods
and *every other* edge from either block gets longer. Meanwhile the larger object
packs worse, so area goes up too, and a looser packing lengthens wire again. The
one edge bought is the smallest of the three effects.

## 4. 🔑 The pattern this completes

Three independent attempts to improve hpwl, on three different levers, all failing
the same way:

| attempt | lever | hpwl |
|---|---|---|
| **L272** hint feeds the wire term | better *information* at scoring time | 0.2924 → **0.2999** |
| **GUIDE_MED** wire-optimal candidate origin | better *candidate* at scoring time | 0.2484 → **0.2538** |
| **L280** mutual-top-1 compound items | *commitment* — the lever L276 pointed at | 0.2484 → **0.2860** |

**Every intervention that makes the greedy more wire-aware makes wire worse.** Not
one of them merely failed to pay; each degraded the quantity it targeted.

The reading that fits all three is M27's, re-confirmed on the graded corpus with
three mechanisms it never saw: **the shipped greedy sits on the (area, HPWL)
frontier**, and these perturbations do not move along it — they move *off* it. The
score's own arithmetic explains why they cannot help by accident: `Cost` prices
hpwl and area identically, so a mechanism must buy one for less than it spends on
the other, and all three spend on both.

⇒ L276's "commitment-limited" diagnosis is **correct as a diagnosis and closed as a
prescription**. The blocks really are committed too early; committing them jointly
is not the repair, because a joint commitment is a worse object to pack.

## 5. What is left on this axis

Nothing cheap. What survives L276 §3's list, with L280 now removed from it:

1. **Unit RELOCATION in topology space** — move one unit to a different position
   in the ordering, which flips every pair involving it at once, then re-solve the
   LP.

   ⚠️ **Correction to how this report and L276 first described M64.** M64 was *not*
   a single-pair flip. `m64_flip_probe.py`'s own docstring: a target is a UNIT pair
   and "**ALL block pairs spanning the two units get their separation row
   REPLACED** by direction k". It was already a coordinated multi-pair move — so
   "multi-pair coordinated exchange" is not the untried thing; *relocation* is.

   And M64's death cause makes relocation MORE attractive, not less. 459/529
   attempts (**86.8 %**) were LP-infeasible, and M64_REPORT §4 attributes that to
   "the fixed-disjunct chains of the other ~3000-4900 pairs plus the envelope/bbox
   geometry", having disproved boundary equalities as the cause (15 infeasible
   attempts re-solved without them: **0/15** became feasible).

   🔑 But the probe's own HONEST-SCOPE note says forcing one direction on every
   member pair means "slide all of A past all of B on that axis — **stronger than
   the evaluator's pairwise requirement**; a mixed per-pair topology could be
   feasible where this is not." So a large part of that 86.8 % may be **self-
   inflicted by the move's semantics** rather than a property of the instance. A
   relocation in a sequence-pair ordering is by construction a realisable topology,
   which is exactly the over-constraint M64 imposed on itself.

   ⇒ The prior is better than the re-audit's "most likely ≈0", but this is a real
   build on a 33 KB LP tool, not a knob.
2. A **different placer**, which is M27/L129 and priced at 1.745 against 1.237.

Both are large. Neither is a knob.

## 6. Honest limits

1. Mutual top-1 is one grouping rule. A weaker one (group only when the top edge
   dominates the second by some margin) would touch fewer blocks — but that
   reintroduces a fitted constant, and the failure here is not marginal: +4.62 %
   with both terms worse is not a tuning problem.
2. The compound item's internal arrangement comes from `make_group_item`, which was
   designed for constraint-given clusters. A better internal layout for synthetic
   pairs might reduce the damage; it cannot plausibly reverse a 4.6 % loss whose
   mechanism is loss of placement freedom.
3. Measured on the graded shape; the OOS heavy-band arm is reported in §7 for the
   corpus record, per L275's rule that a candidate be measured on both.

## 7. OOS heavy band — for the record

Same binary, OOS heavy band (40 cases, n >= 101, sample s1):

| | gross | deployable (post shape-LP) | hpwl | area | vrel |
|---|---|---|---|---|---|
| ship | — | — | 0.2924 | 0.2300 | 0.0893 |
| `l280` | **+4.6696 %** | **+3.7240 %** | **0.3088** | **0.2934** | **0.0967** |

All three terms degrade on both corpora, and the two headline numbers agree to
0.05 pp (in-set +4.6163 %, OOS +4.6696 %).

🔑 **That agreement is itself worth recording.** L275's whole finding was that the
two corpora disagree — 4/4 sign flips — for mechanisms whose value scales with a
gap. This one does not disagree at all, because it is not harvesting a gap: it is
removing placement freedom, and that costs the same everywhere. **Corpus
sensitivity is a property of the mechanism, not of the measurement**, and a
mechanism that reads the same on both is telling you something about itself.

## 8. Files

```
l280_patch.py          pristine constructive.cpp -> constructive_l280.cpp (3 patches, md5-guarded)
constructive_l280.exe
results_L280_inset.json    l280_inset.log
l280_oos40.pkl             l280_oos40.log
```

Nothing shipped; `constructive.cpp` and `build_submission.D/` untouched.
