# L279 — the preplaced boundary violations are the area deficit wearing a different hat

L277 left one lead open and called it the cheapest thing on the violation axis:
**23 of the 59 in-set boundary violations sit on `preplaced` blocks**, whose
positions are given by the problem, so satisfying them is a question about where
the bounding box lands rather than about placement. L136 found and fixed exactly
one such family before (`MARGIN` 1e-4 against the scorer's 1e-6, worth +0.5972 %),
so the question was whether another one is hiding there.

**Answer: none of the 23 is independently fixable, and the reason is that they are
not a violation defect at all — they are the density deficit, counted in the
violation term.**

---

## 1. The partition

Two independent tests per violation:

* **HARD** — does another **preplaced** block extend beyond this one on the side it
  must touch? Its position is given too, so the bbox edge can never retreat past
  it: unsatisfiable for anyone, us or the label.
* **LABEL** — does the ground truth satisfy the same constraint? The label is a
  real legal layout, so a pass means the requirement is reachable.

| | count |
|---|---|
| nobody can satisfy it (HARD, label fails too) | 3 |
| label fails it too, not provably hard | 4 |
| **label SATISFIES it → reachable** | **16** |

So 16 of 23 looked like money on the table.

### 1.1 🚨 The bug the partition caught in its own test

The first run put **10 rows in both HARD and label-SATISFIED**, which is
impossible — the label shares our preplaced positions, so anything structurally
blocked for us is blocked for it.

Cause: the test counted `fixed` blocks as immovable. It is
`is_preplaced` that pins a **position**; `is_fixed` pins only the **shape**
(`constructive.cpp:1745-1747` sets `placed[i]=1` for `is_preplaced` alone). With
`fixed` removed from the test the contradiction goes to **0** and HARD drops
17 → 3.

The self-contradiction line was in the report format from the start, which is the
only reason it was caught rather than published.

## 2. Why the 16 are not money on the table

For each violation, count what actually lies beyond the block on the side it must
touch:

    blocks sticking out past it:  min 6   median 12   max 20
    violations where every outlier is MOVABLE (no preplaced blocker):  21/25

It is not one block in the way. A **slab of a dozen blocks** extends past the block
that is supposed to define that edge. So the fix is not a nudge — the bbox has to
retreat. By how much:

    bbox shrink required on that side:  median 14.4 %   mean 14.3 %   max 27.1 %
    our area_gap on those same cases:   median 0.1634   mean 0.1602

🔑 **A 14 % shrink of one side is the area gap.** Satisfying these violations and
closing the area deficit on that side are the same act. There is no cheap version:
the preplaced boundary violation is a *symptom* of packing at ~81 % utilisation
against the label's 96.6 %, not an independent defect with its own repair.

That is also why the label satisfies them: it packs at 96.6 %, so its bbox edge
lands on the preplaced block for free.

## 3. What this changes

**The axes are not mechanistically independent.** The graded-shape headroom
decomposition — hpwl 10.41 % / area 5.67 % / vrel 2.81 % — is arithmetically sound
(it is just the cost formula), but it must not be read as three separable projects
whose returns add. Concretely:

* 23 of 59 boundary violations (worth **+1.19 %** of the +2.81 % vrel prize) are
  redeemable only by closing the area gap;
* so the vrel prize that is reachable *without* touching density is at most
  **+1.62 %**, and L277 already measured that the mechanism for the movable half —
  post-hoc snapping — nets **+0.0012 %**, because on the heavy cases it pays more
  in hpwl than the violation is worth.

Combined with L275 (the area routes are corpus artefacts on the graded shape) and
L276 (hpwl is topology, ~99 % of it survives exact optimisation inside our
topology), the violation axis closes the same way: **its reachable part is small,
and its large part is the density problem under another name.**

## 4. What remains, honestly

* **3 violations are provably unsatisfiable by anyone.** Nothing to do.
* **4 more are not satisfied by the label either.** No evidence they are reachable;
  they may be, but there is no witness.
* The **16** are a re-statement of the density target. If a density mechanism ever
  works on the graded corpus, these come along free — and should be *counted* as
  part of its prize rather than as a separate one. That is the practical use of
  this report: it stops the next person double-counting them.

## 5. Honest limits

1. The LABEL test is an oracle (it reads ground truth). It diagnoses; it can never
   be part of a shipped mechanism. Same status as M26 / M68 / M79's oracle probes.
2. "Label satisfies it" shows the constraint is reachable *in some legal layout*,
   not that it is reachable from ours at acceptable cost in hpwl and area — and
   §2 is precisely the evidence that it is not.
3. The shrink figure is the shrink needed on **one side**, assuming everything else
   holds. A real layout achieving it would redistribute, so 14.4 % is an indicative
   scale, not a construction.

## 6. Files

```
l279_preplaced.py    the partition (HARD / label) + the outlier and shrink analysis
```

Read-only. Nothing shipped; `constructive.cpp` and `build_submission.D/` untouched.
