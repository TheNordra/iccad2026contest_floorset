# L278 — do NOT drop the MIB twins. In-set cannot vote on them.

A corpus re-audit recommended removing the 8 shipped L124 MIB-bucket twin profiles
(`_M124_SRC`, pool indices 94–101, cores-gated ≥ 40), on the grounds that their
in-set quality value is provably zero while their runtime cost is real — the
textbook L275 shape.

**The first half is true and the conclusion does not follow. Verified here, and the
recommendation is rejected.**

---

## 1. Both halves of the claim, measured directly

| | |
|---|---|
| twins are shipped | `_M124_SRC = (16, 31, 87, 88, 89, 90, 92, 93)`, pool at n=120 goes **51 → 43** with `ICCAD_M124_TWIN=0` |
| in-set quality with twins OFF | **1.22632513 vs 1.22632513 — 0/100 movers, +0.0000 %** |
| in-set MIB violations | **0**, provable floor 0 (measured independently in L277) |

So yes: on the in-set 100 the twins do exactly nothing.

## 2. 🚨 Why that is uninformative rather than negative

The twins exist to bucket MIB groups. The two corpora differ *pathologically* on
exactly that constraint:

| | in-set | held-out |
|---|---|---|
| boundary | 78 (83.0 %) | 109 (19.4 %) |
| grouping | 16 (17.0 %) | 32 (5.7 %) |
| **MIB** | **0 (0.0 %)** | **420 (74.9 %) — all 100 cases, median 4 per case** |

The cause is recorded and structural: an MIB group can only collapse to one shape
if the members' ±1 % area windows intersect, and **100 % of in-set MIB groups can
do that while only 2.5 % of held-out ones can** (median target-area span 5.8×).

⇒ **The twins' in-set delta could not have come out positive.** There was nothing
for them to fix. That is handoff trap #7 — *a metric can be bounded by
construction; ask whether your discriminator could have come out the other way* —
and here the answer is no.

Their measured value where the antecedent exists: **OOS s1 +1.2005 % / s2
+0.4697 %** (L124 R4), re-priced at L179 to +0.9090 % / +0.4237 %.

And the graded corpus is not in-set-like on this axis. Back-deriving vrel from
`cost = (1 + 0.5(h+a))·exp(2·vrel)`:

    beta hidden   vrel 0.042518   88/100 cases carry a violation
    in-set 100    vrel 0.014073   54/100

i.e. the corpus that gets graded carries **~1.8× the in-set violation density**
after de-versioning (3.0× raw, but beta ran M73). Whether any of that surplus is
MIB is undetermined — but in-set's zero is the one value we know is unrepresentative.

## 3. 🚨 And their cost cannot be measured on this box

Turning the twins off read **+2.9 % slower** (127.4 s → 131.1 s), which is inside
the 4.0 % run-to-run spread measured on the baseline itself. Removing 8 of 51
profiles cannot actually be slower; the measurement is simply blind.

It is blind for a structural reason. This box has **32 physical cores**, and
`ICCAD_ADAPTIVE_CORES=48` changes only which profiles the pool *selects* — it does
not create 48-way parallelism. The grader runs the profile phase on **48 cores**,
so:

    51 profiles / 48 cores  ->  TWO waves
    43 profiles / 48 cores  ->  ONE wave

Crossing that boundary is potentially a large wall change, and it is exactly the
thing a 32-core box cannot show. Same family as
`[[route-a-inverts-below-48-cores]]`.

⇒ The re-audit's "RF bill 0.5296 pp, corpus-insensitive" is a *model* output, not a
measurement, and the sign of the local number is wrong.

## 4. Verdict

**Undetermined, and therefore: do not act.** Removing them would trade a measured
gain where the antecedent exists (OOS +0.42…+1.20 %) for a wall saving that cannot
be measured here and a hidden-set MIB exposure that is unquantified.

## 5. 🔑 The rule this corrects — my own, from L275

L275 concluded: *"OOS guards against over-fitting, the in-set guards against
difficulty mismatch; a candidate must be positive on BOTH."* Applied naively that
rule kills the twins, and it would be wrong.

The corrected form:

> **A corpus can only vote on a mechanism whose antecedent it contains.** Before
> reading a null as a rejection, count the instances the mechanism acts on. Zero
> instances is not evidence of no effect; it is absence of evidence, and the two
> look identical in a portfolio delta.

This is the mirror of L275's own failure mode. L275 caught mechanisms that were
*overstated* because OOS had 70 % more of the gap they harvest. The same logic run
backwards catches mechanisms *understated* because a corpus has none of it —
in-set MIB being the extreme case, at exactly zero.

Practical form for the next candidate: alongside the portfolio delta, always report
**how many instances of the antecedent the corpus contained**. L277's inventory
(59 boundary / 22 grouping / 0 MIB) is that number for the violation family, and it
is what makes the twins' in-set null readable.

## 6. Files

```
results_L278_notwin.json   in-set 100 @48c with ICCAD_M124_TWIN=0
l278_notwin.log
```

Nothing changed. `build_submission.D/` and `constructive.cpp` untouched.
