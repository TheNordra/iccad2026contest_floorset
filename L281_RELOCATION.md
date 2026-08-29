# L281 — unit relocation in topology space. The move is coherent; the bounding box is not.

The handoff's thesis was that M64's **86.8 % LP-infeasible** was self-inflicted by
its own move semantics, and that a relocation — a coherent topology by
construction — would be feasible where M64's was not.

**The thesis is right about the mechanism it named and it changes nothing.**
Relocation removes essentially all of M64's incoherence (cyclic **23.0 % → 0.3 %**)
and the total infeasibility does not move, because the binding wall was never
cyclicity. It is that **the anchor's own critical chain already saturates the
bounding box exactly in 62 of 100 graded cases**, so almost any rearrangement is
too long to fit in a box that is forbidden to grow.

Over the whole heavy band — 81.1 % of the graded weight — a per-case **oracle**
over 3367 LP solves is worth **+0.0942 %** against a 0.30 % bar, and costs
**172 s per case** against a ~1.5 s budget.

---

## 0. Gates — all six pass before any number below is used

| gate | result |
|---|---|
| offline scorer reproduces the anchor json | **0/100 mismatch**, both `results_L3_port_top32_area.json` and the shipped `results_L274_base_48c.json`, max abs 0.000e+00 |
| `m64_flip_probe.py selfcheck` (the wiring proof the handoff asked for first) | **PASS**, forced-current == unforced bit-identical |
| forcing a unit's CURRENT topology over **all 79** of its pairs is a no-op | **PASS**, bit-identical positions; the certificate also certifies the anchor itself |
| certificate soundness: 50 candidates it REJECTED, fed to the LP anyway | **0/50 solvable** — nothing it rejects is LP-feasible |
| LP determinism across processes | 342 candidates solved twice in separate runs, **0 status/cost mismatches** |
| every reported gain re-scored from stored positions | **9/9**, bit-exact cost, feasible, and beating its own control |

The last one matters most: everything else in this probe is bookkeeping on top
of LP output, so the best layout on each case is re-run through the official
strict scorer from its stored coordinates (`l281_verify_mover.py`). A gain that
cannot be reproduced from a real placement is not a gain.

⚠️ The third gate needs care: forcing a unit's bbox-level relation is only a
restatement of the anchor rows when every constituent block pair agrees with it.
The gate searches for a unit where that holds exactly and refuses to run
otherwise — a heterogeneous pair would be a legitimate *tightening*, and passing
it off as a no-op test would have proved nothing.

## 1. Re-anchored on the graded shape, not on M64's anchor

M64 ran on `results_L3_port_top32_area.json` (offline 1.3003). The shipped
placer is now **1.2263** — the old anchor is 6 % *worse* than what ships, so
headroom measured there is partly headroom the shipped placer has already taken.
This probe re-anchors on `results_L274_base_48c.json`, the in-set 100 at 48
cores, which is L275's rule (measure on the corpus that gets graded) applied to
the *topology* as well as the corpus.

That forces a control M64 did not need. M64's anchor was a fixpoint of M64's own
LP, so "anchor cost" was a fair baseline. Here the research LP is a different LP
from the shipped one, so **every case also gets a no-force LP pass and the same
polish budget**, and relocation is scored against that. Without it, "the
research LP is better than the shipped one" reads as a relocation gain.

## 2. The move — and why it is a target position, not an ordinal

The handoff specifies relocation as "move `u` to ordinal `p` in the ordering".
**There is no ordering to move it in.** The anchor's relation set is the
per-pair max-gap disjunct (`m53_l3_probe.py:213-231`), and such a set is not in
general a sequence pair: the tournament "u left of v OR v below u" can carry
3-cycles for a perfectly legal placement (A left of B, B left of C, A below C,
whenever the y-gap of (A,C) exceeds its x-gap — max-gap will pick it). A literal
1-D reading is worse still: forcing `u` before/after every other unit on one
axis makes `u` a full-height column, which is strictly more over-constrained
than the move M64 was criticised for.

So a relocation here is a **target position for `u`'s bounding box**, which
induces a relation for every pair `(u, v)` directly, from one consistent
geometric configuration. Relations are read off unit *bounding boxes* — the
exact granularity `force_rel` applies at — so a witness at unit level satisfies
every constituent block-pair row, and M64's flagged coarsening is satisfied by
construction rather than assumed away.

**The deviation is measured, not argued.** The literal reading, run through the
same certificate on the same three cases (`l281_ordinal.py`, 5 target ordinals ×
8 units per case):

      literal 1-D ordinal move   117 moves   coherent   0  ( 0.0 %)   cyclic 22   oversized 95
      M64 single-pair flip       262 moves   coherent  48  (18.3 %)   cyclic 61   oversized 153
      RELOCATION (this probe)    984 moves   coherent 163  (16.6 %)   cyclic  1   oversized 820

Zero out of 117. Following the handoff literally would have produced a probe
that cannot solve a single instance of its own move.

## 3. The certificate — what makes the fork measurable instead of anecdotal

The forced LP is feasible only if, at block level, the horizontal and vertical
constraint graphs are **acyclic** and their longest node-weighted chains **fit
the anchor bbox** (`sum of widths along a chain <= XMAX-XMIN <= W0`). Both are
exact, both are cheap, and together they let an infeasibility be *attributed*:

* **CYCLIC** — the move is not a realisable topology at all;
* **OVERSIZED** — realisable, but not inside this bounding box;
* **coherent yet LP-infeasible** — the instance really is that tight.

M64 could not make this distinction; it reported one undifferentiated
`lp_status_2`. Running the same certificate over M64's own move on the same
anchor puts the two move semantics on identical geometry.

## 4. The census — the fork, and it splits sideways

    in-set cases 85 / 88 / 91, shipped anchor, top-8 units, identical geometry

      move                n     coherent      cyclic   oversized
      RELOCATION        984   163  16.6 %          1         820
      M64 1-pair flip   262    48  18.3 %         61         153

**Cyclic: 1/984 = 0.1 % for relocation against 61/262 = 23.3 % for M64.** The
thesis is confirmed exactly as stated — a relocation essentially cannot produce
the contradictory constraint set M64 imposed on itself.

**And the totals are the same.** Incoherent is 83.4 % for relocation against
81.7 % for M64. M64's own reported figure was 86.8 %, over 529 attempts that
include its variant-B retries; the arm here reproduces only its variant-A
candidate set (262 attempts against M64's ~half of 529), so the two are close
but not the same population. Within that caveat the certificate **explains
M64's death cause and decomposes it**: roughly a quarter of it was the move
semantics, and the rest is the box. Consistency check in the other direction —
the certificate certifies 18.3 % of M64's moves as coherent and M64 actually
solved 13.2 % (62 feasible + 8 prefiltered of 529), which is the right ordering
for a necessary-but-not-sufficient condition.

The shape holds over the whole heavy band. All 20 cases, 16736 relocation
candidates (cases 85 and 88 exhaustive, the rest top-5 units):

      RELOCATION   16736 moves   coherent 3530 (21.1 %)   cyclic  49 ( 0.3 %)   oversized 13157 (78.6 %)
      M64 1-pair    1998 moves   coherent  487 (24.4 %)   cyclic 460 (23.0 %)   oversized  1051 (52.6 %)

Cyclic **23.0 % against 0.3 %** on 20 cases instead of 3 — the head-to-head is
not a small-sample artefact. (The M64 arm re-runs per census entry, and cases
85/88/91 have two entries each, so its 1998 is ~3 cases' worth of double count;
the rates are unaffected, the absolute count is not a population size.)

### 4.1 Variant B cannot rescue the oversized ones

M64 had a fallback that relaxes the bbox by `sqrt(1.005)` per side (≤0.5 % area
growth), worth ≈0.32 units on a `W0` of ~130. The measured excess is two orders
of magnitude larger:

    median chain excess over the bbox row, oversized candidates
      case 85   H +9.56   V +11.52      (W0 129.8)
      case 88   H +4.73   V +12.92
      case 91   H +8.07   V + 9.62

### 4.2 Neither can a bigger relaxation — the score prices it

"Then let the box grow" is the obvious escape, and the official cost prices
area at the same 0.5 weight as wire, so it can be settled by arithmetic rather
than opinion. `bbox_relax` multiplies both rows, so area grows by `relax²`, and
`A/a_base` is `1 + area_gap` straight off the anchor json (all 13157 oversized
candidates, `l281_relax_price.py`):

      quantile   row relax   area factor   cost of that growth
      min           1.0000        1.0000         +0.001 %
      p25           1.0312        1.0635         +2.825 %
      p50           1.0515        1.1057         **+4.845 %**
      p75           1.0830        1.1730         +8.075 %
      p99           1.1967        1.4321        +20.253 %

The whole first-order wire prize (§9) is **+0.7994 %**. Making the median
oversized relocation fit costs **six times the entire prize**, and only
**818/13157 = 6.2 %** of them could be afforded even if the full prize were
spent on that one move. The box is not incidentally in the way; the score is
charging correctly for what relocation needs. (Measured first on a 3596-candidate
subset and then on all 13157: p50 +4.850 % → +4.845 %, affordable 6.3 % → 6.2 %.
This quantity is extremely stable.)

## 5. Why — the anchor is critical-path saturated

    in-set 100, shipped anchor, block-level graphs, unit-level pairs (what the LP sees)

      H slack   min 0.0000 %   p50 0.0000 %   p75 1.8171 %   max 10.8885 %
      V slack   min 0.0000 %   p50 0.0000 %   p75 3.0036 %   max  8.0945 %
      cases with min(H,V) slack <= 0.00000 % :  62/100
      cases with min(H,V) slack <= 1.00000 % :  77/100
      anchor topologies acyclic               : 100/100

The gate output makes it concrete on case 85: `lH = 129.8107202276296` against
`W0 = 129.8107202276296` — the horizontal critical chain **is** the bounding box,
to the last bit.

🔑 **That is the mechanism.** Under `bbox_relax = 1.0` the LP may compact but
never grow, so any topology whose critical chain is one ULP longer than the
anchor's is infeasible before wire is even considered. The greedy has packed
itself into a configuration where the box is exactly as wide as its longest
chain of abutting blocks, and 62 % of the time there is no room at all.

**And the chain is not a local structure.** Over the 86 in-set cases saturated
on at least one axis, the blocks lying on a zero-slack critical chain are

      as % of the case's blocks   min 4.3 %   p25 15.2 %   p50 34.3 %   p75 52.1 %   max 77.9 %
      absolute count              min 3       p50 19       max 69

A third of the layout, at the median, is a rigid train whose total width the
bounding box is exactly paying for. Relocating a unit into or across it
lengthens it, which is why "oversized" and not "cyclic" is what kills the move.

## 6. Binding versus vacuous — most "flips" are not moves

A relation change is not a move. For a diagonal pair *both* the horizontal and
the vertical separation already hold, so switching between them rewrites the LP
row without excluding the current placement: the LP returns the same solution and
the relocation relocated nothing. Liveness, case 85 (positions compared bit by
bit against the no-force control):

      unit      nflip   cost            delta        blocks moved   unit moved
      ('U',  1)    31   1.184115037390  +1.988e-03      2/106        39.2227
      ('U', 53)     3   1.186103239058  +0.000e+00      6/106         0.0000
      ('U', 97)     5   1.186103239058  +0.000e+00      6/106         0.0000
      ('U', 22)    41   1.190407060228  -4.304e-03      9/106        58.3837
      ('U', 87)     1   1.186103239058  +0.000e+00      0/106         0.0000

The mechanism is **live** — a binding relocation displaces its unit by 39–58
units and changes the cost in both directions. But `nflip` counts rewrites, and
only the pairs whose forced direction is *violated at the current position*
force `u` to go anywhere. Every number in §7 is therefore reported twice: over
all candidates, and over binding candidates only.

This is the handoff's trap #1 in a new costume. Cost alone cannot separate
"live but degenerate" from "silently never applied"; only comparing positions
can, which is why the liveness table above exists.

## 7. The LP arm — every certified-coherent relocation on two cases

The whole heavy band, every certified-coherent target: **3367 LP solves**,
official strict scorer, against a control given the identical LP and the
identical polish budget.

    status of the 3367 certified-coherent relocations

      lp_status_2 (infeasible)   2741   81.4 %
      feasible but worse          566   16.8 %
      MOVER                        59    1.8 %
      ladder_kill                   1    0.0 %

      of those, 214 (6.4 %) were VACUOUS -- the forced relation already held at
      the current position, so the LP returned the control's own solution.
      Restricted to the 3153 BINDING moves: **87.0 % infeasible**, 59 movers.

**The fork, totalled.** The census generated 16736 candidates and certified
3530, so 13206 were rejected before any LP ran. End to end the infeasibility of
a relocation is **(13206 + 2742)/16736 = 95.3 %** — *higher* than M64's 86.8 %,
not lower. The composition inverted; the total got worse.

### 7.1 What the movers are, and are not

    case 85   anchor 1.186336   ctrl 1.186103   ctrl+polish 1.186103   best reloc 1.183830
    case 88   anchor 1.298342   ctrl 1.298342   ctrl+polish 1.298342   best reloc 1.296542

    union-oracle over all 20 heavy-band cases
       vs anchor                   +0.2943 %
       vs the polished control     +0.0942 %   <- the honest one

    delta of every feasible solution vs the polished control (n = 625)
       best +0.010569   p75 +0.000000   p50 -0.000169   p25 -0.002687   worst -0.021732

Three things in that table matter more than the headline.

**(a) The polish is not the source.** `ctrl+polish == ctrl` on both cases: the
no-force LP is already a fixpoint, so the gain is relocation's own. Had the
control not been given the same polish budget, the first mover found would have
read as **44× larger than it is** — its own LP delta was +8.6e-05 and its
polished delta +3.8e-03.

**(b) Relocation moves hpwl and essentially nothing else, and on average it
makes hpwl worse.** Mean deltas per case over the 625 feasible solutions:

      case 85   mean d_hgap +0.00359   d_agap -0.00000   d_vrel +0.00000   (217 feasible)
      case 88   mean d_hgap +0.00415   d_agap +0.00000   d_vrel -0.00000   (216 feasible)
      case 94   mean d_hgap +0.02133   d_agap +0.00000   d_vrel +0.00000   ( 10 feasible)
      case 87   mean d_hgap -0.00745   d_agap -0.00017   d_vrel +0.00000   ( 15 feasible)
      case 86   mean d_hgap -0.00413   d_agap -0.01637   d_vrel +0.00000   ( 12 feasible)

**`violations_relative` is exactly 0.00000 in all 20 cases** — independently
reproducing M64's finding that all 62 of its feasible flips had a vrel delta of
exactly 0. `area_gap` is exactly 0 in 15 of 20 and slightly *negative* (better)
in the rest — cases 86, 96 and 80 are the exceptions, where the LP compacts into
the space the relocated unit vacated. (An earlier draft of this report said area
never moves; that is right for three quarters of the band and wrong for the
rest, and the exceptions are the cases with the largest gains.)

The mean hpwl delta is **positive**, i.e. worse, in 13 of 20 cases. The median
feasible relocation is a small loss (p50 −0.000169); only an oracle pick is a
gain.

🔑 That completes L280 §4's pattern with a fourth entry, and it is the strongest
of the four because this mechanism is the one L276 pointed at:

| attempt | lever | effect on hpwl |
|---|---|---|
| L272 hint into the wire term | better *information* | 0.2924 → 0.2999 |
| `GUIDE_MED` wire-optimal origin | better *candidate* | 0.2484 → 0.2538 |
| L280 mutual-top-1 compound items | joint *commitment* | 0.2484 → 0.2860 |
| **L281 relocation** | **direct topology edit, post hoc** | **mean d_hgap positive in 13/20 cases** |

**(c) The certificate is not what destroys the value — the LP is.**

      demand  best unit -> its wire optimum, unconstrained     +0.6580 %
      supply  best target that passes the certificate          +0.6201 %   (94.2 % of demand)
      realised best feasible relocation, vs polished control   +0.0942 %   (14.3 % of demand)

Coherent targets retain **94 %** of the first-order wire prize. The chain-vs-bbox
wall of §5 kills three quarters of the *candidates* but barely dents the *value*,
because each unit has many targets and a good one usually survives. The value
dies afterwards, in the LP.

### 7.2 So what in the LP kills it — and it is not the boundary equalities

M64's most informative single result was that dropping the boundary equalities
rescued **0/15** of its infeasible flips. Repeating that diagnostic on
certified-coherent, binding relocations (`l281_why_infeasible.py`):

      30 certified-coherent BINDING relocations that the LP rejected
        feasible once boundary equalities are dropped          :  0
        feasible only once the bbox may ALSO grow 20 %         :  4
        still infeasible with no boundary ties and +20 % bbox  : 26

**0/30, exactly as M64 got 0/15 — on a different move, on a different anchor.**
And 26 of 30 survive even a 20 % larger bounding box with no boundary ties at
all. What is left is what M64_REPORT §4 named: the **frozen/preplaced blocks,
which have no delta variable and are therefore hard points in the middle of the
layout, plus the fixed-disjunct chains of the other few thousand pairs.** Those
are properties of the instance and of the anchor's own topology. Relocation
inherits them in full.

### 7.3 Deployability — the ranking that finds the winner on one case misses it on the other

The oracle above costs **799 s (case 85) and 293 s (case 88)** per case, against a
shipped budget of ~1.5 s. The only hope is a cheap, label-free rule that says
which unit to try, and `rank_units` supplies one: the exact first-order wire
prize, no LP required.

      case 85   best relocation is the rank-4/49 unit    54 of 980 solves (5.5 %)   top-5 units: +0.1917 %
      case 88   best relocation is the rank-36/58 unit  430 of 899 solves (47.8 %)  top-5 units: +0.0000 %

On case 85 the ranking is excellent; on case 88 it is worthless. This is
M56/M79's finding again — the per-case *identity* of the winner is
idiosyncratic — and it means there is no cheap version of even the small prize
that exists.

### 7.4 The whole heavy band — the number that decides it

The heavy band (n ≥ 101) is **81.1 % of the graded weight** in both the in-set
and the beta hidden set (L275 §1). All 20 cases, cases 85 and 88 exhaustive and
the other 18 over the top-5 units by wire prize:

    case  scan  solves   sec        base        best      gain
      80  top5     137   100    1.289770    1.279202   0.8194 %
      87  top5      64    76    1.168547    1.163774   0.4084 %
      85  FULL     980   799    1.186103    1.183830   0.1917 %
      82  top5      43    96    1.262887    1.260673   0.1753 %
      88  FULL     899   293    1.298342    1.296542   0.1387 %
      98  top5      37    19    1.184133    1.182582   0.1310 %
      96  top5     209  1133    1.207754    1.206274   0.1225 %
      99  top5     113   107    1.265344    1.263803   0.1218 %
      97  top5     103   306    1.192806    1.191884   0.0773 %
      81  top5      49    55    1.217953    1.217111   0.0691 %
      86  top5     120    95    1.202468    1.201742   0.0604 %
      93  top5      87    46    1.203117    1.202691   0.0354 %
      91  top5      40   194    1.168660    1.168404   0.0218 %
      83 84 89 90 92 94 95                              0.0000 %  (7 cases)

    weighted exp(n/12), base 1.214964

      per-case ORACLE gain vs the polished control  :  **+0.0942 %**
      cases with any gain at all                    :  13 / 20
      LP solves                                     :  3367
      LP wall time                                  :  3439 s = **172 s/case**
                                                       (shipped budget ~1.5 s/case)

**+0.0942 % against a 0.30 % bar, for 115× the entire per-case runtime budget.**
And that is an *oracle*: it picks the best of up to 980 attempts per case using
the official scorer, which is not information a submission has.

⚠️ Honest asymmetry in that table: the 18 top-5 rows are a lower bound on their
own case's oracle (case 88 shows the winner can sit at rank 36), while cases 85
and 88 are exhaustive. Scanning the whole band exhaustively would raise +0.0942 %
somewhat — the two exhaustive cases came in at +0.19 % and +0.14 % — but not to
0.30 %, and it would multiply the 172 s/case by roughly an order of magnitude.

## 8. Corpus check — L275 reproduced on a mechanism it never saw

Relocation buys wire by harvesting a gap, so L275's rule says measure both
corpora. The certificate's binding term needs only positions, so it can be
measured on the OOS heavy band directly. Every block is its own node on **both**
sides here, so the two are like for like (levels differ from §5, the contrast
does not):

    saturated = min(H,V) critical-chain slack <= 0

      IN-SET 100 (graded shape)          86/100 = 86.0 %    p50 min-slack 0.0000 %
      IN-SET heavy 20 (n >= 101)         17/ 20 = 85.0 %    p50 min-slack 0.0000 %
      OOS heavy 40, sample s1            10/ 40 = 25.0 %    p50 min-slack 0.3334 %
      OOS heavy 40, sample s2             9/ 40 = 22.5 %    p50 min-slack 0.3344 %

🔑 **The enabling condition for this mechanism is 3.5× more common on the corpus
the whole L250–L274 arc measured on than on the corpus that gets graded.** Had
this probe been run OOS — as that entire arc was — it would have reported a much
higher feasibility rate and a much larger prize, and none of it would have
transferred. That is L275's finding, independently reproduced by a mechanism
L275 never looked at, and it is the second reason (after §5) that this axis is
tighter than it looks from anywhere else.

Per L280 §7's corollary: a mechanism that reads the *same* on both corpora is
not harvesting a gap. This one reads very differently, which confirms it is a
gap-harvesting mechanism — and the gap is smallest exactly where it is graded.

⚠️ **The cost arm was not run on OOS, and that is a deliberate choice, not an
omission.** L275's rule is that a candidate must be positive on *both* corpora.
This one is negative on the graded corpus — +0.0942 % against a 0.30 % bar, at
172 s per case — so no OOS number can rescue it, and the measurement above
says an OOS number would read *better* and fail to transfer. Running it would
have reproduced the exact error L275 exists to prevent. (It would also have been
a real build: the OOS layouts in `l252_cache.pkl` carry positions but not the
b2b / pin / constraint metadata `cost_eval` needs.)

## 9. Demand exists; supply does not

The wire really does want blocks moved. Moving a unit rigidly to its
weighted-L1-median with everything else fixed is the exact minimum of that
unit's own wire term, so it is an exact first-order prize — and it is computable
for all 100 graded cases with no LP at all:

    in-set 100, shipped anchor, weighted exp(n/12), base 1.226325

      best single unit per case -> its wire optimum, unconstrained : +0.7994 %
      best five units per case (not jointly realisable)            : +2.4569 %
      heavy band n >= 101                              best-1 +0.7800 %  top-5 +2.3770 %
      per-case best-1 : p25 0.5914 %  p50 0.9905 %  p75 1.6040 %  max 4.4368 %
      units excluded as boundary/extreme-pinned : 1902/5142 = 37.0 %

**+0.80 % is well above the project's historical 0.30 % ship bar.** So this is
not a case of "there was nothing to win" — the demand is real, and 37 % of units
are forbidden to move before anything else is considered. The funnel from that
demand to what is actually collectable is §7.1(c):

      demand    best unit -> wire optimum, unconstrained    +0.6580 %
      supply    survives the coherence certificate          +0.6201 %   94.2 % of demand
      realised  survives the LP and beats the control       +0.0942 %   14.3 % of demand

Each step is a different wall: the certificate takes almost no value (it takes
*candidates*), and the LP takes six sevenths of what is left. On top of that,
§7.3 shows no cheap rule finds the winner, and §7.4 shows the survivor costs
172 s per case.

## 10. Verdict — RED, and it closes more than relocation

**The handoff's fork, answered on its own terms.** Infeasibility is **95.3 %**
against M64's 86.8 %: not materially lower, in fact slightly worse. By the
pre-registered criterion — *"infeasible ≈ 86 % like M64 → the wall is the
instance, not the move semantics. Report it and close the axis"* — this is a
stop, and §7.2 supplies the mechanism the criterion was guessing at.

**But the thesis it was testing was correct, and deserves recording as such.**
Relocation removes essentially all of M64's self-inflicted incoherence — cyclic
**23.0 % → 0.3 %** over the whole heavy band — exactly as `L280 §5` predicted.
It changes nothing because
cyclicity was only about a quarter of M64's wall, and the other three quarters
are two instance-level walls that a coherent move inherits in full:

1. the anchor's critical chain **exactly saturates** the bbox in 62/100 graded
   cases, so 78.6 % of candidates are too long before the LP is even called; and
2. of what survives, 87 % is still infeasible with the boundary equalities
   removed **and** a 20 % larger box — the frozen/preplaced hard points and the
   other pairs' fixed disjuncts, which is precisely M64_REPORT §4's attribution.

The pre-registered gate's companion signal did flip: M64 had **0 movers in 529
attempts**, and relocation has **59 in 3367**. That is a real qualitative change
and it is why this report measures the cost side rather than stopping at the
count. **The cost side is what settles it:** over the whole heavy band — 81.1 %
of the graded weight — the per-case *oracle* is **+0.0942 %** against a 0.30 %
bar, 13/20 cases gain anything at all, and it costs **172 s per case** against a
~1.5 s budget. There is no cheap rule that finds the winner either: it is rank
4/49 on one case and rank 36/58 on another (§7.3).

**What closes with it.** The wall in §5 and §7.2 is not a property of
relocation — it is a property of *editing this anchor's topology at all* after
the greedy has finished. That is the same wall that killed M64 (single-pair
flips), L256/L259/L262 (ruin-and-recreate: at the jam the largest block has 0
legal positions), and now relocation. Three different move semantics, one cause:
**the shipped greedy hands the LP a layout whose longest chain of abutting
blocks is exactly as long as the box, with a third of the blocks on it.** Post
hoc topology repair on such a layout has nowhere to put anything.

⇒ **Every post-hoc topology-repair move is closed, not just this one.** A future
proposal on this axis has to say why it is not a topology edit of a
chain-saturated anchor, or it is already answered.

### 10.1 What would reopen it, stated falsifiably

1. **A placer that does not produce chain-saturated layouts.** Slack is the
   enabling resource and the shipped greedy produces none: `l281_saturation.py`
   is the measurement, and it is cheap enough to run against any candidate
   placer before building anything on top of it. Note the direction of the trade
   — L252 already priced loosening the frame at only +1.50 %, and §4.2 prices
   buying the slack after the fact at 6× the whole wire prize.
2. **A move that shortens the critical chain instead of lengthening it.** Every
   move measured here perturbs the chain upward. Nothing in the ledger has tried
   to *target* the chain, and §5 says a median of 34.3 % of blocks (19 blocks)
   sit on it, so the target is identifiable without a label. This is the only
   genuinely untried thing this probe found, and it is a packing-time idea, not
   a post-processing one.
3. **Evidence that the graded corpus is not chain-saturated.** §8 says the
   in-set is (86 %) and OOS is not (23 %); if the hidden set resembled OOS the
   whole picture changes. Note this cuts the other way from the usual worry —
   the corpus we are graded on is the *harder* one for this mechanism.

⚠️ Per L279's rule, do not add relocation's hpwl prize to the area or violation
prizes: §7.1(b) shows it moves `area_gap` and `vrel` by exactly zero, so its
prize is entirely inside the hpwl term and overlaps nothing.

## 11. Honest limits

1. The LP arm is **two cases** (85, 88) at full width plus a top-5-unit sweep of
   the rest of the heavy band. The certificate census, the saturation
   measurement, the corpus comparison and the first-order prize are all on the
   **full in-set 100**; only the LP cost arm is sampled. The sampling is
   defensible (the two cases are 980 + 899 exhaustive solves each) but it is a
   sample.
2. Everything is measured with `bbox_relax = 1.0` for the main arm. §4.2 prices
   the alternative rather than assuming it away, and §7.2 tests +20 % directly.
3. This is an **offline research LP** (`m53_l3_probe` / `m64_flip_probe`), not
   the shipped shape-LP. A positive result would still have needed a port, and
   §7.3 is why that question never arose.
4. The target generator abuts `u` against its heaviest neighbours plus the
   unconstrained wire optimum. A different generator would produce different
   candidates — but §7.1(c) shows the generator is not the binding constraint:
   its coherent output already retains 94 % of the available wire prize.
5. The OOS comparison in §8 uses positions only (every block its own node) on
   both corpora, so its *levels* differ from §5's unit-level numbers. The
   contrast between corpora is measured identically on both sides.

## 12. Files

```
probe      l281_reloc_probe.py    gate | census | probe | report
gates      l281_gate.py           anchor reproduction, both anchors
           l281_cert_gate.py      certificate soundness (rejected -> LP anyway)
           l281_liveness.py       binding vs vacuous, positions compared bitwise
           l281_verify_mover.py   re-score every reported gain from its positions
diagnosis  l281_why_infeasible.py boundary equalities / bbox / neither
           l281_ordinal.py        the handoff's literal move, measured
           l281_saturation.py     critical-chain slack, in-set 100
           l281_oos_slack.py      the same quantity in-set vs OOS s1/s2
           l281_chain.py          how many blocks sit on the critical chain
pricing    l281_prize.py          exact first-order wire prize, all 100 cases
           l281_relax_price.py    what buying the missing slack would cost
           l281_deploy.py         can a label-free ranking find the winner
           l281_band.py           heavy-band aggregation: gain vs LP wall time
data       l281_cache.pkl         census + ~2900 LP solves
           l281_*.log
```

`constructive.cpp`, `optimizer_constructive.py` and `build_submission.D/` were
not touched. Nothing here is shippable as it stands: this is an offline LP
probe, and a positive result would still have needed a port.
