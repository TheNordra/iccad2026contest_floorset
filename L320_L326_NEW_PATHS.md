# L320–L326 — the label generator is identified, and it changes the problem

A research session with one brief: find paths this project has **not** walked. The
answer is not a list of tweaks. The dataset's own paper plus six local probes settle
what the target actually is, and most of what we have been doing optimises a
different functional than the one that produced it.

---

## 0. The finding

**FloorSet-Lite's labels are produced by Parquet — Adya & Markov's fixed-outline
B\*-tree simulated-annealing floorplanner — run with an AREA-ONLY objective.
Wirelength is not in it.** The netlist is then *reverse-engineered from the finished
layout*.

FloorSet paper (arXiv:2405.05480), Algorithm 6 line 6, verbatim:

    F <- runSA(W, H, n_parts, A_parts)     # "Run Parquet (Simulated Annealing) for area optimization"

Note the argument list: outline, partition count, aspect-ratio distribution. **No
connectivity.** Then Algorithm 4 (`annotateNETS`) samples the netlist *from* `F`:

    PDist <- pairwiseB2BDistance(F)
    PSim  <- 1 - Normalize(PDist)
    b2bConnectivity <- Sample(F, size=n_b2b_nets, p=PSim)

and Algorithm 5 reads the boundary / preplaced / cluster / MIB constraints off the
same finished layout.

⇒ **The netlist is not a specification to satisfy. It is a noisy readout of the answer.**

---

## 1. Six invariants, measured here, not taken on trust

All on the 100 validation labels (7050 blocks) and one 112-layout training shard.

| # | invariant | measured | probe |
|---|---|---|---|
| 1 | every label coordinate is an integer | **100/100 cases** | L320 |
| 2 | block area equals target **exactly** — zero use of the 1 % slack | **7050/7050** | L321 |
| 3 | (w,h) is an integer divisor pair of A with aspect ≤ 3 | **7050/7050**, median **1** unordered option, 354 blocks have exactly one | L321 |
| 4 | every block is bottom-supported (y=0 or abutting below) | **7050/7050 = 100.0 %** | L322 |
| 5 | `tree_sol` in the training shards obeys the B\*-tree rules | left child `x=x_p+w_p` **4419/4419**; right child `x=x_p` **3869/3869**; y = contour **8400/8400** | L325 |
| 6 | the netlist encodes the label's geometry | weight vs centre distance **r = −0.558**, negative in **100/100**; connected pairs **1.51×** closer, in **100/100**; pin centroid predicts block centre at **0.322×** naive error over 4465 blocks | L324 |

Invariant 4 is the B\*-tree contour rule showing through: y is derived, x is chosen.
Only 92.9 % are left-supported — so **do not left-compact aggressively; that leaves
the manifold the label lives on.**

Two more, from the same probes:
* label utilisation is **p50 0.9693** (min 0.954, max 0.987) — and since w·h = A
  exactly, `area_baseline ≈ Σ area_target / 0.971` is computable **from the input
  alone, to about ±1 %**, with no label.
* **0/19** sampled labels are guillotine/slicing ⇒ recursive-bisection and
  shape-curve methods are the wrong bias and can be dropped.

## 1.1 `tree_sol` — the unused asset

The 1M training shards are a **7-tuple**, and element [4] is `tree_sol`, shape
`[B, n−1, 3]` = (parent, child, left/right flag): **the generator's own B\*-tree for
every sample.** The contest's validation loader (`lite_dataset_test.py`) returns only
`fp_sol` and `metrics` — it **throws the tree away**.

That converts "learn a floorplan" into "**predict a B\*-tree**": a discrete structure
with n−1 labelled edges, 1M supervised examples, and a *deterministic* decode that
always yields a legal, compact, bottom-supported packing. Crucially the error mode is
different from M52's: a wrong edge **perturbs** the layout, it does not destroy it.
M52 died because coordinate imitation had zero error tolerance. This does not.

## 1.2 "HPWL" is not HPWL

`calculate_hpwl_b2b` is **weighted centre-to-centre Manhattan distance over 2-pin
edges** — not half-perimeter over multi-pin nets:

    total_wl += weight * (abs(x2-x1) + abs(y2-y1))     # x1 = pos[i].x + w/2

Consequences: **x and y separate completely**, the objective is convex and piecewise
linear in the centres, and it is *exactly* the quantity Algorithm 4 inverted to draw
the netlist.

---

## 2. 🚨 A framing correction I owe this ledger

I have been saying "beating the label earns nothing, so this is pure reconstruction."
That is **only true at the clamp**. Both gaps are currently strictly positive
(`hpwl_gap` 0.2316, `area_gap` 0.0915 on the `mix` arm), so in the region we actually
occupy

    cost = 1 + 0.5*(our_HPWL/HPWL_label + our_area/area_label)

is a plain weighted sum of **our own** metrics, and the label only fixes the exchange
rate between them. Ordinary floorplan improvement pays, linearly, all the way down to
the label's numbers. The reconstruction framing becomes binding only at the boundary.

## 2.1 And the prize is wirelength, not area

Weighted, on the `mix` arm:

    hpwl_gap 0.2316  ->  contributes 0.1158     (2.5x the area term)
    area_gap 0.0915  ->  contributes 0.0458
    vrel     0.0139  ->  multiplies by 1.0283

    zero the area gap alone -> 1.1474      (not enough)
    zero the HPWL gap alone -> 1.0753      <- would BEAT rank 1's 1.0845

---

## 3. L326 — the decisive precondition experiment, already run

A **label-free** estimator: anchor every block with p2b nets at its weight-weighted
pin centroid, then relax the rest to the b2b-weight-weighted mean of their neighbours
(harmonic extension, anchors pinned). ~40 lines. **No overlap handling, no shapes, no
constraints** — it cannot produce a layout, only a position estimate.

    netlist inversion   mean error 0.1267 of the bbox half-perimeter
    our shipped placer  mean error 0.1161   (given its best translation)
    the crude estimator is closer on 49/100 cases

**A forty-line inversion locates the label's blocks as well as the entire 50-profile
portfolio.** Our placer is not using this signal at all; a trivial one matches it.
That is the strongest single argument that the untried direction is real.

---

## 4. The untried paths, ranked

### Tier 1 — follow from the generator's identity, preconditions verified above

1. **Replay the generator.** Parquet-style fixed-outline B\*-tree SA with an **area**
   objective, on the integer lattice, shapes drawn from the divisor pairs with
   AR ≤ 3, outline bracketed by `Σ A ≤ W·H ≤ Σ A / 0.954` and by the pin ring
   (the pin bbox is ≥ the label bbox in 100/100 cases and is almost certainly the
   sampled outline). This is not "a packer rewrite" — M27 compared representations as
   *search heuristics against our own objective*. This is running **the target
   distribution's own program**.
2. **Netlist inversion as the placement target.** Fit the generative law
   (p and weight vs distance) on the 1M training set, invert to a posterior over the
   label's pairwise distances, MDS to a configuration, and steer the placer to *that*
   rather than to minimum HPWL. Strictly more information than HPWL, which sees only
   the weighted sum. L326 says the crude version already ties our placer.
3. **Supervised B\*-tree prediction from `tree_sol`.** 1M (instance → tree) pairs,
   deterministic decode, repairable errors. The single largest unused asset in the
   dataset.

### Tier 2 — structural: remove a cost term instead of optimising it

4. **Zero-dead-space soft-block partitioning (ZDS / IMP).** Drives `area_gap` to ≈0
   by construction. Precondition theory for when zero deadspace is achievable exists.
5. **Mosaic / rectangular-dual / area-universal layouts.** Eliminates the area term
   structurally rather than minimising it.
6. **CP-SAT / `geost` with boundary + grouping + MIB as HARD constraints.** `V_rel`
   is 8.29 % on the graded corpus = a **1.18×** multiplier; making the three soft
   families invariants removes it. Needs `ortools` (not installed).

### Tier 3 — cheaper, worth a probe

7. Boundary-constrained B\*-tree (exact four-sided feasibility conditions).
8. Hierarchical B\*-tree / symmetry-island machinery — turns grouping and MIB into
   representation invariants (analog-placement literature; same constraint shapes).
9. Window matheuristic: exact MILP re-optimisation of 10–16-block windows — the only
   way to get exact topology optimisation at n=120.
10. Anchored stress majorisation (SMACOF) on the netlist-as-distance-matrix.
11. FunSearch-style LLM evolution of the packer's priority/tie-break function —
    structurally perfect error tolerance, and not bounded by any of our three oracles.
12. Automatic portfolio **construction** (Hydra / Cedalion, MAP-Elites) instead of
    hand-built profiles; and per-case Bayesian optimisation over a *continuous*
    configuration space, whose oracle is unbounded where the 50-profile one caps at
    +2.03 %.

### Explicit negatives from the sweep — do not spend time here

AlphaChip / Circuit Training (three structural mismatches); GNN coordinate regressors
(wrong scale, and it is the closed advisor layer); DREAMPlace / GPU analytical placers
(fixed-die, standard-cell scale, and we do not need speed); min-cost-flow and
displacement-minimising legalisation; Lagrangian soft-module sizing and shape
curves / geometric programming (the perfect-shape oracle is +0.099 %); monolithic
exact MILP/MISOCP/SMT over the whole instance (sizes are nowhere near); multilevel
coarsen–partition–uncoarsen (exists to buy speed); sequence-pair symmetry /
common-centroid conditions (no such constraints here).

---

## 5. What this costs to start

Tier-1 items 2 and 3 need **no new dependency** — they are offline computation over
data already on disk. Tier-2 item 6 needs `pip install ortools`, which would change
the environment and has not been done.

## 6. Files

```
l320_labelprint.py       integer / utilisation / aspect / slicing fingerprint
l321_shapespace.py       the discrete shape space (7050/7050)
l322_blstable.py         bottom-left stability (7050/7050 bottom-supported)
l323_ours_vs_label.py    the same statistics on our own output
l324_netlist_inversion.py   the netlist is a readout of the geometry
l325_btree_verify.py     tree_sol IS the generator's B*-tree
l326_inversion_vs_ours.py   inversion vs our placer as position predictors
```

Nothing shipped or modified.


---

# L327–L329b — Tier 1 ② built and measured: the signal is real, the pipeline is not yet

## A. The generative law, learned from the training set (L327)

336 training layouts, no assumptions about the form of `Normalize()`:

      b2b multiplicity  m=0 -> E[u]=0.3909   m=1 -> 0.2306   m=2 -> 0.1911
      b2b weight (as an integer multiple k of the per-layout base unit)
          k=1 -> 0.3793   k=2 -> 0.2436   k=3 -> 0.2291   k=5 -> 0.2343
          k=8 -> 0.1534   k=9 -> 0.1412   k=12 -> 0.0922
      p2b edge present  -> u_pin = 0.0514  against a 0.4341 baseline   (8.4x)

`u` is the per-layout normalised centre distance. The p2b channel is by far the
sharpest: a pin-connected block sits at ~5 % of the maximum pin distance.

## B. The inversion as a position target (L328)

Scales come from the input only — bbox area = ΣA/0.971 (L320), aspect and absolute
anchoring from the pin ring. Then L1 stress minimisation, with the pin terms
supplying absolute position, so it is trilateration rather than a free embedding.

      inverted generative model   mean error 0.0867 of the bbox half-perimeter
      our shipped placer          mean error 0.1161   (given its best translation)
      L326 crude harmonic         mean error 0.1267   (no distance information)
      inversion closer on 78-82 / 100 cases

🔑 **The inversion locates the label's blocks 25 % more accurately than the entire
50-profile portfolio**, using no label information, no shapes, no constraints and no
placement reasoning.

## C. Realising it — and the null control that reads the result (L329, L329b)

Sequence pair extracted from the target (Γ+ by x+y, Γ− by x−y), decoded by longest
path — overlap-free by construction. Then the same realiser fed the **label's own**
centres and shapes, which is the control that separates a bad target from a bad
realiser:

      what goes in                          hpwl_gap   area_gap
      the label itself (by definition)        0.0000     0.0000
      label centres + label shapes            0.0468     0.1359
      label centres + SQUAREST shapes         0.2160     0.4442
      INVERTED target + squarest shapes       0.6289     0.9915
      our shipped placer (mix)                0.2402     0.1176

Three losses, now separately measured:

1. **The realiser costs 0.0468 / 0.1359 even on perfect input.** Its wirelength is
   excellent — **5× better than our placer's 0.2402** — and its area is slightly
   worse than ours. So for HPWL, which §2.1 showed is the prize, this realiser is
   not the bottleneck.
2. **Shape choice costs more than anything else**: label shapes → squarest shapes
   moves hpwl 0.047 → 0.216 and area 0.136 → 0.444. The paper's "squarest reproduces
   the label 82 % of the time" is true per block and still not good enough — the 18 %
   dominate. ⚠️ This does **not** contradict M79's "perfect shapes = +0.099 %": M79
   fed perfect shapes into **our packer**, which cannot exploit them. In this
   pipeline they are worth an enormous amount.
3. **Target accuracy costs the remainder**: 0.216/0.444 → 0.629/0.992.

## D. Verdict on Tier 1 ②

**The signal is verified; the pipeline is not yet competitive.** The inversion beats
our placer as an *estimator* and loses badly as a *placer*. But the decomposition
says exactly where the loss is, and it is not where I would have guessed:

* the realiser is fine for wirelength and mediocre for area;
* **shape selection is the largest single lever in this pipeline** and is a
  well-posed supervised problem — a ~2-way choice per block with 1M labelled
  examples (`A_parts` in the paper's Algorithm 6);
* target accuracy is second and improvable (joint multiplicity×weight law, better
  scale estimates, iterated refinement).

Next experiment, in order: (i) learn the shape choice from the 1M set instead of
taking the squarest, (ii) re-measure C with learned shapes, (iii) only then invest in
a better realiser — and the natural one is Tier 1 ①, a B\*-tree SA whose cost is
distance-to-target rather than HPWL, which keeps packing quality by construction.

```
l327_fit_law.py / l327_law.json   the learned generative law
l328_invert.py                    inversion -> position target, scored
l329_realise.py                   seq-pair realisation of the target
l329b (inline)                    the null control on the label's own input
```


---

# L330–L332 — Jimmy is right about the distribution shift, and it lands on ONE channel

Reported by a teammate: the public 1M and the set that generated the test data have
very different "T2B relatedness" (97 % vs 30 %), so ML trained on the 1M scores well
in set and badly on beta.

**What is and is not settleable here.** The 1M and the validation 100 both carry a
netlist AND a label, so every statistic of the generative law is directly comparable.
The hidden beta set gives us only per-case OUTCOMES (cost, gaps, runtime) -- no
netlists, no layouts -- so its law cannot be measured from here at all.

## The shift is real, and it is entirely in the PIN channel

      statistic                    validation    1M train    ratio
      p2b edges per block               8.5005      1.3805    6.16x
      p2b blocks covered                0.6221      0.2622    2.37x
      p2b u | connected                 0.0875      0.0368    2.38x
      p2b concentration                 6.2071     13.5987    0.46x
      b2b edges per block              11.4025      6.2218    1.83x
      ---- the LAW, by contrast, is stable ----
      b2b weight-vs-distance r         -0.5564     -0.5629    0.988
      b2b concentration                 1.5209      1.6404    0.927
      utilisation                       0.9708      0.9737    0.997

**The mechanism is the same; the sampling density is not.** In the 1M a pin edge
pins a block to `u = 0.0368` of the maximum pin distance; in the validation set the
same edge is worth only `u = 0.0875`. A model fitted on the 1M will over-trust the
pin channel by a factor of ~2.4, on 6x more edges.

### 🚨 This hits my own L328 estimator

I fitted the p2b law on the 1M (`u_pin = 0.0514`) and applied it to the validation
set, and I weighted the pin terms **3x** because the 1M said they were 8.4x sharper
than b2b. On the validation set they are not that sharp. **The 0.0867 in L328 was
obtained with a miscalibrated prior**, and the direction of the error is the same one
Jimmy describes.

## But the SHAPE channel does not shift

      what the label picks, as a rank among divisor pairs sorted by squareness
                          rank0    rank1   rank2+   landscape   has a real choice
      1M train            0.368    0.469    0.163      0.619          0.952
      validation          0.372    0.458    0.170      0.606          0.950

      a rule learned on the 1M:   0.5240 in-sample  ->  0.5082 HELD OUT   (-3 % rel.)

So **shape learning from the 1M is legitimate and transfers essentially losslessly.**
Nothing about the shape channel is contaminated by the pin-density difference,
because shape selection never reads the netlist.

## ⚠️ Correction: the "squarest reproduces the label 82 %" figure is the UNORDERED one

      squarest, UNORDERED shape {w,h}      1M 0.8286    validation 0.8169
      squarest, WITH orientation           1M 0.3680    validation 0.3716

A layout needs the orientation. Including it, "squarest" is right **37 %** of the
time, not 82 %. The 1M-learned rule gets **50.8 %** held out.

## And 51 % is not enough

      what goes in                          hpwl_gap   area_gap
      label centres + label shapes            0.0468     0.1359
      label centres + LEARNED (50.8 %)        0.1786     0.4384
      label centres + squarest (37.2 %)       0.2160     0.4442
      inverted centres + LEARNED              0.6210     1.0239
      inverted centres + squarest             0.6510     0.9955
      our shipped placer                      0.2402     0.1176

Learning the shapes recovers **22 %** of the hpwl gap between squarest and the label
(0.216 -> 0.179) and almost none of the area gap (0.444 -> 0.438). With inverted
centres it stays at 0.62 -- still worse than the placer we ship.

## What this does to the Tier-1 ranking

The three Tier-1 paths have **very different exposure** to this shift:

| path | reads the netlist? | exposure |
|---|---|---|
| ① replay the generator (area-only B\*-tree SA) | **no** — Algorithm 6 takes no connectivity | **immune** |
| ③ supervised `tree_sol` prediction | layout structure, not netlist density | low, but must be checked |
| ② netlist inversion | **yes, and leans hardest on the pin channel** | **highest** — the p2b term is 8.4x sharper than b2b, and p2b is exactly what shifts |

🔑 **The path I just built is the one most exposed to the problem Jimmy found, and
the path that is immune is the one I ranked first for a different reason.** The
generator never looked at connectivity, so a faithful replay of it cannot be hurt by
the netlist distribution moving.

```
l330_dist_shift.py       1M vs validation, every candidate reading of "relatedness"
l331_shape_transfer.py   shape-choice transfer, and the ordered/unordered split
l332_learned_shapes.py   learned shapes inside the L329 pipeline
```


---

# L333–L336b — Tier 1 ①: the manifold is real, the generator's objective is not ours

Rebuilt the generator's program: B\*-tree SA (the representation L325 verified at
100 % on `tree_sol`), integer lattice, shapes = integer divisor pairs of the exact
area with aspect <= 3, area objective, no connectivity. Self-contained, so immune to
the L330 distribution shift.

## ✅ The density ceiling was a property of OUR packer, not of the problem

      n     iters    label util   B*-tree SA    vs our 0.877
      40     40000     0.9715       0.9145         +0.0375
      40    160000     0.9715     **0.9455**       +0.0685
      80     40000     0.9706       0.8682         -0.0088
      80    160000     0.9706     **0.8867**       +0.0097
     120     40000     0.9663       0.8324         -0.0446
     120    160000     0.9663     **0.8806**       +0.0036

Still climbing at 160 k on all three, in **pure Python**. At n=40 it reaches 84 % of
the way from our utilisation to the label's.

🔑 **This refutes L284's "density ceiling 85.4 %" as a statement about the problem.**
L284 measured the reachable set of *our* 42-profile pool and was right about that;
it is not a bound on the instance. A naive SA in an interpreted language already
beats it at every n given enough iterations.

⚠️ L334 also tested the **fixed-outline** objective (Parquet's actual contribution;
the generator samples W,H first). It came out *worse* — but that comparison was not
iteration-fair (its budget was split across four aspect ratios). Unresolved.

## ❌ But replaying the generator's OBJECTIVE is the wrong thing for our score

      n     util     hpwl_gap    area_gap
      40   0.9145      1.1257      0.0624     <- beats us on area at n=40
      80   0.8682      1.5950      0.1179
     120   0.8324      1.3276      0.1608
     ours  0.877       0.2402      0.1176

**hpwl_gap 1.13–1.60, five to seven times worse than ours.** This is exactly what
§0 predicts and it is not a defect of the implementation: the generator optimised
area alone, and the netlist was drawn *afterwards from its layout*. A different
area-optimal packing is a perfectly good floorplan that the given netlist simply
does not match. Area and HPWL are not aligned here because the netlist was fitted to
one particular area optimum.

## The synthesis this leaves

Keep the **representation** (immune to the distribution shift, and demonstrably
denser than our packer) and drop the **objective** (the generator's, not ours):

> B\*-tree SA on the label's own space — integer lattice, exact-area divisor-pair
> shapes, aspect <= 3 — minimising **area + HPWL**, i.e. what we are actually scored
> on.

That could not be measured here: HPWL inside the SA loop is ~600 edge evaluations
per iteration and pure Python cannot afford it (the run was abandoned). It is a C++
job, and the C++ is the natural home anyway — 160 k iterations took 279 s at n=120
in Python, which at a typical 100x would be ~3 s/case, comparable to the 1.4 s/case
we spend now and affordable against the ~19 s of free RF budget.

```
l333_btree_sa.py      B*-tree + contour decode + SA, area objective
l334_fixed_outline.py the fixed-outline objective (not iteration-fair; unresolved)
l335_scaling.py       the utilisation-vs-compute curve
l336b_cost.py         both gaps for the area-only replay
```
