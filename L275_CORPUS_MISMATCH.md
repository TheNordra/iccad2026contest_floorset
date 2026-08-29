# L275 — the L250–L274 arc was priced on a corpus 22–24 % harder than the graded one

**Every arm in the arc that reads better on the OOS heavy band reads worse on the
corpus the score is actually computed on. Four out of four, no exceptions.**

    100 in-set cases, official strict scorer, 48 cores, weighted exp(n/12)
    OOS columns are the arc's own published numbers.   negative = better.

    arm         OOS s1 hvy    OOS s2 hvy |  in-set 100   in-set hvy20   movers
    ---------------------------------------------------------------------------
    adapt        -1.3785 %          -    |   +1.5578 %     +1.9695 %       84
    nosize       +0.1626 %    -0.0286 %  |   +1.8616 %     +1.9289 %       92
    l271sng      -0.4587 %    -0.1836 %  |   +0.1861 %     +0.2372 %       18
    l269p1       -0.5706 %    -0.5507 %  |   +0.0245 %     +0.0009 %       31
    l269p2       -0.8143 %    -0.7285 %  |   +0.4244 %     +0.4176 %       42

`adapt` — the largest quality gain of the whole arc at −1.3785 % — is **+1.5578 %**
here. `l269p1`, the ship candidate that cleared s2 on two disjoint corpora at 8/8
split-halves, is **+0.0245 %**.

---

## 1. It is not the band. It is the corpus.

The obvious objection is that the arc measured only `n ≥ 101` while the graded
score covers three bands. That objection is **wrong**, and the data closes it two
ways.

**(a) The heavy band is 81.1 % of the graded weight, in both corpora.**

| | heavy cases | heavy share of weighted total |
|---|---|---|
| in-set 100 | 20 of 100 | **81.1 %** |
| beta hidden (the real grader) | 20 of 100 | **81.1 %** |

The beta hidden set has the *same* n-range (21…120) and the *same* count at
n ≥ 101. Measuring the heavy band was a defensible choice.

**(b) Restricting the in-set to the same band does not rescue any arm.** The
`in-set hvy20` column above is the identical band, and every sign is unchanged —
in fact slightly *more* negative. So the flip is not band composition.

**What differs is difficulty.** Same band, same shipped placer:

| corpus | weighted | vs in-set, same band |
|---|---|---|
| in-set 100, heavy 20 | **1.2174** | — |
| **OOS s1 heavy 40** ← the arc's corpus | **1.5116** | **+24.2 % harder** |
| OOS s2 heavy 40 | 1.4868 | **+22.1 % harder** |
| beta hidden heavy 20 (M73 code, version-confounded) | 1.3170 | ≈ +2.4 % after de-versioning |

🔑 **The corpus that gets graded is roughly ten times closer to the in-set 100 than
to the OOS sample the whole arc was measured on.**

## 2. Why difficulty inverts these mechanisms specifically

Every arm in the arc buys **area**. `Cost = (1 + 0.5·(hpwl_gap + area_gap))·exp(2·vrel)`,
so what a density mechanism can harvest is bounded by the area deficit that exists
to be harvested. A corpus 24 % harder has a correspondingly larger deficit, so the
same mechanism finds more to take — while its costs (wire, violations) are much
less corpus-dependent.

That is exactly the shape L274 measured directly on `l271sng`'s s1 → s2 move: the
**area delta transferred almost perfectly** (−0.0109 → −0.0100) while the wire cost
and the violation sign did not. Same phenomenon, one step further along the
difficulty axis.

⇒ **For any mechanism whose value scales with a gap, the measurement corpus's
difficulty is a first-order term, not a detail.**

## 3. What this does and does not overturn

**Overturns:** the deployability of every frame/density candidate in L250–L274.
`adapt`, `nosize`, `l269p1`, `l269p2`, `l271sng` are all negative on the graded
shape. Nothing in the arc is shippable. This is consistent with — and much
stronger than — L274's narrower finding.

**Does not overturn:** the *mechanistic* results. The exchange rate, the
`s_min` ceiling (81.34 → 84.83 % utilisation), the wire-blindness diagnosis, the
isolated per-profile deltas, the identity gates — all of those were correctly
measured on the corpus they were measured on. What was wrong was the inference
from that corpus to the score.

**Does not license optimising on the in-set instead.** The in-set 100 *is* the
local validation set, so tuning on it is textbook in-sample fitting, which is why
OOS was introduced in the first place (M67-D, M74). The correct rule is stronger
than either:

> **OOS guards against over-fitting. The in-set guards against difficulty
> mismatch. A candidate must be positive on BOTH. Neither alone is sufficient,
> and this session now has a counterexample in each direction:**
> * `l271sng` — positive on OOS s1, 4/4 split-halves, **negative on OOS s2**
>   ([[l271-no-constant-still-needs-s2]])
> * `l269p1` — positive on OOS s1 *and* s2, 8/8 halves, **negative in-set**


## 3.5 The corrected target: it is hpwl, and the arc was optimising area

With the corpus fixed, the deficit decomposition changes shape. Shipped placer,
48 cores, weighted `exp(n/12)`:

| corpus | hpwl_gap | area_gap | vrel | **hpwl : area** |
|---|---|---|---|---|
| **graded shape** (in-set 100) | **0.2484** | 0.1355 | 0.0141 | **1.83×** |
| graded shape (heavy 20 only) | 0.2462 | 0.1316 | 0.0118 | 1.87× |
| the arc's corpus (OOS s1 heavy 40) | 0.2924 | **0.2300** | **0.0893** | 1.27× |

The OOS corpus carries **+70 % more area deficit** and **6.3× the violations**.

Prizes on the graded shape (drive one term to zero, hold the others):

    hpwl -> 0    -10.41 %      <- 56 % of the total headroom
    area -> 0     -5.67 %      <- what the entire arc attacked
    vrel -> 0     -2.81 %
    all  -> 0    -18.46 %

(These reproduce CLAUDE.md's own in-set decomposition — 10.15 / 6.00 / 3.57 on
older code — which is a useful consistency check: the *in-set* numbers in the
project's own summary were right all along; it was the arc that moved corpus.)

🔑 **So the arc chose the smaller term and then measured it where it was inflated.**
Area is 31 % of the graded headroom, and the corpus it was measured on had 70 %
more of it than the graded one. Both errors point the same way, which is why five
independent mechanisms all came out positive on OOS and negative here.

⇒ **The corrected target is hpwl.** It is 56 % of the graded headroom and it is the
term the LP cannot repair (7.5 % of hpwl_gap at depth 12, against 49 % of
area_gap — L267_L269 §2.4).

### 3.6 One hpwl arm measured on the corrected corpus, and it is RED

`ICCAD_GUIDE_MED=1` adds a connectivity-weighted L1-median of an item's neighbours
as an extra candidate origin — a mechanism aimed squarely at wire, and one the
ledger never priced on its own.

| | |
|---|---|
| graded shape | **+0.4982 %** (worse); heavy 20 **+0.5501 %** |
| gaps | hpwl 0.2484 → **0.2538** (+0.0054), area 0.1355 → 0.1430 (+0.0075) |
| movers | 78 (31 better / **47 worse**) |

**It makes hpwl worse — the term it exists to improve.** That is M78's "adding
candidates is harmful by default" again: the greedy's `bbox_area_with` is
short-sighted, so handing it a wire-optimal origin buys local wire and loses more
globally. Consistent with L272's failure (the hint fed into the wire term also
made hpwl worse).

⇒ Targeting hpwl is the right *direction*; feeding the existing greedy better wire
information is not the right *mechanism*. Two independent attempts now
(L272 hint-into-wire, GUIDE_MED candidate-seed) both degraded hpwl.

## 4. Why the in-set deserves the weight

It is not merely "another corpus":

* the alpha test set was **bit-identical** to the local validation set, so the
  in-set 100 has actually been a graded set once;
* the beta hidden set has the **same n-range and the same 20/100 heavy split**, and
  the same 81.1 % heavy weight share;
* de-versioned, beta hidden sits ≈ 2.4 % from the in-set and ≈ 22 % from OOS.

## 5. The cheap fix, for whoever picks this up

One 100-case official eval per arm, ~10 minutes, no new tooling:

```
cd iccad2026contest
ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN=<probe> <flags> \
  python iccad2026_evaluate.py --evaluate ../optimizer_constructive.py -o out.json
```

⚠️ **Ambient `ICCAD_FRAME_SCALES` will not work this way** — 44 of 55 profile dicts
set it themselves and `env.update(env_over)` makes the profile win (handoff trap
#2). Ladder arms have to be injected into the profile dicts. The flags used here
(`ICCAD_L267/L268/L269/L271`) are set by no profile, so ambient really reaches the
binary; `nosize`'s 92 movers and `adapt`'s 84 are the liveness evidence.

⚠️ **Local eval forces RF = 1.0**, so these columns are quality only. A mechanism
that trades runtime for quality cannot be judged from them.

## 6. Files

```
l275_inset.sh                    the three-arm in-set driver
results_L275_{adapt,nosize,l271sng,gmed}.json
results_L274_{base_48c,base_48c_rep2,ship_48c,p2_48c}.json
l275_inset.log
```

Baseline determinism: two independent baseline runs differ on **0/100** cases, so
every in-set number above is exact rather than sampled.
