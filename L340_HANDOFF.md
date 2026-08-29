# L340 handoff — C++ B\*-tree SA, area + HPWL. HW sweep and seed noise both done.

Nothing is running, nothing is shipped, nothing in the shipping tree is modified.
Research tools only: `l340_btree.cpp`, `l340_btree.exe`, `l340_run.py`, `l340_seed.py`.

## Headline

At n = 80 the B\*-tree SA beats our shipped packer on **5 of 5 seeds**, by a median
of **−0.0678** quality (1.1500 vs 1.2178), worst seed still **−0.0508**. The same
margin appears at n = 40 and n = 120. **It is a real win and it is not a lucky draw.**

**And it is not deployable.** Matching our packer needs **2.8× / 27× / 56×** the compute
budget at n = 40 / 80 / 120, and the requirement grows fastest at the large n where
`exp(n/12)` puts the score weight. At the ~1.4 s per-case budget the SA *loses* at every
n — by +0.096, +0.225 and +0.438. The quality question is answered; the runtime question
is answered too, and the answer closes the line at this iteration count.

Two things from the previous version of this document are now **wrong**:

1. **`HW*` is not the optimum.** It is beaten at every n, and at n = 80 it *loses to
   our own packer* (1.2249 vs 1.2178). The gradient-matching derivation is valid only
   at an interior optimum; area is already against a wall (util 0.91–0.93 against the
   0.9455 the manifold reaches) while hpwl is not, so the linear weight misprices the
   reachable frontier.
2. **The per-cell numbers carry ~0.03–0.09 of noise**, which is the same size as the
   differences between HW cells. Every HW ordering claim below N = 5 seeds is unreadable.

## Usage

    cd ship_final                     # REQUIRED, see traps
    $PY l340_run.py  <n-list> <iters> <HW-multiples>
    $PY l340_seed.py <n> <HW-multiples> <iters> <n-seeds>

`$PY` = `C:\Users\.01\anaconda3\envs\floorset\python.exe`.

### Three traps that cost time this session

* **`cd ship_final` every single time.** `l340_run.py` resolves three paths relative
  to cwd — the `LiteTensorDataTest` symlink, `iccad2026contest`, and `EXE` — and all
  three only exist there. Bash cwd resets to `C:\ICCAD_ml` between calls.
* **`| tee` masks failure as exit 0.** A whole sweep died in 0 s with
  `can't open file` and reported success. Use `${PIPESTATUS[0]}`.
* **Build needs msys on PATH** (`windows-msys-path-silent-sa-fallback`); `ICCAD_CXX`
  does *not* work around it:

      $env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
      g++ -O3 -std=c++17 -o l340_btree.exe l340_btree.cpp

## The HW sweep — 2M iterations, seed 1, quality (lower is better)

    n      HW/HW*:   0.5      1.0      2.0      4.0      8.0   |   ours
    40                 -    1.1012   1.0848  *1.0410*  1.0534 |  1.1140
    80              1.2324  1.2249  *1.1404*  1.1845   1.1634 |  1.2178
    120             1.2649  1.1913   1.1566   1.1503  *1.1397*|  1.2136

Read this table for the *level*, not the *ordering*. The three n put their optimum at
HW = 4, 2 and ≥8 with no common pattern, n = 80 is non-monotone (4 worse than both
neighbours), and the 1→2 step improves **both** hpwl_gap and area_gap at n = 80 and
n = 120 — dominance, where a weight change should produce a trade. Those are noise
signatures, and the seed probe below confirms it.

`hpwl_gap` clamps to 0.0000 at n = 40 for HW ≥ 4: the SA matches or beats the label's
own HPWL there. Beating the label earns nothing (the clamp, per `l320-l326`), which is
why HW = 8 is worse than HW = 4 at n = 40 — the extra weight buys no score and only
costs area.

**Determinism: confirmed.** n=80/HW=2 and n=120/HW=2 were each run twice in separate
invocations and returned bit-identical util / hpwl_gap / area_gap. All spread below is
the seed, not the environment.

## The seed probe — `l340_seed.py 80 2,4 2000000 5`

    HW = 2*HW*   1.1404  1.1426  1.1500  1.1669  1.1670    median 1.1500  sd 0.0129
    HW = 4*HW*   1.1354  1.1845  1.1848  1.2112  1.2210    median 1.1848  sd 0.0332

    HW2 vs HW4:  mean diff −0.0340, SE 0.0159, t −2.13 (Welch df ≈ 5.2, p ≈ 0.08)
                 range-based test: NOT READABLE at N = 5.

**Seed 1 was the best of 5 at HW = 2 and the 2nd of 5 at HW = 4.** That single fact
generated the whole apparent HW = 2 ≫ HW = 4 ordering, the n = 80 jaggedness, and the
dominance anomaly. HW = 2 is *probably* better than HW = 4 (p ≈ 0.08) but it is not
established.

Higher HW is noisier: sd triples from 0.0129 to 0.0332 between the two cells.

### What this does to the headline number

    vs ours 1.2178 at n = 80
      sweep headline (seed 1, and min over the HW axis)   −0.0774   <- biased
      median of the HW = 2 cell                           −0.0678   <- the estimate
      WORST of 5 seeds                                    −0.0508   <- the floor
      5/5 seeds beat ours; the worst is 3.9 sd below ours

**`min-of-N` bias applies on the HW axis, not only on seeds** — this is the L296/L298
rule in a place it had not been applied. Taking the best of 5 HW cells is itself a
selection, so **every starred number in the sweep table is an upper bound on the win,
not an estimate.** The n = 40 (1.0410) and n = 120 (1.1397) bests have *not* been
de-biased; only n = 80 has. Expect them to give back a similar ~0.010 when measured.

## The iterations sweep — `l340_iters.py 40,80,120 2 10k..2M 3`, HW = 2*HW\*

Median of 3 seeds per point. This is the measurement that decides the line.

    n=40  (ours 1.1140)          n=80  (ours 1.2178)         n=120 (ours 1.2136)
    iters  median   time         iters  median   time        iters  median   time
     10k   1.3754    0.2s <=B     10k   1.5742    0.4s <=B    10k   1.6512    0.8s <=B
     30k   1.2505    0.4s <=B     30k   1.4425    1.2s <=B    30k   1.5209    2.4s
    100k   1.2104    1.3s <=B    100k   1.3318    3.8s       100k   1.4023    7.8s
    300k   1.1018    3.9s  WIN   300k   1.3239   11.5s       300k   1.3177   23.3s
      1M   1.0673   13.6s  WIN     1M   1.2187   38.5s ~tie    1M   1.2110   78.1s ~tie
      2M   1.0848   26.6s  WIN     2M   1.1500   76.6s  WIN     2M   1.1566  156.8s  WIN

`<=B` = fits the ~1.4 s per-case budget.

**At the budget, it loses at every n — by +0.096, +0.225 and +0.438.** The affordable
point is 100k at n = 40, 30k at n = 80 and 10k at n = 120.

### The number that closes this: required speedup to break even

    n = 40    100k affordable -> ~300k needed      1.4 s -> 3.9 s     2.8x
    n = 80     30k affordable ->   ~1M needed      1.4 s -> 38.5 s   27.5x
    n = 120    10k affordable ->   ~1M needed      1.4 s -> 78.1 s   55.8x

**The multiplier grows super-linearly in n (2.8 → 27 → 56 across a 3× change in n),
and `exp(n/12)` puts the score weight at exactly the n where it is worst.** Those two
compound in the wrong direction. C++ already bought ~100× over the Python prototype;
another 30–60× is not an implementation-tuning result.

**Parallel restarts do not rescue it either.** Running many seeds inside the same wall
budget and keeping the best is min-of-N, and the observed within-point range is far too
small: at n = 80 / 30k the three seeds span 1.4038–1.4617, so even a very lucky draw
lands ~0.19 above our 1.2178. Closing that needs ~4× the entire observed range. And our
own 1.4 s already uses the cores (41-profile portfolio), so the comparison is
wall-clock-fair as it stands.

### Two corrections this sweep forces

* **n = 40 is saturated by ~1M, not "still improving at 2M".** The previous claim came
  from one seed at `HW*`. At 3 seeds the n = 40 median goes 1.0673 (1M) → **1.0848 (2M)**
  — it got *worse*, well inside the 0.06–0.07 spread. It is n = 80 and n = 120 that are
  still genuinely improving at 2M.
* **Fewer iterations is noisier as well as worse** (n = 40 spread 0.0705 at 10k against
  0.0267 at 2M in the 5-seed cell), so the cheap end of this curve is exactly where
  single-seed points would have misled most.

⚠️ Confound this sweep cannot see: HW is fixed at 2×HW\*, and the best weight may itself
move with the iteration budget — a coarser search may want a different weight. Given the
size of the gap at the budget (+0.10 to +0.44), no plausible weight retune closes it.

## Known limits before this could ever be a candidate

1. **Runtime — measured, and it is the wall this line dies on.** Breaking even with our
   own packer needs **2.8× / 27× / 56×** more compute at n = 40 / 80 / 120 (above), and
   the requirement grows fastest where the score weight is. This is no longer an open
   question with an unknown answer; it is a closed one with a bad answer.
2. **Preplaced blocks are not honoured.** A B\*-tree cannot express a fixed coordinate,
   so these layouts are **not submittable**. Real engineering gap, not a detail.
3. **`HW*` is an oracle** (uses `hpwl_L`), and the sweep shows the optimum is not at
   `HW*` anyway, so a label-free proxy now has to hit a target we cannot yet locate.
   `area_L` is label-free (`ΣA/0.971`, L320); `hpwl_L` is not.
4. **Three cases, one instance each.** No OOS, no Linux lane. Determinism ✓, seed
   spread ✓ (n = 80 only).

## Where this leaves the line

**Quality: established.** The B\*-tree manifold genuinely reaches better layouts than our
packer on our own objective — 5/5 seeds at n = 80, median −0.0678, replicated in the same
direction at n = 40 and n = 120.

**Deployment: closed at 2M iterations, on runtime.** Not by a factor that engineering
closes.

What is left, in the order it is worth anything:

1. **Nothing on the weight axis.** It is inside the seed noise and it cannot move a gap
   of +0.10 to +0.44.
2. **A fundamentally cheaper search, or nothing.** Not a faster implementation — a
   different algorithm that reaches the same manifold in ~1/30th the moves. Incremental
   cost evaluation is the obvious candidate and was explicitly rejected in the source
   comments ("a B\*-tree move relocates many blocks at once"), so this is a rewrite, not
   a tuning pass.
3. **Preplaced is still unsolved and still blocks submission regardless of runtime.**
4. If the line is revived, **de-bias n = 40 and n = 120** (their sweep bests are still
   min-over-HW) before quoting any number.

## Why this line exists at all

L333–L336b: the B\*-tree manifold reaches 0.9455 utilisation at n = 40 in pure Python
and was still climbing — above our shipped packer's 0.877 and above the 85.4 % ceiling
L284 measured, which is a property of our pool's reachable set and not of the instance.
The generator's own objective is area-only, and replaying it gives `hpwl_gap` 1.13–1.60
against our 0.240, because the netlist was sampled *from* the generator's finished
layout (paper Alg. 4). So: keep the representation, use our objective. That is
`l340_btree.cpp`.

Full context: `L320_L326_NEW_PATHS.md`.
