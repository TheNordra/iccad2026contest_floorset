# L119 — six routes measured to the end, one free speedup shipped

Session of 2026-08-11. Shipped: `6201d12` (L116) and `d1fd804` (L117), both
pushed on `l113-route-a`. Everything else in this report is a measurement, and
most of it is a RED that should stop someone re-trying it.

## Where the score actually is

```
score = quality × RF          RF = max(0.70, (s·t/M)^0.30)
```

| | total | avg cost | avg runtime |
|---|---|---|---|
| 32-core default | `1.293461035226291` | 1.3301 | 0.93s |
| M80 @48c (route A + LP off) | `1.2666234250706565` | 1.3083 | — |
| **shipped @48c (Windows)** | **`1.2367916697725434`** | **1.2695** | 3.22s¹ |
| shipped @48c (Linux, measured) | `1.2362075522257698` | 1.2695 | — |

¹ 32 physical cores with the pool forced to 48 — not a grader wall, not an RF input.

**The time side is spent.** Weighted RF is **0.7042** against a floor of 0.7000,
with 49/100 cases already on it, and the 51 cases above the floor carry only
**8.2% of the weight**. Driving every case to the floor is worth 0.6%.

**The score is 12 cases.** Weight is `exp(n/12)`, so the top-10 hold **54.6%** of
the weighted excess over 1.0 and the top-20 hold **78.6%** — all of them n ≥ 109.
On those cases HPWL is consistently the largest term (gap 0.15–0.36) ahead of
area (0.10–0.23). Any lever that does not move n ≥ 109 is worth approximately
nothing.

## What shipped

**L116 — the separation transitive reduction was never switched on with the
HPWL pruning.** L112 sized the pruning work off "80.2% of rows are HPWL", which
is the PRE-pruning figure. With B=8 live the mix inverts: separation becomes
55.8–72.7% on the heavy cases. `_sep_reduction_mask` (documented EXACT) had been
in the builder all along, but `lp_pass` read `if PRUNE_B is not None and not
sep_trim` — mutually exclusive purely because `solve_pruned` did not forward the
flag. Forwarding it: rows 12497 → 6274 on case 99, objective identical to 4e-15,
end-to-end **0.2670s → 0.1947s (1.37×)** with quality identical to every digit.
Priced with route A that is **grid worst +1.066% → +2.355%**, for nothing.

**L117 — the final tar now runs under the official command on Linux.** Three
lanes pass. The default-core lane is bit-exact against `results_M74_default`
(|d| = 0, zero ULP warns — even case 84, which M67-C expected to warn), which is
the first proof that the bundled ELF actually *executes* rather than merely
being an ELF. The corrupt-binary lane falls through to the on-site g++ chain and
lands on the same cost. **The 48-core lane does not reproduce**: scipy 1.15.3 /
py3.10 on Windows versus 1.18.0 / py3.14 on Linux land on different optima of the
same degenerate LP — 92/100 cases agree to <1e-9, 8 move, positions apart by up
to 11.5. Bit-equality is the wrong gate there, so `judge48()` checks the
invariant the LP was shipped on instead (every case feasible, none worse than the
pre-LP anchor, total still ahead), which holds on both stacks. **The grader's
score will not be a fixed digit string — report the gain as +2.35% to +2.40%.**

## The six routes, and why each stops

**1. LP depth — RED, and the old pricing was wrong.**
`rows_for_k(l97, k)` slices `passes[:k]` and takes the last kept cost. Fed a file
with ONE pass per case — which is what `results_L100_lp_speed.json` is — it
returns k=1 quality for *every* k, so depth got charged its time and credited
none of its quality. That is exactly the shape of the k-ladder on record. No
results file in the screen tree had multi-pass data, so `results_L118_depth12.json`
was produced (100 cases × k=12, prune+sep_trim, min of 3, 741 passes / 655 kept).
Re-priced with real quality the verdict **survives** — k=1 is still RF-optimal —
but it is now supported rather than accidentally right:

| k | wtLP | quality | grid worst |
|---|---|---|---|
| **1** | 0.2184s | 1.236783 | **+1.674%** |
| 2 | 0.4381s | 1.222835 | −2.680% |
| 4 | 0.9018s | 1.207581 | −10.592% |
| 12 | 1.8781s | 1.197354 | −23.690% |

Break-even for k=12 is now **8.43×** (the old 5.1× is stale — L116 made k=1
cheaper too, which *raised* the bar).

**2. Per-case adaptive depth — RED.** An oracle that knows M and picks the best
k per case scores **+1.681%** against the shipped **+1.674%**: **+0.007pp**. Its
depth histogram is `{1: 96, 2: 3, 3: 1}`.

**3. Machine-speed calibration (M77 route Ⓐ) — capped at +1.06pp.** Narrowing
s from [0.5, 2.5] to [1.0, 1.1] moves grid worst +1.674% → +2.738%, and the best
depth is still k=2 at quality 1.2228 — **above 1.2**.

**4. Lagrangian relaxation / exact area — the cheap version is dead.**
Position-only optimisation (rho=0) is worth **+0.6223%** and **saturates in one
pass** (k=4 and k=12 return the same 1.257987), so the half of an LR
decomposition that becomes an O(V+E) longest path is chasing 12% of the value;
88% is in the shapes. The trust region is a cliff, not a slope: rho=0.06 keeps
100/100, **rho=0.10 keeps 0/100** (the area band's rho²·p slack exceeds the
official ±1%). But shapes cannot be post-processed at all — walking the LP's own
solution further along its own direction gives **0/100 kept at s=1.25**, and even
at s=1, swapping the linearised h for an exact-area h (a pure aspect trade at
constant area) gives **0/100**. The LP sits on a vertex where separations are
exactly tight, so any shape perturbation — in any direction, however small,
however good for area — breaks abutment. A real LR would have to solve positions
and shapes jointly, inside a budget of **<1.8× one LP pass** (k=12 quality at 1×
pass cost is worth +3.14pp; at 2× it is already negative). Estimated 8–15 days,
outcome unknown until the end.

**5. Soft violations — the lever is real but lives in the topology layer.**
The official cost is `(1 + 0.5(hpwl+area)) × exp(2·V_rel) × RF`; violations enter
as an *exponent*. Zeroing them is worth +3.041% in-set (quality → **1.199181**,
across 1.2) and +11.762% for MIB alone on the held-out corpus. The two corpora
disagree completely — in-set is 83% boundary and **0% MIB**, held-out is 74.9%
MIB across all 100 cases — so **L115's "MIB RED" only proved that this corpus has
nothing to take**, and cannot be extrapolated to the hidden graded set. But:

  * **MIB — RED.** Collapsing a group needs a median **69.4%** width change
    (p10 50.3%, p90 84.2%), i.e. ~12 passes at rho=0.06, and **52% of groups
    contain a fixed/preplaced member** that forces the shape outright.
    Reshape-only repair: 100 groups need it, **0 are repairable**.
  * **boundary — RED, three ways.** Translation-only reaches +0.341% held-out
    against a +4.834% bound. Inside the LP with a slack column priced from 0.5 to
    32 — a 64× range — **not one violation closes** (77 of 78 remain): touching
    is a 0/1 predicate with a 1e-6 threshold and a linear price only ever
    shortens the distance. Forced hard, **55/100 cases go LP-infeasible**, and
    exactly 54 cases have a violation — so essentially every case that has one
    cannot satisfy it. Under fixed topology the room does not exist.
  * Composition of the 78 in-set violations: **38 (49%) on preplaced blocks**,
    3 in no movable unit, 37 movable, 40 needing a corner (2+ edges). The
    preplaced 38 are repairable in principle by pulling the *envelope* in to meet
    them (the bbox is a min/max, and shrinking it is what the area term wants
    anyway) — implemented and measured, still nothing.

**6. Spending the runtime that sits under the floor — RED.** 91.8% of the weight
is already at the floor, so that time is free until `s·t/M > 0.3046`. Giving each
case the deepest k that fits under the floor at a design speed:

| design s | k>1 cases | s=1.0 | s=2.5 | grid worst |
|---|---|---|---|---|
| shipped | 0 | +2.53% | +1.68% | **+1.674%** |
| 1.00 | 35 | **+4.19%** | −6.31% | −6.311% |
| 1.50 | 5 | +2.95% | −0.39% | −0.390% |
| 2.00 | 1 | +2.58% | +1.42% | +1.422% |

No design speed beats the shipped configuration on grid worst, and the risk is
lopsided: designing for s=1 buys +1.66pp at the calibrated point and loses
7.99pp at the edge. κ = 3.161 is not a guess — it was back-solved from the alpha
round's real result (Rank 3, official 1.0286, cost-weighted RF 0.7081) — so s=1
*is* the calibration and [0.5, 2.5] is its uncertainty band. Every remaining
lever is now a bet on that band.

## Traps found the hard way

1. **`rows_for_k` degrades silently** on a one-pass file (see route 1). It prints
   a clean, monotonically falling ladder while crediting depth nothing.
2. **A probe must mirror `lp_pass`'s cluster re-freeze loop.** `hard_ok` does not
   test contiguity, so split clusters slip past it and get rejected later by the
   proxy: the first long-step probe scored 66/100 at s=1 where `dep_case` gets
   100/100. Any probe of this kind needs `s=1 must reproduce k=1` as its own gate.
3. **`hard_ok` sees none of the soft constraints** — not MIB, not boundary, not
   contiguity. Only the official `evaluate_solution` does.
4. **`is_feasible` is not just overlap.** It also covers fixed-shape and preplaced
   dimensions; a first cut that reshaped locked MIB members turned 63/100
   held-out cases infeasible (cost 1.53 → 7.62).
5. **`dep_case`'s guard structurally cannot accept a violation repair.** It is
   `better = hpwl or area improved` and `worse = any of hpwl/area/vrel worsened`,
   so a pass that only improves vrel fails the `better` test. Anything aimed at
   violations has to widen that guard first.
6. **Do not fit the cost function.** A least-squares fit gives
   `1 + 0.488h + 0.514a + 2.584v` with residual 2.7e-2 — right direction, wrong
   model. The real one is multiplicative with `exp(2·V_rel)`.

## State

Offline twin `l100_lp_speed.py` carries an L119 knob `BND_W` (boundary-repair
slack price; `None` = shipped behaviour, `0` = hard constraints). It was left in
place because it is the measurement apparatus for a route that is now RED and
should not be rebuilt from scratch to re-check. **Default path verified unchanged:
quality 1.236783247, kept 100/100.** The ship tree was not re-extracted, so
nothing in the submission moved.
