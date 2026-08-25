# L203–L205 — the gate shape is exhausted; the bet's downside was never drawn

**Written overnight 2026-08-25 while the L199 gates and L200 Linux lanes ran.
Deadline 2026-08-28.** Nothing here has been shipped. Two of the three findings
are *negative* (an axis closes); the third argues the shipped risk posture is
backwards, and names the one measurement that can settle it.

---

## 0. TL;DR

| # | finding | verdict |
|---|---|---|
| L203 | the one-parameter gate family `s` is the wrong *shape* — the objective is separable in `n`, so the optimal gate is a per-`n` sign test, not a threshold in `t/M` | 🟡 **real but worthless.** Transfers cleanly (98/100 agreement across disjoint samples) and still changes **no rank in either regime**. Axis closed. |
| L204 | the handoff's "downside protected" rests on the loss branch being `ra = 1.0`. It is not bounded there. The package stops beating beta at **ra = 1.052** — route A may cost **5 % more wall** before we are worse than our own graded submission | 🔴 **the tail is unpriced and it reaches rank 7** |
| L205 | route A's premise is *idle cores*. `solve()` submits **51 profiles at once** and the grader has **48 cores**, so the grader is **saturated** and there are none. Route A's win condition reduces to a workload ratio that **is** measurable here | ⏳ running |

---

## 1. L203 — the gate family is the wrong shape, and it does not matter

L196/L196b swept `gate_s(n) = 1 iff t_pool(n) + dt_lp(n) <= 0.3046·M(n)·s` and
cross-validated `s`, landing on 1.2. That search was over the wrong set. Both
halves of the objective are affine in the gate:

```
qual(g) = SUM_n QGAIN(n)·g[n]
rf(g)   = rf(0) - SUM_n RFCOST(n)·g[n]
```

The runtime factor is applied **per case**, not to a pool, so there is no shared
budget and this is not a knapsack. The optimal gate is the sign test
`g[n] = 1 iff QGAIN(n) > RFCOST(n)`, and no single threshold in `t/M` contains
it except by accident.

### The units error the ORACLE row caught

The first version of `l203_marginal_gate.py` printed **ORACLE below the shipped
s=1.2**, which is arithmetically impossible under separability. Cause: `RFCOST`
lives on the 100-row beta corpus (exactly one case per block count) while
`QGAIN` was being read off a 240-case sample carrying 2–4 replicates per block
count. Dividing one by 240-case sums and the other by 100-row sums makes the
comparison depend on how many replicates a block count happened to draw.

Put both per-case-at-`n` and the weights cancel exactly:

```
turn n on  iff  meangain(n)/qo  >  q_n·delta_n/BETA_NUM
```

**Printing a known upper bound next to the honest rows is what made a silent
normalisation bug visible.** It is worth keeping in any table of this shape.

### The result

Every candidate below is built **once** (fit on one 240-case sample) and scored
on the **other**, then scored in both regimes — because the package carries one
table and cannot know at runtime which regime it is in.

| candidate | on | NEUTRAL graded | rk | BET .68 graded | rk |
|---|---|---|---|---|---|
| **time gate s=1.2 (SHIPPED)** | 63 | 0.91491 | 4 | 0.87974 | 2 |
| time gate s=1.0 | 30 | 0.91806 | 4 | 0.89215 | 3 |
| time gate s=1.5 | 85 | 0.92588 | 4 | 0.87075 | 2 |
| marginal @ra=1.00 smooth-4 | 60 | **0.91460** | 4 | 0.88415 | 2 |
| marginal @ra=0.68 smooth-4 | 86 | 0.92652 | 4 | 0.87085 | 2 |
| marginal @ra=0.84 smooth-4 | 74 | 0.91674 | 4 | 0.87531 | 2 |

* The marginal rule is **not** noise fitting: two tables fitted on disjoint
  samples agree on **98/100** block counts, and ORACLE sits only ~0.07 pp above
  the honest rows.
* It is nonetheless **worth nothing**: +0.03 pp at neutral, −0.44 pp under the
  bet, and **no rank changes anywhere in the table**.
* `s = 1.2` is confirmed as the right *risk* choice, not merely the CV pick:
  `s = 1.5` is better under the bet (0.87075) but leaves only **0.08 %** of
  margin over beta at neutral, and both are rank 2 anyway.

**Rank 1 is not reachable on this axis.** The oracle bet-regime gate lands at
0.87051 against an r1 threshold of 0.85863.

### Side result: the wall is max-setter bound, and that confirms pruning is closed

The uncontended data settles a structural question for free. Per case,
`D_max = 1.492·D_mean` while `W/48 = 1.0625·D_mean`, so **`D_max > W/48`: the
48-core wall is set by the slowest single profile**, not by total work. That
makes "drop the max-setter" the natural runtime lever, and the oracle bound is
real — dropping the slowest profile *per case* would cut the weighted wall 3.21 %
(≈ +0.96 % of score), three of them 8.23 % (≈ +2.47 %).

**31 distinct profiles set the wall across the 100 cases**, the most frequent in
only 10 of them (8.6 % of weight), and the argmax agrees across three runs in
just 6/100. So "drop *the* max-setter" is genuinely unreachable — you cannot
identify it.

> 🚨 **This section originally concluded from that "the lever is closed, which
> strengthens L138/L139". That was wrong, and §6 is the correction.** Dropping
> the slowest *k* does not require identifying any particular profile: with
> `D_max/D_2nd = 1.017` the top is a plateau, so a table built from the mean of
> three runs still captures **35–49 %** of the oracle saving under
> cross-validation. Measured, k=8 is worth **+0.87 pp NET** and survives both
> OOS samples. Diffuseness kills the k=1 form and says nothing about k=8 — I
> generalised from one measurement to a family it does not cover.

### The `s` grid was too coarse to have proved 1.2; a 0.05 grid does

L196 sampled `s ∈ {0.8, 1.0, 1.2, 1.5, …}` — one point in the whole peak
region. Swept at 0.05:

| s | on | NEUTRAL graded | rk | BET graded | rk |
|---|---|---|---|---|---|
| 1.10 | 49 | 0.91545 | 4 | 0.88876 | 3 |
| **1.15** | 56 | **0.91463** | 4 | 0.88425 | 2 |
| **1.20 (shipped)** | 63 | 0.91491 | 4 | **0.87974** | 2 |
| 1.30 | 71 | 0.91582 | 4 | 0.87843 | 2 |
| 1.50 | 85 | 0.92588 | 4 | 0.87075 | 2 |
| 1.55 | 87 | 0.92706 | **5** | 0.87080 | 2 |

The neutral optimum is at `s = 1.15`, by **+0.03 pp** — well inside the ~0.2 pp
spread between the two CV directions — while `s = 1.20` is **+0.45 pp** better
under the bet. **Keep 1.20.** Two further facts the coarse grid hid: rank 2
under the bet is only reached from `s ≥ 1.15`, and the neutral branch falls
behind beta at `s ≥ 1.55`.

> Reusable: when an optimisation is separable, sweep the *shape* before tuning
> the *parameter*. A one-parameter family that has been cross-validated to death
> still only tells you the best member of the family you happened to write down.

---

## 2. L204 — 🚨 the downside is not neutrality

The handoff prices route A at 1.00 ("neutral") and 0.68 ("the bet") and
concludes the package is hedged because the neutral branch still beats beta by
+1.19 %. Two things are wrong with that frame.

### (a) neutral is not a branch of the bet — it is a different package

`_shape_lp_on()` and `_route_a_default()` fire on the same `>=40`-core gate, so
"route A neutral" is *exactly* what shipping with route A **off** produces. That
is a code default we can set with certainty. The choice is therefore

* **A** route A ON → `ra` unknown, outcome anywhere on the curve below
* **B** route A OFF → `ra = 1.00` exactly, **0.91491, rank 4, no tail**

Option B's number is the same 0.91491 the handoff calls "the downside" — but as
a **guarantee**, not as a floor that depends on route A being merely useless
rather than harmful.

### (b) the curve is steep and the tail is deep

| ra | graded | rank | |
|---|---|---|---|
| 0.68 | 0.87974 | 2 | the bet |
| 0.90 | 0.89656 | 3 | |
| **1.00** | **0.91491** | **4** | **route A OFF ships exactly here** |
| 1.05 | 0.92619 | 4 | −1.22 pp vs route-A-OFF |
| 1.10 | 0.93763 | 5 | **worse than beta** |
| 1.35 | 0.99369 | 7 | |
| 2.90 | 1.24839 | 7 | the multiplier measured on this box |

**Crossings:** rank 2 holds only while `ra ≤ 0.841`; rank 3 while `ra ≤ 0.917`;
**beta is lost at `ra = 1.052`.** The sensitivity is structural, not an artefact:
RF enters as `(t/med)^0.3`, so a 5 % wall increase costs 1.5 % of score, and the
whole margin over beta is 1.26 pp.

**Sensitivity to my own modelling choice.** `ra` above multiplies the *whole*
pool wall, which overstates the risk: route A replaces the profile-execution
phase but not the serial proxy/post-process tail on the main thread, which M47
measured at 29 % of the n>100 wall. With `t = POOL·(1−φ + φ·ra) + dt·g`:

| φ | beta lost at | rank 3 holds to | graded @ ra=1.35 |
|---|---|---|---|
| 1.00 | ra = 1.052 | ra ≤ 0.917 | 0.99369 |
| 0.71 (M47) | ra = 1.073 | ra ≤ 0.884 | 0.97121 |
| 0.50 | ra = 1.103 | ra ≤ 0.835 | 0.95473 |

I expected φ to roughly double the margin. **It does not** — 5 % of wall becomes
10 % of wall, and that is all. φ rescales the perturbation but not the 1.26 pp
margin over beta, and `t^0.3` spends that margin at 0.3 % of score per 1 % of
wall either way. **The conclusion is insensitive to φ.**

### (c) expected value

Option B is a certainty at **0.91491**. Option A's expectation, for
`p = P(route A delivers 0.68)` and three assumptions about the loss branch:

| p | loss = 1.0 | loss = 1.35 | loss = 2.9 |
|---|---|---|---|
| 0.5 | 0.89733 | 0.93672 | 1.06406 |
| 0.8 | 0.88677 | 0.90253 | 0.95347 |
| 0.9 | 0.88326 | 0.89113 | 0.91660 |

If the loss branch really is 1.0, the bet pays from `p ≳ 0.2`. If it is 1.35,
you need `p ≳ 0.75`. If it is 2.9, even `p = 0.9` loses to the certainty.

**So the entire decision reduces to: can route A be WORSE than neutral on the
grader?** The handoff never asks this. L205 answers it.

---

## 3. L205 — route A's premise fails on the grader, and that is checkable here

Read out of the code, not assumed:

* `solve()` submits **all** profiles at once —
  `ThreadPoolExecutor(max_workers=len(profiles))` — and the 48-core
  configuration has **51 profiles** (verified: `_pool_indices(n)` returns 51 for
  every `n` in 21…120). The grader has 48 cores. **51 runnable on 48 is
  saturated**, so route A's stated premise — "route A only converts IDLE cores
  into wall" — has no idle cores to convert.
* `_route_a_cores()` **deliberately ignores** `ICCAD_ADAPTIVE_CORES` and sizes
  the frame queue from the real core count, so route A *cannot* oversubscribe.
  My first hypothesis — that the 2.9× came from oversubscription and could be
  fixed by gating on physical cores — is **wrong, and the code says so in its
  own docstring**. What costs is not concurrency, it is **work**: L110 measured
  route A doing **1.44×** the work of the plain path.

On a saturated box the two makespans are

```
plain    ~ max( D_max , W/cores )     one subprocess per profile, frames serial
route A  >= 1.44 * W/cores            same frames, one shared queue
```

so route A wins iff the profile durations are imbalanced enough for the long
pole to dominate:

```
D_max / D_mean  >  1.44 * 51/48  =  1.53
```

**`D_max/D_mean` is dimensionless.** Every profile on this box is slowed by the
same oversubscription factor, so the ratio survives the transport that the wall
does not — the ledger's own `f`-free discipline, applied to a different
quantity. This is why "route A cannot be measured here" was true of its *wall*
and false of its *win condition*.

### 🚨 The first measurement was corrupted, and the corruption was not visible

v1 put the instrument in `optimizer_constructive.py` and printed
`[proftime] …` to stderr from all 51 profile threads. It produced a clean-looking
table with `weighted mean ra = 0.980`. It was wrong:

```
grep -c 'proftime]'          5100   = 100 cases x 51 profiles, all emitted
lines matching the record     4588   = 10% shredded by concurrent stderr writes
```

**Neither bias is recoverable after the fact, and they point opposite ways:**
losing a record biases `D_max` **down** (the lost one may have been the maximum,
which favours the plain path) *and* `k` **down**, which lowers `W` and so
favours route A. A second bug rode along: the verdict column compared against a
hardcoded `1.44·51/48 = 1.53` while the per-case `ra` used the measured `k`, so
the table said "4 of 100 can win" next to a mean `ra` below 1 — **an internal
contradiction that is the only reason the record loss was noticed at all.**

v2 fixes both: records go to a file under a `threading.Lock` (all 51 are threads
in one process, so a lock is exact), the threshold is per-case `1.44·k/48`, and
`l205_imbalance.py` now **asserts every block count has all 51 records** before
printing anything.

> Reusable: an instrument that loses samples silently is worse than no
> instrument. The completeness count is not a formality — it is the only thing
> standing between "0.980" and a wrong decision about the whole package.

### The instrument does not ship

v1 edited the shipped tree, which would have forced a re-stage and a second
hour of Linux lanes to prove inertness. v2 puts it in `optimizer_l205probe.py`
instead, and the shipped tree was reverted and **verified byte-identical**:

```
rebuilt op_wrapper md5   bb44bb147231fee7bc9670cdc28448bc
staged/verified md5      bb44bb147231fee7bc9670cdc28448bc   MATCH
```

So the artefact that ships is exactly the one that passed the eight in-set gates
and the five Linux lanes. No re-stage, no re-verify, and `l207_wsl_final.sh` is
needed only if someone later changes the tree.

The probe runs with `ICCAD_ROUTE_A=0` where the L199 gates ran with it on, which
makes the pair a free re-test of the ledger's "route A is result-neutral" claim.

### The measurement, corrected

Two independent runs, 5100/5100 records each, completeness asserted:

| run | ratio min | p25 | **median** | p75 | route A wins | weighted share | **mean ra** |
|---|---|---|---|---|---|---|---|
| parallel r1 | 1.169 | 1.298 | **1.377** | 1.439 | 2/100 | 1.7 % | **1.0639** |
| parallel r2 | 1.144 | 1.276 | **1.349** | 1.391 | 1/100 | 2.9 % | **1.0778** |

Threshold `1.44·51/48 = 1.530`. **The imbalance is nowhere near enough**: the
median case is at 1.36 against a bar of 1.53, and the two cases that clear it
carry under 3 % of the weight.

`ra ≈ 1.064–1.078` against L204's beta crossing at **`ra = 1.052`** — route A
lands **just past the point where the package is worse than our own graded
submission**, and strictly worse than the certain route-A-OFF configuration in
every scenario.

**This is the sign the corrupted v1 got backwards** (0.980, "route A saves
2 %"). The whole shipping decision rests on which side of 1.052 this number
falls, and a 10 % silent record loss was enough to move it across.

### The threshold's inherited constant, checked against a direct measurement

The bar `1.44·k/48` carries L110's work multiplier, which I inherited rather
than measured — and the verdict is sensitive to it: at `WORK = 1.30` the bar
falls to 1.381, right on the median, and at 1.20 route A wins.

It is measurable here, from runs that already existed. Four 100-case
evaluations, same tree, same 51-profile pool, same flags, **differing only in
route A**:

| arm | route A | wall |
|---|---|---|
| `l199_det1` / `det2` | **on** | 7:06 / 7:01 |
| `l205_r1` / `r2` | **off** | 2:45 / 2:47 |

Both configurations saturate this box — route A off is 51 subprocesses on 32
logical cores, route A on is a queue capped at `_route_a_cores()` = 32 — so the
wall ratio *is* the work ratio: **2.52–2.58×**. That independently reproduces
the ledger's 2.9×, from matched arms rather than from a single noisy timing.

**This makes the verdict robust rather than fragile.** Both available estimates
of the multiplier (L110's 1.44, this box's ~2.5) sit at or above 1.44, and the
bar is monotone increasing in it, so route A loses under either — by 10 % at
1.44, by a wide margin at 2.5. Flipping the verdict needs `WORK < 1.30`, and
nothing points there.

> ⚠️ The 2.5× is an **upper** bound for the grader, not a transportable value:
> route A spawns one subprocess per *frame* where the plain path spawns one per
> *profile*, and Windows process creation is far more expensive than Linux's.
> The verdict below keeps L110's 1.44.

Two further readings from the same table: the gated LP costs only ~14 s of wall
across 100 cases here (7:06 vs `l199_lpoff`'s 6:52), while route A costs over
four minutes.

### The one bias that could still rescue route A, and how it is being removed

Two biases remain, pointing opposite ways:

* **against** route A — the model credits it with *perfect* packing of the frame
  queue, which `_run_profile_route_a`'s own submission rule (stop once the
  prefix holds `max_trials` successes) does not give. True `ra` is **higher**.
* **for** route A — this box runs 51 profiles on 32 logical cores (1.6×
  oversubscribed) where the grader runs 51 on 48 (1.06×). Under heavy
  oversubscription short profiles finish early and hand their share to the long
  ones, **compressing `D_max/D_mean` toward 1**. True `ra` is **lower**.

The second one is not negligible: the verdict flips if the true median rises
from 1.377 to 1.530, an **11 %** move. So it gets measured rather than argued —
`ICCAD_PROF_SEQ=1` runs the profiles **one at a time**, with no scheduler in the
way, giving the workload's own imbalance.

**It was real, and it changes the verdict.**

| run | ratio min | p25 | **median** | p75 | A wins | weighted share | **mean ra** |
|---|---|---|---|---|---|---|---|
| parallel r1 | 1.169 | 1.298 | 1.377 | 1.439 | 2/100 | 1.7 % | 1.0639 |
| parallel r2 | 1.144 | 1.299 | 1.362 | 1.411 | 2/100 | 3.6 % | 1.0671 |
| **sequential** | 1.191 | 1.418 | **1.492** | 1.565 | **30/100** | **32.3 %** | **1.0021** |

Compression, measured per block count: **median 1.088** (p25 1.051, p75 1.134).
The contended runs were understating the imbalance by ~9 %, exactly as the
mechanism predicted.

**Route A is neutral, not harmful.** `ra = 1.0021` sits inside the beta margin
(crossing at 1.052), and the earlier "it costs 6.4 %" reading was the
compression artefact. Sequential vs shipped results: cost 100/100, positions
100/100 — the probe does not perturb what it measures.

The uncontended median ratio 1.492 is still below the 1.530 bar, but by 2.5 %
rather than 11 %, and a third of the weight now sits on cases that clear it.

---

## 4. What is verified as of writing

`l199_verdict.py`: **ALL PASS**, G2c included — nothing provisional.

| gate | result |
|---|---|
| G1 determinism | cost 100/100, positions 100/100 |
| G2a L147 hatch | 63/63 identical to the anchor on the block counts the gate keeps |
| G2b skipped == no-LP | 37/37 identical to the `lpoff` arm on the block counts it drops |
| **G2c L147 + gate off** | **100/100 identical to `results_L165_l147off.json`** — decisive: the 63/100 reading was the gate and nothing else |
| G3 gate fired | default **63** = the table's 1-set exactly; `LP_GATE=0` → **100**; `SHAPE_LP=0` → **0** |
| G4 map is flat | `k1` bit-identical to `det1`; passes spent `{1: 63}` |
| G5 feasibility | 100/100 in all 7 arms |
| G6 gate cost | −3.3998 % vs the ungated LP, 37 moved (exactly the gated-off set) |
| G7 LP value | **+1.6417 %**, 62 better / 0 worse |
| G8 hb predictor | −0.0227 % (was −0.0512 % on L172's map) |

Cross-session: `det1` is bit-identical to the previous session's
`_l198_gateon.json`, 100/100.

### Linux, all five lanes (`l200_wsl_verify.sh`, 32-core WSL, py3.14 / scipy 1.18)

| lane | result |
|---|---|
| 1 — 48c, LP off | **PASS**, `+0.0000%` vs the Windows LP-off base — bit-identical with the LP off, as L153 recorded |
| 2 — 48c, `SHAPE_LP_L147=0` | **PASS**, LP ran on exactly the 63 selected block counts |
| 3 — 48c, shipped default | **FAIL, on the stale threshold only** — see below |
| 4 — determinism, same run twice | **PASS**, cost 100/100 **and positions 100/100** |
| 5 — `ICCAD_LP_GATE=0` liveness | **PASS** — default == the table's 1-set, gate-off == all 100 |

**LANE 5 is the one that could have failed on L196** and it is green on Linux:
the gate fires on exactly the right set, and the kill switch widens it to 100.
Lanes 1–4 would all pass unchanged if the table were inert.

**LANE 3's only failure is `--live-min 1.5`**, a floor set when the LP ran on all
100 cases. Everything of substance passed: feasible 100/100, **0 regressions
against the pre-LP base at the strict `budget = 0`**, total `1.23922` vs pre-LP
`1.26025` = **+1.6686 %** ahead. The measured gap to the control is **+0.7418 %**
on Linux (Windows in-set predicted +0.6761 %), against a failure mode — L147
silently not applying — that produces 0.000 %. `l207_wsl_final.sh` re-runs the
lane at 0.40.

7/100 cases differ from the Windows arm by >1e-9 (L119 recorded 8/100 from the
scipy-version divergence on a degenerate LP), the largest being case 9 at
|d| = 0.157. **G-B passing at `budget = 0` means no case — case 9 included — is
worse than its own pre-LP value**, so the divergence is about *which* equivalent
optimum the LP lands on, not about safety.

### Two anchors that were stale, repointed rather than relaxed

* `l117_linux_verify.py:_lp_liveness` hard-failed when the stats file had fewer
  lines than cases — correct behaviour for a *correct* L196 package, and it
  would have failed LANE 3 an hour into the run. It now parses `_L196_LPGATE`
  **out of the tar** and asserts the multiset of block counts matches exactly,
  which also catches a table firing on the wrong 63.
* `results_L165_l147off.json` predates the LP gate, so the flat bit-compare
  reads 63/100 on a package where nothing is wrong. The 37 that differ are
  **exactly** the 37 the gate drops (both set differences empty). Split into
  G2a/G2b/G2c, which together assert strictly more than the original did.

### One phantom flag

L177's `det1`/`det2` were distinguished by `ICCAD_SHAPE_LP_NOOP`, **which does
not exist in the tree** (grep: 0 hits). Harmless there — both arms were the
default, which is what a determinism test wants — but the L199 arms differ by
tag only.

---

## 4b. 🚨 The route A decision, stated as what it actually is

`ra = 1.0021`. **Route A's expected effect on the score is indistinguishable
from zero.** Not harmful — the 6.4 % cost was a compression artefact — and not
helpful either.

Both sides of the model are lower bounds, so neither direction is strict:

* `route A ≥ 1.44·W/48` assumes perfect packing of the frame queue, which its
  own submission rule (stop once the prefix holds `max_trials` successes) does
  not give → true `ra` **higher**.
* `plain ≥ max(D_max, W/48)` assumes the 3 profiles that do not fit on 48 cores
  land on short tails → true `ra` **lower**.

They partly cancel, and the residual is smaller than the distance to any rank
boundary. **So this is not a question about expected value. It is a question
about variance**, and that makes it a decision rather than a measurement:

| | outcome |
|---|---|
| **route A OFF** | certainty: **0.91491, rank 4**, beats beta by 1.26 pp |
| **route A ON** | a lottery centred on the same place: rank 3 needs `ra ≤ 0.917`, rank 2 needs `ra ≤ 0.841`, **rank 5 starts at `ra ≥ 1.10`**, rank 7 at 1.35 |

The right answer depends on the payoff, which is the user's to weigh:

* **If placing 4th is meaningfully better than 7th** — keep certainty, turn
  route A off. Same expected score, no tail.
* **If only the top 3 pay** — a certain 4th is worth exactly what 5th is worth,
  so the lottery's upside is free and the tail costs nothing. Keep route A on.

I have **not** made this change. Turning it off is one line —
`_route_a_default()` returning 0 — but it would invalidate the package that has
just passed eight in-set gates and five Linux lanes, so it needs the full chain
again (~1.5 h, scripts ready: `l199_gates.sh` → `l199_verdict.py` →
`l207_wsl_final.sh`). It must be a **code default**: the grader strips `ICCAD_*`,
so an env-only kill switch is inert in the package (L158).

⚠️ One correction to the ledger either way: `HANDOFF_2026-08-25.md` §5 says
route A "cannot be measured here and both school machines are unavailable …
This is closed: it stays a bet." That is true of its **wall** and false of its
**win condition**. The condition is a workload ratio, it transports off this
box, and it has now been measured three times.

---

## 5. Recommendation, for review

1. **Ship the L196 package** — gates are green and the Linux lanes are running.
   Nothing in L203/L204 argues against the LP gate or `s = 1.2`; L203 confirms
   both under a strictly more general family.
2. **Decide route A explicitly, on the variance question in §4b** — it is no
   longer an unmeasured bet, and it is no longer "unfavourable" either. `ra` is
   1.0021: turning it off buys certainty at the same expected score, and that
   is worth doing iff 4th place is worth more than 7th. My own read is **turn it
   off**: the margin over beta is 1.26 pp, RF spends it at 0.3 % of score per
   1 % of wall, and nothing in three measurements suggests route A pays. But it
   is a preference over outcomes, not a fact, so I have not applied it.
4. **G8 / the hb predictor**: −0.0227 % in set. Per this ledger's own rule
   ("never act on an in-set null" — the twins moved 0 in set and are worth
   +0.67 % OOS) that is **not** grounds to pull it. It would need the two OOS
   samples re-run in the L196 configuration, ~2.3 h, for ~0.02–0.05 %. Lowest
   priority of anything open.

---

## 6. L211-L213 — the pool drop: the one rank-improving route that survived

### Where it came from

The uncontended durations answered a structural question for free: the 48-core
wall is **max-setter bound** (`D_max = 1.492*D_mean` vs `W/48 = 1.0625*D_mean`),
so dropping the slowest profiles is the only runtime lever left after L155/L156
closed the LP ones. Three facts made it worth pricing rather than dismissing:

* `_pool_indices` takes **block_count**, so a drop table can be per-`n`. The
  ledger's L138/L139 closure is about *fixed global* drop sets.
* M41/M42 already took the free part — max-setters the proxy never selects — so
  what remains costs quality, and the question is only *how much*.
* The portfolio takes the **min over 51 candidates**, so removing one costs
  nothing unless it was the winner.

### What it is worth

| k | pool | quality in set | moved | case wall | **NET** | graded | rank |
|---|---|---|---|---|---|---|---|
| 0 | 51 | — | — | — | +1.260 % | 0.91491 | 4 |
| **8** | **43** | **−0.1242 %** | 12 | **−5.50 %** | **+2.133 %** | 0.90684 | 4 |
| 12 | 39 | −0.2438 % | 20 | −6.68 % | — | — | 4 |
| 20 | 31 | −0.3943 % | 32 | −10.31 % | — | — | 4 |

**Out of sample, both disjoint 240-case samples, 0 infeasible in either:**
s1 **−0.3852 %**, s2 **−0.2125 %**, mean **−0.2989 %** = **2.41×** the in-set
cost — the same order as the thin pool's 1.9×. **NET +2.133 % against today's
+1.260 %: +0.87 pp, about 3× the project's 0.30 % ship bar.** Rank does not
change; **rank 3 needs +2.942 %** and the best extrapolated point is +2.33 %.

### Why the OOS cost is 2.41× and why it had to be measured

The table is keyed on block count and the in-set corpus has **exactly one case
per block count — the very case whose durations built it.** The OOS samples
carry 2–4 *different* floorplans at each block count, where a dropped profile
may be the winner. The move rate rises **12 % → 20 %** accordingly. This is
precisely how L138/L139 died, and an in-set-only reading would have been
worthless.

### Two model errors found by checking rather than assuming

1. **The max-setter is not identifiable from one run.** The argmax agrees across
   three runs in only **6/100** block counts, because the top of the duration
   distribution is a **plateau** — median 2 profiles within 2 % of `D_max`, 7
   within 10 % — and run-to-run wall noise swamps the gaps. `D_max/D_2nd` is
   **1.017**. This is also *why route A cannot win*: `D_max/D_mean = 1.49` comes
   from a long **left** tail of fast profiles, not from one slow outlier, and
   load balancing needs a long pole to balance. The table is therefore built
   from the mean of three runs and cross-validated: a table fitted on the two
   parallel runs captures **35–49 %** of the oracle saving on the sequential one.
2. **The LP gate is a static table.** `_L196_LPGATE` was computed at the
   *pre-drop* pool times and does not recompute at runtime, so the shipped
   package does **not** get the "gate widens for free" compounding I had
   modelled. Checked rather than assumed — and it turns out immaterial: rebuilt,
   the gate fires on 66 instead of 63 for NET +2.131 % against +2.133 %. The
   extra quality is exactly cancelled by the extra RF. **No gate rebuild.**

### How it ships

`_L211_POOLDROP` — 100 block counts × 8 ORIGINAL `_PROFILES` indices, a **code
default** (the grader strips `ICCAD_*`, so an env-only mechanism is inert in the
package — L158). Cores-gated `>= 40` and fail-closed: the durations were
measured in the 48-core configuration where the pool is 51, and below the gate
the pool is a different 35 where these indices would drop the wrong profiles.
Read **after** the `ADAPTIVE_POOL=0` early return, like M41/M42 and unlike the
additive M72/M76/M80 tiers, so the full-pool probe path still measures the full
pool. Kill switch `ICCAD_L211_POOLDROP=0`.

Verified in all four directions before any run: 48 cores → 43, kill switch → 51,
32 cores → no drop, `ADAPTIVE_POOL=0` → 57 untouched.

**G10** is the only gate that can kill this table, and it is a *pair*: the kill
switch arm must reproduce `results_L209_det1.json` bit-for-bit (catches a table
that drops too much) **and** the default must differ from it on exactly the 12
cases the drop was measured to move (catches a table that never loaded). Either
half alone is passed by one of the two failure modes.
