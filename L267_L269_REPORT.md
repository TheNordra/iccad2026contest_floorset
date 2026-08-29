# L267 / L268 / L269 — the frame axis reopens; the ordering axis closes with a number

**Two verdicts.**

1. **§2.1 adaptive frame search is real and, in the right deployment form, affordable.**
   `l269p2` — bisection *inside* the trial loop, budget 2 — is **−0.8143 %** of true cost
   on the 40-case heavy band, at a measured **1.0101×** max-setter wall ⇒
   **NET +0.52 … +0.66 pp**, against a 0.30 % ship bar.
   **✅ CONFIRMED ON s2** (the disjoint 240-case corpus, `worker_10..19`):
   **−0.7285 %**, isolated **−0.8270 %** (647 better / 345 worse), and
   **positive on 8/8 split-halves across the two corpora**. See §1.6.
   Not deployed: it is a `constructive.cpp` change and the shipping tree is frozen.

2. **§3.4 big-first commitment order is closed, and now it has an exchange rate.**
   The order really is the density constraint — it moves the packer's ceiling further
   than anything ever measured here (util 81.34 → 85.15 %). It still cannot pay,
   because **it buys area at ~1.2× its own price in wire**, and area and wire enter
   the cost function identically.

---

## 0. Instruments and gates

One probe binary, `constructive_l270.exe`, branched from the shipping
`constructive.cpp` (md5 `e2c7b2f418ef2b70b6bff99f7adfbd37`), carrying three
independent mechanisms plus the L252 emitters. `constructive.cpp` was not modified.

| gate | arm A (flags off) | arm B (flag on) | reading |
|---|---|---|---|
| `--flags ICCAD_L252` | **102/102 PASS** | **102/102 PASS** | recompile clean, emitters stderr-only |
| `--flags ICCAD_L267` | 102/102 PASS | **0/102 FAIL** | mechanism live on every profile |
| `--flags ICCAD_L268` | 102/102 PASS | **0/102 FAIL** | mechanism live |
| `--flags ICCAD_L269` | 102/102 PASS | **0/102 FAIL** | mechanism live |

For an *instrument* arm B must PASS; for a *mechanism* arm B must **FAIL** — a passing
mechanism arm is a silent no-op. Both directions are exercised here.

**A fourth gate, free, inside the measurement**: every quality run carries a `ship`
arm (probe, no flags). It reproduced the cached baseline at **+0.0000 %, 0 layouts
changed**, in all three runs — so the harness is measuring the shipped placer.

---

## 1. §2.1 — the adaptive frame search

### 1.1 The first form was right about quality and wrong about cost

`L267`: a pre-pass bisects for the per-case cliff, then emits a ladder anchored there.
No fitted rung positions — every rung is derived per case, so L266's failure mode
cannot apply.

| | |
|---|---|
| quality, 40 cases, official strict scorer | **−1.3785 %** |
| isolated mechanism (same profile) | −1.4193 %, 1908/2040 changed, **1151 better / 757 worse** |
| split-half | **+1.63 / +1.14** (alternating), **+1.22 / +1.61** (by size) — no flip |
| gaps | hpwl 0.2924→**0.2774**, area 0.2300→**0.2107**, vrel 0.0893→0.0890 |

That is the largest quality gain on this 40-case anchor in the whole L250–L268 arc
(next best: `dense`, −0.7447 %), and the only arm that improves **all three** terms.

**But the wall killed it: 1.2417× on the max-setter ⇒ −3.649 pp ⇒ NET −2.27 pp.**

### 1.2 🚨 A tooling error, and the reconciliation that settles it

The first deterministic proxy read **1.063×** and appeared to contradict the stopwatch.
It was mis-specified, not more trustworthy:

> `_pkmax` = max over profiles of *pack count*. The pack max-setter is **prof 33**,
> which is the **slowest-ranked 51 of 51 by time** (0.298 s against the max-setter's
> 0.793 s). A profile is slow because its packs are *expensive*, not because it does
> many of them. Taking a max along the wrong axis picks a cheap, loopy profile.

Correct proxy (`l269_wallproxy.py`): weight each profile's own **measured** shipped
time by that profile's own pack ratio, *then* take the max.

| | est × max | est × sum |
|---|---|---|
| corrected pack proxy, `adapt` | **1.2310×** | 1.1390× |
| stopwatch, `adapt` | **1.2417×** | 1.1070× |

**The two instruments agree to 1 pp.** The stopwatch was right; L267's pre-pass form
is genuinely unaffordable. Cutting the probe budget does not rescue it (`adaptp2`
1.1125×, `adaptp3` 1.1242×) — the problem is structural.

### 1.3 The structural fix: bisect *inside* the trial loop

`L269`: no separate probe pass. The loop proposes its own next frame by bisecting
between the loosest scale known to fail and the tightest known to pack, **anchored on
the aspect that just succeeded** (which makes it monotone — it can never land looser
than the shipped ladder's own first success, the failure mode that cost L267 13/40
cases). A proposal that packs becomes a real trial — work the loop would have spent
on another aspect anyway. Only the *failing* proposals are new cost.

| arm | quality | isolated | × max (stopwatch) | × max (corrected proxy) | NET |
|---|---|---|---|---|---|
| `adapt` (pre-pass, 5) | −1.3785 % | −1.42 % | 1.2417 | 1.2310 | **−2.27 pp** |
| `l269` (in-loop, 5) | −0.6499 % | **+2.20 %** | 1.0253 | 1.0123 | +0.27 pp |
| **`l269p2` (in-loop, 2)** | **−0.8143 %** | **−0.7220 %** | **1.0101** | 1.0190 | **+0.52…+0.66 pp** |
| `l269b` (start loosest) | +2.4381 % | +10.95 % | 1.0544 | 0.8947 | RED |

Total work actually **falls**: `l269` 0.9670×, `l269p2` 0.9741×.

### 1.4 🔑 Fewer proposals is better, and the isolated column says why

`l269` (5 proposals) has a *better* headline than nothing but its **isolated** delta is
**+2.20 % with 753 better / 784 worse** — a coin flip. Its portfolio gain is being
manufactured by the selector picking the lucky ones, not by the mechanism.
`l269p2` (2 proposals) is **−0.72 % isolated, 626 better / 384 worse** — the mechanism
itself is better.

Mechanism: every proposal that packs consumes one of `max_trials`(=4) slots. Spend 5
on scale-bisection at one aspect and the **aspect search is crowded out**; spend 2 and
you keep two aspect trials plus two tighter scales.

### 1.5 Transfer

| arm | alt H1 | alt H2 | size H1 | size H2 |
|---|---|---|---|---|
| `l269` | +0.782 | +0.524 | +1.188 | **−0.125 ← FLIPS** |
| **`l269p2`** | **+1.183** | **+0.463** | **+1.041** | **+0.487** |

`l269p2` is positive on 4/4. `l269` flips on the by-size split.

⚠️ **Honest limit on the constant.** `PROBES=2` was chosen on this sample. The
*mechanism* transfers (7 of 8 half-measurements positive across both budgets); the
*budget* is not well determined — selection between the two transfers at 66–88 %,
and picks a different one in each direction. The claim that survives is
"in-loop bisection with a small budget", not "2 is the right number".

⚠️ **Concentration.** `l269p2` moves 22/40 cases (57.1 % of weight), 14 better /
8 worse, and the **top-3 cases carry 55 % of the gain**. Both bands are positive
(heaviest 20 +1.041 pp, lightest 20 +0.487 pp).

### 1.6 ✅ s2 — the disjoint corpus agrees

`PROBES=2` is a constant chosen on s1, and the project's own rule for that is an s2
re-score. `l252_frames.py --sample s2` first captured a fresh shipped-placer baseline
for the top-40 heavy cases of `worker_10..19` (append-only into `l252_cache.pkl`;
`l252_cache.pkl.s1only` is the pre-append backup, same md5 as the original).

| | s1 (`worker_0..9`) | **s2 (`worker_10..19`, disjoint)** |
|---|---|---|
| portfolio | **−0.8143 %** | **−0.7285 %** |
| isolated mechanism | −0.7220 % (626 better / 384 worse) | **−0.8270 % (647 / 345)** |
| split-half, alternating | +1.183 / +0.463 | +0.369 / +1.102 |
| split-half, by size | +1.041 / +0.487 | +0.578 / +0.955 |
| `ship` arm vs cached base | +0.0000 % | +0.0000 % |
| exchange ratio Δhpwl / −Δarea | 0.24 | 0.59 |
| NET | +0.52 … +0.66 pp | ≈ **+0.46 pp** |

**Positive on 8/8 halves across two disjoint corpora, and the *isolated* mechanism is
slightly better on s2 than on s1.** Both clear the 0.30 % bar.

🔑 Note the headwind this passed against: **s2's own cliff is further from the shipped
ladder than s1's** — s2 baseline `s_min` 1.1284 (util 78.5 %) vs s1's 1.1088 (81.3 %),
`s_landed` 1.1391. A mechanism whose whole job is to put a rung on the cliff has *less*
reachable grain on s2, and it still transferred at 89 % of magnitude.

s2's gaps: hpwl 0.2778 → 0.2944, area 0.2482 → **0.2201**, vrel 0.0812 → **0.0797**.
Same shape as s1: buys area, pays a fraction of it in wire, does not touch violations.

### 1.7 The probe-budget sweep — the constant is decided by the ISOLATED column, not the headline

All five budgets, one run, one base (so the comparison is internally exact).
`l269p2` reproduced its earlier value to the digit (−0.8143 %), which is the
replication check.

| budget | portfolio | **isolated** | better : worse | layouts moved | est × max | NET | split-half |
|---|---|---|---|---|---|---|---|
| **p1** | −0.5706 % | **−1.8798 %** | **2.22 : 1** | 683/2040 | **0.9990** | **+0.59 pp** | 4/4 positive |
| **p2** | −0.8143 % | −0.7220 % | 1.63 : 1 | 1010/2040 | 1.0190 | **+0.53 pp** (+0.66 by stopwatch) | 4/4 positive |
| p3 | **−0.9482 %** | **+0.3306 %** 🚨 | 1.31 : 1 | 1263/2040 | 1.0195 | +0.65 pp | 4/4 positive |
| p4 | −0.6470 % | +1.5209 % | 1.04 : 1 | 1438/2040 | 1.0121 | +0.47 pp | **flips** (−0.125) |
| p5 | −0.6499 % | +2.2044 % | 0.96 : 1 | 1537/2040 | 1.0123 | +0.27 pp | **flips** (−0.125) |

🔑 **The isolated column is monotone in the budget and changes sign between p2 and p3**:
−1.88 → −0.72 → **+0.33** → +1.52 → +2.20, with the better:worse ratio decaying
2.22 → 1.63 → 1.31 → 1.04 → 0.96. That is not a noisy curve with a lucky peak — it is a
monotone trend with a mechanism behind it: every proposal that packs consumes one of
`max_trials`(=4) slots, so a larger budget fills all four trials with scale-variants at a
single aspect and crowds the aspect search out entirely.

🔑 **So the headline and the mechanism disagree, and they disagree informatively.**
p3 has the best portfolio number *while making most of the layouts it touches worse*
(+0.33 % isolated) — its gain is the selector picking lucky candidates out of a noisier
pool, not the mechanism working. p1 and p2 are the only budgets where the portfolio gain
and the mechanism point the same way.

⇒ **The defensible range is p1–p2; p3 and above are not**, and the criterion that says so
is reproducible on any future knob. Among p1/p2 the NET spread (+0.53…+0.66 pp) is inside
the instruments' own disagreement, so the headline cannot choose between them.

### 1.8 ✅ Both open questions closed — and `p1` wins

The two missing measurements were taken on an idle box (min-of-3 for the wall, and a
full s2 re-score). Both went p1's way.

**Wall, min-of-3, exclusive box** (ship max-setter prof 1 at 0.789 s):

| arm | × max | × total | NET wall term |
|---|---|---|---|
| **l269p1** | **0.9887** | **0.9668** | **+0.171 pp (a bonus, not a cost)** |
| l269p2 | 1.0183 | 0.9819 | −0.276 pp |
| l269p3 | 1.0196 | 0.9806 | −0.295 pp |

**`p1` is measurably FASTER than the shipped placer**, on both the max-setter and total
work — the corrected proxy said 0.9990× and the stopwatch says 0.9887×. The mechanism:
one proposal *replaces* ladder frames rather than adding to them, so a tight frame that
packs early lets the loop reach `max_trials` sooner and walk fewer shipped rungs.

**Final table — quality × cost × transfer, two disjoint corpora:**

| budget | s1 quality | s2 quality | transfer | wall × max | **NET s1** | **NET s2** | isolated | halves |
|---|---|---|---|---|---|---|---|---|
| **p1** | +0.5706 pp | **+0.5507 pp** | **96 %** | **0.9887** | **+0.742 pp** | **+0.722 pp** | **−1.88 % (2.22:1)** | **8/8** |
| p2 | +0.8143 pp | +0.7285 pp | 89 % | 1.0183 | +0.538 pp | +0.453 pp | −0.72 % (1.63:1) | **8/8** |
| p3 | +0.9482 pp | — | — | 1.0196 | +0.653 pp | — | **+0.33 %** 🚨 | 4/4 (s1 only) |

🏆 **`l269p1` is the ship candidate.** It wins on every axis that matters and it wins the
one that decides: **NET +0.742 pp (s1) / +0.722 pp (s2)**, ~2.4× the 0.30 % bar, on both
corpora. It also has the tightest cross-corpus agreement of any arm measured this session
(−0.5706 → −0.5507), the strongest isolated mechanism, and a wall that is *negative cost*.

🔑 **And the headline was the wrong guide throughout.** Ranked by portfolio quality the
order is p3 > p2 > p4 ≈ p5 > p1 — exactly backwards from NET, because the quality ranking
prices neither the wall nor whether the mechanism is doing the work. The two columns that
did choose correctly were **isolated cost** and **measured wall**, and both pointed at the
*smallest* budget from the start.

---

## 2. §3.4 — big-first ordering, and the exchange rate that closes it

### 2.1 The reachability claim is TRUE, and bigger than anything measured here

Dense 26-rung ladder, 40 cases, `s_min` = tightest packable frame:

| arm | s_min | util | area | tighter / looser / same | packs |
|---|---|---|---|---|---|
| shipped order | 1.1088 | 81.34 % | — | 0/0/40 | 1.000× |
| `big1` global area desc | 1.0837 | **85.15 %** | **−4.48 %** | 27/2/11 | 0.753× |
| **`nosize` (L268=4)** | **1.0857** | **84.83 %** | **−4.12 %** | **25/0/15** | 0.758× |
| `hoist1` largest one item | 1.1091 | 81.29 % | +0.06 % | 3/5/32 | 1.004× |
| `hoist3` largest three | 1.1085 | 81.38 % | −0.05 % | 9/10/21 | 1.019× |
| `bigfree` (parameter-free) | 1.1090 | 81.31 % | +0.03 % | 0/1/39 | 1.002× |

For scale: L262's eviction moved this ceiling 81.6 → 82.5 %; L252 called the residual
"the cliff, unreachable". **It is reachable — by reordering.**

🔑 **92 % of it is one tie-break.** `nosize` keeps the `bscore` boundary-class key
completely intact and removes *only* "compound cluster items before singles". It gets
−4.12 % against `big1`'s −4.48 %, with **0 cases looser** (big1 has 2) and 24 % fewer packs.

🔑 **The gentle variants are empty.** L260's "displacing exactly ONE placed block opens
a slot, 8/8 cases" does **not** invert: hoisting the largest one or three items moves
`s_min` by ±0.06 %. And `bigfree`'s antecedent is nearly empty — almost no free item is
larger than the largest boundary item (0 tighter / 1 looser / 39 same).

### 2.2 Half the damage hypothesis was right, and the wrong half is the fatal one

| arm | hpwl | area | vrel | portfolio |
|---|---|---|---|---|
| ship | 0.2924 | 0.2300 | 0.0893 | — |
| `big1` (drops bscore) | 0.3656 | 0.2244 | **0.0998** | +4.9845 % |
| `nosize` (keeps bscore) | 0.3332 | **0.1972** | **0.0884** | +0.1626 % |
| `nosize269` (+ adaptive frame) | 0.3669 | **0.1685** | 0.0913 | +0.9752 % |

✅ **The violation damage is entirely the `bscore` key, and it is fully avoidable.**
`nosize` keeps it and vrel is *better* than shipped (0.0884 vs 0.0893). This confirms
M78's `anch_ord4` (+1.069 %) and `WIRE_ORDER`'s vBd 390 from the other side: those
measured the cost of dropping the key; this measures the cost of keeping it, which is zero.

❌ **The wire damage is not avoidable.** `nosize` keeps `bscore` and still pays
hpwl 0.2924 → 0.3332. The greedy's wire term scores a candidate against
**already-placed** neighbours only; compound cluster items *are* the connectivity
anchors, so demoting them makes every early placement wire-blind, and the clusters
then land in leftover space away from their own neighbours.

### 2.3 🔑 The exchange rate — this is the closing argument

`Cost = (1 + 0.5·(hpwl_gap + area_gap))·exp(2·vrel)` prices hpwl and area
**identically**. So a mechanism only pays while `Δhpwl < −Δarea`:

| arm | Δarea | Δhpwl | **wire paid per unit of area bought** | predicted | measured |
|---|---|---|---|---|---|
| `adapt` | −0.0193 | **−0.0150** | **−0.78** (improves both) | −1.419 % | −1.379 % |
| `dense` (L264) | −0.0306 | +0.0031 | **0.10** | −0.773 % | −0.745 % |
| `nosize` | −0.0328 | +0.0408 | **1.24** | +0.137 % | +0.163 % |
| `nosize269` | −0.0615 | +0.0745 | **1.21** | +0.918 % | +0.975 % |
| `both` | −0.0491 | +0.1308 | 2.66 | +7.216 % | +7.326 % |
| `big1` | −0.0056 | +0.0732 | 13.07 | +4.859 % | +4.984 % |

The model reproduces every measured arm to **≤0.15 pp**, so the decomposition is sound.

⇒ **Tightening the frame is a paying trade (ratio 0.10, or negative for `adapt`).
Reordering to make a tighter frame reachable is not (ratio 1.2), and it is not close
to close — it is 20 % the wrong side of a line that is exactly 1.0.**

That is why L255's prize curve was never collectable by this route: the prize was
priced as area alone, and the order that buys the area spends 1.2× of it on wire.

### 2.4 What this closes

* Big-first ordering in every form measured: global by area, global by max dimension,
  hoist-K, the parameter-free threshold, and the `bscore`-preserving tie-break removal.
* The **repair** route with it. Independently established this session from the corpus:
  the LP is topology-preserving by construction (`_pick = _g.argmax` freezes the
  disjunct off the input positions), recovers only **7.5 %** of `hpwl_gap` at depth 12
  against 49 % of `area_gap`, and **nothing in the corpus repairs vrel** (C++ post-passes
  fix 57/161486 = 0.04 %; compaction *adds* 80; L135 moved 423 blocks and removed 0).
  And the arithmetic closes it before the mechanism argument is needed: giving `big1` a
  **perfect** hpwl repair still leaves it **+1.90 %** worse than shipped.
* A per-case gated / twin deployment: the oracle over {base, big1} is **−0.41 %**,
  over {base, both} **−0.18 %** — a perfect per-case gate cannot reach the ship bar.

### 2.5 What it does not close

The density is genuinely there — `nosize269` reaches **area_gap 0.1685**, the best ever
measured on this packer, against the shipped 0.2300. Any mechanism that could reach that
density *without* scrambling wire visibility would be worth ~−1.9 %. Nothing in the
corpus does, and the LP cannot.

---

## 3. The affordable-set deployment (measured, not adopted)

The grader's profile phase is max-bound at n=120, so a mechanism that only slows
profiles which are **not** the max-setter costs nothing on the deciding number.

From `l267_wall.pkl` × `l267_q40.pkl`, no new solver time:

| margin | \|A\| | pool oracle | new max | sum work |
|---|---|---|---|---|
| 1.00 | 26 | −1.58 % | **1.0000×** | 1.0137× |
| **0.60 – 0.95** | **20** | **−0.85 %** | **1.0000×** | 1.0076× |
| (adapt on all 51) | 51 | −2.32 % | 1.2417× | 1.0938× |

**|A| = 20 is a plateau across margins 0.60–0.95** — those 20 profiles run `adapt` in
≤0.80× the max-setter's time, so a 25 % timing error cannot flip membership. That is a
structurally separated set, not a fitted boundary. The greedy |A| = 26 has **12 profiles
within ±5 % of the boundary** and should not be used.

🚨 **But L258 already built exactly this gate** (`l258_gate.py:38-41`,
`safe = [p for p in mn if mn[p] <= max_off]`) and measured **0 % split-half transfer in
both directions**. Its diagnosis may not carry — L258's gain lived in 9 idiosyncratic
cases of 40, whereas `adapt` changes 1908/2040 layouts with 1151 better — but that is an
argument for measuring it, not for assuming it. **And the max-setter's identity is not
stable**: prof 100 / 93 / 7 / 40 / 3 across five captures.

Given `l269p2` reaches NET +0.5…+0.7 pp with **no gating at all**, this route is now a
fallback rather than the plan.

---

## 4. Honest limits

1. **Heavy band only** — 40 cases, n ≥ 101, sample s1. The deployed score is 100 cases
   across three bands.
2. **No s2.** `PROBES` is a fitted integer; the project's own rule asks for s2.
3. **Wall is 3 cases × min-of-2** on a box with ≥20 % run-to-run spread. Only ratios are
   claimed, and they are corroborated by an independent deterministic proxy built from
   40 cases.
4. **`l269p2`'s gain is concentrated**: top-3 cases carry 55 %.
5. **Not deployable from here.** Both candidates are `constructive.cpp` changes. Per L158
   an env-only mechanism is inert in the package, so shipping means a code default —
   which forces a `bin/constructive_linux` rebuild on a Linux box, and the shipping tree
   is frozen at `build_submission.D` (48c Linux `1.2264069637381392`, rank 2).

## 5. Files

```
l267_patch.py        constructive.cpp -> constructive_l267.cpp (8 patches, md5-guarded)
constructive_l270.exe   the gated probe: L252 emitters + L267 + L268(1..6) + L269

l267_cliff.py        s_min + pack bill, both mechanisms, one pass   -> l267_cliff.pkl
l268_screen.py       the cheap ordering screen (s_min only)         -> l268_screen.pkl
l267_quality.py      generalised arms, official scorer, per-profile proxy metrics
                     + optional --gate for affordable-set simulation
l267_wall.py         same-batch min-of-N wall, env-flag arms
l269_wallproxy.py    the CORRECTED deterministic wall proxy (read its docstring)
l267_splithalf.py    transfer check, any arm set

l267_q40.pkl  l269_q40.pkl  l268_q40.pkl     40-case quality captures
l267_wall.pkl l267_wall2.pkl                  per-profile timings
l267_cliff.log l268_screen.log l267_q40.log l269_q40.log l268_q40.log
l267_wall.log l267_wall2.log
```
