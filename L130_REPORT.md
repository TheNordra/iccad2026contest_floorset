# L130 — the GORDIAN alternation, and a floating-point coin flip

Continues L129. The submission was **not touched**: everything here is offline,
behind env knobs that default OFF, and `l129_global_placer.py` has never been in
`build_submission/`.

**Verdict: the alternation is a large SOLO win and RED at the portfolio gate.**
Weighted solo cost −5.1% on the common set with every deficit term improving,
and the gate moves the wrong way (+0.011% → +0.003%, bar 0.05%).

---

## 0. Gate 0 — priced before it was built

L125's precondition is now a standing rule, so `l130_gordian_price.py` ran first.
The alternation only touches stage A; stages C and D run once on whatever centres
stage A hands over, so the added price is bounded by (levels) × (one constrained
solve).

| stage | weighted share of a case |
|---|---|
| `lp_polish` | **95.80%** |
| `legalise` | 2.54% |
| `refine_area` | 1.14% |
| `global_place` | 0.29% |
| `spread` | 0.17% |
| `build_units` | 0.07% |

Stage A is **0.46%** of the runtime. Replacing its one dense solve with ~6 levels
of KKT solves measured **0.256 ms/case = 0.061%**. GREEN, and the gate cost two
minutes.

🔑 **The price gate also says where the next runtime work is, and it is not here.**
96% of this placer's time is the shipped LP being called six times per case.

## 1. What was built

`gordian()` in `l129_global_placer.py`, behind `L129_GORDIAN=1` (default OFF so
the v6 anchor stays reproducible from the same file). Stage A's Laplacian
assembly was factored out (`assemble` / `fixed_centres` / `unit_map`) and is
shared; the refactor was verified a no-op by reproducing `1.708551` /
hpwl_gap 0.4265 / area_gap 0.4297 to every digit.

Each level adds one area-weighted centre-of-gravity equality per region and
re-solves

    min x'Lx - 2b'x  s.t.  Ax = u   =>   [[L, A'], [A, 0]] [x; lam] = [b; u]

as one dense (F+R) system, factored once for both axes. Regions are then split
again **on the coordinates that solve just produced**. That is the whole
difference from v6, which solves once and does all its spreading in a single
destructive bisection.

Constraining only the FREE units is deliberate: a region whose area is mostly
preplaced would otherwise demand its few movable units average out to a point far
outside the box, and the next partition would read that excursion as an ordering.

## 2. Three things that had to be fixed, and what each cost

### 2.1 The alternation does not know what a boundary block is (`_bnd_rank`)

First working version, 30 cases: hpwl_gap 0.4265 → **0.4172** and area_gap
0.4297 → **0.3851**, both better — and vbnd 2.0775 → **2.8490**, total
1.7086 → **1.7767**, worse.

Nothing in the QP objective or its constraints encodes "a needL unit must have no
x-predecessor". The free solve is *right* to pull such a unit inward, and
`legalise` then reads the centre order and finds it cannot honour the
requirement.

Fix: the partition sorts boundary-requiring units to the matching end of every
split, so needL lands in the leftmost leaf column and is extremal in the centre
order. This is handoff 08-15 §3.1 for the third time — **a stage that improves
one term of a multiplicative cost must re-enforce every constraint the stages
around it rely on.**

### 2.2 A centre-of-gravity row is an average (`_snap_leaves`)

A unit may sit far outside its region as long as its region-mates compensate, and
`legalise` reads those coordinates twice: once to pick each pair's cheaper axis
from the overlap of their real (w,h), once to orient it. The first read is
meaningless when centres are excursions.

Snapping every free unit to its leaf region centre — keeping only the ORDER the
alternation produced — is worth **−0.017 weighted cost and +1 case of coverage**
on the n≥80 bucket. Small, kept, and not a retreat to `spread`: the leaf a unit
lands in was decided by log2(U) solves that each knew about wirelength.

### 2.3 🚨 `grouping_violations` was partly a floating-point coin flip

The emit did `lx[k] + off_i`. The evaluator builds a block's far edge as `x + w`.
Floating-point addition is not associative, so `(lx + ox) + w` and `lx + (ox + w)`
differ by an ULP: a packing that abuts exactly in exact arithmetic lands
**±2.84e-14** off in doubles.

Measured on case 66, same packing, relative offsets identical to 1e-6:

| arm | abutment error | shapely `unary_union` | vgrp |
|---|---|---|---|
| v6 | **+2.84e-14** (a GAP) | two geometries | **1** |
| GORDIAN | **−2.84e-14** (a hair of OVERLAP) | merged | **0** |

`_emit_abut` (`L129_EXACT_ABUT=1`) replays the shelf accumulation in absolute
coordinates, so the next member's low edge is computed by the same `+ w` the
evaluator uses for the previous member's high edge and the polygons touch
bit-exactly.

Isolated, identical 63-case sets, abut OFF → ON:

| | v6 arm | GORDIAN arm |
|---|---|---|
| cost | −0.032 | **−0.112** |
| vgrp | −0.479 | **−0.823** |
| area_gap | +0.007 | **−0.088** |
| hpwl_gap | −0.012 | −0.023 |
| runtime | +1.11 s | +2.18 s |

The vgrp half is fully explained by the coin flip. The area half is a downstream
consequence through `lp_polish` that is **not** isolated to a mechanism here —
what is ruled out is `hard_ok` rejection (0 rejections in all four arms), and
what is observed is the LP returning `None` on 4 of 24 passes in the unabutted v6
arm versus 0 of 24 abutted. The runtime rises because the LP now completes passes
it used to fail.

🔑 **This confounded the whole comparison.** Before the fix the alternation looked
like it traded hpwl for area (area_gap +0.063 against v6). After it, area_gap is
**−0.047**. Every vgrp delta measured between two stage-A forms before this fix
was that noise plus whatever was real.

## 3. Where the alternation actually stands

Full 100, common feasible set (58 cases), both arms with exact abutment:

| metric | v6 | +GORDIAN | delta |
|---|---|---|---|
| **cost** | 1.713803 | **1.625743** | **−0.088 (−5.1%)** |
| hpwl_gap | 0.467743 | **0.403280** | −0.064 |
| area_gap | 0.443687 | **0.396636** | −0.047 |
| violations_relative | 0.078546 | **0.074183** | −0.004 |
| vgrp | 0.419971 | 0.429299 | +0.009 |
| runtime | 13.54 s | 13.94 s | +0.40 s |

**Better on all three deficit terms, winning 40/58 cases.** hpwl_gap −13.8% is
the thing L128 said was needed and could not get: its two cheap routes (IRLS
reweighting, HPWL in the flip search) were both null *because they operate
downstream of the ordering*, and this is the first mechanism that changes the
ordering itself.

Headline solo numbers, full 100:

| variant | weighted solo cost | coverage | gate |
|---|---|---|---|
| v6 (anchor) | 1.744529 | 64/100 | +0.010% |
| v6 + abut | 1.712144 | 64/100 | **+0.011%** |
| GORDIAN + snap | 1.738042 | 63/100 | +0.003% |
| **GORDIAN + snap + abut** | **1.625964** | 63/100 | +0.003% |

## 4. 🚨 Why a −5.1% solo win moves the gate the WRONG way

| | gate | beats portfolio on |
|---|---|---|
| v6 + abut | **+0.011%** | 67 (n=88), 28 (n=49), 10 (n=31) |
| GORDIAN + abut | **+0.003%** | 58 (n=79), 23 (n=44), 27 (n=48) |

Selection efficiency is **100.0%** in both — the proxy lands exactly on the
per-case oracle, so the shortfall is candidate quality, not arbitration (M76/M77
holding up again on a genuinely heterogeneous candidate).

The candidate is at 1.63 against a portfolio at **1.2935**, so it contributes
only where it happens to beat the portfolio outright — 3 cases either way. And
the winning sets are **disjoint**, so the difference between the two gate numbers
is not a mechanism:

🔑 **v6's entire gate is one case.** Case 67 carries `w*d` 0.000112 of a total
delta of 0.000137 — **82%**. GORDIAN loses case 67 (1.2953 → 1.4621) and drops
case 28 from coverage, and that alone is the whole +0.011% → +0.003%. This is
L127's tally-shape lesson in a new place: **when the gate is one case, the gate
is measuring luck.** Both arms are RED against the 0.05% bar and neither is
within 4× of it.

Deploying both as two separate candidates is worth roughly the union, ~+0.013%.
Still RED.

## 5. Size structure (measured before the abutment fix, and it is why --minn exists)

| n range | cases | v6 | +GORDIAN | delta | share of weight |
|---|---|---|---|---|---|
| 0–40 | 18 | 1.8269 | 1.6516 | −0.1753 | 0.2% |
| 40–60 | 13 | 1.6813 | 1.5574 | −0.1239 | 0.8% |
| 60–80 | 10 | 1.6834 | 1.4495 | −0.2339 | 2.7% |
| 80+ | 17 | 1.7481 | 1.7772 | +0.0291 | **96.3%** |

🔑 **The weight is `exp(n/12)`, so the n≥80 cases are 96.3% of the score and the
first 30 cases are 3.7%.** A 30-case sample said −9.6% while the graded mix said
+0.003%. `--minn` / `--maxn` were added to `l129_global_placer.py` for this
reason; iterating on `--limit 30` measures the part nobody is graded on.

## 6. Reproduce

```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l130_gordian_price.py --limit 30
```
```bash
L129_GORDIAN=1 L129_EXACT_ABUT=1 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l129_global_placer.py run --out results_L130.json
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u m77_ml_candidate_probe.py score results_L130.json --cores 48 --dt 0
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l130_cmp.py results_L129g.json results_L130.json
```

Judge feasibility with `SolutionMetrics.is_feasible`, never with cost. Compare
only on the COMMON feasible set — coverage is a free variable for this placer and
each file's own headline is a different case mix.

## 7. What this closes and what it opens

**Closed:** "hpwl needs a genuinely better ordering" is no longer open — the
alternation produces one, worth −13.8% hpwl_gap, and it is cheap (0.06% of
runtime). The idea that the alternation trades hpwl for area is **closed as an
artefact**; with exact abutment it wins both.

**Open, and in priority order:**

1. **`L129_EXACT_ABUT` should be measured on the SHIPPED path, not just here.**
   It is a pure floating-point correctness fix worth −0.032 solo cost and −0.479
   vgrp on the v6 arm, and nothing about it is specific to this placer — any
   emitter that writes `origin + offset` for abutting blocks has the same bug.
   That is the highest-value item on this list and it is not a GORDIAN question.
2. **Coverage is fragile and it is where the gate lives.** The two arms drop five
   cases each, disjointly (v6 loses 37/48/58/77/85, GORDIAN loses 28/33/52/57/63).
   Losing one heavy case costs more gate value than a 5% solo improvement earns.
   `legalise` non-convergence is the single cause and has never been investigated.
3. **Case 67 specifically.** It is 82% of the gate. Why v6 wins it and the
   alternation does not is one case's worth of work and would answer whether the
   alternation's loss there is structural or incidental.
4. **The area half of the alternation.** The CoG row controls only a first
   moment; leaf region aspect is set by cut order and has no relation to the
   unit's own aspect, which lengthens compaction chains. Aligning split direction
   with unit aspect rather than region aspect is the obvious next lever.

**Not worth more investment on current evidence:** pushing the alternation's
solo quality further. At 1.626 against a portfolio at 1.2935 the gate cannot see
solo improvements at all — it only sees the ~3 cases where the candidate wins
outright. Solo quality stops mattering until it is close to 1.29.
