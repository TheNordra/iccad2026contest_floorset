# L147 — L122's tangent cut, ported and GREEN: +2.08~2.31% OOS

The shape axis was closed by L122 on a runtime argument. The contest published
the per-case median runtimes of the beta hidden set, that argument is now
measurable instead of modelled, and it was wrong. This is the port, the four
gates, and the handover.

**Scope: Python only.** `optimizer_constructive.py` is the single file changed.
`bin/constructive_linux` md5 `6d43cf2cbfd9e4d578cd692277a7f868` **unchanged
throughout** — that is the proof this lane needs no ELF rebuild.

## 1. Why the old verdict was wrong

L122 measured the mechanism and killed it on **grid worst**: the arm was priced
across a grid of machine speeds `s in {1, 1.5, 2, 2.5}` and judged on the worst
cell, because nobody knew how fast the grader was relative to us.

| arm | s=1 | s=1.5 | s=2 | s=2.5 | grid worst |
|---|---|---|---|---|---|
| shipped LP (k=1) | +2.568% | +2.721% | +2.417% | +2.248% | +2.248% |
| R=1.5 | **+4.620%** | +2.531% | +0.667% | −0.072% | **−0.072%** |

The medians make that grid obsolete: our own beta per-case runtimes were measured
**on the grader**, and `RF = max(0.7,(t/M)^0.3)` is now computable per case.
`l146_rf_price.py` reproduces the graded `total_score` 0.9245185859 against the
official 0.9245183670 — **2.4e-7 relative**, exactly the rounding of a CSV with
three decimals.

⚠️ Two corrections to that tool, both mine, both load-bearing:

* it divided each case's cost by its own RF, but the harness already forces
  RF=1.0 (`iccad2026_evaluate.py:924-940`) — so `_total(1.0)` reproduced
  `raw_score` as an **algebraic identity** and tested nothing. Fixed; the
  validation above is against `total_score`, which is a real check.
* pricing an added-time mechanism by its **average** is the wrong shape. The
  measured per-case `dt` is right-tailed and lands on the big-n cases, which have
  the least slack. New entry point `price_seconds()` applies the vector case by
  case and reports a permutation band.

## 2. The port

`optimizer_constructive.py:2137-2959` is an AST-sliced extraction of
`teammate_m71_screen/l100_lp_speed.py`; the functions correspond 1:1 and are
byte-identical outside flag hunks. So the port ADDS FLAGS to our own builder:

| anchor | change |
|---|---|
| `build_and_solve` signature | `area_R=None, area_g=1.05, area_tol=None, area_price=0.0` |
| after `resh = reshapeable(...)` | `rho = max(rho, area_R - 1.0)` when `area_R` is set |
| the `area_band` loop | forked band ↔ **tangent cuts** of `w*h >= A(1-tol)` |
| after the `a_base` block | the area **price** objective loop |
| the shape bounds | a box on the SHAPE `(w/R - w, w*R - w)`, not a rho trust region |
| `lp_pass` / `solve_pruned` | `**kw` threaded through all 6 call sites |
| `_shape_lp` | reads `ICCAD_SHAPE_LP_{R,G,TOL,PRICE}`, all default OFF |

Plus one diagnostic: `ICCAD_SHAPE_LP_STATS` appends `n kept` per case. The
tangent arm drops the upper area bound and leaves `hard_ok` to adjudicate, and a
**rejected case loses the whole shipped LP gain, not just the increment** — so
kept-rate is a gate and has to be observable.

🔑 `area_g` is the row-count knob, and it was not treated as one before.
`steps = ceil(2*ln(R)/ln(g))` rows per reshapeable unit — 18 at (R=1.5, g=1.05).
Moving to (g=1.10, tol=0.006) cuts that to 10 **and tightens** the worst true
area from 0.99140 to 0.99163. That is the shipped setting here.

## 3. The four gates

**Gate 1 — flag-off bit-equality.** The only admissible bit gate (once rows are
added the matrix is reformulated and the LP is degenerate). Flags off, 48c:
total `1.2284738198320346`, **0/100 costs and 0/100 positions differing** from
`results_L136_48c_anchor.json`. Re-run after the kept-rate counter landed
(`ctrl2`): identical again, so "edited while measuring" is closed.

**Gate 1b — the R→1 freeze invariant**, against the true pre-LP layout
(`ICCAD_SHAPE_LP=0`), not against the shipped band:

    LP off (pre-LP)        1.257055419130
    R->1 frozen            1.251376791662   +0.4517%   shapes max rel 1.021e-07
    shipped band (ctrl)    1.228473819832   +2.2737%   shapes max rel 6.000e-02
    R=1.5 g=1.10           1.197768284824   +4.7164%

1.021e-07 is exactly the R=1.0000001 bound and 6.000e-02 is exactly rho — both
branches do what they say. 🔑 And the decomposition is worth keeping: **the
shipped LP is worth +2.27% on this tree, only 0.45pp of it from translation —
~80% is shape freedom, and the tangent cut doubles the whole LP's value.**

⚠️ My first version of this test compared against the shipped band and read
"max rel 6.4e-2 → FAIL". That was the control's own reshaping, not a bug. An
invariant needs the right reference.

**Gate 2 — in-set official eval @48c.**

| arm | total | vs L136 | feasible | LP kept | worse than pre-LP anchor |
|---|---|---|---|---|---|
| ctrl | 1.228473819832 | — | 100/100 | 100/100 | 2 |
| R=1.2 | 1.206793557252 | +1.7648% | 100/100 | 97/100 | 0 |
| R=1.3 | 1.202496563446 | +2.1146% | 100/100 | 99/100 | 0 |
| **R=1.5 g=1.10 tol=0.006** | **1.197768284824** | **+2.4995%** | 100/100 | 98/100 | 0 |

Offline predicted +1.728% / +2.294% for R=1.2 / R=1.3; measured +1.765% /
+2.115% ⇒ **transfer 92-102%**, so the stale 0.926 deflator is retired.
Side finding: `ctrl` — i.e. the package already uploaded — is itself worse than
the pre-LP anchor on 2 cases, an independent reproduction of the teammate's
judge48 result that the old invariant is one no shipped package satisfies.

**Gate 3 — the deployed price**, min-of-3, arms interleaved, exclusive box:

    per-case added time   min -0.310  p50 +0.047  p90 +0.293  max +1.092  sum +9.67s
    whole-solve wall      1.0310x
    control's own spread  p50 2.8%  max 8.9%      <- why min-of-3
    RF cost               -0.9726%  (permuted p50 -0.2335% / p05 -0.5785%)
    NET (in-set quality)  +1.5269%   bar +0.80%  -> PASS

🚨 The identity join is 4x worse than a random re-assignment and below its p05:
the cost is concentrated **exactly on the cases that carry the weight**, because
those are the low-slack ones. Band-gating cannot fix it (n>100 only moves the RF
cost from −0.97% to −0.79%).

**Gate 4 — OOS 240 x 2 @48c**, harness `l140_oos_soft_audit.py` (it restores
every `ICCAD_*` around `m77_oos_probe`'s import-time strip; `l137_oos_ab.py`
captures only the HINT knobs and would have produced a silently empty A/B):

| | s1 | s2 (disjoint) |
|---|---|---|
| cost | 1.467038 → 1.436588 **+2.0756%** | 1.471125 → 1.437085 **+2.3138%** |
| area_gap | 0.194170 → **0.155269** | 0.199268 → **0.157418** |
| hpwl_gap | 0.272043 → 0.262011 | 0.276489 → 0.264949 |
| vrel | 0.085869 → 0.085313 | 0.085773 → 0.084938 |
| better / worse | 219 / 19 | 229 / 11 |
| feasible | 240/240 | 240/240 |
| n>110 subset | +2.0892% | +2.2382% |

Both samples positive, 83% and 93% of the in-set delta (bar was 60%), 480/480
feasible. The gain lands on **area_gap, −20%** — which is where L128's deficit
decomposition said it was.

**NET on OOS quality: +1.10% (s1) / +1.34% (s2)** against a 0.30% ship bar.

## 3.5 What it costs in wall clock, and whether we keep the 0.70 floor

**On the grader** (beta per-case runtimes + the measured per-case dt):

| | cost-weighted RF | cases off the floor | total runtime |
|---|---|---|---|
| today (L136) | 0.700042 | 1/100 | 52.07 s |
| **with L147** | **0.706849** | **9/100** | **61.74 s** (+18.6%) |

So we do NOT keep the floor everywhere: nine cases leave it (n = 71, 79, 83, 92,
94, 99, 106, 112 and one more, all with t/M already 0.30-0.41). That is exactly
the −0.97% RF term Gate 3 priced, and the +2.08~2.31% quality still dominates it
by better than 2:1.

🚨 **The transfer model matters and the obvious one is wrong.** Priced by the
per-case *ratio* (1.0310x) instead of the added *seconds*, the same change reads
"RF 0.700000, 0/100 off the floor, no cost at all". That is wrong, and the reason
is visible in an `ICCAD_SHAPE_LP=0` run:

    local, same box, 48c pool shape
      LP off          299.49 s
      shipped LP      317.01 s     the LP itself costs +17.52 s
      + tangent cut   326.68 s     the extra rows cost  +9.67 s

    grader beta       52.07 s total, WITH the shipped LP already inside it
      => ~35 s of C++ pool (48 real cores) + ~17 s of LP

The pool is C++ across 51 parallel profiles and runs ~8.5x slower on this
32-core box than on the grader's 48; **the LP is single-threaded Python+scipy and
runs at the same speed on both.** The added work is entirely LP, so it transfers
in SECONDS, not as a ratio. Pricing it by ratio would divide the real cost by
about six.

**Local wall by generation** (48c pool shape, one run each unless noted; the
control's own run-to-run spread is 2.8% p50 / 8.9% max, so only the L147 row is
min-of-3 and therefore exact):

| package | local total | per case |
|---|---|---|
| L114 (the beta-era submission) | 261.66 s | 2.617 s |
| M80 | 132.06 s | 1.321 s |
| L136 (currently on Drive) | 296.71 s | 2.967 s |
| L137 (teammate's, not uploaded) | 334.35 s | 3.343 s |
| **L147 (this change)** | **323.09 s** | **3.231 s** |

Against the beta-era package we are +23% slower, but **only +3.1 percentage
points of that is L147** (317.01 → 326.68 on the same box, min-of-3); the rest
arrived with L136 and L137.

## 4. The one real risk, quantified

The medians are the **beta** field's. If the final field gets faster they drop,
and our slack drops with them (quality is unaffected; only the RF term moves):

| final medians vs beta | RF cost | NET | cases still floored |
|---|---|---|---|
| 1.00x | −0.97% | **+1.53%** | 99/100 |
| 0.90x | −1.57% | +0.93% | 99/100 |
| 0.85x | −1.91% | +0.59% | 94/100 |
| 0.80x | −2.27% | +0.23% | 90/100 |
| 0.70x | −3.02% | **−0.52%** | 82/100 |

Break-even is about **0.78x**: the field would have to get ~22% faster for this
to turn negative. The observed direction is the opposite — the leaders are the
SLOW submissions (rank 1 runs 169 s at RF 0.824), because they are buying quality
with runtime, which is the same trade this change makes.

R=1.3 was measured as the hedge and is **dominated at every median scenario**:

| medians vs beta | R=1.3 RF / NET | R=1.5 RF / NET |
|---|---|---|
| 1.00x | −0.691% / +1.424% | −0.973% / **+1.527%** |
| 0.90x | −1.213% / +0.901% | −1.573% / **+0.927%** |
| 0.80x | −2.029% / +0.086% | −2.270% / **+0.230%** |
| 0.75x | −2.472% / −0.357% | −2.620% / **−0.121%** |

🔑 The reason is `area_g`, and it is the most exportable thing in this report:
R=1.5 at g=1.10 emits **10 rows per reshapeable unit**, R=1.3 at the default
g=1.05 emits **12** (and R=1.5 at g=1.05 would emit 18). The stronger arm is also the cheaper one (added time +9.67s
vs +9.90s). **The row-count knob, not the aspect range, is what a shape
mechanism costs.** No hedge is needed; R=1.5 / g=1.10 / tol=0.006 is the pick.

## Combined with L137 (measured, not assumed)

All the numbers above were taken on the **L136 base**, because that is what our
tree carried while the work was done. Re-measured on the teammate's head with
L137's defaults ON (48c in-set, official eval):

| | total | vs L136 | feasible |
|---|---|---|---|
| L136 (uploaded) | 1.228473819832 | — | 100/100 |
| L137 (teammate) | 1.227176561424 | +0.1056% | 100/100 |
| L147 on L136 | 1.197768284824 | +2.4995% | 100/100 |
| **L137 + L147** | **1.196679286011** | **+2.5881%** | 100/100 |

Additive prediction +2.6051%, measured +2.5881% ⇒ the overlap is **0.017pp, i.e.
99.3% additive**, which is what the mechanisms predict: the hint moves anchors
during packing, the tangent cut moves shapes in the post-pack LP.

⚠️ The wall figures in that run (312.3 s combined vs 323.1 s for L147 alone) are
single runs and the control's own spread is 2.8% p50 / 8.9% max — do not read a
speed-up into them. Only the min-of-3 numbers in §3 are timing evidence.

## 5. Reproduce

```bash
cd /c/ICCAD_ml/ship_final/iccad2026contest && ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py -o ../results_L147_r15g.json
```
```bash
cd /c/ICCAD_ml/ship_final && ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l140_oos_soft_audit.py run --sample s1 --cores 48 --out l147_oos_s1_r15g.json
```
```bash
cd /c/ICCAD_ml/ship_final && bash l147_gate3.sh && "C:/Users/.01/anaconda3/envs/floorset/python.exe" l147_price.py --quality 2.4995
```

Artefacts: `results_L147_*.json` (arms + 3 timing reps each), `l147_*_stats.txt`
(kept-rate), `l147_oos_s{1,2}_r15g.json`, `l147_gate3_price.txt`, and the
harnesses `l147_arms.sh` / `l147_gate3.sh` / `l147_gate4.sh` / `l147_price.py`.
