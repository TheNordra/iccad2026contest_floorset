# L295–L300 — what beating rank 1 costs, and the first configuration that lands on it

The brief was "how do we pass rank 1". It has an exact answer, and answering it
exactly changed which mechanism is worth building.

    rank 1   total 0.858632   raw 1.084488   cwRF 0.7917   110.9 s
    us       projected  0.871 – 0.879                       45.2 s

**We need 1.4 – 2.3 % of raw quality and we cannot buy any of it with runtime.**

**`ICCAD_LP_GATE=0` (+ `ICCAD_SHAPE_LP_ITERS=2`) closes 70–80 % of that gap** and
is the first configuration in this project's history that lands in the same room
as rank 1 instead of 2–4 % away. Whether it *crosses* depends on which of the
project's two RF pricing methods you use:

      projected graded total          baseline A   baseline B   vs rank 1
      shipped                          0.878564     0.870684    -1.38 .. -2.27 %
      gate0 + k=2, f-free ratios       0.862607     0.862203    -0.42 .. -0.46 %
      gate0 + k=2, imported f = 3.17   0.858444     0.858042    **+0.02 .. +0.07 %**
                                                    rank 1  =   0.858632

The f-free (same-box ratio) method is the sounder of the two and it says **short
by 0.45 %**. Report it that way. §6.

> ⚠️ **Label collision.** The parallel session also used "L295" (for the
> `ICCAD_LP_GATE=0` OOS validation, files `l295_gate0_oos.sh`, `l295_s[12].log`).
> Nothing was overwritten — this file's `l295_rank1.py` / `l295_geom.json` are
> distinct — but the number is shared between two pieces of work.

> 🚨 **SUPERSEDED IN PART by `L302_L313_F_PINNED.md` (same session).** `f` has
> since been measured rather than imported: **2.38–2.84 overall, 1.62–2.13 on the
> heavy band**, against the 3.17 used below. Re-priced at the measured f, the
> §4 candidate is **1.2–2.8 % short of rank 1, not past it**, and `gate0 + k=2`
> loses to `gate0` alone. §1's target arithmetic, §2's violation inventory, §3's
> corpus projection and §5/§7's RED verdicts are unaffected — none of them use f.

---

## 1. The target, stated exactly (`l295_rank1.py`)

`total = Σ w_i q_i RF_i / Σ w_i`, `w_i = exp(n_i/12)`, `RF_i = max(0.7,(t_i/M_i)^0.3)`.
**Gate:** recomputing our beta row from `beta_evaluation_results.json` + the
2026-08-23 medians reproduces the leaderboard to 5e-7 (`0.9265867` vs
`0.9265861`) and the wall to 4 decimals (52.0712 s).

    quality still needed to BEAT each rank, at runtime scale s
    (s = 1 is the beta package's 52.07 s; the shipped package is s = 0.868 = 45.2 s)

      s      wall    cwRF   floor    vs rank1  vs rank2  vs rank3
     0.50   26.0s  0.70000  100/100   -2.264%   +1.100%   +2.369%
     0.87   45.2s  0.70004   98/100   -2.269%   +1.095%   +2.363%   <- shipped
     1.00   52.1s  0.70161   82/100   -2.487%   +0.869%   +2.134%
     1.20   62.5s  0.70872   62/100   -3.467%   -0.144%   +1.108%
     1.40   72.9s  0.72341   38/100   -5.426%   -2.171%   -0.944%
     2.00  104.1s  0.79287   11/100  -13.712%  -10.742%   -9.622%

🔑 **Runtime is worth nothing in either direction from here.** Running twice as
fast (26.0 s) improves the target by **0.005 pp** — we are already on the floor.
Running slower is charged immediately: +17 s costs 1.20 pp, and the marginal
price rises from **0.03 %/grader-second at 45 s to 0.19 %/s at 63 s**.

⚠️ **This retires the "≈19 s of free budget" framing from L285.** That number is
the distance to *losing rank 2*. For a **rank-1** attack the budget is the
per-case distance to the RF floor:

    total free grader seconds before ANY case leaves the floor   20.66 s
      of which in the 20 cases carrying 81 % of the weight        8.33 s
      per heavy case                mean 0.42 s, min 0.054 s (case 112)
    at f = 3.17 that is ~1.3 LOCAL seconds per heavy case

That per-case slack table is why two arms with similar added time can have RF
bills differing by 13× (§5.1).

## 2. 🚨 The graded corpus's violations, recovered exactly

`violations_relative` is `V / N_soft` with both integers, so
`Fraction(v).limit_denominator()` inverts it. **88 of 88 non-zero values resolve
exactly.** That gives the violation mass, the count and `N_soft` for all 100
hidden cases — data that was sitting in a file nobody had inverted.

| | in-set 100 | **beta hidden (graded)** | OOS s1 240 |
|---|---|---|---|
| weighted vrel | 0.01407 | **0.04252** | 0.08620 |
| cases carrying one | 54/100 | **88/100** | 240/240 |
| soft violations (reduced count) | 66 | **152** | — |
| heavy band n≥101 | **11** over 20 cases | **36** over 20 cases | — |
| worth if driven to 0 | −2.81 % | **−8.29 %** | −16.0 % |

`N_soft` is the same order at matched `n`, so the hidden set does **not** carry
more soft constraints — **we violate ~2.3× more of them**. In-set `N_soft`
decomposes as **boundary 54 % / grouping 37 % / MIB 9 %**, and all 100 in-set MIB
groups are collapsible, which is why in-set MIB is 0 and why L278 was right that
the in-set cannot vote on the twins.

### 2.1 The prize is brutally concentrated

      rank  case   n  N_soft   this one   cumulative
         1    95  116     18     0.629 %      0.629 %
         2    90  111     14     0.545 %      1.175 %
         3    98  119     28     0.543 %      1.718 %
         4    96  117     33     0.359 %      2.076 %
         5    97  118     53     0.261 %      2.337 %
       152     1   22     23     0.000 %      8.287 %

      band  21- 50 :  31 violations worth 0.028 %
      band  51- 80 :  40 violations worth 0.243 %
      band  81-100 :  45 violations worth 1.548 %
      band 101-120 :  36 violations worth 6.468 %

🔑 **Five of 152 violations are the entire rank-1 gap.** Two multipliers stack:
`exp(n/12)`, and `vrel = V/N_soft` — one violation on a case with `N_soft = 14`
costs `exp(2/14) − 1 = 15.3 %` of that case against 3.1 % at `N_soft = 65`.
**The most expensive violations sit on heavy cases that have the fewest soft
constraints to satisfy in the first place.**

⚠️ Honest limit on the *count*: a reduced fraction `1/14` can be `1/14` or
`2/28`. The **mass** figures (−8.29 %, the band split, "25 % of violations") are
exact because they use only `vrel`; the **count** figures are lower bounds and
the per-violation prizes are correspondingly upper bounds.

## 3. 🚨 The in-set understates every violation-trading mechanism by ~3×

`l296_project.py` summarises an arm by a geometry factor `g` and a violation
factor `phi`, then applies them to the graded corpus's own per-case
`(hgap, agap, V/N_soft)`. `l299_project2.py` does it per component, because the
corpora also differ in mix (in-set hgap:agap = 65:35, graded 51:49).

    exchange rate ON THE GRADED CORPUS
      1 % of geometry        -0.175 %
      10 % of violations     -0.870 %      <- 5x more per relative point
      25 % of violations     -2.157 %      <- the whole rank-1 gap
      50 % of violations     -4.255 %

Applied to arms that are already measured, **three published verdicts move**:

| arm | in-set | graded | why |
|---|---|---|---|
| L137 GORDIAN hint OFF | −0.004 % ("harmless") | **+0.128 % (worse)** | the hint buys violations (`phi` 1.023) — it is *not* the component failing to pay for itself |
| M80 tier OFF | +0.439 % | **+0.967 %** | the tier buys violations (`phi` 1.086) |
| L296 A1 (§7) | **+0.480 % (RED)** | **−0.140 % (green)** | trades geometry for violations at 0.88 |
| L277 post-hoc snap | +0.0012 % | −0.0014 % | flips, but it only removes **0.04 %** of the violation mass — noise either way |

⚠️ CLAUDE.md's "下一步 4" flagged the corpus asymmetry in words in August. What
is new is that it is a **number applied to specific arms**, computed from the
graded corpus's own per-case data rather than from a proxy corpus.

⚠️ And the mechanism gap is still real: **no mechanism this project has ever
built removes more than ~1 % of the violation mass.** The shape LP does not do it
either — it changes the count from 48 to 47 on the cases it runs on. Violations
are 8.29 % sitting on the table with no tool that reaches them; §4's candidate is
a **geometry** mechanism (area gap ×0.61).

## 4. 🏆 The candidate: `ICCAD_LP_GATE=0` + `ICCAD_SHAPE_LP_ITERS=2`

`_L196_LPGATE` switches the shape LP off for 29 block counts carrying 44.2 % of
the graded weight. Ungating it was the parallel session's open item (L294);
stacking depth on top of the ungated LP had not been tried.

    in-set 100, official evaluator, ICCAD_ADAPTIVE_CORES=48, all 100/100 feasible

      arm                          in-set total    vs ship     local wall
      shipped (gate on, k=1)       1.226325126      0.0000%    130.0 / 132.5 s
      gate0                        1.199000373     -2.2282%    147.8 / 143.8 s
      gate0 + k=2                  1.189471885     -3.0052%    167.2 s
      gate0 + k=4                  1.187314246     -3.1811%    211.7 s

Priced onto the graded corpus (`l297_rank1_price.py`, `l299_project2.py`), with
the **imported** `f = 3.17`:

      arm            g_h      g_a      phi  |    RF   | baseline A | baseline B | grader s
      gate0        0.9412   0.7134   0.9914 | +0.97 % |  0.860219  |  0.857882  |  50.8 s
      gate0+k=2    0.9208   0.6144   0.9882 | +1.87 % |  0.858444  |  0.858042  |  56.1 s
      gate0+k=4    0.9133   0.5969   0.9882 | +6.79 % |  0.897876  |  0.897824  |  70.3 s
                                                        rank 1 = 0.858632

and with the **f-free same-box ratio** method (`l301_ffree.py`, the method
`l294_final.py` §(b) established; dt averaged over all four ship×arm pairings):

      arm          dt local   -> grader   implied f |    RF   | baseline A | baseline B
      gate0        +15.78 s     +6.66 s      2.37   | +1.32 % |  0.863151  |  0.860805
      gate0+k=2    +28.69 s    +11.77 s      2.44   | +2.36 % |  0.862607  |  0.862203
      LP k=2 alone  +8.23 s     +4.23 s      1.95   | +0.03 % |  0.875296  |  0.868145

* **k=4 blows past the RF floor on the heavy cases and lands at rank 4** under
  either method — that one is decided.
* **gate0 vs gate0+k=2 is a wash.** The depth adds 0.78 pp of in-set quality and
  the RF bill takes it back. Under the f-free method with baseline B, gate0
  *alone* is the better arm (0.860805 vs 0.862203) — and it is the one that is
  already OOS-validated (s1 +2.4648 % / s2 +2.2373 %, transfer 100–111 %,
  147 movers all better, 0 worse — the parallel session's L295).
* The gain is **area**: `g_a = 0.6144` is a **38.6 % cut in the area gap**, by far
  the largest single-component move any mechanism in this project has produced.
* Quality saturates: k=4 buys 0.18 pp more quality for 3.6× the RF bill.

The two baselines are:

* **A — flat.** Apply −4.97 % (L287's 93 % transfer of the in-set −5.34 % since
  real M73) to the graded raw → shipped 0.878564.
* **B — per component.** Apply the measured ship-vs-M73-like factors (hgap
  ×0.9587, agap ×0.7335, vrel ×0.9378) to the graded corpus's own per-case values,
  times the 0.97248 that no kill switch can revert (real M73 in-set 1.295548 →
  M73-like 1.259898) → shipped 0.870684.

### 4.1 What is verified and what is not

| | |
|---|---|
| in-set quality | measured, official evaluator, 100/100 feasible |
| determinism | gate0 bit-identical over 2 repeats; gate0+k=2 repeat in §8 |
| deployment class | `ICCAD_LP_GATE` and `ICCAD_SHAPE_LP_ITERS` are **wrapper-only**; `ICCAD_SHAPE_LP_ITERS` appears 0 times in `constructive.cpp` — **no ELF rebuild** |
| runtime headroom | 56.1 grader s vs the 64.1 s rank-2 threshold |
| **OOS s1 / s2** | **NOT RUN** — `l287_transfer.py --arms ship,g0k2` is the next measurement |
| Linux lane | not run |
| HiGHS `time_limit` | not audited for the ungated path (CLAUDE.md standing rule) |

## 5. Why selective ungating does not help (`l298_selective_ungate.py`)

The gate is per block count, so the L157 move — ungate only where the added time
fits under the RF floor — was the obvious refinement. **It loses** (baseline A):

      ALL 29 (= ICCAD_LP_GATE=0)   quality -2.51 %  RF +0.97 %  -> 0.864877
      RF-SAFE (dt fits slack)      quality -0.97 %  RF +0.00 %  -> 0.870064
      GREEDY top-16 by dq/dt       quality -2.42 %  RF +0.94 %  -> 0.865334

The quality and the bill sit on the **same** cases: 112 (slack 0.054 s, spends
0.485), 117 (0.239 / 0.280), 118 (0.271 / 0.403), 120 (0.611 / 0.654). There is
no subset that keeps the gain and drops the bill, and the greedy variants are the
project's five-times-burned case-idiosyncratic shape anyway.

## 6. 🚨 The whole claim rests on `f`, and here is the grid (`l300_sensitivity.py`)

    projected graded total; ** = BOTH baselines beat rank 1 (0.858632)

      gate0 + LP k=2            f = 2.20    2.71     3.17      4.00
        transfer 100 %              0.8733  0.8639  0.85804**  0.85257**
        transfer  93 %              0.8753  0.8660  0.86006    0.85458**
        transfer  85 %              0.8777  0.8683  0.86237    0.85687**
        transfer  75 %              0.8806  0.8712  0.86526    0.85974

      gate0 alone
        transfer 100 %              0.8683  0.8615  0.85788*   0.85457**
        transfer  93 %              0.8698  0.8630  0.85937    0.85605*

**`f = 3.17` is the right value for these numbers, and it is measured, not
assumed.** L157 §5h derived it from a same-package whole-case ratio: the beta
package reconstructed from `7f38893` runs 141.07 s in WSL against 52.07 s on the
grader (f = 2.71, per-case p25 2.33 / p50 2.71 / p75 3.20, flat across bands),
and the Windows LP runs 1.17× slower than WSL, so **Windows-second → grader-second
= 1.17 × 2.71 = 3.17**. Our `dt` is Windows-measured, so 3.17 is the matching
constant; 2.71 is the WSL one and using it here would be a unit error.

⚠️ What the grid says plainly: **rank 1 needs f = 3.17 *and* near-full transfer.**
At L287's measured 93 % package transfer the candidate lands 0.16 % short of rank
1 and comfortably rank 2. Treat "beats rank 1" as a **coin flip that is finally
being flipped**, not as a result.

⚠️ Second-order caveat on `f`: it was measured on a package with **no LP**
(`_shape_lp` count 0 in `7f38893`), so it is the ratio for the *packing* phase
applied to *LP* seconds. The 1.17 Windows/WSL correction is LP-specific, but the
2.71 is not.

### 6.1 The f-free method disagrees, and it is the sounder one

`l294_final.py` §(b) — the parallel session's — removes `f` entirely: express the
LP's added time as a **fraction of that case's local wall on this box**, then
apply the fraction to the **grader's own** measured per-case time. The machine
factor divides out and the only external input is our own grader runtime vector.
`l301_ffree.py` applies it to the combined arm:

      implied f: gate0 2.37,  gate0+k=2 2.44,  LP k=2 alone 1.95

That is **23 % below the imported 3.17**, and it is enough to move the verdict:

      gate0 + k=2      imported f=3.17   0.858444 / 0.858042   beats rank 1
                       f-free ratios     0.862607 / 0.862203   short by 0.45 %

🔑 **Which is right is not decidable from here, and the difference is the whole
claim.** The ratio method's implied `f` is an LP-weighted mean of the per-case
whole-case ratio, which is more faithful per case; the imported 3.17 additionally
carries the measured fact that the **LP specifically** runs 1.17× slower on
Windows than on WSL, which the ratio method cannot see. The truth is somewhere in
[2.4, 3.2]. **Report the conservative one.**

## 7. L296 — the selection proxy's area/HPWL exchange rate is wrong by ~5000×, and fixing it is strictly worse

The true cost is
`(1 + 0.5·hgap + 0.5·agap)·exp(2V/N) = (0.5/A_L)·(area + (A_L/H_L)·hpwl)·exp(2V/N)`,
so `layout_score`'s `area + hw*hpwl` has exactly the right **form** and `hw`
should be the per-case `A_label / H_label`. From the dataset's own `metrics`
tensor on the in-set 100:

      A_label / H_label   min 25.6   p50 290.0   max 3095
      shipped hw          0.06 (n<116) / 0.12    ->  median ratio 4833x

`hw·hpwl` carries **0.08 % of `layout_score`** (max 0.65 %), while across a real
51-candidate set the hpwl term varies *slightly more* than the area term in
true-cost units (0.420 vs 0.397). `layout_score` is in practice a pure area
minimiser plus a near-lexicographic violation term; ranking real candidates with
it instead of the true proxy costs **+11.5 %** and agrees with the true argmin on
**9/80** cases.

A label-free deterministic estimator recovers the right value:
`hw* = 3.3675·sqrt(1.035·ΣA)/Σw` (HPWL = total net weight × a characteristic
length; that length scales as `sqrt(area)`), accurate to **5.2 % median / 12.6 %
p90**.

**Measured, and RED.** `constructive_l296.exe`; identity gate 102/102 PASS with
the flag off, liveness 44/51 profiles move with it on:

      shipped                          1.226325126     0.0000 %
      ICCAD_LS_C=0.0067  (hw x10)      1.226325126     0.0000 %  bit-identical
      ICCAD_LS_C=0.0337  (hw x48)      1.228302252    +0.1612 %
      ICCAD_LS_C=0.337   (hw x480)     1.228049934    +0.1406 %
      ICCAD_LS_C=1.0     (hw x1400)    1.228422689    +0.1710 %
      ICCAD_LS_C=3.3675  (hw = r*)     1.232210358    +0.4799 %
      + multiplicative violation term  1.234334073    +0.6531 %

No interior optimum. The diagnosis is the useful part:

      weighted           hpwl_gap   area_gap     vrel
      shipped              0.2484     0.1355   0.01407
      ICCAD_LS_C=3.3675    0.2563     0.1475   0.01239

**Raising the weight on HPWL made the final HPWL worse.** `layout_score` runs
before compaction, `hpwl_push` and the shape LP; pre-refinement HPWL is
anti-predictive of post-refinement HPWL. Fourth independent time this project has
moved an HPWL knob and got HPWL back worse (L276, L280 ×3).

Also measured and ~neutral: `ICCAD_LS_GF=150000` (charge grouping fragments the
same as boundary misses, which is what the true cost does — `csc_of`'s own comment
says the 150000 : 6500 split is wrong): in-set **−0.0037 %**, graded **−0.019 %**,
RF +0.002 %. Free, real, and far too small to matter.

## 8. Determinism — PASS

    gate0 + k=2   run 1  1.189471885   run 2  1.189471885
                  per-case cost differences 0/100, feasible 100/100 both
                  wall 167.15 / 165.97 s (0.7 % spread)

`l294_gate0.json` / `_r2.json` are likewise bit-identical (`1.199000373` twice).
So the ungated in-window LP does not trip CLAUDE.md's HiGHS `time_limit`
non-determinism hazard on this corpus — though the flag itself has still not been
audited in the source.

## 9. Order for the next session

1. **Decide gate0 vs gate0+k=2, and lean gate0.** The combination is a wash on
   score, `gate0` alone is already OOS-validated on both samples, and it is the
   smaller change. Running `l287_transfer.py --arms ship,g0k2` would settle it,
   but the prior is that it will not separate them.
2. **Pin `f`.** §6 shows the rank-1 claim lives and dies on it, and the current
   value is a packing-phase ratio applied to LP seconds.
3. Determinism ×2 and a HiGHS `time_limit` audit on the ungated path.
4. Linux lane (`l117_linux_verify.judge48()` invariants, not bit-equality).
5. Do **not** reopen L281–L284, the `hw` axis (§7), or selective ungating (§5).

## 10. Files

```
l295_rank1.py            the rank-1 target as a function of (quality, runtime)
l296_project.py          corpus projection: (g, phi) -> graded total, + sensitivity
l297_rank1_price.py      projection + corrected RF, priced against the leaderboard
l298_selective_ungate.py per-block-count subsets of ICCAD_LP_GATE=0
l299_project2.py         per-component projection, baselines A and B (--base=)
l300_sensitivity.py      the f x transfer grid
l301_ffree.py            the same-box-ratio (f-free) price, carried to rank 1
l295_geom.json           per-case A_label, H_label, N_soft, r*
constructive_l296.cpp/.exe   ICCAD_LS_C / ICCAD_LS_MULT / ICCAD_LS_BV / ICCAD_LS_GF
                             (all default to the shipped constants; gate 102/102)
l296_A1/A2/c*.json  l296_sweep.sh
l297_ship/ship2/g0k2/g0k4/gf.json  l297_g0k2_r2.json  l297_combo.sh
```

Nothing shipped. `constructive.cpp` md5 `e2c7b2f4…`, `op_wrapper.py` md5
`1c326784…`, both untouched. `constructive_l296.cpp` is a separate probe source
and is not on the deployment path.
