# L285 — the shipped package is 13–15 % FASTER than the one that scored 52.07 s

The question was whether the current package's 48-core runtime has drifted since
beta, because the rank is sharply sensitive to it and beta's 52.07 s predates the
shape LP. It has drifted — **downwards**.

    same machine, same session, back to back, ADAPTIVE_CORES=48, x2 repeats

      arm                                  in-set total       runtime
      shipped default                      1.226325126    129.69 / 128.20 s
      shipped, shape LP off                1.258974453           117.86 s
      beta configuration (M73-like)        1.259897682    149.43 / 149.97 s

      shipped / beta-cfg over all 4 pairings : **0.8548 – 0.8679**
      projected grader runtime : **44.5 – 45.2 s**  (was 52.07 s; rank 1 spends 110.9 s)

The L223/L231 REFINE cuts (heavy 4→2, mid 8→2) save more than the shape LP costs.

    cwRF 0.70004    **98 / 100 cases sit on the RF floor**
    projected total 0.87511   ->  **rank 2**, margin +0.01307 over the next team
    losing rank 2 needs a **1.43×** slowdown from here

**Runtime is not a risk.** The margin is wide enough that the conclusion survives
any plausible error in the ratio: even at 1.0× (no speedup at all) the package is
still rank 2, and the threshold is 1.23×.

---

## 1. Why it had to be measured this way

A direct 48-core measurement is not available here and a naive one is actively
misleading:

* this box has **32 logical cores**, and `ICCAD_ADAPTIVE_CORES=48` changes pool
  *selection* only — route A then fans the frame-trial loop 48 ways onto 32
  cores, so the raw local 129.7 s is an oversubscription artefact;
* CLAUDE.md's standing warning is *"品質可信、時間不可信"*.

So the sound quantity is a **same-machine ratio**, in which the machine-speed
factor cancels, applied to the one absolute we own: our own 52.07 s **as measured
by the grader**. The beta arm is reconstructed from the shipped code with the
project's own kill switches — `ICCAD_SHAPE_LP=0`, `ICCAD_ROUTE_A=0`,
`ICCAD_M80_TIER=0`, `ICCAD_HINT_MODE=0`, `ICCAD_L223_REFINE_HEAVY=4`,
`ICCAD_L231_REFINE_MID=8`.

**Gate.** The shipped-default arm reproduces the anchor exactly
(`1.226325126` vs `results_L274_base_48c.json`'s `1.2263251265`) and both arms
are **bit-identical across repeats**, so what was timed is the shipped
configuration and the quality columns carry no noise. Runtime spread across
repeats is **1.2 %** (shipped) and **0.4 %** (beta-cfg) — far tighter than
CLAUDE.md's ≥20 % warning, which was about comparing *different* configurations
whose true difference was small.

⚠️ The beta arm reverts the flag-gated additions but **not** M74's constant regen
(`_BIG_REDUNDANT_IDX` membership, tier-3 cores-gating). It is "M73-like", not M73.

## 2. Headroom, priced against the real grader data

`beta_2026-08-16/beta_evaluation_results.json` holds our per-case runtimes **as
the grader measured them**, and the 2026-08-23 republication holds the medians it
prices against. Sweeping a slowdown `s` on that vector
(`RF_i = max(0.7, (s·t_i/M_i)^0.3)`):

    gate: recomputed beta row   raw 1.3206649447   cwRF 0.701606
                                total 0.926586663   vs leaderboard 0.9265861161

    lose rank 2 : s = 1.23x  (64.1 s)
    lose rank 3 : s = 1.35x  (70.5 s)
    lose rank 4 : s = 1.57x  (81.9 s)

We measured **s = 0.855–0.868**. The headroom is **1.43×**, and 98/100 cases are
on the RF floor where the derivative is exactly zero — roughly **19 seconds of
genuinely free budget**.

⚠️ The rank also depends on the quality transfer: the in-set 48c total went
1.295548 (M73) → 1.226325 (now), −5.34 %, and that factor is *assumed* to carry
to the hidden set. It is the softer of the two assumptions — if only 75 % of the
gain transfers we are rank 3, at 50 % rank 4. L275's corpus warning applies.

> ### ⚠️ CORRECTION (same day, L287-L291): this section understates the error badly
>
> The baseline is only the *secondary* cause. The dominant one is that
> `l276_price.py` (and `l146_rf_price.py`) add **local dt seconds** to **grader
> seconds** with no machine factor — while `l172_depthmap.py` carries exactly
> that factor and has done since L161: `F = 3.17  # dev-box LP second -> grader
> second`. Restoring it, LP k=2's RF bill goes from **−0.4816 %** to
> **−0.0146 %**, a **33×** over-charge, and the arm is **NET +0.2929 %**, not
> RED. See `L287_L291_TRANSFER_AND_PRICING.md`. The "~40 %" below is the
> baseline half only.

## 3. 🚨 A systematic correction: every RF bill this project has printed is too large

`l276_price.py` prices added seconds against `load()`, which takes the runtime
vector straight from the **beta** results — i.e. the M73 package. The shipped
package is 13–15 % faster, so every case sits lower on the `(t/M)^0.3` curve and
more of them are on the floor. Re-pricing L276's own arm with its own tool:

    LP k=2                 graded total       RF        NET     floor
      beta baseline          0.9265867    -1.2611%  -0.9536%   82/100   <- L276 as published
      SHIPPED baseline       0.9245123    -0.7568%  -0.4493%   98/100
      hypothetical 1.20x     0.9359879    -2.3437%  -2.0362%   62/100

**The bill is 40 % smaller than published from the baseline alone.** ⚠️ With the
missing machine factor `f = 3.17` also restored it is **33× smaller** and the
verdict **flips to GREEN (+0.2929 %)** — see the correction banner above. The
"verdict does not change" sentence that stood here was wrong.

### 3.1 An error of mine, caught by the project's own tool

My first attempt at this re-pricing used a `dt` median over a ±2 block_count
window and reported LP k=2 as **NET +0.12 %, GREEN**. That is wrong by a factor
of four. The in-set has ~one case per block_count, so `dt_by_n`'s `mean` is that
case's actual dt, while my window-median smoothed away the fat tail — dt is
p50 +0.018 s but max +0.745 s, and the expensive cases are the big-n ones with
the least slack. This is verbatim the trap `l276_price.py`'s own docstring warns
about: *"an added-time distribution is not a ratio"*. Use the tool.

## 4. Files

```
l285_runtime_headroom.py   slowdown sweep against the real grader runtimes + medians
l285_lp_on.json / _r2      shipped default, two repeats
l285_lp_off.json           shipped with ICCAD_SHAPE_LP=0
l285_betacfg.json / _r2    the M73-like reconstruction, two repeats
```

Nothing was shipped or modified. `constructive.cpp` md5 `e2c7b2f4…`,
`op_wrapper.py` md5 `1c326784…`, both unchanged.
