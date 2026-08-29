# L294 — `ICCAD_LP_GATE=0` is the biggest quality lever left, and it prices GREEN

The handoff's §3.1 open item, measured. The gate switches the shape LP **off**
on 29 block counts carrying **44.2 %** of the graded weight (8 of the heavy 20).
It exists purely to buy runtime — the exact quantity L287/L291 showed was being
over-charged **33×** — so the trade had to be re-decided under corrected pricing.

    in-set 100, official evaluator, ADAPTIVE_CORES=48, 2x2 sandwich

      arm             total          quality    dt local   feasible   grader s
      ship (k=1)   1.226325126      0.0000%       0.00 s    100/100      45.2 s
      LP k=2       1.222554152     +0.3075%      +9.36 s    100/100      48.1 s
      **gate0**    **1.199000373** **+2.2282%** **+15.78 s** **100/100** **50.2 s**

**Quality +2.2282 %** — 7.2× LP k=2, and the largest single in-set move in this
ledger since M71. **29 of 29 gated-off cases moved, 29 better / 0 worse**, which
is structural, not luck: `_shape_lp` accepts a layout only when the shapely
proxy strictly improves and `hard_ok` passes, so the mechanism cannot make a
case worse — only slower.

**NET +0.91 % … +1.42 %**, against the project's 0.30 % bar.

---

## 1. Gates

| gate | result |
|---|---|
| **G0 liveness** | `ship` LP ran on exactly **71** distinct block counts, `gate0` on **100**; `gate0 \ ship` **equals the gated-off set exactly** |
| **G1 anchor** | `ship` = **1.226325126**, bit-equal to the shipped anchor |
| **G1 determinism** | `ship` vs `ship_r2` **100/100** cost and positions; `gate0` vs `gate0_r2` **100/100** cost and positions |
| **G2 feasibility** | **100/100 in all four runs** |
| **G3 monotonicity** | 29 moved, **0 worse**; every moved case is on a gated-off `n` |

G0 is the one that mattered. `_shape_lp_maybe` never raises by design (handoff
trap 5), so a dead flag is indistinguishable from a decision not to act — the
stats-file line count is the only thing that can tell them apart, and it is the
same check L205's LANE 5 used.

## 2. The cost, measured twice because the verdict now depends on `f`

LP k=2 was GREEN across the whole defensible `f` bracket (break-even 1.14), so
the machine factor did not matter. **For the ungate it does**: break-even
`f` = **1.99**, which sits just inside the bottom of L287 §3's own bracket
[1.91, 2.87].

### (a) imported `f` — as `l293_frontier.py` prices

      f          RF          NET     grader s   on RF floor
      1.00   -6.4523%    -4.2241%       61.0 s     73/100
      1.91   -2.4287%    -0.2006%       53.5 s     76/100
      2.71   -1.0885%    +1.1397%       51.0 s     82/100
      3.17   -0.8063%    +1.4219%       50.2 s     84/100

### (b) same-box ratios, where `f` cancels

L173 §4 withdrew a number for mixing a Windows wall with a WSL-calibrated `f`.
That risk is avoidable here: express the LP's added time as a **fraction of that
case's local wall on this box**, and apply the fraction to the grader's own
measured per-case time. Every ratio is same-box; the only external input is the
grader's runtime vector, which we own.

    LP share of local wall  p50 1.4 %   max 70.7 %
    local +15.78 s  ->  grader +6.66 s      (implied f = 2.37)
    quality +2.2282%   RF -1.3167%   NET **+0.9114%**   grader 51.9 s   81/100 on floor

**Control:** the same method on LP k=2 returns NET **+0.2792 %** against the
published **+0.2929 %** — agreement to **0.014 pp**. That is what says the
method is sound rather than convenient.

⇒ **NET +0.91 % … +1.42 %.** Take **+0.91 %** as the number: it is the one that
imports no cross-box constant.

### What the bill actually is

Runtime is **not** the binding constraint — 50.2 s at f = 3.17, 51.9 s by
ratios, 53.5 s even at f = 1.91, all far under the 64.1 s rank-2 threshold. The
whole bill is cases **leaving the RF floor**: 98/100 → 81–84. The LP is
**50–71 % of the base wall** on the heavy cases (n=99: +1.61 s on a 2.19 s
case), which is exactly why L196 built the gate — and exactly what the 33×
over-charge made look unaffordable.

## 3. The full ungate is at the optimum — no cut point to fit

Ranking the 29 by graded value per local second and walking the prefix:

      keep    quality       dt s        RF        NET   grader s
        5    +1.3018%     3.96 s   -0.2660%   +1.0358%     46.4 s
       10    +1.8150%     7.06 s   -0.6683%   +1.1468%     47.4 s
       15    +2.1239%    12.19 s   -0.9353%   +1.1887%     49.0 s
       29    +2.2282%    16.24 s   -1.0145%   +1.2137%     50.3 s

Flat from keep ≈ 10 onward. So there is nothing to gain from a partial ungate,
and that matters for a reason beyond convenience: **`ICCAD_LP_GATE=0` deletes a
fitted 29-entry table rather than adding one.** L271's concentration warning
does not apply to a change whose entire content is the removal of constants.

## 4. Why L205's −3.3998 % read larger than the whole LP

The handoff flagged this as needing a sanity check before belief. Both halves
are now explained:

* L205's gate had **37** block counts off; the shipped one has **29**.
* The shipped LP is worth **+2.59 %** in set (`ship` vs `noLP`, L287). Ungating
  adds **+2.23 %** on top, so the ungated LP is worth ≈ **+4.8 %** against no LP
  at all, and the gate was giving up **86 %** of the mechanism's value.

The number meant what it said. What was wrong was the price, not the quantity.

## 5. Deployment class

`ICCAD_LP_GATE` appears **0 times** in `constructive.cpp`; `_lp_gate_ok` is
wrapper-only, both in the tree and in the shipped `op_wrapper.py`. So this is a
**Python-only change with no ELF rebuild** — the same class as the LP depth
candidate.

⚠️ **An env var will not ship it.** L158's rule: the grader strips every
`ICCAD_*`, so a mechanism reachable only through the environment is inert in the
package. Shipping means changing the **code default** — `_L196_LPGATE` to all
1s, or `_lp_gate_ok` to return True.

## 6. L295 — OOS: positive on BOTH samples, and it transfers at 100–111 %

`l287_transfer.py --arms ship,lp2,gate0`, the full deployable pipeline, 240
cases per sample, `ship` re-used from cache:

      sample     ship        gate0      ship vs gate0   movers   transfer
      in-set   1.226325   1.199000        +2.2282%      29/100      --
      **s1**   1.470262   1.434895      **+2.4648%**    73/240   **111 %**
      **s2**   1.465254   1.433190      **+2.2373%**    74/240   **100 %**

      by band          s1        s2
      light n<=60   +2.134%   +2.031%
      mid 61-100    +1.982%   +1.945%
      heavy n>=101  +2.520%   +2.272%

**L275's both-corpora rule is satisfied outright, with room.** s1 and s2 agree
to **0.23 pp**, all three bands are positive in both, and the gain transfers at
**100–111 %** — it does not decay off the in-set at all.

For scale: on the same two samples LP k=2 is +0.4842 % / +0.4891 %. The ungate
is **≈5×** that.

### Liveness and safety, checked on the OOS side too

| | s1 | s2 |
|---|---|---|
| errors / infeasible, all three arms | **0 / 0** | **0 / 0** |
| movers | 73/240 | 74/240 |
| every mover on a gated-off block count | **yes** | **yes** |
| better / worse | **73 / 0** | **74 / 0** |
| cases on a gated-off `n` that moved | 73 of 74 | **74 of 74** |

The mover count is itself the liveness proof: 29 of 100 block counts are gated
off, and 30.4 % / 30.8 % of the held-out cases moved — the arm reached exactly
the cases the table names and nothing else. `set_arm()` now also asserts
`_lp_gate_ok(38)` flips with the arm, at the source, before any solve.

**0 worse out of 147 held-out movers** confirms the in-set monotonicity was
structural, not a small-sample artefact.

## 7. What is NOT measured

1. **Composition with LP k=2.** They are independent knobs — the gate decides
   whether the *first* pass runs, `k` decides how many run where it already
   does. The pair is unmeasured and is **not** the sum.
2. **Linux lane** (`l117_linux_verify.judge48()` invariants).
3. **`f` is still bracketed, not pinned** — and it now matters, which it did not
   for k=2. The OOS result does not help here: it confirms the **quality** half
   on three corpora, while `f` prices the **cost** half.
4. dt's own spread: the four pairings run 12.97–18.59 s (±18 %) against wall
   noise of 1.65 s (ship pair) and 3.97 s (gate0 pair).

## 8. Files

```
l294_lpgate.sh      the 2x2 sandwich runner, with the stats liveness gate
l294b_det.sh        the gate0 determinism repeat
l294_gate.py        gates + pricing + the partial-ungate frontier
l294_final.py       the 2x2 dt and the two independent pricings
l294_{ship,ship_r2,gate0,gate0_r2}.json / .log / _stats.txt
l295_gate0_oos.sh   the OOS driver; l295_s1.log / l295_s2.log
l287_transfer.py    + the `gate0` arm and a source-level `_lp_gate_ok` assert
l287_cache.pkl      now 2880 keys (s1 x 8 arms partial, s2 x 3)
```

Nothing shipped or modified. `constructive.cpp` md5 `e2c7b2f4…`,
`op_wrapper.py` md5 `1c326784…`, both unchanged.
