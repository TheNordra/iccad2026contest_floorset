# Handoff 2026-08-25 — route A OFF + pool drop, both verified end to end

> **FINAL 2026-08-25 14:50.** Two changes shipped this session, each gated and
> Linux-verified from scratch. Final identity:
>
> ```
> op_wrapper / op_src     09faf9b39514270c144f77c4743e6b86
> cadc1075.tar.gz         2683751f6e8458c449175fcaa2c649fe
> bin/constructive_linux  bc9912072cd97b45b47a03adec7170ce   (never touched)
> ```
>
> | change | what it does | worth |
> |---|---|---|
> | **route A OFF** (`_route_a_default()` -> 0) | removes a bet with `ra = 1.0021` (measured) and an unbounded tail | certainty at 0.91491 instead of a lottery |
> | **`_L211_POOLDROP`** (8 of 51 profiles per block count) | cuts the max-setter-bound 48-core wall by 5.50% | **+0.87 pp NET**, OOS-validated on two disjoint samples |
>
> **Final: NET +2.133 % vs beta**, against +1.260 % at the start of the session.
> **Rank is unchanged at 4** — rank 3 needs +2.942 % and nothing available
> reaches it. See `L203_L205_REPORT.md` §6 for the full pricing.
>
> * **In-set: ALL PASS**, 10 arms, 11 checks including **G10** (the pool-drop
>   pair: kill switch bit-identical to the pre-drop anchor AND the default moves
>   exactly the 12 measured cases) and **G2c'** (the L147 hatch re-tested in the
>   anchor's own pool configuration: 100/100).
> * **Linux: all five lanes PASS** on both packages.
> * ⚠️ The local score gets *worse* (1.23922 -> 1.24076) and that is correct:
>   the local harness forces RF=1.0, so the wall gain is structurally invisible
>   and only the −0.1242 % quality cost shows. The gain is a MODELLED RF number
>   on a MEASURED quality cost — the same split every runtime lever in this
>   ledger has.
>
> Everything below is the overnight analysis that produced both decisions.

> **UPDATE 2026-08-25 11:05.** The user took the §4 decision: **turn route A
> off.** Done, re-staged, and re-verified from scratch. New identity:
>
> ```
> op_wrapper / op_src   28d9273601c8670bd003b231dac742af
> cadc1075.tar.gz       bf98ebb4f8d1148e9345d85f63471c62
> bin/constructive_linux  bc9912072cd97b45b47a03adec7170ce   (unchanged)
> ```
>
> * **In-set: ALL PASS**, 8 arms, and a new **G9** — every arm bit-identical to
>   its L199 counterpart on cost AND positions. Route A off moved the wall and
>   nothing else, which is the third independent confirmation that it was
>   result-neutral (L177 det1/det2, the L205 probe, now G9).
> * **Linux: all five lanes PASS**, including LANE 3, which failed only on the
>   stale `--live-min 1.5` and passes at the derived 0.40 (measured +0.7418 %).
>   Total `1.2392183582600296` — bit-identical to the route-A-on Linux run.
> * The lane verifies its own preconditions: `route A off marker: 1`,
>   `_PROF_TIMING: 0`, tar md5 == staged tree.
>
> Everything below is the overnight analysis that produced the decision.

---

# (original) the package is verified; route A is now a choice, not a bet

**Deadline 2026-08-28.** Continues `HANDOFF_2026-08-25.md`, which left three
things to do. All three are done. Full evidence in `L203_L205_REPORT.md`.

---

## 0. One-paragraph version

The L196 package is **re-staged, fully gated, and Linux-verified** — eight in-set
gates ALL PASS, five Linux lanes green (the only FAIL was a threshold inherited
from a configuration that no longer exists, corrected and explained). Two new
axes were explored and both **close**: the LP gate's *shape* is provably
suboptimal but worth no rank, and the `s` parameter is confirmed at 1.20 on a
grid ten times finer than the one that chose it. The one thing that moved is
route A: the ledger called it unmeasurable, **its win condition turned out to be
measurable here**, and it comes out at `ra = 1.0021` — neutral. That converts it
from an unpriced bet into an explicit **variance choice**, which is yours to
make and is the only open item.

---

## 1. Package identity — verified, do not rebuild without re-verifying

```
op_wrapper.py / op_src.py   bb44bb147231fee7bc9670cdc28448bc
cadc1075.tar.gz             f6fadf263a0821bda4ac4ad344675430
bin/constructive_linux      bc9912072cd97b45b47a03adec7170ce   (unchanged)
_L157_DEPTH                 {1: 100}   flat
_L196_LPGATE                63 on / 37 off, 8 of them n>100
requirements.txt            104 bytes, scipy listed
vendor/                     1536 files
```

The tree, the staged directory and the tar all hash to the same `op_wrapper`.
**This is the artefact that passed everything below.**

## 2. What was verified

**In-set (`l199_gates.sh`, 7 arms + G2c; scored by `l199_verdict.py`): ALL PASS.**

| gate | result |
|---|---|
| G1 determinism | cost 100/100, positions 100/100 |
| G2a/b/c L147 hatch | 63/63 on the kept counts, 37/37 == the no-LP arm on the dropped ones, and **100/100 vs the anchor with the gate killed** |
| G3 the gate fired | **63** = the table's 1-set *as a set*; `LP_GATE=0` → 100; `SHAPE_LP=0` → 0 |
| G4 the map is flat | `k1` bit-identical to `det1`; passes spent `{1: 63}` |
| G5 feasibility | 100/100 in all 7 arms |
| G6 gate cost | −3.3998 % vs the ungated LP (exactly the 37 dropped) |
| G7 LP value | **+1.6417 %**, 62 better / 0 worse |
| G8 hb predictor | −0.0227 % |

Bonus: `det1` is bit-identical to the previous session's `_l198_gateon.json`.

**Linux (`l200_wsl_verify.sh`, 5 lanes):** 1 PASS (`+0.0000%` vs the Windows
LP-off base), 2 PASS, 3 **FAIL on a stale threshold only**, 4 PASS (determinism
100/100 on cost *and* positions), 5 **PASS — the gate is live on Linux**, firing
on exactly the right set and widening to 100 under the kill switch.

**LANE 5 is the only lane that could have failed on L196.** Lanes 1–4 all pass
unchanged if the table is inert.

## 3. Three stale anchors, repointed rather than relaxed

Each of these read FAIL on a *correct* package because it encoded a
configuration that no longer exists. None was loosened.

1. **`l117_linux_verify.py:_lp_liveness`** hard-failed when the stats file had
   fewer lines than cases — now correct behaviour for L196. It parses
   `_L196_LPGATE` **out of the tar** and asserts the multiset of block counts,
   which also catches a table firing on the *wrong* 63. Would have failed LANE 3
   an hour into the run.
2. **`results_L165_l147off.json`** predates the LP gate, so the flat compare read
   63/100. The 37 that differ are **exactly** the 37 the gate drops (both set
   differences empty). Split into G2a/G2b/**G2c**, the last of which reproduces
   the anchor 100/100 with the gate killed — so the hatch is provably untouched.
3. **`--live-min 1.5`** was set when the LP ran on all 100 cases. Under L196 the
   shipped band is **+0.7418 %** ahead of the control on Linux (+0.6761 % in
   set), not +1.5 %. `l207_wsl_final.sh` uses **0.40** — 59 % of the measured
   gap, against a failure mode (L147 not applying) that produces 0.000 %.

One phantom: L177's `det1`/`det2` were distinguished by
`ICCAD_SHAPE_LP_NOOP`, **which does not exist in the tree** (grep: 0 hits).
Harmless there; the L199 arms differ by tag only.

## 4. 🚨 The only open decision: route A

The ledger closed route A as unmeasurable. That is true of its **wall** and
false of its **win condition**, which is a workload ratio and transports off
this box:

* `solve()` submits **all 51 profiles at once**; the grader has 48 cores, so it
  is **saturated** and route A's premise (convert *idle* cores into wall) has no
  idle cores.
* `_route_a_cores()` deliberately ignores `ICCAD_ADAPTIVE_CORES`, so route A
  cannot oversubscribe. What costs is **work**: 1.44× (L110).
* Therefore route A wins case `n` iff `D_max/D_mean > 1.44·51/48 = 1.53`.

Measured three ways (`optimizer_l205probe.py`, instrument **not** in the shipped
tree; 5100/5100 records each, completeness asserted):

| run | median ratio | wins | weighted | **mean ra** |
|---|---|---|---|---|
| parallel r1 / r2 | 1.377 / 1.362 | 2/100 | 1.7 / 3.6 % | 1.0639 / 1.0671 |
| **sequential (uncontended)** | **1.492** | 30/100 | 32.3 % | **1.0021** |

The parallel runs **compress the imbalance by 8.8 %** (measured directly), which
is why they read "route A costs 6.4 %". Uncontended, **`ra = 1.0021` — neutral**.

Independent check on the inherited 1.44: four 100-case evaluations differing
only in route A gave ON 7:06/7:01 vs OFF 2:45/2:47 = **2.5×** on this saturated
box, reproducing the ledger's 2.9× from matched arms. (Windows spawns a process
per *frame*; treat 2.5 as an upper bound, not a transportable value.)

### The choice

| | outcome |
|---|---|
| **route A OFF** | certainty: **0.91491, rank 4**, beats beta by 1.26 pp |
| **route A ON** | a lottery on the same centre: rank 3 needs `ra ≤ 0.917`, rank 2 needs `ra ≤ 0.841`, **rank 5 starts at `ra ≥ 1.10`**, rank 7 at 1.35 |

* **4th meaningfully better than 7th** → turn it off. Same expected score, no tail.
* **Only the top 3 pay** → a certain 4th is worth what 5th is worth, so the
  lottery's upside is free. Leave it on.

**My read: turn it off** — the margin over beta is 1.26 pp, RF spends it at
0.3 % of score per 1 % of wall, and nothing in three measurements says route A
pays. But it is a preference over outcomes, not a fact, so **I did not apply
it.** One line: `_route_a_default()` returns 0. Must be a **code default** — the
grader strips `ICCAD_*` (L158). Re-verification chain, ~1.5 h:
`l199_gates.sh` → `l199_verdict.py` → `l207_wsl_final.sh`.

## 5. Two axes that closed (no action)

* **The LP gate's shape.** The objective is separable in `n`, so the optimal gate
  is a per-`n` sign test, not any threshold in `t/M`. It transfers cleanly
  (98/100 agreement across disjoint samples) and **changes no rank in either
  regime**: +0.03 pp neutral, −0.44 pp under the bet.
* **`s`.** Swept at 0.05 where L196 sampled one point in the peak. Optimum is
  1.15 by **+0.03 pp** — inside the ~0.2 pp CV spread — while **1.20 is +0.45 pp
  better under the bet**. Keep 1.20. Rank 2 needs `s ≥ 1.15`; the neutral branch
  falls behind beta at `s ≥ 1.55`.

**Rank 1 is unreachable on this axis**: the oracle bet-regime gate lands at
0.87051 against an r1 threshold of 0.85863.

## 6. Silent-failure modes found this session

* **An instrument that loses samples silently is worse than none.** v1 printed
  `[proftime]` to stderr from 51 threads: 5100 emitted, **4588 parseable**. The
  two biases (D_max down, k down) point *opposite* ways, so the run cannot be
  repaired after the fact — and it flipped the sign of the route A verdict.
  Caught only because the same table printed "4 of 100 can win" next to a mean
  `ra` below 1. **v2 asserts completeness before printing anything.**
* **Print a known upper bound next to the honest rows.** L203's first version had
  ORACLE scoring *below* the shipped gate, which is arithmetically impossible
  under separability — that is what exposed a units error comparing a 100-row
  corpus against a 240-case one.
* **Do not instrument the shipping artefact.** v1 edited
  `optimizer_constructive.py`, which would have forced a re-stage and a second
  hour of Linux lanes to prove inertness. The probe copy costs nothing and the
  shipped tree stayed byte-identical to what was verified.
* **`bash -lc` is not enough for WSL.** MSYS rewrites `/mnt/...` inside the
  quoted string and the failure mode is a variable expanding to *empty*, so the
  script runs and measures nothing (that is `l187_wsl_verify.log`'s 99 bytes).
  Working form: `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl -d Ubuntu -- bash -lc '...'`.

## 7. Still open, lowest priority

**G8 / the L171 hb predictor**: −0.0227 % in set on the L196 configuration
(−0.0512 % on L172's map). Per this ledger's own rule — never act on an in-set
null; the twins moved 0 of 100 in set and are worth +0.67 % OOS — that is not
grounds to pull it. It would need the two OOS samples re-run in this
configuration (~2.3 h) for ~0.02–0.05 %.

## 8. New files

```
l199_gates.sh / l199_verdict.py      in-set gates, 7 arms + G2c, 8 checks
l201_g2c.sh                          the decisive L147 arm
l200_wsl_verify.sh                   5 Linux lanes (supersedes l187)
l207_wsl_final.sh                    same, with the two stale thresholds fixed
l203_marginal_gate.py                gate-shape analysis + decision table
l204_routea_risk.py                  the route A payoff curve and phi sensitivity
l205_imbalance.py / l205b_compare.py the win-condition measurement
l205_run.sh / l205b_seq.sh           its two runners (parallel / uncontended)
optimizer_l205probe.py               probe copy; NOT the shipping tree
L203_L205_REPORT.md                  full evidence
```
