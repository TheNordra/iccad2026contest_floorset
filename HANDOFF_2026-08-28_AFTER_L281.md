# Handoff 2026-08-28 — after L281-L286. Quality is out of experiments; runtime is verified safe.

The previous handoff named one task, the relocation probe. It is done and it is
**RED** (`L281_RELOCATION.md`). The one idea L281 left open — shortening the
critical chain rather than lengthening it — was then built and measured too, and
is **also RED** (`L282_CHAIN_SHORTENING.md`). Read both; §3 below is the state
of the axis after them.

---

## 0. Shipping state — unchanged, still frozen

| | |
|---|---|
| uploaded | `build_submission.D/cadc1075.tar.gz`, Drive **Final** |
| identity | `op_wrapper.py` md5 **`1c326784de7cd9246cd1f380e2842668`** |
| source | `constructive.cpp` md5 **`e2c7b2f418ef2b70b6bff99f7adfbd37`** |
| 48c Linux | **1.2264069637381392**, feasible 100/100 |
| graded | **0.87818**, **rank 2**, margin 1.00 pp over r3 |

**Nothing was shipped, staged, or modified this session.** `constructive.cpp`,
`optimizer_constructive.py` and `build_submission.D/` are byte-identical to how
the previous handoff left them. L281 is an offline probe; it never had a
deployable form to consider.

## 0.1 Runtime verified (L285) — and a stale-cache correction that hit two of my own reports

**The shipped package is 13-15 % FASTER than the one that scored 52.07 s.** Same
machine, back to back, two repeats each, with the beta arm reconstructed from the
shipped code's own kill switches:

    shipped default        1.226325126  (== the anchor, bit-exact)   129.69 / 128.20 s
    shipped, SHAPE_LP=0    1.258974453                                      117.86 s
    beta config (M73-like) 1.259897682                               149.43 / 149.97 s
    ratio over all 4 pairings 0.8548 - 0.8679  ->  grader **44.5 - 45.2 s**

L223/L231's REFINE cuts save more than the shape LP costs (LP = +11.83 s = +10 %,
buying +2.59 % quality). Priced on the real grader runtimes and the 2026-08-23
medians: cwRF **0.70004**, **98/100 cases on the RF floor**, total **0.87511**,
**rank 2** with +0.01307 of margin. **Losing rank 2 needs a 1.43x slowdown.**
Runtime is not a risk, and there are ~19 seconds of free budget.

🚨 **`l276_price.py` prices against the BETA runtime vector, so every RF bill this
project has printed is ~40 % too large.** Re-priced at the shipped baseline, LP
k=2 goes from NET −0.9536 % to **−0.4493 %** — still RED, but scale the baseline
before pricing anything marginal.

🚨 **`audit_cache_ship.pkl` is STALE and it invalidated the numbers in L283 and
L284.** Signature REFINE 6/4 vs the shipped 2/2, no shape LP; its pool-best
reproduces the shipped per-case cost on **4/100** cases (weighted 1.292646 vs
1.226325). Both reports now carry correction banners. **The shipped weighted
utilisation is 85.16 %, not the 80.91 % L284 reported** — and since the stale
pool's own maximum was 85.36 %, the capability-limit conclusion survives and is
stronger. L281 and L282 are unaffected (they anchor on the shipped positions).

⇒ **New standing gate before using any audit cache: check that its pool-best
reproduces the shipped per-case cost.** Pinning the exe md5 in the signature was
not enough, because nobody checked the cache against the shipped result.

## 0.2 The package's OOS transfer (L286) — first measured, and it is ~zero on the heavy band

L285's rank projection assumed the in-set gain since M73/M74 carries to the
hidden set. **That had never been tested**: every OOS number in the ledger is one
mechanism against the base of its day, never the cumulative package. Two caches
share all 80 OOS case keys, so it needed no placer runs:

    OOS heavy band n=111..120, 40 cases/sample, m67._cost throughout
           pre-LP     post-LP    M74-era   now vs M74   LP moved
      s1  1.511619   1.486674   1.473374     +0.90 %     20/40
      s2  1.486799   1.447403   1.456133     -0.60 %     20/40
      in-set 48c: M74 1.293461 -> now 1.226325 = **-5.19 %**

**The in-set gain is not visible on this band** (mean ≈ +0.15 %, signs disagree).

⚠️ **Read it as a caution, not a forecast.** L275: this band is 22-24 % harder
than the graded corpus while beta hidden sits ≈2.4 % from the in-set — the hidden
set resembles the in-set, not this band. Reading it as "the gain will not
transfer" repeats L275's error in the opposite direction. Range stays **rank
2-4**, centre moves from "rank 2" to "rank 2-3". **The floor is unchanged**: zero
transfer lands where beta landed (rank 4), and the runtime half is certainly
better (§0.1).

**Two silent artefacts had to be cleared first, and both are worth carrying:**

1. The first read looked like a **2.5 % OOS regression**. It was pre-LP vs
   post-LP: `l252_cache`'s records are raw per-profile positions, `m67`'s costs
   are the wrapper's full output. Recomputing the l252 base with `m67._cost`
   reproduces **L275's published 1.5116 / 1.4868 to six figures**, which proves
   L275's "shipped placer" row is the pre-LP one. The LP is worth +2.59 %, i.e.
   the entire discrepancy. **No regression.**
2. Adding the LP back first reported **"LP moved 0/40"** — impossible for a
   +2.59 % mechanism. `import m67_oos_probe` **strips ICCAD_\*** and
   `_shape_lp_maybe` **never raises by design**, so the flag was dead and every
   case came back unchanged while the table printed normally. Setting the env
   **after** the imports and asserting `oc._shape_lp_on()` gave 20/40.

⇒ **Standing rule: set `ICCAD_*` after importing any probe module, and assert the
flag is live before the measurement loop — never infer liveness from the output.**

Four limits on the headline (L286 §4): band mismatch (n=111..120 only), version
mismatch (m67 is 41-profile M74-era, beta shipped M73), sample sign disagreement,
and assembly mismatch. Settling it needs both packages through
`m67_oos_probe --force-cores 48` end to end on the full 240 — which changes no
shipping decision, only the sharpness of the rank prediction.

## 1. What L281 settled

**The fork the handoff posed.** Relocation's end-to-end LP-infeasibility is
**95.3 %** against M64's 86.8 % — not materially lower, slightly worse. That is
the pre-registered "close the axis" branch, and L281 §7.2 supplies the mechanism
the branch was guessing at.

**The thesis underneath it was correct.** Relocation removes essentially all of
M64's self-inflicted incoherence, exactly as `L280 §5` argued:

    heavy band n>=101, identical geometry, same certificate
    move                     coherent    cyclic    oversized
    M64 single-pair flip      24.4 %     23.0 %     52.6 %
    RELOCATION                21.1 %      0.3 %     78.6 %

It changes nothing because cyclicity was only a quarter of the wall.

**The wall, named.** The shipped anchor's longest chain of abutting blocks is
**exactly as long as the bounding box** in **62 of 100** graded cases (case 85:
`lH = 129.8107202276296` against `W0 = 129.8107202276296`), and a median of
**34.3 %** of blocks sit on that zero-slack chain. The LP may compact but never
grow, so a topology whose chain is one ULP longer is infeasible before wire is
considered.

⇒ **This closes post-hoc topology repair as a family, not just relocation.**
M64 (single-pair flips), L256/L259/L262 (ruin-and-recreate), L281 (relocation) —
three move semantics, one cause. A new proposal on this axis must say why it is
not a topology edit of a chain-saturated anchor.

**The cost side, for completeness.** **3367 LP solves over the whole heavy band**
(n >= 101 = 81.1 % of the graded weight; cases 85 and 88 exhaustive, the rest
top-5 units by wire prize), against a control given the identical LP *and* the
identical polish budget:

    per-case ORACLE gain vs the polished control  :  +0.0942 %   (bar 0.30 %)
    cases with any gain at all                    :  13 / 20
    LP wall time                                  :  172 s/case  (budget ~1.5 s)

That is an *oracle* — it picks the best of up to 980 attempts per case with the
official scorer, which a submission does not have — and there is no cheap rule
that finds the winner: it is rank 4/49 on one case and rank 36/58 on another
(M56/M79 again). Every reported gain was re-scored from its stored coordinates:
**9/9 bit-exact, feasible, and beating its own control.**

## 2. Three things worth carrying forward regardless

1. **`l281_saturation.py` is a cheap pre-screen for any candidate placer.**
   Slack in the critical chain is the raw material for every topology-editing
   idea, and the shipped greedy produces none. Run it before building anything
   on top of a new placer.
2. **A changed relation is not a move.** For a diagonal pair both separations
   already hold, so rewriting that LP row does not exclude the current placement
   — the LP returns the control's own solution and `nflip` lies. 6.4 % of
   L281's certified-coherent candidates were vacuous; excluding them moved the
   infeasibility from 81.4 % to 87.0 %. Cost alone cannot detect this; only
   comparing **positions** can (`l281_liveness.py`).
3. **The corpus asymmetry runs the *unhelpful* way here.** Chain saturation is
   86 % in-set and 22–25 % OOS. Anything measured on the OOS heavy band will
   report a far higher feasibility rate and a far larger prize for this class of
   mechanism, and none of it will transfer — L275, independently reproduced by a
   mechanism L275 never saw.

## 3. What is actually left — L282 tried the last open idea and it is also RED

L281 §10.1 item 2 named one untried thing: **shorten** the critical chain
instead of lengthening it. **L282 built and measured it: RED.** Full report in
`L282_CHAIN_SHORTENING.md`.

Gate 0 (no LP, 100 cases) said the redundancy is real — the best single unit
shortens the binding row by p50 0.82 % / p90 5.64 %, worth an optimistic
**+0.6282 %**, and the chain (not the frozen span) is the binding floor in
**93/100** cases. So it was worth building. Then:

    9 heavy cases, 413 chain-shortening relocations
      LP-infeasible                            374/413 = 90.6 %
      of the 39 that solved, cost got worse     31/39
      union-oracle vs the polished control      +0.0057 %   (1 of 9 cases)

**Two unrelated death causes, both worth keeping:**

1. **The journey, not the destination.** These moves *shrink* the box, so the
   box cannot be the obstacle — yet 90.6 % never solve, dropping boundary
   equalities rescues **0/30** (third reproduction: M64 0/15, L281 0/30), and
   **7/30 solve only when the bbox may GROW 20 %**. The unit cannot reach its
   target through a layout whose other ~4900 pairs are frozen at their anchor
   disjuncts.
2. **The exchange rate is 2.74 : 1 against.** Candidates were picked for a
   median predicted shrink of 4 %, and the median realised `area_gap` change is
   **exactly zero** — the LP *declines the shrink* in 28/39 cases. Where it does
   take it, one unit of area costs **2.74** units of wire, and the score prices
   them identically. This is L268's packing-time exchange rate (1.2 : 1) again,
   worse.

⇒ **The bounding box is not what the score is charging us for.** Slack was the
wrong thing to want, which retires L281 §10.1's framing as well.

### 3.1 The one thing still genuinely untested

**The packing-time version.** Death cause 1 — every other pair frozen at its
anchor relation — **does not exist during construction**. A greedy that never
builds a long chain is untested by anything here.

⚠️ But death cause 2 does carry over: 2.74 : 1 is a property of the geometry and
the scoring formula, not of the move, and L268 already measured 1.2 : 1 against
*at packing time* while achieving the largest utilisation gain in the project's
history. So the prize is bounded by an exchange rate now measured as unfavourable
twice, independently, at both ends of the pipeline.

**That question has since been answered — L283, and it did not need a flag.**
`audit_cache_ship.pkl` already holds 42 profiles x 100 in-set cases = **4200
independently generated layouts**, which are different packings rather than
compressions of one another, so they trace the generation-side frontier
directly. All 4200 re-scored with the official scorer:

    cheapest hpwl_gap paid per area_gap bought, per case
      p50 **+0.977**  (break-even)   vs L282's squeeze rate 2.74
      84/100 cases have a denser layout in their own pool
      25/84 are outright NEGATIVE -- denser AND better wire

**So yes: a shorter chain can be had without paying wire.** Death cause 2 really
was an artefact of squeezing. And it changes nothing, because on those same 25
cases `vrel` rose in **25 of 25**:

    quality bracket gained  +2.5564 %
    violation cost paid     +6.8343 %     ratio **2.67 : 1** against
    NET                     +4.0602 %  (worse)
    paid in: boundary 67.3 %, grouping 32.7 %, MIB 0 %

| route to density | currency | rate against |
|---|---|---|
| L282 squeeze the committed layout | hpwl | 2.74 : 1 |
| L268 big-first commitment order | hpwl | 1.2 : 1 |
| **L283 generate a denser layout** | **violations** | **2.67 : 1** |

⇒ **The currency changed and the price did not.** A packing-time chain rule is
not blocked by wire — it is blocked by boundary and grouping constraints, and
must be measured against `vrel`, not `area_gap`. This is L279's identity
("preplaced boundary violations *are* the density deficit") measured from the
opposite direction and priced.

⚠️ The mechanism is NOT that the chain is built from pinned blocks: chain blocks
are boundary-constrained at 36.0 % against a 34.1 % base rate, **enrichment
1.06x**. The whole packing is loose, and the looseness is what buys
soft-constraint satisfaction.

### 3.2 The target, restated so it is falsifiable

Not "shorten the chain". It is:

> **a mechanism that raises utilisation while holding boundary and grouping
> violations fixed.**

The label proves such layouts exist — recomputed from the dataset, its
utilisation is **97.1 %** against our 84.5 %, and it buys that 12.6 pp with
**+0.03222** of vrel while driving both gaps to zero. It is on the same trade,
at a better rate.

### 3.3 L284 — and that target is not merely unaffordable, it is unreachable

L283's target assumed density is *available* and merely expensive. L284 measured
whether it is available, on the same 4200 layouts, with no new placer runs:

    utilisation   shipped **80.91 %**   pool max **85.36 %**   label **96.60 %**
    the pool closes only 28.3 % of the density gap to the label
    the shipped layout is already at the **86th percentile** of its own pool

    cost vs density, every layout binned relative to its case's selected one
      d_util  -8pp    -4pp    -2pp   **0**    +2pp    +4pp    +6pp    +8pp
      d_cost +12.52% +14.38% +9.65% **+7.35%** +10.61% +10.77% +11.33% +13.24%
      d_vrel +0.0202 +0.0271 +0.0232 +0.0193  +0.0310 +0.0350 +0.0438 +0.0467

**The cost curve is a U and its minimum is exactly where we ship**, with `vrel`
climbing monotonically above it. Density and violations are welded together
above 81 % for this packer and the portfolio already sits on the optimum. The
densest layout is the cheapest on **0/100** cases.

🔑 **L268 reproduced from cache**: its big-first ordering reached 85.2 %
utilisation and was called the largest such gain in the project's history; the
stock pool's own maximum is **85.36 %**. That mechanism was reaching somewhere
42 shipped profiles already reach.

⚠️ **A correction worth carrying**: the within-case correlation between
utilisation and cost is negative in 93/100 cases, which reads as "go denser".
It is wrong — 2331 pool layouts are sparser than the selected one and only 687
are denser, so the correlation is reporting the sparse side of a U. **A
correlation over a U-shaped relationship reports whichever side has more mass.**

⇒ So the answer is a statement about the packer, which is M27/L129.

**And say so plainly.** The previous handoff's §7 anticipated this: if
relocation closed, the quality side is exhausted at this placer. It has closed.
The remaining item is `M27`/`L129` — a different placer, priced at 1.745 against
the shipped 1.237, whose own memory names full GORDIAN alternation as the
unfinished work. Producing another knob instead would be the wrong answer.

## 4. Traps this session paid for (in addition to the previous handoff's list)

1. **The handoff's own literal instruction was unimplementable.** "Move `u` to
   ordinal `p` in the ordering" has no referent: the anchor's max-gap disjunct
   set is not a sequence pair (it can carry 3-cycles), and a literal 1-D reading
   makes `u` a full-height column. Measured: **0/117 coherent**, i.e. worse than
   the move it was meant to improve on. L281 §2 documents the substitution.
2. **The control needs the same search budget as the arm.** L281's first mover
   read +8.6e-05 from its own LP and +3.8e-03 after polish. Polishing only the
   arm would have reported the mechanism as **44× larger** than it is. The
   no-force LP turned out to be a fixpoint already, so the polish contributed
   nothing to the control — but that had to be measured, not assumed.
3. **`m64_flip_probe.py`'s default anchor is 6 % worse than what ships.** It
   points at `results_L3_port_top32_area.json` (1.3003) while the shipped placer
   is 1.2263, so headroom measured there is partly headroom already taken. Both
   anchors reproduce bit-exactly under `cost_eval`, so `--anchor` is a one-word
   fix — but the default is wrong.
4. **A necessary condition must be gated as one.** L281's certificate discards
   78 % of candidates without an LP. That is only legitimate because 50 rejected
   candidates were fed to the LP anyway and **0/50** were solvable
   (`l281_cert_gate.py`). Any prefilter that saves this much time needs that gate.

## 5. Files added this session

```
report    L281_RELOCATION.md  L282_CHAIN_SHORTENING.md  L283_DENSITY_CURRENCY.md
          L284_DENSITY_CEILING.md  L285_RUNTIME_VERIFIED.md  L286_PACKAGE_TRANSFER.md
probes    l281_reloc_probe.py       gate | census | probe | report
          l281_gate.py              anchor reproduction, both anchors
          l281_cert_gate.py         certificate soundness
          l281_liveness.py          binding vs vacuous, positions compared bitwise
          l281_why_infeasible.py    boundary equalities / bbox / neither
          l281_ordinal.py           the handoff's literal move, measured
tools     l281_saturation.py        critical-chain slack, in-set 100
          l281_oos_slack.py         the same quantity in-set vs OOS s1/s2
          l281_chain.py             how many blocks sit on the critical chain
          l281_prize.py             exact first-order wire prize, all 100 cases
          l281_relax_price.py       what buying the missing slack would cost
          l281_deploy.py            can a label-free ranking find the winner
          l281_band.py              heavy-band aggregation: gain vs LP wall time
          l281_verify_mover.py      re-score every reported gain from its positions
L282      l282_chain_gate.py        Gate 0: chain redundancy + frozen span, no LP
          l282_chain_probe.py       chain-targeted candidates, area-ranked
          l282_why_infeasible.py    boundary ties / bbox growth / neither
          l282_cache.pkl            413 LP solves
L283      l283_generate_vs_squeeze.py  re-score all 4200 pool layouts
          l283_cache.pkl               4200 scored layouts
L284      l284_density_ceiling.py      density ceiling vs label, cost-vs-density U
L285      l285_runtime_headroom.py     slowdown sweep vs real grader runtimes
          l285_{lp_on,lp_off,betacfg}*.json
L286      l286_transfer.py             cache join + LP re-application, liveness assert
data      l281_cache.pkl            census + 2900+ LP solves
          l281_*.log
memory    l281-relocation-red · chain-saturation-closes-topology-repair
          changed-relation-is-not-a-move · l282-chain-shortening-red
          density-is-paid-in-violations · l284-density-ceiling
          l285-runtime-verified · l286-package-transfer
```
