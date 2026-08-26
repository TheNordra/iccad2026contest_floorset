# Handoff 2026-08-28 — shipping is CLOSED; this is the research handoff

**Read `HANDOFF_2026-08-27.md` first for the shipping story.** This file is only
about the open research question, and it starts by telling you what NOT to
re-open, because this session spent most of its time re-closing axes that
earlier documents had left ambiguous.

---

## 0. The shipping state — do not touch it

| | |
|---|---|
| uploaded | `build_submission.D/cadc1075.tar.gz`, **408,795 B**, in the Drive **Final** folder `1FDF1doINpSBKcr2OpL9PI19H04YLXyC4` |
| identity | `op_wrapper.py` md5 **`1c326784de7cd9246cd1f380e2842668`** |
| 48c Linux | **1.2264069637381392**, feasible 100/100, LP on 71 block counts |
| projected | NET **+5.224 %** vs beta, graded **0.87818**, **rank 2**, margin over r2 **1.08 pp** |
| verified | 10 in-set arms ALL PASS · 5 Linux lanes ALL PASS · 20-clause compliance check 19/20 (the 1 is the nt-gated msys path, which was also in the M73 package that was graded for Beta) |
| commits | `0f207ec` → `366fa3d` → `6280824` on `l113-route-a` |

**Verified round-trip**: the file downloaded back from the Drive is byte-identical
to the local artefact on all six graded files.

---

## 1. 🚨 The one thing that reframes the whole project

Nobody had decomposed the beta leaderboard into quality and runtime. Do it and
the strategy inverts:

| rank | total | raw = quality | cwRF = total/raw | runtime | quality vs us | on the floor |
|---|---|---|---|---|---|---|
| **1** | 0.85863 | **1.0845** | **0.7917** | 110.9 s | **−17.9 %** | no |
| 2 | 0.88819 | 1.2077 | 0.7354 | 110.7 s | −8.6 % | no |
| 3 | 0.89933 | 1.2848 | 0.7000 | 24.5 s | −2.7 % | yes |
| **4 (us)** | 0.92659 | **1.3207** | **0.7016** | 52.1 s | 0.0 % | **yes** |
| 10 | 1.05983 | 1.5140 | 0.7000 | 14.1 s | +14.6 % | yes |

**Rank 1 runs 2.13× slower than us and still wins by 7.3 pp, on quality alone.**
Our raw is the second-worst in the top ten. Across the top ten, **quality spans
40 %** (1.0845 → 1.5140) while **the runtime factor spans at most 12 %**
(0.70 → 0.89) — and we have already taken essentially all of the runtime axis.

    to reach rank 1 at our cwRF 0.70423:   raw <= 1.2192
    our projected raw:                     raw  = 1.2470   -> 2.2 % of QUALITY
    with rank 2's quality and OUR runtime: 1.2077 x 0.70423 = 0.85050 -> rank 1

🔑 **The runtime work is banked, not wasted** — we are 4.4 pp better than rank 2
on cwRF. What is missing is 2.2 % of placement quality.

⚠️ And "spend runtime to buy quality", which is rank 1's playbook, is **already
priced on our package and rejected**: gate at 100 on (LP everywhere) scores
+3.659 % / 0.89268 / **rank 3**, worse than the shipped 71. We lack a converter.

---

## 2. What this session CLOSED, each with a number

Do not re-open any of these without new evidence of the specific kind named.

| axis | verdict | the number |
|---|---|---|
| LP speed (coverage side) | closed | gate 100 on = rank 3; 71 is the optimum |
| LP speed (rate side) | closed | ceiling at **infinite** LP speed is graded 0.85752 — beats r1 by 0.0011, i.e. **inside the model's own ±0.3 pp error bar**. At 3× further it is still rank 2 |
| LP Python half | banked 1.170× | Python half 1.553×, ceiling 1.48×; every remaining item < 0.05 pp |
| LP solver method | closed | scipy's `"highs"` already resolves to dual simplex (`highs` == `highs-ds` to 0.1 %, 0/12 layouts move); `highs-ipm` is 5 % slower |
| HiGHS `devex` | measured, declined | OOS GO (s1 +0.0002 %, s2 +0.0240 %) but only **+0.03…+0.06 pp**; wired as `ICCAD_LP_EDGE_WEIGHT`, default off, verified a no-op |
| `prune_B` 8→4 | 1.055× | ≈+0.08 pp; one case's objective moved 9.1e-3 (freeze-set path) and would have to be ruled out |
| repair rounds (L247) | closed | 25.7 % of builds are repairs and the forced set IS predictable (smallest-margin 10 % pre-empts 64 %), but that costs +7.5 % of rows on **every** build → best ≈1.07×, +0.10 pp |
| pool additions (L248) | closed, with a CURVE | K=6 costs **9.2 %** of heavy-band wall for +0.364 % quality = **NET −1.03 pp**; every K negative. Closes L125 beam twins and further L124 twins too |
| boundary overlay (L249) | closed | `ICCAD_BND_ABUT` as a global overlay: **+0.0039 %** on the portfolio and **0 movers of 20 above n=100** |
| mid/heavy REFINE | done | both bands at 2; the mid band is now at the floor (5.5 % of the remaining deficit) |
| REFINE 2→1 | dead | L219: +21.77 % vs +21.52 % of wall — a quarter of a percent |

**Remaining RF deficit: 0.601 pp**, 93.8 % of it above n=100 and **46.5 % of it in
the single case n=112** (t/med 0.385 against a floor at 0.3046, and its LP gate is
*off*, so it is the pool, not the LP).

🚨 **A wrong argument worth remembering**: I reasoned that the pool is max-bound
until ~65 profiles (`sum/48 = 2.501 s` vs `d_max = 3.204 s` at n=120) so the first
additions would be nearly free. **That is wrong** — 51 profiles on 48 cores is
*already* oversubscribed, so six more do not fill idle slots, they grow the second
wave from 3 to 9, on top of L167's serial proxy tax. Measured, not modelled.

---

## 3. The research question, and where it stands

**The gap is generation, not selection.** Measured on the shipped 51-profile pool,
OOS s1, heavy band (n ≥ 101, 40 cases, weighted `exp(n/12)`), true cost via the
official strict scorer with label baselines (`l250_selection.py`):

    proxy pick (what we ship)   1.511619
    oracle over the same pool   1.511432   SELECTION loss  +0.0124%
    the LABEL itself            1.245233   GENERATION loss +17.6124%

    the proxy picks the true best on 39/40 cases; its worst miss ranks 3rd of 51

So **CLAUDE.md:488 holds on the current tree** — a better selector buys nothing,
and the whole deficit is that the pool does not *contain* better layouts.

🔑 **And 17.6 % is the same order as rank 1's 17.9 % lead over us**, which puts
**rank 1 roughly at the label's level**. Whatever they do, it reconstructs close
to ground truth.

### 3.1 Which term the 17.6 % lives in (L251) — and one axis flips sign

Same 40 cases, pricing "set this ONE term to the label's value, leave the others":

| term | ours (median) | label | worth if zeroed |
|---|---|---|---|
| **hpwl_gap** | **0.2766** | 0.0000 | **+11.5708 %** |
| **area_gap** | **0.2256** | 0.0000 | **+9.1807 %** |
| vrel | 0.0857 | 0.1061 | **−3.8753 %** |
| all three | — | — | +17.6226 % |

(The terms multiply, so they do not sum. Feasible 40/40 both sides.)

🚨 **The violation axis is not a deficit — it is a surplus. We beat the label on
vrel in 31 of 40 cases**, and forcing our vrel to the label's value would cost us
**3.88 %**. L128 measured this term at **+3.57 % positive** on the pre-L124 tree;
L124's MIB twins plus L131/L136's correctness fixes have since overtaken the
ground truth. **This retires CLAUDE.md's "next step 4 (violation axis)" as a
source of gain against the label** — it is already won, and any further work
there is fighting for the residual against ourselves, not against the label.

**The deficit is hpwl (+11.57 %) and area (+9.18 %), and they are coupled.**
`area_gap = 0.2256` is the same fact CLAUDE.md records as "our utilisation 82.2 %
vs the label's 96.6 %": our layouts are **22.6 % larger in bbox and 27.7 % longer
in wire** than ground truth. A bigger outline spreads blocks apart, which is most
of why the wire is longer — so these are not two independent 10 % prizes.

Captures are cached in `l251_cache.pkl` (keyed `(sample, case_key)`), so any
follow-up decomposition is free.

---

## 4. 🚨 The trap that ate an hour, and will eat yours

`m67_oos_probe.py:61-63` **deletes every `ICCAD_*` from the environment at import
time** ("shipped defaults only"). So:

```python
os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
import m67_oos_probe as m67          # <- silently undoes it
import optimizer_constructive as oc
oc._pool_indices(120)                # -> 13, not 51
```

`_effective_cores_hi()` then reports this box's 32, **all four ≥40-core tiers
switch off**, and the pool is 13 instead of 51. The first run of `l250` measured
that 13-profile pool and printed a completely plausible table (selection loss
+0.0006 %, generation +21.17 %) — the numbers only looked wrong because the
candidate count was printed next to them.

**Set the env AFTER importing m67, and assert the pool size.** `l250`/`l251` both
now refuse to run if `len(_pool_indices(120)) != 51`. This is the third appearance
of this family (`[[probe-import-time-silent-nooks]]`).

Related: `l124_r3_scale._capture` keys profiles by `tuple(sorted(env.items()))`,
so if you set the screened flag to a value that makes a twin identical to its
host, the two collapse and you silently capture fewer candidates.

---

## 5. Where to go next

The three perfect-information bounds (perfect ordering **+0.005 %** M26, perfect
seed **+0.001 %** M68, perfect shape **+0.099 %** M79) say every *per-block*
decision handed to this packer is exhausted, and L250 now adds that *selection*
over the pool is exhausted too. What is left is the packer's reachable set.

Ranked by what I would try, with the reason each is not already dead:

1. **Finish L251.** Which term the 17.6 % lives in decides everything below. If
   it is hpwl-dominated the lever is topology; if area-dominated, the frame.
2. **The FRAME search has never had a perfect-information bound, and L251 now
   points straight at it.** M26/M68/M79 bound ordering, seed and shape — all
   *given* the frame. `frame_candidates()` chooses the outline, `area_gap` is
   0.2256, and nothing downstream can fix a wrong outline. The probe is M79's
   shape: feed the label's bbox as the only frame and measure.
   ⚠️ **Strong prior against it, from CLAUDE.md**: the teammate's
   `oracle_pack_ceiling.py` did exactly this on *their* packer — with the true
   bbox, the 42 cases that fit scored **1.017 each (better than fp_sol
   verbatim)**, but **58 % could not be packed at all**, and any slack that made
   them fit destroyed quality (×1.15 → 98/100 but hpwl 0.4511). *"The cliff is
   not a slope."* If our greedy behaves the same way, the frame is not the lever
   and the real statement is **the packer cannot reach the label's 96.6 %
   density** — which is M27 from the other side. Measuring it on OUR packer is
   still worth one probe, because that result was on a different packer and the
   42 % that DID fit scored better than verbatim.
3. **The label's own layouts are in the pool's reach?** L128 says topology cannot
   be transplanted (perturb 2 % and it breaks) — but that measured *transplant*,
   not *imitation*. The question L128 did not answer: how far in edit distance is
   our best layout from the label's, and is there a monotone path?
4. **Do NOT** re-run: anything in §2; any fp_sol-supervised ML (user ruling
   2026-08-05 — offline oracle probes with labels are fine, and that is what
   L250/L251 are); pool pruning; `ICCAD_ANCHOR_W` sweeps.

## 6. Files added this session

```
l230_calib.sh l230_gate.py            the LP-gate re-derivation (4 arms x 3 reps)
l231_midband.sh l231_score.py         the mid-band REFINE sweep
l232_pool.sh l232_score.py            pool restore pricing (all RED)
l233_oos.sh l233_score.py             mid-band OOS + GO/NO-GO
l234_implement.sh                     mid band + gate adds, both kill switches
l235_patch.py l235_lpbench.py         the LP Python rewrite + its identity gate
l235_timers.py                        the phase-timed probe
l236_gate.py l237_ship.sh             pricing and shipping the speedup
l238_gates.sh l238_verdict.py         the 10-arm in-set suite
l238_wsl_final.sh                     the five Linux lanes
l239_solver.py l240_highsopts.py      solver method / HiGHS options
l241..l243                            devex, measured and declined
l244_variantD.sh l244b_wsl.sh         variant D built and Linux-verified
l246_compliance.py                    20-clause check of the UPLOADED artefact
l247_patch.py                         repair-round structure
l248_patch.py l248_score.py           the lens-D size curve
l249_bndabut.sh                       the boundary overlay
l250_selection.py                     SELECTION vs GENERATION  <- start here
l251_terms.py                         the per-term decomposition (in flight)
```
