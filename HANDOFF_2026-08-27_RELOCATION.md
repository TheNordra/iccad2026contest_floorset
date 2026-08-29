# Handoff 2026-08-27 — one task: the RELOCATION probe

You are starting the last concrete experiment on the quality side. Everything else
this session touched is closed, and §3 is the reason there is only one item left.

**Read `L276`, `L280` §5, and `L275` before writing code.** §5 of this file is the
methodology those three established; ignoring it is how the previous arc lost ~25
experiments to a corpus error.

---

## 0. Shipping state — frozen, do not touch

| | |
|---|---|
| uploaded | `build_submission.D/cadc1075.tar.gz`, Drive **Final** |
| identity | `op_wrapper.py` md5 **`1c326784de7cd9246cd1f380e2842668`** |
| source | `constructive.cpp` md5 **`e2c7b2f418ef2b70b6bff99f7adfbd37`** (unmodified all session) |
| 48c Linux | **1.2264069637381392**, feasible 100/100 |
| graded | **0.87818**, **rank 2**, margin 1.00 pp over r3 |
| local anchor | in-set 100 @48c weighted **1.226325** (`results_L274_base_48c.json`, carries positions) |

L274 built a full replacement package from `l269p1`, passed both equivalence
gates, measured it, and **rejected it** (+0.0245 % on the graded shape). Do not
re-open packaging without a mechanism that is positive on the graded corpus.

⚠️ WSL Ubuntu **does** exist now (unlike the L137-era note), but it carries glibc
2.43 against the shipped ELF's 2.34 floor. A rebuild needs `-static` (verified
working, 1.49 MB, smoke passes). That is a link-model change to the shipped
artefact and has never run on the grader.

## 1. THE TASK — unit RELOCATION in topology space

**Move one unit to a different position in the ordering — which flips every pair
involving that unit at once — then re-solve the LP and score it.**

### 1.1 Why this specific move, and why now

`L276` measured that the LP, minimising the official 0.5/0.5 hpwl+area objective
*inside our own topology*, removes only **0.9–1.3 %** of `hpwl_gap` while removing
3.8–4.6 % of `area_gap`. Wire is an **adjacency** problem, not a coordinate one.
hpwl is **56 %** of the graded headroom.

Every non-topology attempt has now failed, and all three failed by making hpwl
*worse*:

| attempt | lever | hpwl |
|---|---|---|
| L272 hint → wire term | information at scoring time | 0.2924 → 0.2999 |
| `ICCAD_GUIDE_MED=1` | candidate at scoring time | 0.2484 → 0.2538 |
| L280 mutual-top-1 grouping | **commitment** | 0.2484 → **0.2860** |

### 1.2 🚨 The correction that makes this worth doing

The ledger (and my own first draft of L276) called M64 *"single-pair topology
flips, 529 flips, 0 movers"*. **That is wrong.** `m64_flip_probe.py`'s docstring:
a target is a **UNIT pair**, and *"ALL block pairs spanning the two units get their
separation row REPLACED by direction k"*. M64 was already a coordinated multi-pair
move. So "multi-pair coordinated exchange" is **not** the untried thing.

M64's result: 529 attempts, 0 movers, **459/529 = 86.8 % LP-infeasible**.
`M64_REPORT.md` §4 attributes that to the fixed-disjunct chains of the other
~3000–4900 pairs plus envelope/bbox geometry, and *disproves* boundary equalities
as the cause (15 infeasible attempts re-solved without them → **0/15** feasible).

🔑 **But the probe's own HONEST-SCOPE note says forcing one direction on every
member pair means "slide all of A past all of B on that axis — stronger than the
evaluator's pairwise requirement; a mixed per-pair topology could be feasible where
this is not."**

⇒ A large share of that 86.8 % may be **self-inflicted by the move's semantics**.
A relocation in a sequence-pair ordering is, by construction, a *realisable*
topology — it cannot produce the contradictory constraint set M64 imposed on
itself. That is the whole thesis of this probe, and the first thing to test.

### 1.3 What already exists (do not rebuild it)

```
m64_flip_probe.py   33 KB, OFFLINE, never shipped
  build_and_solve_flip(ci, P, freeze_units, force_rel=...)   :110   force_rel is a
                                                  dict keyed by unit-PAIR -> direction
  lp_pass_flip(...)                                          :291
  cost_eval                                                  imported from m53_l3_probe
                                                             (official strict scorer)
  modes: selfcheck | pilot | ...
```
`selfcheck` forces the CURRENT direction of a homogeneous pair and must be a no-op
— run it first, it is the wiring proof.

### 1.4 Suggested first step, and the kill criterion

1. **Run `m64_flip_probe.py selfcheck`** unchanged. If it is not a no-op, stop and
   fix the harness before anything else.
2. **Add a relocation move**: for unit `u` and target ordinal `p`, emit the
   `force_rel` entries for *every* pair `(u, v)` implied by moving `u` to position
   `p` in the ordering — i.e. `u` before `v` for all `v` now after it, and after
   `v` for all `v` now before it. That is a coherent ordering, not an arbitrary
   constraint set. Everything downstream (`build_and_solve_flip`, the LP, the
   prefilter, `cost_eval`) already accepts a `force_rel` dict.
3. **Measure the infeasibility rate first, before any cost number.** That single
   number decides the thesis:
   * infeasible ≈ 86 % like M64 → the wall is the instance, not the move semantics.
     **Report it and close the axis.** This is a genuine and publishable negative.
   * infeasible materially lower → the move is realisable where M64's was not, and
     the cost distribution becomes worth measuring.

**Do not** spend LP time on cost until the feasibility question is answered — it is
cheaper, and it is the fork.

### 1.5 Honest prior

The re-audit that proposed this estimated *"0 … +0.3 % in-set, most likely ≈0, with
a fat right tail"*. §1.2 argues the prior should be better than that, but nobody has
measured it. Budget it as a real build on a 33 KB LP tool, not as a knob.

## 2. Anchors and corpora

| | file / value |
|---|---|
| in-set 100 @48c, current shipped code | `results_L274_base_48c.json` (positions + `violations_relative`) |
| weighted total | 1.226325 · gaps hpwl 0.2484 / area 0.1355 / vrel 0.0141 |
| in-set determinism | two baseline runs differ on **0/100** cases |
| OOS heavy 40 (s1) | `l252_cache.pkl` + `l271_quality.py --limit 40` |
| graded-shape headroom | hpwl **−10.41 %**, area −5.67 %, vrel −2.81 %, all −18.46 % |

One 100-case official eval, ~10 min:

```
cd iccad2026contest
ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN=<probe> <flags> \
  python iccad2026_evaluate.py --evaluate ../optimizer_constructive.py -o ../out.json
```

## 3. What this session closed (do not re-open)

| axis | verdict | where |
|---|---|---|
| the whole L250–L274 frame/density arc | **corpus artefact** — 4/4 sign flips on the graded shape | L275 |
| LP depth k=2/4 | NET −0.95 % / −3.09 % on current medians | L276 §2 |
| violation axis, post-hoc snap | +0.0012 % — the weighting defeats it | L277 |
| preplaced boundary violations | = the density deficit; needs a 14.4 % bbox shrink | L279 |
| connectivity grouping | **+4.6 % on both corpora**, all three terms worse | L280 |
| `ICCAD_BND_ABUT` | removes **0** in-set violations; its +0.0039 % is hpwl | verified from `results_L249_abut.json` |
| MIB twins removal | **undetermined — do NOT drop them** | L278 |

## 4. 🚨 Traps this session paid for

1. **`l252_identity.py` is value-blind.** It sets every flag to `"1"`. A binary that
   accepts only a different value reads as arm-B PASS — i.e. "silent no-op" — when
   the flag was never set. Use `l271_liveness.py`, which runs the arm with its real
   values and demands a traceable reason for every byte-identical pair.
2. **Reason-taxonomy order matters.** Testing "no failed frame → EMPTY" before
   "retry lost → LOST" understated a blast radius by 2×.
3. **`is_fixed` pins the SHAPE, `is_preplaced` pins the POSITION**
   (`constructive.cpp:1745-1747`). Treating `fixed` as immovable made 10/23 rows
   come out both HARD and label-satisfied — impossible, and that self-contradiction
   is the only reason it was caught.
4. **`l146_rf_price.py` reads the 2026-08-19 medians.** They were republished
   2026-08-23, all 100 lowered (p50 ×0.7418). It understates an RF bill by ~6.6×
   and flipped LP k=2 from NET −0.95 % to +0.12 %. Use **`l276_price.py`**.
5. **Local runtime cannot see the 48-core wave boundary.** This box has 32 physical
   cores; `ICCAD_ADAPTIVE_CORES=48` changes pool *selection* only. 51 profiles on 48
   cores is two waves, 43 is one — invisible here.
6. **Ambient `ICCAD_FRAME_SCALES` is a no-op** for the 44/55 profiles that set it
   themselves (`env.update(env_over)` — the profile wins). `ICCAD_L2xx` flags are
   set by no profile, so ambient does reach the binary.
7. **Local eval forces RF = 1.0**, so quality columns are RF-free. A mechanism that
   trades runtime for quality cannot be judged from them.

## 5. The three methodology rules — these are the session's real output

**(a) L275 — measure on the corpus that gets graded.** OOS s1/s2 heavy is **+22–24 %
harder** than the same band in-set and carries **+70 % more area_gap** and 6.3× the
vrel; beta hidden sits ≈2.4 % from in-set. Any mechanism whose value scales with a
gap is overstated there. Measure **both** corpora from the first arm.

**(b) L278 — a corpus can only vote on a mechanism whose antecedent it contains.**
In-set MIB = 0 (100 % of in-set MIB groups collapse to one shape; only 2.5 % of
held-out ones do), so the MIB twins' in-set null **could not** have been positive.
Zero instances is absence of evidence, not evidence of no effect, and the two look
identical in a portfolio delta. **Always report the antecedent count next to the
delta** — L280 did: 1276 pairs, 100/100 cases, 36.2 % of blocks.

**(c) L280 — corpus sensitivity is a property of the mechanism.** L280 read +4.6163 %
in-set and +4.6696 % OOS — agreement to 0.05 pp — because it does not harvest a gap,
it removes placement freedom, and that costs the same everywhere. A mechanism that
reads the same on both corpora is telling you which class it is in.

Corollary for the relocation probe: it changes topology to buy wire, so it is a
*gap-harvesting* mechanism and **must** be checked on both corpora.

⚠️ And the axes are **not mechanistically additive** (L279): 23 of 59 boundary
violations are redeemable only by closing the area gap, so the vrel and area prizes
overlap by +1.19 %. If relocation works, count its violation and area effects as
part of *its* prize, not as separate ones.

## 6. Files added this session

```
reports   L271_L272 · L274_SHIP_DECISION · L275_CORPUS_MISMATCH · L276_HPWL_IS_TOPOLOGY
          L277_VIOLATION_AXIS · L278_TWINS_UNDETERMINED · L279_PREPLACED_IS_DENSITY
          L280_GROUPING_RED
probes    l271_patch.py (l271/l272/l273.exe) · l280_patch.py (l280.exe)
          l274_ship_patch.py + l274_gate.py + constructive_ship.exe (the rejected package)
tools     l271_liveness.py   liveness with an ordered reason taxonomy
          l271_quality.py    arms + FREE post-LP deployable column
          l271_exchange.py   exchange-rate predictor (reproduces arms to <=0.031 pp)
          l276_price.py      exact RF on the 2026-08-23 medians, from a measured dt
          l277_vio_prize.py  violation inventory + CLEAR-and-soft upper bound
          l277_snap.py       post-hoc boundary snap + same-path control
          l279_preplaced.py  HARD / label partition
data      results_L274_base_48c.json (the anchor) · results_L275_* · results_L276_k{2,4}
          results_L278_notwin.json · results_L280_inset.json · l280_oos40.pkl
memory    l271-no-constant-still-needs-s2 · l274-ship-decision-keep-D
          l275-arc-priced-on-wrong-corpus · l276-hpwl-is-topology
          l277-violation-axis-graded · l278-corpus-can-only-vote-if-antecedent-present
          l279-preplaced-is-density · l280-grouping-red
```

Nothing owned by the concurrent session was modified.

## 7. If relocation also closes

Then the quality side is exhausted at this placer, and what remains is
`M27`/`L129` — a different placer, priced at 1.745 against the shipped 1.237, whose
own memory names **full GORDIAN alternation** (solve → partition → re-solve under
region constraints → re-partition) as the unfinished work. That is a project, not
an experiment. Say so plainly rather than producing another knob.
