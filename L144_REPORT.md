# L144/L145 — the boundary axis, instrumented: scoring is closed, availability is the gap

Follow-on from `L140_REPORT.md`, which established that boundary is the only
violation family with held-out headroom (+2.44% movable / +4.49% all, on OOS s1).
This report answers *why* the violations exist and screens three mechanisms.

**Nothing in the shipped path was touched.** Every measurement runs on probe
binaries (`constructive_l144*.exe`); `constructive.cpp`, `constructive.exe`,
`bin/constructive_linux`, `optimizer_constructive.py` and every `results_*.json`
/ `*.pkl` were md5-verified unchanged at the end of the session.

## 1. The instrument

`constructive_l144.cpp` = `constructive.cpp` + `ICCAD_BND_TRACE` (stderr counters
only). Off-path gate: **612/612 byte-identical** to the shipping exe over 12 OOS
cases x the full 48-core pool, re-run after every edit.

Over 745411 boundary-item placements (24 OOS s1 cases, full pool):

    a compliant (bp==0) candidate EXISTED       584001   78.3%
    the chosen candidate WAS bp==0              584001   78.3%
    existed but was NOT chosen                       0    0.0%
    no compliant candidate at all               161410   21.7%

    the three repair passes fix                  57 of 161486   (0.04%)
    compaction, on selected layouts            2069 -> 2149     (+80)

🔑 **The greedy takes the compliant slot every time it has one** (heavy band:
7613/7616, so ">=99.96%" is the honest bound). This is why `ICCAD_BP_WEIGHT` is
inert in *both* directions — the decision the weight governs never happens. The
entire gap is that 21.7% has nothing compliant to choose.

## 2. Who blocks the slot (`constructive_l144b.cpp`, gate 408/408)

Classifying the obstruction for every failed placement:

| blocker | light band (n=21-28) | heavy band (n=101-104) |
|---|---|---|
| **another boundary block on the SAME side** | **48.7%** | **63.2%** |
| a preplaced block | 22.0% | 7.8% |
| a boundary block of a different side | 15.0% | 8.1% |
| a plain non-boundary block | 11.3% | 18.8% |
| no exact-edge candidate was generated | 3.0% | 2.1% |

**Candidate generation is exonerated** (3.0%/2.1%, and the clamp never fires):
`item_candidates` does emit the exact edge coordinate. Boundary items block each
other, on their own side, having each individually taken the best compliant slot
available at their turn.

⚠️ And the edge is often not merely fragmented but *full at that moment*: total
free length on the strip is short of the item's extent in **57.0%** (light) /
**47.6%** (heavy) of misses. Any reservation/contiguity mechanism can therefore
address at most **43-52%** of the gap.

## 3. Is it even satisfiable? (`l144_feas_probe.py`, 480 held-out cases)

Blocks on one side share that side's coordinate, so their intervals along the
edge must be disjoint — feasibility is 1-D interval packing. Validation first:
`v_bnd` recomputed from positions matches the official field on **480/480 cases**.

| regime | s1 sides infeasible | s2 |
|---|---|---|
| achieved shapes, achieved bbox | **1 / 960** | 2 / 960 |
| achieved shapes, free frame at the same bbox AREA | 0 / 960 | 0 / 960 |
| free aspect <= 2.5 (= the shipped LR 2.50 / TB 0.40) | 0 / 960 | 1 / 960 |

**The violating cases are not the infeasible ones**: 149/150 (s1) and 138/139
(s2) violating cases have zero capacity obstruction on any side, carrying 100.0%
of the violating weight. Sides that missed run at demand/capacity 0.663 versus
0.651 for sides that did not — statistically indistinguishable.

**Provably forced: 10/254 (s1) and 10/229 (s2) — 3.9% / 4.4%.** Two structural
families only: two preplaced blocks required onto the same side at different
coordinates (7 s1 / 8 s2, exact conflicts), and preplaced blocks pinning both
ends of one axis while the perpendicular side over-subscribes (1 each).

⇒ **96% of the boundary deficit is satisfiable, and neither frame proportion nor
block aspect is the lever** — every side already fits. What binds is *when* the
edge is claimed.

## 4. Where a compliant position gets lost (`constructive.cpp` audit)

Attribution of the 254 s1 violations by what could even reach them:

| class | count | reachable by |
|---|---|---|
| preplaced (position pinned) | **106 (41.7%)** | **no repair pass** — all three skip preplaced |
| pure-movable cluster member | 69 (27.2%) | `final_group_boundary_nudge` only |
| single, `cluster==0` | 68 (26.8%) | `final_boundary_nudge` + `final_single_edge_escape` |
| mixed (anchored) cluster member | 10 (3.9%) | none (`has_pre` -> `continue`) |

Two structural findings worth more than the repair passes:

* **Opposite-edge pairs inside one compound item are unsatisfiable in every
  frame.** An item holding both an L member (flush at `ox=0`) and an R member
  (flush at `ox+bw=it.w`) needs `x=0` and `x=fw-it.w` at once, i.e. `it.w==fw`,
  and `frame_candidates()` never sizes a frame to an item's width. Measured: 19
  pure-movable clusters, **24 violations, 14.7x the rate of other clusters**.
* **Every non-flush cluster member violates — all 28 of them** (a member whose
  offset leaves it inside the item's own bbox can never reach the frame edge;
  the negative candidate coordinate is clamped to 0). M71 already took −1.59%
  out of this channel; this is the residual.

Together **47 of 79 movable cluster violations (59.5%) are structurally
impossible at pack time** — a subset of the 21.7% that has nothing to do with
edge occupancy.

Also corrected here (both were wrong in `L140_REPORT.md` §8, now fixed there):
`compact_layout` accepts on `csc_of`, not `layout_score`, so one extra violation
costs only ~3.8% of `area + hw*hpwl`; and "existed but not chosen = 0" is
">=99.96%".

## 5. Three mechanisms screened

All three: new `.cpp`/`.exe`, flag default 0, off-path gate byte-identical
(408/408 each), single-profile A/B, then an independent adversarial verifier.

| mechanism | flag | first verdict | verifier |
|---|---|---|---|
| low-corner contiguity gradient | `ICCAD_BND_EDGE_RUN` | RED (mine) | — |
| **abut-or-corner (discrete)** | `ICCAD_BND_ABUT` | RED | **REFUTED** |
| largest-edge-demand-first within the bscore class | `ICCAD_BND_DEMAND_ORDER` | RED | CONFIRMED |
| boundary-aware compaction guard | `ICCAD_BND_COMPACT_SAFE` | RED | CONFIRMED |

### 🚨 5.1 The light-band trap, again

The abut screen was run on `specs[0:16]` and `specs[0:128]`. `m77._specs()`
returns cases sorted by **n ascending**, and scoring weights by `exp(n/12)`:

| slice | n | share of the 240-case weighted score |
|---|---|---|
| `specs[0:16]` | 21-28 | **0.0126%** |
| `specs[0:128]` | 21-84 | **2.74%** |
| `specs[192:240]` | 109-120 | **69.81%** |

Re-run at full strength on all 240 cases, solo:

| | cost | boundary | runtime ON/OFF |
|---|---|---|---|
| profile 0 | **+0.379%** | 523 -> 515 | 0.996x |
| profile 1 | **+0.503%** | 524 -> 512 | 1.003x |

Jackknife: the sign survives dropping **any** single case (min +0.259%). A
placebo (a different perturbation of the same branch, same 240 cases) gives
**−3.66%**, so the gain is mechanism-specific, not "any perturbation helps".

🔑 **A mechanism was declared dead on 0.0126% of the score.** This is the same
failure this whole session is about — the in-set/light-band distribution is the
one where the effect does not exist — and it is now on record twice.

⚠️ Known confound: turning `ICCAD_BND_ABUT` on also switches boundary selection
from soft-weighted (`BP_W`) to hard lexicographic. `ICCAD_BND_ABUT=1e-12`
isolates that switch and has not been run yet.

⚠️ Solo is not the deployable form: the shipped 48-core portfolio already carries
254 boundary violations versus 523 solo, i.e. **profile diversity alone removes
51%** of them, so the portfolio may absorb the gain. That is what the twin screen
in §6 measures.

## 6. Twin screen (L124 R3 protocol)

`l124_r3_scale.py --flag ICCAD_BND_ABUT --bin constructive_l144v1.exe --on-val
200 --cache l144_twin_cache.pkl --nmin 101`, heavy band, s1 and s2, plus the
cross-sample transfer run. **A fresh cache file — L124's own 60 MB cache is not
touched.**

Heavy band (n>100, 89.6% of the weight), 80 cases per sample, full 48-core pool
captured twice (flag OFF and ON) per case:

| K appended | s1 pick / s1 score | s2 pick / s2 score | **s1 pick / s2 score** |
|---|---|---|---|
| 4 | +0.2310% | +0.2445% | +0.0796% |
| 8 | +0.3729% | +0.2961% | **+0.0796%** |
| 16 | +0.5484% | +0.3233% | +0.1377% |
| all 51 (unaffordable) | +0.5484% | +0.3233% | — |

**Cross-sample transfer is 27%** (L124's twin transferred at 80-83%), and the
winner tally is `3,2,2,2,1,1,1,1` — nearly all 1s, which is exactly the
noise-fitting signature `[[l127-twin-screen-line-exhausted]]` recorded. Even the
K=all upper bound is +0.32% on s2, and it blows the RF budget.

### 6.1 The other deployment form: global overlay (M71-style)

The same cache holds the ON capture of every profile, so the RF-free form — flag
on for the whole pool, pool size unchanged — prices for nothing
(`l144_global_overlay.py`):

| | s1 | s2 |
|---|---|---|
| OFF (today) | 1.506759 | 1.506753 |
| **GLOBAL ON** | 1.501520 **+0.3477%** | 1.506017 **+0.0488%** |
| per-case oracle (ceiling of any gate) | +0.5484% | +0.3233% |
| cases better / worse / same | 21 / 12 / 47 | 14 / 25 / 41 |
| n>110 subset | +0.5355% | +0.2000% |

Both samples are positive — no L123-style sign flip — but **s2 is +0.0488%,
six times under the 0.30% OOS bar**, and the per-case oracle (which is not
decidable at run time anyway, M56/M79) tops out at +0.32%.

### 6.2 Verdict: RED for shipping, in both forms

Solo the mechanism is real and robust (+0.379%/+0.503% over 240 cases, jackknife
stable, placebo −3.66%). It does not survive deployment because **the portfolio
has already bought most of it**: the shipped 48-core pool carries 254 boundary
violations against 523 for a single profile, i.e. profile diversity alone removes
51% of them. A mechanism that removes ~1.5% of the solo violations (523 -> 515)
has little left to add on top of that.

🔑 **The generalisable finding: on this axis, solo gains transfer to the
portfolio at roughly one part in seven** (+0.38% solo -> +0.05% deployed on s2).
Any future boundary mechanism should be screened at the portfolio level from the
start; a solo A/B is not evidence of a deployable gain — which is the mirror of
`[[m77-*]]`'s "solo cost and portfolio value are not monotonically related".

## 7. What is left on the boundary axis (ranked, from the L145 audit)

The abut term attacks the *occupancy* half of the gap. The audit found three
populations it never touches, each with a bigger reachable count and each fixable
at construction time (no RF cost):

| # | mechanism | reachable | est. pp | why it is unattacked |
|---|---|---|---|---|
| **R1** | **preplaced extent floor** — make a boundary-constrained preplaced block's coordinate a hard bound on the movable packing region for that axis | **100 of 108** preplaced side-misses | **≈ +1.90** | no repair pass may touch preplaced (41.7% of all violations); the packing region always starts at (0,0) and `frame_candidates()` has no origin |
| **R2** | **opposite-edge pair in one compound item** — append `it.w` / `it.h` as frame candidates when such an item exists | 24 (19 clusters, 14.7x the rate of others) | ≈ +0.40 | provably unsatisfiable in every frame the placer can build today |
| **R3** | **non-flush cluster members** — repair the EXPOSE snap variant instead of discarding it whole on any overlap (`constructive.cpp:649`) | 28 (100% violation rate) | ≈ +0.46 | M71 took −1.59% from this channel; this is the residual |

R2 ∪ R3 = 47 of 148 movable violations (31.8%). ⚠️ Every estimate above is an
upper bound that assumes removing the violation changes nothing else, and §6.2
says to discount any solo estimate heavily — R1's estimate in particular is a
re-pack, not a nudge (median 18 blocks / 25.2% of area currently sit beyond the
pin), so it will move area and hpwl.

## 8. Files

New only: `constructive_l144{,b,v1,v2,v3}.cpp`/`.exe`, `l144_bnd_trace.py`,
`l144_edge_run_ab.py`, `l144b_gate.py`, `l144_feas_probe.py`, `l144v{1,2,3}_*.py`,
`l144_verify0_*.py`, `l141_edge_capacity.py`, `l143_edge_occupancy.py`,
`l142_arm_cmp.py`, and the logs `l144_*.log`.
