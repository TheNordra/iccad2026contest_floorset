# L140 — the soft-violation audit on HELD-OUT data: MIB is a mirage, boundary is real

Handoff 2026-08-19 §5.2: *"start by re-running `l135_soft_audit.py` against an
OOS sample rather than the in-set 100 — the in-set audit found only 16 grouping
and 78 boundary, and the OOS vrel is 6.9x higher, so the same audit there should
surface far more."* It does — **2.7x more boundary and 944 MIB violations that
in set do not exist at all** — but the largest family turns out to be
arithmetically unreachable, and the reachable one is not where the in-set audit
said it was.

Everything here is READ-ONLY. The submission is untouched.

## 1. What was built

| file | what it does |
|---|---|
| `l135_soft_audit.py` | refactored into `audit_case()` / `report()` so the SAME audit code serves both corpora. In-set output is **bit-identical** to before the refactor (18 grouping / 56 boundary / corridor CLEAR 2/17 / slide CLEAR 21/56). Two columns added: boundary rows are now classified `preplaced`/`MARGIN`/`soft`, and MIB groups get a **provable floor** |
| `l140_oos_soft_audit.py` | `run` (solve the OOS sample with the SHIPPED optimizer, dump positions + official per-case metrics), `audit` (replay the L135 audit on that dump), `inset` (same audit + pricing on an in-set results json, so both columns come out of one code path) |
| `l141_edge_capacity.py` | is a boundary miss a CAPACITY failure (the required blocks do not fit along the edge) or a PACKING failure (they fit and were not put there)? |

Cases and loading are `m77_oos_probe._specs` + `m67_oos_probe._load_case`, i.e.
the same 240 s1 cases every historical OOS number used, plus the disjoint s2.
Runs are at `ICCAD_ADAPTIVE_CORES=48` (the grader's pool shape — M76: OOS numbers
move 2.7x with pool shape), on the shipped L136 tree with the hint OFF.

**Validation.** Per-case boundary/grouping/MIB counts from the audit match the
evaluator's own counts on **all 580 cases, zero mismatches** (the audit asserts
this case by case), and the in-set weighted total it reconstructs, 1.228474,
is the graded L136 anchor.

## 2. The violation mix inverts out of sample

| | in-set 100 | OOS s1 240 | OOS s2 240 |
|---|---|---|---|
| weighted cost | 1.228474 | 1.467038 | 1.471125 |
| weighted vrel | 0.014039 | 0.085869 | 0.085773 |
| boundary | **81.1%** (56) | 26.1% (254) | 21.1% (229) |
| grouping | 18.9% (18) | 6.7% (58) | 4.2% (54) |
| **MIB** | **0.0%** (0) | **67.2%** (944) | **74.6%** (976) |
| cases with ≥1 violation | 53/100 | 240/240 | 240/240 |

🔑 **The dominant OOS family has exactly zero in-set instances.** No amount of
in-set auditing could have found it — this is handoff §4's point in its sharpest
form, and it is the reason this audit had to move corpora rather than get more
careful.

⚠️ Two bookkeeping corrections to handoff §4 while we are here:

* the in-set audit numbers quoted there (16 grouping / 78 boundary) are from the
  older L114 anchor; on the shipped L136 anchor they are **18 / 56**;
* the OOS vrel quoted there (0.0967) is at the local default core count. At the
  **grader's 48c pool shape it is 0.0859** (s1). The 6.9x becomes **6.1x**.

## 3. 🚨 Two thirds of the OOS violation mass cannot be removed

The MIB rule is `Σ(distinct shapes − 1)` per group, and identical shapes imply
identical **areas** — while soft-block area is a **hard** ±1% constraint. So two
members can share a shape only if their target-area windows intersect:

    a_hi / a_lo <= 1.01 / 0.99 = 1.0202

Greedy interval cover over each group's target areas gives the **minimum**
number of distinct shapes, hence a provable lower bound on that group's
violations. Measured:

| | s1 | s2 |
|---|---|---|
| MIB violations achieved | 944 | 976 |
| provable floor | **933** | **955** |
| recoverable | 11 | 21 |
| groups above the floor | 11 / 240 | 21 / 240 |
| target-area span, median | 5.44x | — |
| groups collapsible to ONE shape | **0.0%** | — |

Priced on the official weighted total:

| | s1 | s2 |
|---|---|---|
| MIB → 0 (**unreachable**) | +10.9483% | +12.0518% |
| **MIB → provable floor** | **+0.0941%** | **+0.2322%** |
| boundary → 0 | +4.4910% | +3.5998% |
| grouping → 0 | +1.1589% | +0.7406% |
| all soft → 0 | +15.9459% | +15.8550% |

🔑 **`vrel → 0` is worth 15.9% out of sample and 12.1pp of that is arithmetically
impossible.** Handoff §4's "on OOS violations are the LARGEST axis (17.58%)" is
true of the *measurement* and false as a *target*. This is the same correction
`[[soft-violations-are-the-big-lever]]` made in set for MIB, now measured on 480
held-out cases at the current tree: **L124's bucketing already took this family
to within 1.2–2.2% of its floor.**

⚠️ The floor is a lower bound on the true floor: it assumes free aspect choice
and ignores locked (preplaced/fixed) members, whose shapes cannot be chosen at
all. A locked-aware floor can only be higher, so "recoverable" is an upper bound.

## 4. Grouping is closed for the same reason it was in set

58 / 54 violations, worth +1.16% / +0.74%, and they are geometrically
unavailable: the corridor between the split components is **blocked on 53/54
(s1) and 48/52 (s2)** split groups. The ULP family that L131 fixed has exactly
**one** residual instance out of sample (gap 1e-9, worth +0.0001%) — consistent
with `[[l131]]`'s finding that the family is closed.

## 5. 🏆 Boundary IS reachable — and it is a packing failure, not a geometry one

Blocks required to touch the left edge all sit at `x = x_min`, so their
y-intervals are disjoint and `Σh_i ≤ H` is a hard property of the achieved frame.
That makes the same "provable floor" question askable per side:

| | in-set | s1 | s2 |
|---|---|---|---|
| side-misses | 73 | 307 | 276 |
| **forced by capacity** | 2 | **1** | **2** |
| packing-avoidable | 71 | **306** | **274** |
| demand/capacity, median | 0.669 | 0.656 | 0.650 |
| sides over capacity | 2/400 | 1/960 | 2/960 |

🔑 **The blocks that must touch an edge occupy about two thirds of it, and 99.6%
of the misses are on sides with room to spare.** Unlike MIB and grouping, nothing
structural is stopping these.

And it is not a near-miss that a nudge could fix — the misses are **large**:

    boundary miss distance   median 27.56 units = 1.3552x the block's own size
    misses under 5% of block size                          1% of them
    misses under 1% of block size                          0  (prize +0.0000%)

which is why all three of L135's post-process repairs failed: by the time the
layout exists, the block is a block-width away from where it belongs. **This is a
construction-time decision.** Sub-prizes (s1):

| subset | count | worth |
|---|---|---|
| not preplaced (movable) | 148 | **+2.4448%** |
| preplaced | 106 | +2.0854% |
| slide path CLEAR | 73 | +1.1978% |
| CLEAR **and** movable | 50 | +0.8094% |

The preplaced half cannot move (hard constraint), but its violations are still
ours: the bbox is set by *our other blocks* overshooting past the preplaced
extent. Correlation with `area_gap` is weak (+0.20), so these are not merely a
symptom of area overshoot.

Violations are spread across bands, and the heavy band that carries 89.6% of the
weight has the most: 1.50 boundary + 3.89 MIB per case (n>100) vs 0.75 + 4.01
(n≤60).

## 6. What this changes

1. **Do not spend time driving MIB down.** 98.8% / 97.8% of it is forced by area
   heterogeneity; the whole remaining family is worth +0.09~0.23%.
2. **Boundary is the violation axis with headroom**: +2.44% (movable) to +4.49%
   (all) out of sample, on a mechanism that is provably available.
3. **It must be attacked in `constructive.cpp`, not in a post-process.** The
   misses are 1.36 block-widths deep and every post-process attempt in L135
   failed for that reason.
4. 🚨 **`BP_WEIGHT` is the first thing to re-price, and the ledger's verdict on it
   is in-set-only.** "BP_WEIGHT 雙向封卷: 30000→1M 無變化" was measured where
   boundary violations are 56 and worth +2.3%; out of sample they are 254 and
   worth +4.5%. Same trap as this whole session: an axis judged on the one
   distribution where it barely exists.

## 7. Reproduce

```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l140_oos_soft_audit.py run --sample s1 --cores 48 --out l140_oos_s1_c48.json
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l140_oos_soft_audit.py audit l140_oos_s1_c48.json --sample s1 --dump l140_audit_s1.json
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l141_edge_capacity.py l140_oos_s1_c48.json --sample s1
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l140_oos_soft_audit.py inset results_L136_48c_anchor.json
```

Each `run` is ~17 min for 240 cases (1009 s s1 / 1018 s s2, forced 48c on the
32-core box); the audits are seconds. Artefacts: `l140_oos_s{1,2}_c48.json`
(positions + per-case metrics), `l140_audit_s{1,2}.json` (violation rows),
`l140_run_s{1,2}.log`, `l140_audit_s{1,2}.log`, `l141_s2.log`, `l140_inset.log`.

## 8. L142/L143 — the boundary knob is dead in BOTH directions, and here is why

Four screening arms on the same OOS s1 subset (80 cases, 48c pool shape), plus a
geometry probe. `l142_arm_cmp.py` gates every arm on "did the positions actually
change" first, so a stripped knob cannot masquerade as a null result.

| arm | positions changed | cost | boundary | note |
|---|---|---|---|---|
| `ICCAD_BP_WEIGHT=1000000` | **0/80 — bit-identical** | +0.0000% | 60 → 60 | genuinely inert |
| `ICCAD_BP_WEIGHT=1` (20 cases) | yes, wholesale | −4.20% | 19 → **17** | penalty OFF does not add violations |
| `NO_PUSH+NO_SWAP+NO_JUMP` | 79/80 | −0.2078% | 60 → 61 | HPWL refinement is not the culprit |
| `NO_COMPACT` | 64/80 | −0.8522% | 60 → 58 | compaction costs 2 of 60, and fixes 9 grouping |

The `BP_WEIGHT=1` arm is the control that makes the 1e6 arm readable: the knob
**does** reach the placer (killing it moves cost 1.856 → 1.934 and hpwl_gap
0.305 → 0.381), so the 1e6 no-op is a real measurement, not
`[[probe-import-time-silent-nooks]]`.

🔑 **The penalty is inert upward AND downward.** Turning it off entirely leaves
the boundary count where it was. Whatever decides these violations, it is not the
scoring weight — which also means the ledger's in-set verdict on `BP_WEIGHT`
happens to survive out of sample, for a reason nobody had measured.

`l143_edge_occupancy.py` says why it cannot be a scoring problem, on the final
layout of all 307 s1 side-misses:

    edge FRAGMENTED (no contiguous gap big enough)     44  (14.3%)
    a big enough gap EXISTED                          263  (85.7%)
    largest free run / violator's own extent    median 2.036
    blocks holding constrained edges     2012 entitled vs 220 squatters (9.9%)

So the edge is not contested (90% of it is held by blocks that are themselves
boundary-constrained) and the gap is typically **twice** the size of the block
that needed it. Meanwhile `item_candidates` (`constructive.cpp:781`) *does*
generate the exact edge coordinate for any item with a boundary member.

⚠️ **Two corrections to this section, from the L144/L145 follow-up** (detail in
`L144_REPORT.md`):

* "compaction is downside-protected by `layout_score`'s `150000*bv`" is **wrong**.
  `compact_layout` accepts on `csc_of` (`constructive.cpp:1394-1402`), i.e.
  `(area + hw*hpwl) * exp(2*(bv+gf)/nsoft)`; the `150000*bv` term gates FRAME
  SELECTION only. With `nsoft` median 52, one extra boundary violation is bought
  by an area+hpwl drop of ~3.8% — which a directional pack clears routinely. That
  is the whole +80.
* "existed but not chosen = 0" holds on the light band but is 3/7616 on the heavy
  band, so the honest statement is **>=99.96%**, not 100%.

⇒ The remaining explanation is **timing**: at the moment that block is packed,
the edge slot is occupied; the gap that the audit sees only opens later, once
compaction and the rest of the pack have shifted things. A weight cannot fix an
alternative that does not exist yet, which is exactly why the knob is inert in
both directions.

**Next mechanism to try (a `constructive.cpp` change, not a knob):** give
boundary-constrained items priority in the pack order, or reserve the edge run
they need, so the compliant candidate is still free when they are placed. Gate
it, keep it bit-identical when off, and price it on OOS s1+s2 at 48c against
`l140_oos_s{1,2}_c48.json` — the +2.44% (movable) / +4.49% (all) upper bound from
§5 is what it is competing for.
