# L137 — the GORDIAN hint reaches the shipped placer: quality GREEN, net unresolved

Implements the 08-17 plan. The uploaded L136 submission is untouched, everything
is behind knobs that default OFF, and the hint-off path was re-verified
bit-identical to L136 on **every** iteration.

    in-set 48c   1.2284738198320346 -> 1.2279370648317838   +0.0437%
    OOS s1 240   1.563347           -> 1.561957             +0.0889%
                 hpwl_gap 0.313485  -> 0.311636
                 area_gap 0.256896  -> 0.252881
    48c wall     4.225s             -> 4.299s               +1.76%

**Quality is real, OOS-confirmed, and moves the terms the mechanism predicts.**
**Net after runtime is NOT resolved** and is not claimed either way.

## 0. Phase 0 — the gates were measuring nothing

`make_submission.py verify` and `l113_ship_gate.py` compared bit-exactly against
`results_L114_48c_lp_anchor.json`, i.e. the PRE-L136 artefact, so after the L131
and L136 fixes shipped they FAILED BY DESIGN on every case those fixes improved.
A gate that always fails is not a gate. Re-anchored to
`results_L136_48c_anchor.json` / `results_L136_default.json`; both now PASS (48c
cost+positions 100/100, 32c bit-exact). The old anchors are kept — they still
describe real shipped artefacts.

## 1. The lever

`estimate_anchors()` (`constructive.cpp:368`) anchors a block to the
connectivity-weighted centroid of its **already-placed** neighbours, and it runs
once, when only the PREPLACED blocks are down. A block with no preplaced
neighbour and no pin therefore gets `sw==0`, anchor weight 0, and **no pull at
all**. The file's own M9 header has said so since the beginning: *"the first
blocks are placed blind to HPWL."*

`_gordian_hint()` supplies what is missing — a globally optimised centre for
every block, from the quadratic-solve / partition alternation L130 validated —
through `_serialize_input`'s `gnn_hint` channel, which this binary has always
emitted and never read.

**Block-level, not L129's rigid units.** The C++ forms its own items and consumes
anchors per block, so units would just be rebuilt on the other side; cluster
members are collapsed onto a shared centroid instead, which is the part that
mattered. ~200 lines into the shipped source rather than ~350, numpy-only, inside
the try/except discipline the file already uses for scipy.

## 2. 🚨 Two silent semantic bugs, both found by measuring

Neither raised, neither crashed, both simply made the result worse.

| iteration | 48c total | vs baseline |
|---|---|---|
| floating origin | 1.2344230472301545 | **0.48% WORSE** |
| origin at 0 | 1.2311612296653325 | 0.219% worse |
| frame-relative | **1.2279370648317838** | **+0.0437%** |

1. **The region box floated.** It was anchored at the solve's own `min(cx)`,
   which is correct for L129 (it placed into its own frame) and wrong here:
   `constructive.cpp` packs into `[0,fw] x [0,fh]` — its LEFT test is literally
   `fabs(x - 0.0)` — and preplaced blocks sit at absolute `tx/ty`. The pull was
   dragging every block toward a meaningless point.
2. **The coordinates were absolute.** The C++ does not pack into *one* frame; it
   tries a set (scales 1.05–2.10 × several aspects) and keeps the best. One
   absolute hint silently assumes one of them and fights the rest — the hint box
   is square while e.g. case 54 packs into 141×219. Emitting `[0,1]²` and scaling
   by `(fw,fh)` at the use site, where the frame is known, is worth another 0.26%.

🔑 **Every gain in this line came from repairing a semantic error, never from
tuning a weight.** Had I swept `ICCAD_ANCHOR_W` first, I would have been fitting
noise on top of two coordinate-frame bugs and would have concluded the mechanism
was dead.

The hint is deliberately kept OUT of `anchors[]`: that array is absolute, the
hint is frame-relative, and `set_item_anchor()` averages member anchors — mixing
the two spaces there produces a meaningless centroid.

## 3. 🚨 My Phase 1 pricing was 3× low, and the rule is the reason

Gate 0 measured the hint COMPUTATION: 19.7 ms weighted, 0.467% of the per-case
wall. The measured effect at 48c is **+1.76%**.

The gap is not the computation — it is that **with anchors set the C++ takes
different, slower paths**. L125's rule was always "price what the mechanism costs
when DEPLOYED", and I priced what its code costs to run. Narrowing the rule to
the new code is how a 3× miss looks from the inside.

**Where the cost is** — not spread, but two cases:

| case | share of the weighted runtime delta |
|---|---|
| 90 (n=111) | **190%** |
| 91 (n=112) | 73% |

Over 100% because **the rest get faster**. And the quality gains come from
different cases entirely: case 92 is `0.674s -> 0.674s` unchanged while its cost
goes 1.2385 → 1.2016.

**Why case 90 is slower, and it is not a defect.** The refine loop
(`constructive.cpp:1980`) has no convergence test — its only early exit is
`run_frame` FAILING. Measured on case 90: at `REFINE_ITERS=1` hinted and
unhinted are identical (0.230s vs 0.236s); at the default 12 the baseline spends
0.035s more and the hinted run 0.227s. So without the hint the frame fails and
the loop aborts; with it the frame packs and all 12 passes run. **The extra time
is the hint enabling refinement that used to give up** — which is also where the
quality comes from.

## 4. 🚨 The tier deployment is worse on BOTH axes, and my reasoning was wrong

The plan called for adding hinted profiles as a tier rather than a global
overlay, on the argument that at ≥40 cores the wall is the max-setter (M67-E), so
a 0.38s profile against case 90's 2.93s wall is free. Built it exactly like
M124's tier (indices appended unconditionally, call-time gate, `_M55_BASE_LEN`
and `_BIG_REDUNDANT_IDX` verified undisturbed):

| | cost | vs baseline | 48c wall | cases changed |
|---|---|---|---|---|
| baseline L136 | 1.2284738 | — | 4.225s | 0 |
| **global overlay** | **1.2279371** | **+0.0437%** | 4.299s (+1.76%) | 49 |
| L137 tier (+4 profiles) | 1.2284464 | +0.0022% | **5.350s (+26.63%)** | 2 |

🔑 **The max-setter premise has a precondition I quoted the conclusion without
checking: profiles ≤ cores.** The 48c pool is already **51 profiles on 48 cores**.
Four more crosses the core count, so part of the pool runs in a second wave and
the wall goes from ~one max profile to ~two. `W = max(max dt, sum dt/cores,
sum pt)` is written in this repo's own comments, and adding profiles pushes the
second AND third terms — the serial proxy chain gets one more evaluation per
added profile. It is a cliff, not a slope.

The tier also delivers almost no quality: the proxy takes a hinted profile on
**2/100** cases, because 4 hinted recipes compete against 51 existing ones. The
global overlay changes 49 cases precisely because every profile gets the hint.

The tier code is kept (`_L137_IDX`, `ICCAD_HINT_POOL`, default OFF) — it is the
measurement that establishes the above, and it is bit-identical when off.

## 5. Where this actually stands

**Settled:** the alternation improves the shipped placer's quality, out of
sample, on the terms the mechanism predicts. This is the first time this project
has moved `hpwl_gap` on the shipped path — L128 closed the cheap analytical
routes, L130 proved the mechanism inside a candidate, L134 closed that candidate
on runtime, and it now runs inside the C++ for ~0.5% of a case.

**Unsettled:** the net. +0.0889% OOS quality against a +1.76% wall costs
`0.3 × 1.76% ≈ 0.53%` on cases that are NOT at the runtime floor and **nothing**
on cases that are. Alpha measured cost-weighted RF 0.708 against a 0.70 floor,
i.e. most weight is floored, but that anchor is from the M10-era 14-profile
submission and **the Beta results that would settle it have not arrived**
(08-15 §5.2, still open). Layering another estimate on that estimate is exactly
the L135 mistake, so it is not done here.

**Next, in order:**

1. **Bound the runtime instead of tuning quality.** Cases 90/91 are the entire
   cost and the cause is known: more refine passes now succeed. A hinted-run cap
   on `REFINE_ITERS` would bound it directly, and the quality of the capped arm
   is one gate run to measure.
2. **Re-calibrate with Beta results** when they arrive — that converts the
   0–0.53% band into a number and settles the net by itself.
3. Do NOT sweep `ICCAD_ANCHOR_W` before 1 and 2. Every gain so far came from
   fixing semantics; the weight is the last thing to touch, not the first.

## 6. Reproduce

```bash
PATH="/c/msys64/ucrt64/bin:$PATH" "C:/Users/.01/anaconda3/envs/floorset/python.exe" l113_ship_gate.py --cores 48
```
```bash
PATH="/c/msys64/ucrt64/bin:$PATH" "C:/Users/.01/anaconda3/envs/floorset/python.exe" l113_ship_gate.py --cores 48 --env ICCAD_HINT_MODE=1
```
```bash
ICCAD_HINT_MODE=1 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l137_oos_ab.py --sample s1
```

⚠️ `l137_oos_ab.py` captures `ICCAD_HINT_MODE` **before** importing
`m77_oos_probe`, which deletes every `ICCAD_*` at import. The first version did
not, and produced two byte-identical 240-case arms — a clean, plausible,
completely empty A/B. Third instance this session of the harness silently
neutralising the thing being measured (`--dt 0`, the stale packaged binary, this).
The defence is the same each time: print requested-vs-actual and require the arms
to differ before believing either.
