# L125 — a bounded beam in the packer: RED

Route 3 of `HANDOFF_2026-08-13.md`. **Nothing shipped and nothing shippable.**
`constructive.cpp`, `optimizer_constructive.py`, `bin/constructive_linux` and the
submission package are untouched — the beam lives only in `constructive_l125.cpp`
/ `.exe`, so the shipping exe's md5 never moved and every offline cache stays
valid.

**The verdict is not "the beam doesn't work".** It works: its quality ceiling is
the highest any mechanism has shown on the twin screen, higher than the MIB
bucketing that shipped yesterday. It dies on price, and it dies on price *at every
possible beam cost* — including one no beam can achieve. That is what makes the
finding worth more than the route.

---

## 1. What was built

`constructive_l125.cpp` = the current shipping source + a bounded beam inside
`pack_in_frame`. The item loop commits the single best candidate per item; the
beam carries `W` partial layouts across the first `K` committing items, then lets
each complete greedily and returns the completion the frame loop itself would
prefer (`layout_score` of the nudged layout — the same comparison `run_frame`
makes between frames).

| knob | default | meaning |
|---|---|---|
| `ICCAD_BEAM` | 0 | master switch; 0 leaves `pack_in_frame` untouched |
| `ICCAD_BEAM_W` | 2 | beam width |
| `ICCAD_BEAM_K` | 8 | number of branching item levels |
| `ICCAD_BEAM_SEL` | 0 | 0 = score the nudged completion, 1 = raw |

Prefix cost of a partial layout is `bbox_area + the non-area terms the greedy step
score already carries`. Within one parent this ranks exactly as the greedy step
score does — greedy's `area` term *is* the child's bbox area — so W=1 reproduces
the greedy pick including its first-minimum tie-break.

The beam is a **separate function**, not a modification of the greedy path, so the
off-path guarantee is structural rather than argued.

### Gate 0 — bit identity (PASS)

100 cases × 6 shipped recipes, official input including `target_positions`:

| arm | result |
|---|---|
| `ICCAD_BEAM` unset | **600/600 bit-identical** to `constructive.exe` |
| `ICCAD_BEAM=1 ICCAD_BEAM_W=1` | **600/600 bit-identical** |

The second row is the one that earns its keep: a width-1 beam *is* the greedy, so
it proves the duplicated placement arithmetic did not drift.

## 2. The mechanism is live, and the deployment form decides the sign

W=2, K=8, same 100 × 6 grid (`l125_ab_w2k8.log`):

| | |
|---|---|
| (case,recipe) whose output changed | **374/600** |
| solo true cost | **219 better / 155 worse** / 226 equal |
| best-of-6-recipes, weighted, **replace** form | **−0.3981%** (worse) |

L123's shape exactly: forcing the mechanism on every profile loses while the
per-case signal underneath is positive. So the verdict had to come from the
**twin** form.

## 3. Quality: the best twin-screen ceiling measured so far

`l124_r3_scale.py --flag ICCAD_BEAM --bin constructive_l125.exe`, OOS sample s2,
80 cases, per-case proxy arbitration over the real 51-profile pool ∪ appended ON
twins:

```
  K=0 (today, pool unchanged)   1.509085
  K=4                           +0.3275%
  K=8                           +0.5719%
  K=16                          +0.7442%
  K=all                         +0.8017%   (ceiling)
```

**24 twins ever win** and the win counts are flat (3,2,2,2,2,2,2,2,…) — the value
is spread, not concentrated, which is why the curve keeps climbing. Against the
two mechanisms already judged by the same machine on the same sample:

| mechanism | K=8 (s2) | ceiling | verdict |
|---|---|---|---|
| L124 MIB bucketing | +0.4712% predicted → **+0.4697% measured** | — | shipped |
| L126 anch_cross | +0.2633% | +0.2860% | RED |
| **L125 beam** | **+0.5719%** | **+0.8017%** | RED — on price |

## 4. Price: the pool has no room for a slower twin

### 4.1 The beam costs ~2× serially, and B0's model was right

The A/B run read p50 **2.30×**, but that run used 11 parallel workers. Measured
**serially**, all 51 pool profiles × 20 heavy cases (`l125_dt_cache.pkl`):

    p10 1.82   p50 2.06   p90 2.23   max 4.06

So B0's assumption of `m = W = 2.0` was correct and the 2.30 was contention.
**Timing under load is not timing** — L100's rule again, from the other side.

### 4.2 The structural fact that makes a twin set cheap

At 48 cores the wall is the max-setter (M67-E, 100/100), so
`new wall = max(old wall, m · max_{p∈S} dt_p)`: **a twin set's dRF is the dRF of
its slowest member.** Once a source is affordable, adding more affordable sources
is *free*. There is no K-vs-RF trade-off at all — only source eligibility.

### 4.3 …and the one that kills it

Eligibility is brutal, because the pool's runtime distribution is **bimodal**.
Measured per-profile weighted dRF of appending its beam twin (`l125_afford.log`):

```
   13 profiles      0.0000%   (0 walls raised)
   #33              0.0267%   (1 wall raised)
   ---------------- cliff ----------------
   #96             11.4540%   (18 walls raised)
   ... 35 more, up to 19.2%
```

There is **nothing between 0.03% and 11.45%**. M41/M42/M45 already pruned this
pool by runtime, so what survives is either far below the wall or *is* the wall.
Only **14 of 51** profiles have the headroom to be made 2× slower — and only
**one** of them (#95) is in the ≥40-core tiers where the beam's value lives.

Restricting the twin sources to what can actually be paid for:

| sources | quality ceiling |
|---|---|
| all 24 winners | +0.8017% |
| base profiles only (`--allow-max 41`) | +0.1868% |
| **affordable at the measured 2.06×** (14) | **+0.0956%** |

### 4.4 A cheaper beam does not rescue it

The obvious escape — make the beam cheaper (e.g. beam only the initial pack and
not the REFINE guide passes, ≈1.2×) — was priced before being built, and it fails:

| assumption | eligible sources | quality ceiling | RF cost | NET |
|---|---|---|---|---|
| measured 2.06×, zero-cost sources | 14 | +0.0956% | ~0.000% | **+0.10%** |
| hypothetical **1.15×**, zero-cost sources | 16 | +0.1398% | ~0.000% | +0.14% |
| hypothetical 1.15×, 0.30% RF budget | 23 | +0.2152% | 0.30% | **−0.08%** |
| no budget | 24 | +0.8017% | 11.5–19% | ruin |

A 1.15× beam is cheaper than any W=2 beam can be — W=2 pays two completions on the
branched portion by construction. Every row is under the 0.30% bar, and every row
is an **in-sample-selected upper bound** (L124's cross-sample transfer was 80–83%,
so multiply by ~0.8). No K, no beam efficiency, no selection method rescues a
ceiling that is already below the bar at zero cost.

⚠️ Note the shape of the eligibility trap: *per case* 44–47 of 51 profiles have
15% of headroom, but the source must be affordable on **every** case at once, and
the weighted intersection is 16. Same trap B0 recorded from the other direction
("the intersection of never raising any wall is empty").

## 5. 🔑 What outlives the route

**The twin screen has a second precondition nobody had written down: the ON side
must be time-neutral.** L124 shipped because MIB bucketing costs nothing — it is a
construction-time decision. Any mechanism whose ON side is *slower* can only be
applied to the ~quarter of the pool that has wall headroom, and here that quarter
carried **12%** of the twin value (+0.0956 of +0.8017). Before building anything
for the twin screen again, price the eligible-source set FIRST — it is a pure
cache computation (`l125_beam_price.py afford`, ~45 min of serial timing once) and
it would have killed this route before a line of C++ was written.

This also closes the packer-search axis in the same shape L122 closed the shape
axis: **not because the mechanism is inefficient, but because there is no time to
spend.** L122's rule was "anything future here has to be essentially free
(≈1.0×)". L125 sharpens it: *free is not sufficient either* — it has to be free
**on the profiles that are at the wall**, and at 48 cores that is 35 of 51.

## 6. Honest range

- the K-curves are **in-sample selection on s2**; s1 was not run, because a
  ceiling below the bar cannot be rescued by a second sample (L126's logic)
- affordability is decided on **in-set** heavy-band dt and applied to an **OOS**
  quality curve — profile speed is a property of the profile, but it is a splice
- the timing pass is **reps=1**; nothing here turns on a marginal number (the
  cliff is a factor of 400), so the ≥3-rep median was not spent
- the pre-registered risk — `layout_score` is not monotone on partial layouts, so
  the beam's pruning criterion is unreliable — **is** visible in the data (155 of
  374 changed pairs got worse), but it is not what killed the route

## 7. Artefacts

| file | what |
|---|---|
| `constructive_l125.cpp` / `.exe` | the beam, probe-only |
| `l125_beam_probe.py` | `offpath` / `ab` / `sweep` |
| `l125_beam_price.py` | `measure` / `afford` / `rank` / `set` — **B0 written down for the first time**, plus the whole-pool timing it lacked |
| `l125_dt_cache.pkl` | serial dt, 20 heavy cases × 51 profiles × both arms |
| `l125_beam_cache.pkl` | the twin screen's s2 capture (80 cases × 51 × 2) |
| `l124_r3_scale.py` | gained `--allow` / `--allow-max`: restrict twin sources to an affordable subset |
