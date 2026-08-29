# L256 — shrink + guided re-pack: built, live, mechanism POSITIVE, deployment NEGATIVE

**Status: the mechanism L252–L255 pointed at is implemented and measured.**
Isolated, it improves true cost by **−0.2506 %** on the layouts it touches
(154 better / 105 worse of 259, ≈3.0σ from a coin flip). Deployed as a global
overlay it is **NET +0.505 % — worse** — and the reason is not the mechanism: it
is the proxy re-ranking the pool on a shifted `hmin`.

Nothing shipped. `ICCAD_L256` defaults off; `constructive.cpp` untouched.

---

## 1. What it does

After the frame trial loop picks a winner, shrink that frame a step and re-pack,
guided by the layout we already have. Accept only if the in-C++ true-cost proxy
improves; stop at the first refusal.

```
ICCAD_L256=1          enable            ICCAD_L256_STEP=0.99   linear shrink/step
ICCAD_L256_RUIN=0.12  ruin budget       ICCAD_L256_ITERS=40    max steps
ICCAD_L256_MODE=1     1=guided re-pack, 0=seed the kept blocks
ICCAD_L256_DBG=1      why did it stop?
```

Built by `l256_patch.py` from the pristine shipping `constructive.cpp`
(md5 `e2c7b2f4…`), 6 patches. Re-run on the FINAL binary
(`constructive_l256.exe` md5 `a4306c7d2eb3d4ad72670b683e1eb99b`):

| arm | result | reading |
|---|---|---|
| A — flag OFF vs stock | **102/102 PASS** | the recompile is clean, default path is the shipped placer |
| B — flag ON vs stock | **45/102** | **this is the liveness signal, not a failure** — 57 of 102 profile-layouts change |

⚠️ For the L252/L254 *instruments*, arm B passing was the requirement (they only
write to stderr). For a *mechanism*, arm B must fail — otherwise it is a silent
no-op. The first L256 build passed arm B 102/102 and that is exactly how the
frame-clamp bug in §2.1 was caught.

## 2. Three things that had to be fixed, in order

### 2.1 The frame clamp — the bug that made it look like a packer limit

`frame_candidates()` floors every frame at `max(pre, max_block) + FRAME_EPS`. The
first version shrank `(fw, fh)` without re-applying that floor, so it routinely
demanded a frame **narrower than the widest block** — the pack failed instantly on
every case and read as *"the packer is already at its limit"*. It is arithmetic,
not a limit. With the floor applied, cases immediately started accepting 2, 6, 11
consecutive shrinks.

⚠️ Generalisation worth keeping: **any code that constructs a frame outside
`frame_candidates()` must re-apply its clamps.** The clamp is 30 lines away from
the thing it constrains.

### 2.2 Mode 0 (seed the kept blocks) fails on every case

The first design tore out the overflowing items and re-seeded everything else at
its **old coordinates**. That fails 100 % of the time, and L254 says exactly why:
the freed space is fragmented, and a fragmented pocket is precisely what this
greedy cannot use. Mode 1 re-packs everything with the previous layout as a guide
— which is REFINE's own `use_prev`/`prev_pos` machinery pointed at a *tighter*
frame instead of the same one — and works. Mode 0 is kept only as a control.

### 2.3 The accept test was never the binding constraint

`layout_score` has no meaningful wire term and a 150000×bv weight, so I switched
the accept test to `csc_of()` — `(area + hw·hpwl)·exp(2(bv+gf)/nsoft)`, the
true-cost-shaped proxy compaction already uses. The re-scored result was
**bit-identical, every digit, all 12 cases.** The loop almost always ends on
*repack FAILED*, not on a reject, so the criterion had nothing to decide. Kept
(it is the more correct test) but it bought nothing.

## 3. The measurement, and the mistake it corrected

First pass, portfolio-level on 12 heavy cases, official strict scorer:

    weighted base 1.508991 -> 1.516611   +0.5050%   better 3  worse 3  same 6

Then the per-case table showed what that number actually was:

| | |
|---|---|
| every case whose cost moved | **had a different profile selected** |
| every case that kept its profile | moved by **exactly 0.000** |

🚨 **So the portfolio delta was measuring the proxy re-ranking, not the shrink.**
The Python proxy's `hmin` is the pool-wide minimum HPWL; L256 changes some
candidates, `hmin` moves, and the ordering of all 51 re-shuffles. This is the
M80 `hmin` coupling, and it makes a global-overlay A/B on 12 cases roughly a coin
flip (3 better, 3 worse).

Isolating the mechanism — same profile, before vs after:

    profiles whose layout actually changed   259 / 612   (42% liveness)
    weighted true cost over those            1.697641 -> 1.693387   -0.2506%
    better 154   worse 105

154/259 against a coin flip is ≈3.0σ, so the mechanism is genuinely more often
right than wrong. **It is a real, small, positive effect.**

### 3.1 It has to go deep to pay

| `ITERS` | isolated mechanism | better/worse |
|---|---|---|
| 1 | **+0.1458 %** (worse) | 129 / 130 |
| 3 | +0.1713 % (worse) | 143 / 116 |
| 40 | **−0.2506 %** (better) | 154 / 105 |

A single small shrink re-packs the whole layout through the guide and disturbs it
without recovering enough area to pay for the disturbance. Only a long chain of
shrinks wins. `ITERS = 40` is not binding — runs end on *repack FAILED* — so
−0.2506 % is the mechanism's ceiling on this band, not a tuning artefact.
`STEP` ∈ {0.98, 0.99, 0.995} moves the same 2 of 5 sampled cases and changes the
total shrink by <1.5 pp, so the outcome is not a step artefact either.

## 4. Where it stands against the prediction

L255 priced perfect relocation of the last 2 % of area at **+1.22 %** of quality
and called that an *upper* bound. L256 realises **+0.25 %** of it in isolation and
**loses it again** at deployment. That is consistent, not contradictory: the
mechanism is inside the predicted band, at the low end of it.

## 5. What would have to change for this to pay

* **Twin deployment** (the L124 pattern — the mechanism's RED may be its
  deployment form's RED): keep the original profiles and add L256 variants so the
  proxy arbitrates instead of being perturbed. That removes the coin flip. ⚠️ It
  costs pool size, and L248's curve prices +6 profiles at 9.2 % of heavy-band wall
  for +0.364 % quality = **NET −1.03 pp** — so the quality gain would have to be
  several times what §3 measured.
* **A stricter accept test.** 105 of 259 accepted shrinks make true cost worse, so
  `csc_of` is wrong ~40 % of the time. Its area and hpwl terms are not normalised
  the way the true cost's gaps are; L114 solved the area half with the structural
  bound `ΣA/0.968` and there is no equivalent for hpwl.
* Neither is priced for runtime. The shrink adds packs on top of the frame loop.

## 6. Honest limits

* 12 cases, sample s1, heavy band only. No OOS, no s2, no runtime.
* The −0.2506 % is over the 259 layouts that changed, **not** over the portfolio;
  those are different populations and only the portfolio one is scoreable.
* `--iters/--step/--ruin` were tuned on the same 12 cases the result is reported
  on. The direction (deep beats shallow) is robust; the magnitude is not.

## 7. Files

```
l256_patch.py       pristine constructive.cpp -> constructive_l256.cpp (6 patches)
l256_dbg.py         one case, full stop-reason trace
l256_score.py       portfolio gate + the isolated per-profile comparison
l256_iso.pkl        {rows, per} -- per-case and per-profile before/after
l256_identity.log   arm A 102/102 PASS
l256_score.log l256_iso.log l256_it1.log l256_it3.log
```
