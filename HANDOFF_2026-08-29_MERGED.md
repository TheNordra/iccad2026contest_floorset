# Handoff 2026-08-29 — deadline moved to Aug 31, and there is a live ship candidate for the first time in weeks

Two sessions ran in parallel on this tree. This merges them. **Read §1 first: it
reconciles their two conflicting answers to the same question, and the
reconciliation is itself the most reusable thing here.**

> 🗓️ **Final deadline extended to 2026-08-31 23:59 (GMT+8)** (organiser mail,
> 2026-08-27). Three extra days. That is what makes §3 worth acting on.

---

## 0. Shipping state — unchanged, verified twice this session

| | |
|---|---|
| uploaded | `build_submission.D/cadc1075.tar.gz`, Drive **Final** |
| identity | `op_wrapper.py` md5 **`1c326784de7cd9246cd1f380e2842668`** |
| source | `constructive.cpp` md5 **`e2c7b2f418ef2b70b6bff99f7adfbd37`** |
| 48c Linux | **1.2264069637381392**, 100/100 feasible |

**Nothing has been shipped, staged or modified.** The user supplied
`cadc1075 (2).tar.gz` from Downloads; it is **byte-identical** to
`build_submission.D` (same tar md5 `c7be7452…`, all six file md5s equal) and,
run through the official evaluator from a clean extract, reproduces
`1.226325126` with **0/100 per-case cost differences** and 100/100 feasible. It
compiled its own binary, so it did not hit the silent-SA-fallback trap.

**Runtime is not a risk (L285).** Same machine, back to back, ×2 repeats, with a
beta arm rebuilt from the shipped code's own kill switches:

    shipped default        1.226325126 (== anchor, bit-exact)   129.69 / 128.20 s
    shipped, SHAPE_LP=0    1.258974453                                  117.86 s
    beta config (M73-like) 1.259897682                          149.43 / 149.97 s
    ratio 0.8548 - 0.8679  ->  grader **44.5 - 45.2 s** (was 52.07 s)

cwRF **0.70004**, **98/100 cases on the RF floor**, projected total **0.87511**,
**rank 2**. Losing rank 2 needs a **1.43×** slowdown. There are ~19 s of free
budget, and that free budget is what §3 spends.

## 1. 🚨 The two sessions answered the same question differently. Here is why.

Both asked: *does the in-set gain since M73/M74 survive on a held-out corpus?*

| | corpus | baseline | answer |
|---|---|---|---|
| **L286** (other session) | OOS heavy band n=111..120, 40/sample | `m67_oos_cache_c48` = **M74-era**, cached | transfer **≈ 0 %** |
| **L287** (this session) | OOS s1, **all 240**, all three bands | the **shipped code** with kill switches (M73-like) | transfer **93 %** |

**L287 is the measurement L286 §5 itself prescribed** — "run the shipped package
and an M73 reconstruction through the *same* OOS harness end to end, on the full
240-case sample" — so it removes L286's own limits 1, 2 and 4. Three concrete
reasons the answers differ:

1. 🚨 **L286's reference side is a stale binary.** Its cache carries exe md5
   prefix `dc47a572707c`; the same md5 is in `audit_cache_ship.pkl`, and that
   cache was proven this session to reproduce the shipped per-case cost on only
   **4/100** cases (weighted 1.292646 vs 1.226325, **+5.4 %**). L286 states the
   version mismatch honestly as limit 2 — it is simply larger than it looked.
2. **Different populations.** n=111..120 (40 cases) vs n=21..120 (240, band
   balanced 80/80/80).
3. **Different assemblies.** L286's current side is l252 profile outputs + the
   proxy pick + `_shape_lp_maybe` applied by hand; L287 runs the real wrapper
   end to end on both sides, changing only flags.

⇒ **Use 93 %.** L286's number should be read as what it says on its own tin —
*"the in-set gain is not visible on that band against that reference"* — not as
a transfer coefficient.

### 1.1 And L287's own first answer was wrong too, in the opposite direction

L287 first reported **46.6 %** and it was a basis error, caught by making the
comparison like-for-like:

* the in-set **−5.34 %** is against **real M73**;
* the OOS `m73` arm is **M73-like** — it still carries L131/L136's correctness
  fixes and M74's constant regen, because those are **code, not flags**.

Running the identical arm set on both corpora gives:

    arm       in-set   ship vs arm       OOS   ship vs arm   transfer
    noLP    1.258974     -2.5933%   1.505450     -2.3374%      90 %
    m73     1.259898     -2.6647%   1.507783     -2.4885%    **93 %**
    lp2     1.222554     +0.3085%   1.463177     +0.4842%     157 %
    noM80   1.231706     -0.4369%   1.476732     -0.4381%     100 %
    noHint  1.226272     +0.0044%   1.469135     +0.0767%       --
    refOld  1.226824     -0.0407%   1.469017     +0.0847%       --

🔑 **A transfer ratio is only meaningful if both sides revert the same thing.**
A kill-switch arm and a historical baseline are not the same counterfactual.
Both sessions got a wrong number from a version of that mistake.

**Decomposition:** the shape LP is **2.34 % of the 2.49 % OOS gain** (90 %
transfer); M80 tier transfers at 100 %; the L137 hint is **slightly negative on
both corpora** and costs runtime — the one component not paying for itself.

## 2. 🚨 Two of the three RF pricers omit a machine factor the third one measured

`l172_depthmap.py:39` has carried this since L161:

    F = 3.17    # dev-box LP second -> grader second   (L160 measured 2.71)

**`l146_rf_price.py` and `l276_price.py` do not have it.** They add **locally
measured dt seconds** straight onto the **grader's** per-case runtimes. A time in
seconds is not machine-independent — the project's own
`[[wsl-vs-windows-3x-calibration-trap]]` in a new place.

Second, smaller error: `l276_price.py`'s `load()` takes the baseline runtime
vector from the **beta** results, but the shipped package is 13–15 % faster
(§0), so more cases sit on the RF floor than it thinks.

Restoring both, on LP k=2:

    baseline / f            RF          NET
    beta      / 1.00    -1.2611%    -0.9536%    <- L276 as published, "RED"
    shipped   / 1.00    -0.4816%    -0.1741%
    shipped   / 2.71    -0.0180%    +0.2895%
    shipped   / 3.17    -0.0146%    **+0.2929%**
    beta      / 3.17    -0.0466%    +0.2609%

⇒ **GREEN at every combination of the project's own measured constants.** The
bill was over-charged **33×**, dominated by the missing `f`, not by the baseline.

⚠️ **Correction to L285**, which is in this tree and says "~40 % too large": that
is the baseline half only. L285 now carries a correction banner.

⚠️ **And an error of mine inside the same analysis**: my first re-pricing used a
`dt` **median over a ±2 block_count window** and reported GREEN +0.12 % for the
wrong reason — off by 4×. The in-set has ~one case per block count, so
`dt_by_n`'s `mean` *is* that case's dt; a window median smooths away the fat tail
(p50 +0.087 s, max +0.715 s, and the expensive cases are the big-`n` ones with
the least slack). `l276_price.py`'s docstring warns about exactly this. **Use the
tool, do not re-implement it.**

## 3. The live candidate: shape-LP depth

**Quality, three corpora, all positive** — the first candidate in weeks to
satisfy L275's both-corpora rule outright:

| | LP k=2 |
|---|---|
| in-set 100 (official eval) | **+0.3085 %**, 100/100 feasible |
| OOS s1 (240, full pipeline) | **+0.4842 %** |
| OOS s2 (240, disjoint) | **+0.4891 %** |

s1 and s2 agree to **0.005 pp** and all three bands are positive in both.

**Priced (§2): NET +0.2929 %** at `f = 3.17`.

    depth frontier, in-set 100, official evaluator, priced at f=3.17 on the
    shipped baseline; all arms 100/100 feasible

      arm          total        quality       RF        NET    local dt  grader s
      k=1 (ship) 1.226325126    0.0000%    0.0000%   0.0000%     0.00s     45.2s
      k=2        1.222554152    0.3075%   -0.0146%  +0.2929%     9.36s     48.1s
      k=4        1.221441099    0.3983%   -0.0714%  **+0.3269%** 27.57s    53.9s
      k=8        1.220929833    0.4400%   -1.2774%  -0.8374%    41.03s     58.1s
      k=12       1.220923716    0.4405%   -3.2107%  -2.7703%    57.73s     63.4s

**k=4 is the optimum and clears the project's 0.30 % bar.** The frontier has a
clean interior maximum for two reasons that show up in the table: quality
**saturates by k=8** (0.4400 % vs k=12's 0.4405 % — the LP has converged), while
the RF cost **explodes** past k=4 as cases start leaving the RF floor, where the
derivative stops being zero. k=12 lands at 63.4 s against a 64.1 s threshold.

⚠️ The earlier k=4 figure of +0.3518 % in this session used the **L276-era** dt
(18.86 s). Re-measured back to back it is 27.57 s, so NET is **+0.3269 %**. Use
the fresh number.

**Deployment risk is unusually low:** `ICCAD_SHAPE_LP_ITERS` appears **0 times**
in `constructive.cpp` — it is read only by the wrapper
(`op_wrapper.py:4134`), so changing the default is a **Python-only change with no
ELF rebuild**. The staging chain still has to be re-run.

### 3.1 The bigger unexplored lever: the LP gate

The wrapper carries a **second** gate, `_L196_LPGATE`, which switches the shape
LP **off entirely** for **29 block counts carrying 44.2 % of the graded weight**,
including 8 of the 20 heavy cases (101, 107, 108, 112, 114, 117, 118, 120).
L205's own acceptance table records:

    G6 gate cost   -3.3998 % vs the ungated LP, 37 moved

That gate exists **for runtime** — the exact quantity §2 shows was being
over-charged 33×. `ICCAD_LP_GATE=0` restores the pre-L196 behaviour and has
**not** been re-measured under the corrected pricing. **This is the highest-value
open item.**

⚠️ Sanity first: the LP is worth +2.59 % overall, so a −3.40 % gate cost is
larger than the whole mechanism. Either the number means something narrower than
it reads, or the ungated arm is very expensive. Measure before believing.

### 3.2 What is still missing before any of this can ship

1. **k=4 / ungate OOS validation** (k=2 is done on both samples; k=4 is in-set
   only).
2. **Determinism** — two identical runs must agree bit-for-bit. CLAUDE.md's
   standing rule: *"any in-window LP must not keep a HiGHS `time_limit`"*.
3. **Runtime headroom** — 45.2 s now, threshold 64.1 s. k=4 adds ≈ +6 s grader
   (18.86 / 3.17); the ungated arm is unmeasured and could be much larger.
4. **Linux lane** — `l117_linux_verify.judge48()` invariants, not bit-equality
   (Win scipy 1.15.3 vs Linux 1.18 diverge on degenerate LPs).
5. **`_L157_DEPTH` vs uniform k.** The shipped map is **flat at 1 for every n**
   (flattened in `29d70a3 L205/L213`). L172 had derived a rebuilt map
   `{1:52, 2:18, 3:30}` at OOS +0.4153 % / +0.4452 % with "RF cost exactly zero".
   Whether that beats a uniform `k` under corrected pricing is **unmeasured**.

## 4. Where we actually stand against the field

2026-08-23 leaderboard (the current one). Our beta row is rank 4, raw 1.320665,
52.07 s:

    rank  total      raw        cwRF     runtime
      1   0.858632   1.084488   0.7917   110.9 s
      2   0.888187   1.207716   0.7354   110.7 s
      3   0.899329   1.284755   0.7000    24.5 s
      4   0.926586   1.320665   0.7016    52.1 s   <- us, beta

* Current package projects to **0.87511 → rank 2**.
* **The whole gap to rank 1 is quality**: raw 1.2264 vs 1.0845 = we are 13.1 %
  worse, while we are 11.4 % *better* on cost-weighted RF.
* 🔑 **We do not need 13 %.** Because the speed advantage covers the rest, taking
  rank 1 needs only **0.7 – 2.3 % of raw quality**. §3's candidates are 0.3 – 0.4 %
  each — not decisive alone, but the same order as the gap.
* ⚠️ **Rank 1's raw 1.084488 sits +0.10 % from L128's "label topology + exact
  HPWL LP" bound of 1.083368, and 2.12 % BELOW `fp_sol` verbatim.** They are
  essentially at the reconstruction bound. That is a different class of method
  from ours and it is what M40 declared RED *for our approach*.

## 5. The depth frontier is complete

All four arms landed; the table is in §3 and `l293_frontier.py` reproduces it
(it applies `f = 3.17`, the 0.8679 baseline, and flags any arm crossing the
64.1 s threshold). **Nothing is in flight.**

## 6. The quality/topology axis is closed — do not reopen it

L281–L284, all this session, all RED, all measured on the graded shape:

| | question | answer |
|---|---|---|
| L281 | can topology be edited post hoc? | no — the anchor's critical chain **exactly saturates** the bbox in **62/100** cases; oracle +0.0942 % at 172 s/case |
| L282 | can the chain be shortened? | no — 90.6 % unreachable; where reachable, 2.74 : 1 in wire |
| L283 | is that wire price an artefact of squeezing? | yes — but density then costs **2.67 : 1 in violations** |
| L284 | is density even available? | no — ceiling 85.4 % vs label 96.6 % |

⚠️ **L283 and L284 carry correction banners**: their numbers came from
`audit_cache_ship.pkl`, the same stale cache as §1. The *conclusions* survive
(the shipped weighted utilisation is **85.16 %**, computed from the shipped
anchor, against that pool's own max of 85.36 %) but the tables do not.

⇒ **New standing gate: before using any audit cache, check that its pool-best
reproduces the shipped per-case cost.** Pinning the exe md5 in the signature was
not enough, because nobody checked the cache against the shipped result.

## 7. Traps this session paid for

1. **`m67_oos_probe` strips every `ICCAD_*` at import.** Set flags *after* the
   import and **assert** the gate is armed (`oc._effective_cores_hi() >= 40`).
   Both sessions hit this independently; it produces a table that looks normal
   with every arm a silent no-op. `l287_transfer.py` now asserts at import and in
   `set_arm`.
2. **A stale cache can pass its own signature check.** §6.
3. **A transfer ratio needs both sides to revert the same thing.** §1.1.
4. **Do not re-implement a pricer.** §2.
5. **`_shape_lp_maybe` never raises by design** — a dead flag is indistinguishable
   from a decision not to act.

## 8. Files added by the two sessions

```
reports  L281_RELOCATION · L282_CHAIN_SHORTENING · L283_DENSITY_CURRENCY
         L284_DENSITY_CEILING · L285_RUNTIME_VERIFIED · L286_PACKAGE_TRANSFER
         L287_L291_TRANSFER_AND_PRICING · this file
probes   l281_*.py (8)  l282_*.py (3)  l283_generate_vs_squeeze.py
         l284_density_ceiling.py  l285_runtime_headroom.py  l286_transfer.py
         l287_transfer.py  l290_arms.py  l293_frontier.py
data     l281_cache.pkl · l282_cache.pkl · l283_cache.pkl · l287_cache.pkl
         l285_*.json · l290_inset_*.json · l291_noroutea*.json · l293_k*.json
memory   l281-relocation-red · chain-saturation-closes-topology-repair
         changed-relation-is-not-a-move · l282-chain-shortening-red
         density-is-paid-in-violations · l284-density-ceiling
         l285-runtime-verified · l287-transfer-93-and-rf-overcharge
```

## 9. Recommended order for the next session

1. ~~Finish the depth frontier~~ **done — k=4 wins at NET +0.3269 %** (§3).
2. Measure **`ICCAD_LP_GATE=0`** in-set: quality, dt, feasibility — §3.1.
3. Whichever of {k=2, k=4, ungate, L172 map} prices best, validate on **OOS s1
   and s2** with `l287_transfer.py --arms ship,<arm>`.
4. Determinism repeat, then the Linux lane, then a ship decision **by the user**.
5. Do **not** reopen L281–L284, and do not price anything with `l146`/`l276`
   without restoring `f`.
