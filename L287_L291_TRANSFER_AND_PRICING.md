# L287–L291 — the gain transfers at 93 %, and the project's RF pricing was over-charging twice

Two questions, both raised by L285's rank projection: does the in-set gain since
M73 survive on a corpus the code has never seen, and is the shipped LP depth
really the right one. Both are now measured, and both answers changed a
previously published verdict.

    the SAME seven arms on BOTH corpora
    (in-set 100, official eval | OOS s1 240, full deployable pipeline)

      arm          in-set   ship vs arm        OOS   ship vs arm   transfer
      ship       1.226325       0.0000%   1.470262      0.0000%        --
      noLP       1.258974      -2.5933%   1.505450     -2.3374%      90 %
      m73        1.259898      -2.6647%   1.507783     -2.4885%    **93 %**
      lp2        1.222554      +0.3085%   1.463177     +0.4842%     157 %
      noM80      1.231706      -0.4369%   1.476732     -0.4381%     100 %
      noHint     1.226272      +0.0044%   1.469135     +0.0767%       --
      refOld     1.226824      -0.0407%   1.469017     +0.0847%       --

**The cumulative gain transfers at 93 %**, and essentially all of it is the
shape LP. `lp2` is positive on **both** corpora, which is what L275's rule asks
for, and the corrected RF pricing puts it at NET **−0.17 % … +0.29 %** depending
on one unmeasured ratio.

---

## 1. 🚨 A correction to my own first reading: 46.6 % was a basis error

The first pass reported **TRANSFER 46.6 %** and I took it seriously enough to
move the rank estimate to "2–4, centre 3". It was wrong, and the fault was the
comparison, not the measurement:

* the in-set reference **−5.34 %** is against **real M73**;
* the OOS `m73` arm is **"M73-like"** — the shipped code with the *flag-gated*
  additions switched off. It still carries L131/L136's correctness fixes and
  M74's constant regen, because those are **code, not flags**.

So the OOS side was measuring a smaller delta by construction. Running the
identical arm set on both corpora removes the mismatch: `m73` transfers at
**93 %**, `noLP` at 90 %, `noM80` at 100 %.

🔑 **The lesson is the one this project keeps re-learning in new costumes: a
transfer ratio is only meaningful if both sides revert the same thing.** A
kill-switch arm and a historical baseline are not the same counterfactual.

⚠️ Neither number licenses a rank claim on its own. OOS is drawn from
`floorset_lite` while the in-set, alpha and beta sets are the contest's own, so
a gap here would still be ambiguous between over-fitting and distribution
difference (L275). What changed is that there is **no gap left to explain**.

## 2. The decomposition — it is the shape LP and almost nothing else

Reading the `ship vs arm` column as "what this component is worth":

| component | in-set | OOS | note |
|---|---|---|---|
| shape LP | **+2.59 %** | **+2.34 %** | 90 % transfer; 94 % of the whole OOS gain |
| M80 tier | +0.44 % | +0.44 % | transfers exactly |
| L137 hint | −0.004 % | −0.077 % | **slightly harmful on both** |
| L223/L231 REFINE cuts | −0.04 % | +0.08 % | signs disagree ⇒ noise; they buy runtime, not quality |

The hint being negative on both corpora is small but consistent, and it costs
runtime. It is the one component here that is not paying for itself.

## 3. 🚨 The RF pricer was over-charging twice, and it flips a published verdict

`l276_price.py` prices added seconds against the published medians. Two of its
inputs were wrong in the same direction:

**(a) the baseline runtime vector.** `load()` takes our runtimes from the **beta**
results — the M73 package. L285 measured the shipped package at **0.855–0.868×**
of that (grader ≈ 44.5–45.2 s, not 52.07 s), so every case sits lower on the
`(t/M)^0.3` curve and more of them are on the floor.

**(b) local dt seconds added to grader seconds.** The dt vector is measured on
this box and added directly to the grader's `t_i`. A time in seconds is not
machine-independent — this is the project's own
`wsl-vs-windows-3x-calibration-trap` ("正解是同機比值，f 自己消掉") in a new
place.

Re-priced with a dt measured back to back in this session (`ship` 129.69 s →
`lp2` 139.05 s, +9.36 s):

      dt divided by          RF          NET
      1.00               -0.4816%    -0.1741%   RED
      1.50               -0.0776%    +0.2299%   GREEN
      1.91               -0.0328%    +0.2747%   GREEN
      2.87               -0.0167%    +0.2908%   GREEN

      break-even at d = 1.142

⇒ **LP k=2 is GREEN iff the grader runs the LP at least 1.14× faster per case
than this box.** For reference the packing work runs **2.87×** faster there
(same config, local 149.43 s vs grader 52.07 s), and §5 shows that ratio is not
an oversubscription artefact.

⚠️ **This is a pricing correction, not a measurement of the grader.** `d` cannot
be measured from here: the 2.87× is for the *parallel packing* phase, while the
LP is *serial per case*, and the two need not scale alike — a server part can
have a lower per-core clock than a desktop. The bracket is honest; the point
estimate is not available.

### 3.1 An error of mine inside this same analysis

My first attempt at the re-pricing used a `dt` median over a ±2 block_count
window and reported **NET +0.12 %, GREEN** before any of the above. That was
wrong by a factor of four: the in-set has ~one case per block_count, so
`dt_by_n`'s `mean` *is* that case's dt, and a window median smooths away the fat
tail — dt is p50 +0.087 s but max +0.715 s, and the expensive cases are the
big-`n` ones with the least slack. `l276_price.py`'s docstring warns about
exactly this. **Use the tool.**

## 4. Where that leaves LP k=2 as a candidate

| | |
|---|---|
| in-set quality | **+0.3085 %** (official eval, 100/100 feasible) |
| OOS s1 quality | **+0.4842 %** (240 cases, full pipeline) |
| L275's both-corpora rule | **satisfied** |
| NET after corrected RF | **−0.17 % … +0.29 %**, break-even `d` = 1.14 |
| runtime | +9.36 s local → ≈ +3.3 s grader at d=2.87; 45.2 → 48.5 s against a 64.1 s threshold |
| deployment | **`ICCAD_SHAPE_LP_ITERS` appears 0 times in `constructive.cpp`** — wrapper-only, **no ELF rebuild** |

The downside is bounded and small (−0.17 %); the upside is +0.29 %. It is a
genuine candidate and it is **not** mine to ship — it needs a decision, and it
needs the full staging chain re-run even though the ELF is untouched.

⚠️ The shipped `_L157_DEPTH` is **flat at 1 for every n** (flattened in
`29d70a3 L205/L213`). L172 had derived a rebuilt map `{1:52, 2:18, 3:30}` with
OOS +0.4153 % / +0.4452 % and "RF cost exactly zero". Whether that map beats a
uniform k=2 under the corrected pricing has **not** been measured here.

## 5. Route A costs 1.7 %, not 2.9×, and is quality-neutral

Needed to test whether the 2.87× ratio was an oversubscription artefact (this
box forces 48-way fan-out onto 32 cores):

      shipped default (route A ON)   1.226325126   129.69 s
      route A OFF                    1.226325126   127.55 s
      route A OFF + LP OFF           1.258974453   119.12 s

Route A is **bit-identical in quality** and costs **+2.14 s (+1.7 %)** locally.
So oversubscription does not explain the 2.87×, and the hypothesis that `d ≈ 1`
loses its main support.

## 6. Honest limits

1. `d` is bracketed, not measured (§3). Every NET figure moves with it.
2. OOS is `floorset_lite`; the graded corpus is not. §1's 93 % says the gain is
   not corpus-fragile, which is weaker than saying it will appear on the hidden
   set.
3. One OOS sample (s1). s2 was not run.
4. The `m73` arm is M73-*like*; it does not revert M74's constant regen.
5. Local wall-clock: repeats agree to 1.2 %, far tighter than CLAUDE.md's ≥20 %
   warning, which was about comparing configurations whose true gap was small.

## 7. Files

```
l287_transfer.py       the 7-arm OOS harness (resumable, per-arm cache)
l287_cache.pkl         240 cases x 7 arms
l290_arms.py           the like-for-like two-corpus table + k=2 re-pricing
l290_inset_*.json      lp2 / noHint / noM80 / refOld on the in-set
l291_noroutea*.json    route A separation
l285_*.json            ship / LP-off / beta-config, from L285
```

Nothing shipped. `constructive.cpp` md5 `e2c7b2f4…`, `op_wrapper.py` md5
`1c326784…`, both unchanged.
