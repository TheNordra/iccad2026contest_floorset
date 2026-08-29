# L252 — the frame axis, measured. The counter-prior holds, and now it has a number.

**Verdict: the frame is worth at most +1.50 % of quality, and 83 % of the area
deficit is cliff, not slope.** `frame_candidates()` was the last structural
decision without a perfect-information bound (M26 ordering +0.005 %, M68 seed
+0.001 %, M79 shape +0.099 % all bound decisions taken *given* the frame). It now
has one. No label was read.

---

## 0. The arithmetic that makes the frame axis a single number

`frame_candidates()` (`constructive.cpp:737`) builds every candidate as

    base = sqrt(SumA)      w = base*s*sqrt(a)      h = base*s/sqrt(a)

so a frame's **area is `SumA * s²` and its utilisation is exactly `1/s²`,
independent of aspect**. The whole ladder collapses to one number per rung, and
44 of the 55 profile dicts run the same one:

| rung | s | utilisation |
|---|---|---|
| 1 | 1.00 | 100.00 % (unpackable by construction) |
| 2 | 1.05 | 90.70 % |
| 3 | 1.10 | 82.64 % |
| 4 | 1.20 | 69.44 % |

The label's 96.6 % is `s = 1.0174` — **between rung 1 and rung 2, where the
shipped ladder has no rung at all.** That is what made the axis worth one probe
despite the counter-prior.

## 1. Instrumentation

`constructive_l252.exe`, branched from the CURRENT shipping `constructive.cpp`
(md5 `e2c7b2f418ef2b70b6bff99f7adfbd37`), adds three stderr emitters behind
`ICCAD_L252=1`:

    L252TOT <SumA>              once, from frame_candidates()
    L252FRM <i> <w> <h>         the full ladder, post-sort
    L252TRY <i> <ok> <score>    per trial; ok=0 == the frame did not pack

`ok` is free of charge because the trial loop already `continue`s on a failed
pack **without consuming a trial** — so the loop is an escalating ladder and the
successes it reports are always the tightest ones available.

**Identity gate (`l252_identity.py`) — 102/102 PASS on both arms**, on the two
heaviest cases (n=120) across all 51 pool profiles, compared on raw stdout bytes:

| arm | question | result |
|---|---|---|
| A | probe with the flag OFF vs stock | 102/102 identical |
| B | probe with the flag ON vs stock | 102/102 identical |

B is the arm that matters — it is the configuration the measurement runs in.

## 2. What was measured (OOS s1, n ≥ 101, 40 cases, weighted exp(n/12))

Per case, on the profile the **proxy actually selects** (the shipped selector,
reconstructed exactly as `l250_selection.py` does):

| quantity | s | utilisation | area vs label |
|---|---|---|---|
| the label | 1.0174 | 96.6 % | — |
| **s_fine** tightest packable, dense ladder | **1.1088** | **81.3 %** | **+18.82 %** |
| s_coarse tightest packable, shipped ladder | 1.1222 | 79.4 % | +21.66 % |
| s_landed what `layout_score` selects | 1.1303 | 78.3 % | +23.41 % |
| s_eff the final bbox, after compaction | 1.1260 | 78.9 % | **+22.49 %** |

🔑 **The framework validates against an independent measurement.** `s_eff` is
computed here from raw positions with no label involved, and it reproduces
L251's `area_gap = 0.2256` (+22.56 %) to **0.07 pp**. The s-ladder view of
`area_gap` is correct.

### 2.1 The split

    achieved            +22.49 %  of area over the label
      | selector slack   -1.43 %  s_landed -> s_coarse   (9/40 cases)
      | ladder grain     -2.38 %  s_coarse -> s_fine     (26/40 cases)
    tightest packable   +18.82 %  <- THE CLIFF
      | unreachable      18.82 %  no frame search can cross this
    the label             0.00 %

**Cliff = 18.82 / 22.49 = 83.7 % of the area deficit.** Priced on L251's own
method (`QF = 1 + 0.5*(hpwl_gap + area_gap)`, base 1.2511):

    area_gap 0.2256 -> 0.1882   QF 1.2511 -> 1.2324   =  +1.50 % of quality
    area_gap 0.2256 -> 0.0000   QF 1.2511 -> 1.1383   =  +9.18 %   (L251)

So the frame axis owns **+1.50 pp of the +9.18 pp area prize — 16.3 % of it.**
The other 83.7 % is the packer's reachable set, not the outline search.

## 3. Two granularity artefacts that had to be closed first

The coarse `s_min` is an **upper bound** on the cliff, and `l252_gap.py` shows
why — 27/40 cases have an unresolved interval (median width 0.048, max 0.100)
because 1.05 fails and 1.10 packs. Running the winning profile again on a dense
26-rung ladder (1.00→1.25 step 0.01) moved 26/40 cases, but only from 1.1222 to
**1.1088** — the edge is genuinely sharp.

The other 13/40 cases had **no failure below `s_min` at all**: their tightest
candidate packed. That is not the ladder stopping early, it is the clamp
(`w >= max(pre_w, max_iw) + FRAME_EPS`) — a frame cannot be narrower than the
widest block. Real geometry, and it sits at s = 1.11…1.20 on those cases.

## 4. What this closes, and what it does not

**Closes**: the frame ladder, the frame *selector*, and "the label's density is
reachable if we search the outline better". The real statement is the one
`HANDOFF_2026-08-28_RESEARCH.md` §5 predicted: **this packer cannot exceed
~81.3 % utilisation on the heavy band against the label's 96.6 %** — M27 from
the other side, now with a number. The teammate's *"the cliff is not a slope"*
reproduces on our packer at 83.7 %.

**Does not close**: the +1.50 % is real but is an *upper bound on the lever*, and
it is optimistic in three ways that all point the same way:

* it assumes a tighter frame does not cost hpwl;
* it assumes it does not cost violations — and `constructive.cpp:1900` says the
  opposite outright (`max_trials` is capped at 4 precisely because
  `layout_score`'s 150000×bv weight "picks low-violation but area-bloated
  outlines"), so the looser frame is being chosen *to buy violations*, and
  §3.1 already shows vrel is a **surplus** we would be spending;
* a denser ladder costs wall — every rung below the cliff is a failed pack.

⇒ Anyone re-opening this must price it as an **area/violation trade**, not as
free area. The three-line decomposition above is the right frame for that, and
`l252_cache.pkl` (40 cases × 51 profiles, positions + full ladder) makes any
follow-up free.

## 5. Method notes

* **`_l137_env()` is non-empty at ≥40 cores.** Building the binary's stdin with
  `gnn_hint=None` produces a *different case* than the 48c deployment path. The
  byte-comparison gate in `l252_fine.py` caught it on the first run; without it
  the whole dense sweep would have measured the wrong inputs and printed a
  perfectly plausible table. Third member of the `probe-import-time-silent-nooks`
  family this month.
* **`env.update(env_over)` means the profile dict beats the ambient
  environment** (`optimizer_constructive.py:2178-2179`). Setting
  `ICCAD_FRAME_SCALES` in the shell is a silent no-op for the 44 profiles that
  set it themselves. The sweep injects it into the profile dict instead.
* **The msys g++ needs `C:\msys64\ucrt64\bin` on PATH**, not just an absolute
  path to the exe — otherwise it exits 1 with *empty* stderr, which is exactly
  the silent-compile-failure shape `windows-msys-path-silent-sa-fallback`
  records.
* `s_min` means "the greedy placed every block", not "the result is good". That
  is the right definition for a reachability bound and the wrong one for a
  quality claim.

## 6. Files

```
l252_patch.py       branches constructive.cpp -> constructive_l252.cpp (5 patches)
l252_identity.py    the two-arm byte-identity gate      <- run this first
l252_frames.py      the coarse measurement + l252_cache.pkl
l252_gap.py         is s_min a cliff edge or the ladder's granularity?
l252_fine.py        the dense-ladder sweep (with the input-construction gate)
l252_identity.log l252_frames.log l252_fine.log
```
