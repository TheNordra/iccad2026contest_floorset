# L148/L149 — the ledger re-scanned with the measured medians: five closes, no new opens

L147 proved that at least one ledger RED was an artefact of the old runtime model
(a machine-speed grid, killed by grid-worst). This is the systematic sweep of the
rest of that family with `l146_rf_price.py`. **Nothing new passes the bar; the
shipping config is unchanged.** The value is that five lines are now closed on
measurement rather than on a model.

## 0. The ruler has two scales, and picking the wrong one inverts verdicts

Where the time is added decides how it transfers to the grader:

| work added in | transfers as | because |
|---|---|---|
| the C++ pool (more profiles, deeper REFINE) | **ratio** | 51 profiles in parallel; the wall is the max-setter at 48 real cores, and this 32-core box is oversubscribed ~8.5x |
| the Python post-process (the shape LP) | **seconds, 1:1** | single-threaded scipy; it runs at the same speed on both boxes |

Measured affordability:

    ratio    1.15x -0.028%   1.25x -0.091%   1.50x -0.703%   2.40x -8.070%
    seconds  +0.10s -0.020%  +0.34s -0.343%  +1.00s -5.883%
             +0.10s and +0.20s on n>100 ONLY are exactly -0.0000%

Base for every arm below: **L137 + L147** = `1.196679286011`, 100/100 feasible.

## 1. Closed on quality, not on runtime

| arm | quality | wall | verdict |
|---|---|---|---|
| `ICCAD_REFRAME=1` | **−6.5276%** | 0.843x | **RED** — the first time its quality was ever measured. It is *faster* and much worse: a trade in the wrong direction. |
| full pool + full REFINE (`ADAPTIVE_POOL=0`) | **+0.0028%** | 1.560x | **RED** — the entire pruning family (M41/M42/M45) is worth 0.0028% of quality at 48 cores. Restoring it buys nothing and pays −0.8% RF. The line retires. |
| REFINE band-cut restore (`ADAPTIVE_REFINE=0`) | **+0.0000%**, 0/100 moved | 1.009x | **not applicable** — see §2 |

## 2. 🚨 L137's hint cap silently subsumed the REFINE band-cut

`constructive.cpp:2142`:

    if (HINT_MODE && HINT_REFINE > 0 && REFINE_ITERS > HINT_REFINE)
        REFINE_ITERS = HINT_REFINE;

With L137 on, `REFINE_ITERS` is clamped to 4 **globally**. So M49/M50's band-cut
(4 on n>100, 6 on mid, 12 default) is redundant on the heavy band and *stricter*
than before on the other two: **L137 also cut mid from 6 and light from 12 to 4**,
which was never separately attributed.

Consequently `ICCAD_ADAPTIVE_REFINE=0` is a bit-identical no-op — the flag IS
live (`_band_env(105)` goes from `REFINE_ITERS=4` to empty), the C++ default 12
just gets clamped straight back to 4. Asking the question properly means lifting
the cap, and the teammate's cap sweep predates the tangent cut:

| arm | quality | wall (single run) | RF (ratio) | NET |
|---|---|---|---|---|
| `ICCAD_HINT_REFINE=6` | +0.0089% (8/100 moved) | 1.094x | −0.0136% | **−0.0047%** |
| `ICCAD_HINT_REFINE=12` | +0.0969% (14/100 moved) | 1.323x | −0.2058% | **−0.1089%** |

**cap=4 survives the tangent cut.** The teammate's choice was right and stays.

## 3. The one that looked alive: LP depth composed with the tangent cut

L122 proved R=1.5 dominates k=2..k=12 **standalone**. The composition had never
been measured, and in quality it is additive: k=2 **+0.5967%** (80/100 cases
moved), k=3 **+0.7383%**. Priced on the deployed path (min-of-3, arms
interleaved, exclusive box, seconds model):

    added time  min -0.036  p50 +0.165  p90 +0.485  max +1.667  sum +23.18s
    wall        1.0761x
    RF cost     -1.0560%   (permuted p50 -0.3912% / p05 -0.9719%)
    NET         -0.4593%   bar +0.30%  ->  FAIL

🔑 **The tangent cut has already spent the affordable LP budget.** It adds
+9.67 s for −0.97% RF; k=2 adds **+23.18 s** — 2.4x more time — for +0.60%
quality. The median sensitivity only makes it worse (−1.21% at 0.90x).

## 4. What this leaves

Shipping recommendation unchanged: **L137 + L147**
(`ICCAD_SHAPE_LP_R=1.5 _G=1.10 _TOL=0.006 _PRICE=1.0`), in-set `1.196679286011`,
OOS +2.08~2.31%, NET +1.10~1.34%.

Closed by this sweep: `ICCAD_REFRAME`, the pruning-restore family, the REFINE
band-cut (subsumed), the REFINE cap above 4, and LP depth on top of the tangent
cut. With L147's own closes (R=1.3 dominated, band-gating inert) the
runtime-gated re-scan is finished.

⚠️ Still unmeasured, and the only remaining LP idea with a mechanism behind it:
the tangent cut's RF cost is a *tail* (p50 +0.047 s but max +1.092 s) while the
heavy band's free budget is ~+0.20 s/case. Band-gating `area_g` — coarser
tangents, i.e. fewer rows, on the biggest cases only — targets exactly that tail
and is deterministic. Not attempted here.

## 5. Reproduce

```bash
cd /c/ICCAD_ml/ship_final && bash l148_rescan.sh
```
```bash
cd /c/ICCAD_ml/ship_final && bash l149_chain.sh && "C:/Users/.01/anaconda3/envs/floorset/python.exe" l149_verdict.py
```
