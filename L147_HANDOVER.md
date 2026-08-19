# L147 handover — the tangent cut, ready to ship

**One file changed: `optimizer_constructive.py`. No C++, no ELF rebuild.**
`bin/constructive_linux` md5 `6d43cf2cbfd9e4d578cd692277a7f868` — verified
unchanged from the first edit to the last measurement. Your
`_binary_matches_source()` passes untouched.

## What to set

    ICCAD_SHAPE_LP_R=1.5  ICCAD_SHAPE_LP_G=1.10
    ICCAD_SHAPE_LP_TOL=0.006  ICCAD_SHAPE_LP_PRICE=1.0

All four default OFF/shipped-value; unset `ICCAD_SHAPE_LP_R` restores the shipped
band bit-for-bit (proved, not assumed — see Gate 1). If they ship as defaults,
flip them in `_shape_lp` the way L137's were flipped; the existing
`ICCAD_SHAPE_LP` master switch (cores ≥40, fail-closed) still gates the whole
lane, so nothing fires below 40 detected cores.

## The patch

`git diff e684453 -- optimizer_constructive.py` — 13 hunks, **trial-applied
clean on `origin/l113-route-a` @ `e60f06d`** (all hunks offset +39 lines, no
conflicts, because L137's additions sit above the LP block).

## What it is worth

| | |
|---|---|
| in-set 48c | 1.228473819832 → **1.197768284824** (+2.4995%) |
| OOS s1 240 | +2.0756% (219 better / 19 worse) |
| OOS s2 240 (disjoint) | +2.3138% (229 / 11) |
| feasible | 100/100 in set, 480/480 OOS |
| whole-solve wall | **1.0310x** |
| RF cost (measured medians) | −0.9726% |
| **NET** | **+1.10% (s1) / +1.34% (s2)** — bar is 0.30% |

The gain lands on `area_gap`: 0.194 → 0.155 on s1, 0.199 → 0.157 on s2.

## New anchor to regenerate

The bit-equality gates FAIL by design until re-anchored — this is a
reformulated LP matrix, not a regression:

```bash
cd /c/ICCAD_ml/ship_final/iccad2026contest && ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py -o ../results_L147_48c_anchor.json
```

`results_L147_r15g.json` already on disk is that run; rename/point `l113_ship_gate.py --anchor` at it.

## Linux: use the invariant gate, not bit-equality

The LP is degenerate and Windows/Linux scipy land on different optima (L119:
92/100 agree to <1e-9, 8 move). Judge on **your rewritten judge48**: every case
feasible, none worse than the pre-LP anchor by more than the recorded budget,
total still ahead. Do NOT expect a fixed digit string, and report the gain as a
range.

⚠️ We independently reproduced your judge48 finding along the way: the
**already-uploaded L136** is itself worse than `results_M80_48c_anchor.json` on
2 cases. The old invariant was unsatisfiable for any real package.

## The one risk, quantified

Medians are the beta field's. Quality is unaffected by them; only RF moves.

| final medians vs beta | 1.00x | 0.90x | 0.85x | 0.80x | 0.75x |
|---|---|---|---|---|---|
| NET | +1.53% | +0.93% | +0.59% | +0.23% | −0.12% |

Break-even ≈ **0.78x**: the whole field would have to get 22% faster. Observed
direction is the opposite — the leaders are the slow submissions (rank 1: 169 s,
RF 0.824), because buying quality with runtime is exactly what wins here.

R=1.3 was measured as a hedge and is **dominated at every median scenario**
(+1.42% vs +1.53% at 1.00x, +0.09% vs +0.23% at 0.80x): `area_g=1.10` gives R=1.5
**10 rows/unit** against R=1.3/g=1.05's **12**, so the stronger arm is also the
cheaper one. No hedge needed.

## Combined with L137 (measured, not assumed)

All the numbers above were taken on the **L136 base**, because that is what our
tree carried while the work was done. Re-measured on the teammate's head with
L137's defaults ON (48c in-set, official eval):

| | total | vs L136 | feasible |
|---|---|---|---|
| L136 (uploaded) | 1.228473819832 | — | 100/100 |
| L137 (teammate) | 1.227176561424 | +0.1056% | 100/100 |
| L147 on L136 | 1.197768284824 | +2.4995% | 100/100 |
| **L137 + L147** | **1.196679286011** | **+2.5881%** | 100/100 |

Additive prediction +2.6051%, measured +2.5881% ⇒ the overlap is **0.017pp, i.e.
99.3% additive**, which is what the mechanisms predict: the hint moves anchors
during packing, the tangent cut moves shapes in the post-pack LP.

⚠️ The wall figures in that run (312.3 s combined vs 323.1 s for L147 alone) are
single runs and the control's own spread is 2.8% p50 / 8.9% max — do not read a
speed-up into them. Only the min-of-3 numbers in §3 are timing evidence.

## If you want to re-verify anything

* flag-off bit-equality (12 min): run the eval with no `ICCAD_SHAPE_LP_*` and
  diff against `results_L136_48c_anchor.json` — expect 0/100 cost and position
  differences.
* kept-rate: set `ICCAD_SHAPE_LP_STATS=<file>`; one `n kept` line per case.
  98/100 on the shipped setting. A drop below ~90 means the area price
  mis-scaled and cases are being rejected by `hard_ok`, which loses the whole
  shipped LP gain on those cases, not just the increment.
* OOS: use `l140_oos_soft_audit.py` (restores every `ICCAD_*`), **not**
  `l137_oos_ab.py` — that one captures only `ICCAD_HINT_*` and would hand you a
  byte-identical A/B.

Full evidence: `L147_REPORT.md`.
