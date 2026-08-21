# L156 — A, B and C are all RED. The §5.2 line is closed.

L155 set the bar for `HANDOFF_2026-08-20` §5.2 (make the LP cheaper so depth
k=2 becomes affordable) at **f\* = 1.75×–2.0×**, and proposed four levers.
Three were implemented and measured. **All three buy nothing or less than
nothing, and the reason is structural and shared.**

**Cumulative measured speedup available on this line: 1.06×, against a bar of
1.75×. The line is dead — killed two days ahead of the 08-23 hard stop.**

`optimizer_constructive.py` is back at HEAD; nothing from this report ships.
Tools kept: `l155_lp_rows.py` (census), `l156_lazy_ab.py` (the A/B).

---

## 1. The results

| lever | what it did | measured |
|---|---|---|
| **A** vectorise the prune verifier | replace `solve_pruned`'s Python loop over 350,185 dropped terms with padded numpy column products | **1.007×** — noise |
| **B** lazy envelope rows (9.6% of the matrix) | omit a block's 4 envelope rows when it sits >m·span inside that edge; verify and repair | **0.82× / 0.84× / 0.98×** at m = 0.02 / 0.05 / 0.10 |
| **C** lazy area_tangent rows (14.9%) | emit only the 2j+1 tangents nearest the current width; verify and repair | **0.46× / 0.67×** at j = 2 / 3 |
| **D** push `prune_B` below the shipped 8 | (L155) | 1.06× at B=2, and B=32 is 0.51× |

Every B and C arm is **slower than the shipped matrix**, on the band that
carries the weight (n≥100, min-of-3, `l156_heavy.txt`).

## 2. 🚨 Why A failed: the hypothesis was wrong, and the profiler said so

L155 found 37% of LP wall outside the solver and I attributed it to the
verification loop. **Wrong.** Profiling the LP on n≥110 (`cProfile`, tottime):

    4.277s  scipy _highs_wrapper          (the solver, 50%)
    0.889s  add_hpwl_rows                 (1.511s cumulative, 18%)
    0.742s  build_and_solve's own body
    0.473s  builtins.max                  (181,625 calls)
    0.322s  add_ub                        (0.444s cumulative)
    0.211s  list.append                   (2,335,723 calls)
    0.208s  numpy.asarray
    0.085s  dsize                         (336,847 calls)

**The verification loop does not appear in the top 18 at all.** The 37% is the
row-construction machinery itself — diffuse across `add_hpwl_rows`, `add_ub`,
`max`, and two million `append`s — with no hot spot to fix. A's vectorisation
was measured at 1.007× on 100 cases and left the B=2/B=8 ratio at **0.94 in all
three runs, identical before and after**, i.e. it did not even help in the
situation it was designed for. Reverted.

## 3. 🔑 Why B and C failed — and this is the part that generalises

The base program takes **20 builds over 11 heavy cases, i.e. ~1.8 solves per
case.** A repair round is a *full re-solve*. So the trade is:

    save   8-15% of the rows on one solve
    cost   one entire extra solve when the guess was wrong

and it loses, every time, because the omitted rows are not over-supply — **they
are what defines the optimum.** The LP pushes the bounding box until an envelope
row stops it, and pushes the shape until a tangent stops it. "Only one or two
tangents can bind" is true *at the final point*, and finding out which ones
costs a solve. Row counts summed over rounds show it directly (n≥100):

    arm        rows(all rounds)   builds   t_solve    wall  speedup
    base             190,280         37     7.12s   11.23s   1.00x
    env0.02          216,186         45     8.61s   13.71s   0.82x
    env0.05          210,015         43     8.31s   13.36s   0.84x
    env0.10          184,028         38     7.21s   11.48s   0.98x
    tan2             367,288         74    15.56s   24.54s   0.46x
    tan3             280,912         58    10.07s   16.78s   0.67x

The only arm that reduces total rows at all is `env0.10` (184k vs 190k, −3%) —
and it *still* loses on wall, because it added a build.

⇒ **The whole "remove rows and verify" family is closed**, including the
one-sided-HPWL idea (keep `t ≥ +lin`, verify `t ≥ −lin`) that the row census
made tempting: the kept HPWL terms are precisely the ones whose sign is
uncertain — that is *why* the L112 prune kept them — so it would mis-guess on
about half and repair on nearly every case.

The only remaining way to drop a row is an argument that needs **no verification
round at all**, and L112 deliberately closed that door: it removed the
`|d_u| ≤ prune_B` clamp because a clamp makes the LP a *restriction* whose
optimum can legitimately differ. With position bounds at `±(W0+H0+1)` there is
no a priori "it cannot reach the frontier" argument to be had.

## 4. ⚠️ The exactness gate, and the control that made it readable

The relaxations are exact by construction and the measurement says so — with
one wrinkle worth recording.

On n≥110, `env0.10` and `tan3` each moved the objective on one case:

    case 93 tan3     rel 3.02e-02   1.640178 -> 1.689724   lp_pass attempts 2 -> 3
    case 96 env0.10  rel 1.45e-02   1.568509 -> 1.591329   lp_pass attempts 2 -> 2

Both moved the objective **worse**, which a relaxation cannot do — so neither is
a repair miss. Case 93 is explained by the attempts counter; case 96 is not, so
I ran a **null-arm control**: `env1.0` and `tan9` set the knobs to values that
omit **zero** rows.

    arm       rows      omitted  builds  t_solve   wall   degeneracy  moved
    base     115,341        0      20     4.43s   7.30s       -         -
    env1.0   115,341        0      20     4.42s   7.10s       0         0
    tan9     115,341        0      20     4.42s   7.20s       0         0

Identical row counts, identical build counts, **zero** vertex changes and zero
objective moves. So the plumbing is inert when nothing is omitted, and both
moves come from omission changing which degenerate vertex the solver returns,
which changes `lp_pass`'s cluster-break **freeze set** and hence the program.
That is exactly the caveat `L155_REPORT.md` §6 flagged, now observed.

🔑 **Methodology worth keeping: a null arm — the mechanism switched on but
configured to do nothing — separates "the mechanism perturbed the result" from
"the plumbing perturbed the result".** The attempts counter alone would have
left case 96 unexplained and the report hedged.

## 5. What this leaves

`HANDOFF_2026-08-20` §5.2 was the last open line, and it is now closed on
measurement rather than on the clock:

* the bar is 1.75×–2.0× (L155);
* the levers available are +1.06% (prune_B=2), and everything else measured
  negative;
* and per L155 §1, **an LP speedup has exactly zero standalone value** — we are
  on the RF floor, so there is no partial credit to bank either.

⇒ **k=2 stays unaffordable, and no further work on LP speed is warranted.**

That makes L147 — merged and shipped — the whole of the remaining upside:
**+1.17~1.27% NET, measured, Linux-verified (L153), sitting on a branch.**

## 6. Reproduce

```bash
cd /c/ICCAD_ml/ship_final && PYTHONIOENCODING=utf-8 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l156_lazy_ab.py --arms base,env0.02,env0.05,env0.10,tan2,tan3 --minn 100 --reps 3
```
```bash
cd /c/ICCAD_ml/ship_final && PYTHONIOENCODING=utf-8 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l156_lazy_ab.py --arms base,env1.0,tan9 --minn 110 --reps 1
```

⚠️ `l156_lazy_ab.py` drives `ICCAD_SHAPE_LP_LAZY_ENV` / `_TAN` itself and
**refuses to run if either is set in the environment** — an ambient value would
make every arm identical and report a clean 1.00× across the board.

⚠️ **The two knobs it drives are NOT in the shipping file** — they were reverted
with the rest, so the tool cannot run as committed. Apply the patch first, and
revert it after:

```bash
cd /c/ICCAD_ml/ship_final && git apply l156_lazy.patch
```

`l156_lazy.patch` (213 lines, `optimizer_constructive.py` only) is the exact
mechanism this report measured: `add_ub_lazy` in `build_and_solve`, the two row
families made lazy, a second `force_lazy` repair channel in `solve_pruned`, the
`lp_pass` routing guard, and the two env knobs in `_shape_lp`. It was re-applied
and re-verified against the null-arm control after the revert (identical rows,
identical builds, 0 moved), so it is the measured artefact rather than a
from-memory reconstruction.
