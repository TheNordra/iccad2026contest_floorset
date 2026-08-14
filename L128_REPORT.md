# L128 — Gate 0 for the analytical floorplanner: a topology cannot be transplanted

Route 4 of `HANDOFF_2026-08-13.md`, opened at the user's direction. **Nothing
shipped, nothing built beyond a probe**: `constructive.cpp`,
`optimizer_constructive.py` and the submission package are untouched.

**Verdict: the cheap version of the analytical pipeline is dead, and the reason is
measured, not argued.** The expensive version is not refuted — but it got
measurably harder, and it does not fit in the remaining days.

---

## 1. Why a gate before a solver

Three perfect-information bounds say the bottleneck is *not* what we feed the
packer — perfect order +0.005% (M26), perfect seed +0.001% (M68), perfect shape
+0.099% (M79) — while the gap to the label floor is **10.42%** (1.236792 →
1.107940, both re-measured here). By elimination the deficit is the **topology**,
which is exactly what an analytical floorplanner attacks. That is a good reason to
want one. It is not evidence that one would work, and the pipeline it needs has a
step nobody had measured:

```
analytical solve -> continuous, overlapping positions
                 -> extract a topology
                 -> LEGALISE into non-overlapping rectangles   <- unmeasured
                 -> our post-processing
```

The teammate measured the analogous step through **their** packer and it was a
cliff (`oracle_pack_ceiling`: a full-marks answer legalised to 3.7518; zero slack
put 42 cases at 1.017 each but left 58% unplaceable). Nobody had measured it
through **our** constraint-graph LP, which is a far better legaliser for the job.

**And it needs no new solver.** `build_and_solve` already:
* **derives** its topology from whatever positions it is handed
  (`optimizer_constructive.py:2140` picks the max-gap relation per pair);
* writes separation rows that are exact non-overlap **in the new coordinates**, so
  an *overlapping* input is legalised rather than preserved (the rhs is the
  current gap — negative when they overlap — and the row forces it open);
* minimises **exact** HPWL (an aux column and two rows per (edge,axis) give a true
  absolute value). Only the *area* constraint is linearised, which is what
  `rho=0.06` bounds.

## 2. Harness calibration (both exact)

| arm | weighted total | expected |
|---|---|---|
| `ours` — the shipped 48c anchor | **1.236791670** | 1.2367916697725434 ✓ |
| `calib` — fp_sol verbatim | **1.107940199** | 1.1079 ✓ |

## 3. The gate

Feed the LP the **label's arrangement carrying our shapes**, blending each free
block's aspect ratio from the label's to ours at constant area (`blend=0` is the
label's own shape, `blend=1` is ours; areas are identical to 0 ulp on every
movable soft block, M79). 100 cases, official evaluator, `--scale 1.0`:

| blend | LP infeasible | feasible | weighted total |
|---|---|---|---|
| **0.00** | 2/100 | **100/100** | **1.083368** |
| 0.02 | 11 | 94/100 | 1.350987 |
| 0.10 | 26 | 80/100 | 3.087488 |
| 0.30 | 98 | 2/100 | 9.997733 |
| 1.00 | 100 | 0/100 | 10.000000 |

**A 2% move of the aspect ratios away from the label's is already worse than our
shipped layout** (1.3510 vs 1.2368), and by 30% the LP is infeasible on 98/100.
This is a **cliff, not a slope** — the same shape the teammate found on their
packer, and the same shape M52 recorded as the "zero tolerance band".

### The mechanism, and why it is not a bug in the LP

The label packs at **96.6%** utilisation; we pack at **82.2%** (L95). A layout at
96.6% density has ~3.4% of total slack, and its constraint graph is a **rigid
interlock** — every pair relation is tight. Changing any block's aspect consumes
slack that is not there, and the separation chain has nowhere to give. That is
L121/L122's finding arriving from the other direction: there, pinning shapes into
the LP was infeasible at a **0.05%** shape change and the elastic phase-1 blamed
**separation** rows, not area and not bbox.

⇒ **Topology and shape are not separable.** There is no such thing as "the
label's topology" that can be handed a different set of rectangles. The label's
topology is only feasible for the label's shapes.

### 🚨 A correction I had to make mid-probe

The first version of this curve read **31/100 infeasible at blend=0.05 and
100/100 by blend=0.50** — much sharper than the truth. That was **my harness, not
the geometry**: `decompose()` makes each cluster ONE RIGID UNIT whose member
offsets come from the seed, and separation rows skip intra-unit pairs
(`if ui == uj: continue`). Resizing a clustered member without re-packing the
cluster interior bakes in an overlap that nothing can ever fix. Re-packing a
cluster interior is `make_group_item`'s job, in C++. The table above reshapes only
blocks that are their own unit, and the claim is scoped to them.

The general lesson is the one M75 and M78 both recorded in different words: **a
seed that does not respect the deployed structure produces a confident wrong
number.** Here it would have overstated the cliff by ~10×.

## 4. What this closes, and what it does not

**Closed — the cheap pipeline.** "Analytical stage → extract topology → legalise
with our LP" cannot work. Our LP is a *local repair tool*, not a legaliser: it
requires its input to be essentially legal already. Both legalisers we have (the
greedy packer, via the teammate's probe; and the LP, here) fall off a cliff on a
foreign arrangement.

**Not closed — an analytical solver with its OWN legaliser.** That solver would
never transplant a topology; it would produce its own arrangement and legalise it
with its own shapes, so §3 does not directly refute it. But §3 does raise its
price: to capture the label's HPWL you need the label's *density*, and at 96.6%
utilisation legalisation has no slack to work with. "Reach the topology" and
"reach the density" turn out to be the same requirement, and it is all-or-nothing.
That is a multi-week build (analytical solve + density penalty + a real
legaliser + shape optimisation), against **7 days** in which the submission also
has to be uploaded.

**One number worth keeping.** At blend=0 the LP takes the label layout to
**1.083368**, i.e. **2.2% BELOW the fp_sol verbatim floor of 1.107940**. The
"floor" is not a floor — it is just where the label happens to sit, and exact-HPWL
optimisation under the label's own topology beats it. Unreachable without the
label, but it re-prices the headroom: the true ceiling of a perfect topology is
~12.4% below us, not 10.4%.

## 5. Honest range

- one corpus (in-set 100) and one legaliser; `--scale auto` and `--scale 1.0` were
  both tried and agree
- clustered blocks keep the label's shape throughout §3, for the reason in the
  correction above — the cliff is measured on free blocks only
- `iters=2`; the LP is a fixpoint in 2–4 passes (M53-L3) and infeasibility on pass
  1 is not something more passes fix
- the LP objective is the **shipped baseline-free** one, not the label-derived one
  (`lp-baseline-is-label-derived`), so §3's costs are comparable to the shipped
  number rather than inflated by an oracle constant

## 6. Artefacts

| file | what |
|---|---|
| `l128_analytical_gate0.py` | `calib` / `ours` / `topolab` / `topo`, with `--blend`, `--scale`, `--all-blocks` |
