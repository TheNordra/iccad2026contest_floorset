# L344 — the good trees ARE heading for the label. The label is the direction of improvement.

**Verdict: on the quality axis, path ③'s value proposition holds.** As L340's SA gets
better it moves monotonically toward the label's arrangement, on 3/3 instances, and the
label sits *more centrally* than the good trees sit to each other. **But our own packer
is already 2–6× closer to the label than the SA ever gets, while scoring worse** — so
label-proximity is not by itself the thing that buys score, and the entire remaining risk
for ③ has moved onto the one axis this probe does not measure.

No shipping change. Offline oracle probe — reads labels for diagnosis, trains nothing,
ships nothing, touches no file on the shipping path (2026-08-05 ruling, same standing as
L250–L253). Tool: `l344_treedist.py`, output `l344_out.txt`.

---

## 1. The question, and why it was worth a measurement

`HANDOFF_RESEARCH_AFTER_L340.md` §3: path ③ is "predict the generator's B\*-tree,
supervised on `tree_sol`". L340 established that the manifold *contains* layouts beating
our packer (n=80, 5/5 seeds) and that the line dies on **search cost**, not reachability.
Supervised prediction replaces the search — but only if the label tree is where the good
trees are. The handoff's own framing: *if near, ③ aims right; if far, the model aims at
the wrong target and ③ must be re-scoped to "learn the representation, not the objective"*
(the generator optimised area only, `hpwl_gap` 1.13–1.60 against our 0.240).

The cheap non-circular form: L340 has already found good trees. Measure their distance to
the label. No training, no model, no GPU.

## 2. Metric, and why not raw tree edit distance

A B\*-tree is not a canonical encoding — many trees decode to the same floor — so raw
tree edit distance counts differences that do not exist. The project already owns the
right metric: **L253's `d_hard`**, the max-gap pair relation from
`optimizer_constructive.py:3346-3357`, imported verbatim from `l253_editdist.masks` so
these numbers land on L253's ruler.

`d_hard(A,B)` = fraction of the n(n−1)/2 block pairs whose feasible relation **sets are
disjoint** — pairs that genuinely have to be reordered.

**Two controls, both necessary.**

* **`d_perm`** — the same distance after permuting *which block is which* in A, keeping
  A's geometry untouched. This is chance level **at that layout's own density**, and it
  is required: a loose layout satisfies more relations per pair, which mechanically
  lowers `d_hard` against everything. Every headline below is quoted as `d/perm`.
* **`d(SA_i, SA_j)`** — how far apart two *independently found* good trees are. This is
  the scale that decides whether "near the label" means anything.

Secondary, genuinely tree-flavoured: the B\*-tree left-child rule **is** horizontal
abutment (`x_c = x_p + w_p` with y-overlap), so the abutment-edge Jaccard reads the tree
relation straight off the floor, for label and packer alike.

**Metric sanity (run separately):** `d(lab,lab) = 0.0000` exactly; `d_perm(lab,lab)` =
0.4728 / 0.4193 / 0.3956 = chance; and `relset` (mean relations satisfied per pair) for
the label vs our packer is **1.547/1.533, 1.649/1.662, 1.705/1.690** — agreement to
within 1 %, so the ours-vs-label comparison carries **no density confound at all**.

## 3. The gradient — this is the result

5 seeds per cell, HW = 2·HW\*, same instances L340 used (`config_{40,80,120}`, file 1).
`quality = 1 + 0.5(hpwl_gap + area_gap)`. `d/perm` < 1 means closer than chance.

| iters | n=40 quality → d/perm | n=80 | n=120 |
|---|---|---|---|
| 0 (random tree) | 3.128 → **0.84** | 4.326 → **0.96** | 4.648 → **1.01** |
| 10 000 | 1.375 → 0.92 | 1.574 → 1.03 | 1.657 → 0.92 |
| 100 000 | 1.180 → 0.48 | 1.332 → 0.80 | 1.391 → 0.65 |
| 2 000 000 | 1.073 → **0.34** | 1.150 → **0.60** | 1.154 → **0.24** |

**The random initial tree sits exactly at chance (0.84 / 0.96 / 1.01) and every
improvement in cost is also movement toward the label. 3/3 instances, monotone from
100k on.** The abutment fingerprint moves the same way: 0.0099→0.0472, 0.0016→0.0367,
0.0012→0.0181.

### Is the label inside the good basin? Yes — more central than the basin's own members.

| n | d(SA,label) | d(SA_i,SA_j) | ratio |
|---|---|---|---|
| 40 | 0.1603 | 0.1873 | **0.86** |
| 80 | 0.2572 | 0.4147 | **0.62** |
| 120 | 0.0947 | 0.1370 | **0.69** |

All three **below 1**: two independently found good trees are *further from each other*
than either is from the label. The label is not merely inside the basin, it is nearer its
centre than the samples are to one another. On the pre-registered reading this is the
"near" branch, unambiguously.

## 4. The counter-finding — our packer is already closer, and still worse

| n | d(ours, label) | ours/chance | d(SA@2M, label) | SA/chance | quality ours | quality SA |
|---|---|---|---|---|---|---|
| 40 | 0.0705 | 0.15 | 0.1603 | 0.34 | 1.1140 | **1.0733** |
| 80 | 0.0405 | 0.10 | 0.2572 | 0.60 | 1.2383 | **1.1500** |
| 120 | 0.0570 | 0.14 | 0.0947 | 0.24 | 1.2136 | **1.1537** |

**Our packer is 2.3× / 6.4× / 1.7× closer to the label than the SA's best trees, and
loses to them on cost every time.** (`d(ours,label)` = 0.0705/0.0405/0.0570 independently
reproduces L253's 0.0679 on a different corpus.)

That is not a contradiction; it is the missing half of L252/L253/L284:

| | arrangement like the label | on the B\*-tree manifold | utilisation |
|---|---|---|---|
| our packer | **yes** (7–10× closer than chance) | **no** — bottom-supported 95 / 61 / 76 % | 0.886 / 0.761 / 0.801 |
| L340 SA | partly (0.24–0.60× chance) | **yes** — 100 % at *every* iteration count, incl. iters=0 | high |
| label | yes | yes | ~0.97 |

L253 said *"same topology, different density"*. L344 supplies the other side: **the SA has
the density and lacks the topology; we have the topology and lack the density. Neither of
us has both; the label has both.** That is why "closer to the label" predicts lower cost
*within* the SA family and fails to predict it *across* families — it is one of two axes.

## 5. 🚨 Scope limit — the honest one

**This measures the quality axis only:** `1 + 0.5(hpwl_gap + area_gap)`. There is **no
violation term anywhere in it**, and the SA does not honour preplaced blocks at all
(L340 limit #2 — a B\*-tree cannot express a fixed coordinate; n=40/80/120 have 1/5/6
preplaced blocks, so these layouts are very likely *infeasible*, cost 10.0).

So the smooth monotone gradient found here **does not rebut M52's zero-tolerance cliff**
(one near-miss token → wR 1.232, ideal soft repair still 1.072). M52's cliff lives in
`exp(2·V_soft)` and in the hard preplaced/fixed constraints — the terms L344 does not
touch. The two findings are about different factors of the same cost and are fully
compatible: *quality is smooth around the label; feasibility is a cliff.*

**Do not quote §3 as "M52 was wrong."** It is not; it was measuring the other term.

## 6. What this does to path ③

The pre-registered dichotomy ("near ⇒ ② holds ⇒ ③ replays M52; far ⇒ re-scope") turns out
to be **underspecified**, because the measurement returns *near AND smooth* — a
combination it did not anticipate.

* **Green:** the label tree is where the good trees are heading, the target is a broad
  basin rather than a needle, and a model landing anywhere in it is rewarded
  proportionally. "Learn the representation, not the objective" is **not** forced —
  the generator's area-only objective produced a layout our own objective also likes.
* **Unchanged and now isolated:** both remaining blockers are on the axis §5 excludes —
  (a) preplaced cannot be expressed in a B\*-tree, and (b) M52's violation cliff. Neither
  was made better or worse by this measurement.

⇒ **The whole risk for ③ is now in one place, and it is directly measurable.** The next
probe is the violation version of this one: score L340's 2M layouts with the *full*
official cost (`m52_phase0_probe._cost_strict`, which checks preplaced/fixed hard
constraints — `tree_decode_probe._cost_of` does **not**), and measure how much of the
−0.09/−0.23/−0.06 quality win survives `exp(2·V)`. That is the same shape of question
M52 asked and it needs no training either.

## 7. Correction to L340's baseline (does not change any L340 verdict)

`l340_iters.py:24` hardcodes `OURS = {40: 1.1140, 80: 1.2178, 120: 1.2136}`. Per-case:

| | n=40 | n=80 | n=120 |
|---|---|---|---|
| shipped RF-SAFE (`l313_win48_rfsafe.json`) | 1.1140 | **1.2383** | 1.2136 |
| full-ungate arm (`l294_gate0.json`) | 1.1140 | **1.2178** | 1.2136 |

n=40 and n=120 are identical across arms, so only n=80 discriminates — and L340's value
is the **full-ungate arm, not the shipped one**. The direction is conservative (it
compared against our better number), so nothing L340 concluded flips; but the SA's n=80
margin against what we actually ship is **−0.0883**, not −0.0678. L344 reports both arms.

## 8. Reproduce

```bash
cd ship_final
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l344_treedist.py \
    --ns 40,80,120 --seeds 5 --hw 2 --iters 0,10000,100000,2000000
```

~30 min. Deterministic given seed (L340 confirmed bit-identical re-runs).
`l340_run.run()` was extended additively to return `pos`/`W`/`H`; `l340_seed.py` and
`l340_iters.py` read only `hg`/`ag`/`dt` and are unaffected.
