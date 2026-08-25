# L172 — the medians were replaced, and the shipped depth map went negative

**Date** 2026-08-24 · **Status** two changes staged, in-set and Linux gates NOT yet run

---

## 0. What arrived

Three files, published 2026-08-23, downloaded to `beta_2026-08-23/`:

| file | what it is |
|---|---|
| `C_beta_leaderboard_update_20260823.csv` | an UPDATED leaderboard |
| `C_median_runtimes_beta_hidden_update.csv` | UPDATED per-case medians, plus a new `max_s` column |
| `C_beta_evaluation_report_hidden_final.txt` | the Beta evaluation report, final version |

**Both tables are authoritative on the same evidence the old ones were.** Feeding
the new medians through the published cost formula reproduces our updated graded
`total_score`:

```
raw_score reconstructed   1.3206649447461   official 1.3206649447461247
total with OLD medians    0.9245185859206   old official 0.9245183669982832
total with NEW medians    0.9265866625612   NEW official 0.9265861161320369
```

6e-7 relative, exactly what rounding the published medians to three decimals
predicts. `l172_depthmap.py` does this check on import of the data.

### We are rank 4 now, and it was not us that moved

```
rank 1  0.85863  raw 1.08449  rt 110.91      NEW entry
rank 2  0.88819  raw 1.20772  rt 110.70
rank 3  0.89933  raw 1.28476  rt  24.54      = raw x 0.7 EXACTLY -- floor on every case
rank 4  0.92659  raw 1.32066  rt  52.07      us (was 0.92452, rank 6)
rank 5  0.95071  raw 1.17046  rt 133.15      NEW entry
```

Two entries were replaced; eight are bit-identical to the old table on
`raw_score` and `total_runtime`. Our own `total_score` got 0.22% WORSE without
us touching anything — the medians moved under us.

Rank 3 and rank 10 both satisfy `total = raw * 0.7` to the last digit, i.e. they
sit on the RF floor on all 100 cases. Ours is `0.70161`, 0.23% above it.

### Every one of the 100 medians fell

```
new/old ratio   min 0.484  p10 0.602  p50 0.742  p90 0.878  max 0.943
sum             295.72s -> 216.13s        fell 100 / rose 0 / same 0
```

---

## 1. The consequence: `_L157_DEPTH` had gone from a gain to a loss

The shipped map was derived (L165) as

```
largest k with  t_beta(n) + (dt_tan(n) + (k-1)*dt_pass(n))/f  <=  0.3046 * M(n)
```

on `M` = the 2026-08-19 table. That table is `new / 0.742`, so the shipped map
is a bet that the medians are ~35% higher than the newest measurement of them.

Priced on the new table, with quality arm-mixed from the committed flat
k=1/2/3 OOS arms (`l147_oos_*_r15g` / `l157_oos_*_k2` / `l165_oos_*_k3`,
240 disjoint cases each; the ledger records that arm-mixing reproduces a
really-run gated arm 100/100 on cost AND positions):

```
map              depths            qual s1  qual s2   NET at s_true =
                                                      x1.30   x1.15   x1.00   x0.90   x0.80
k=1 (kill sw)    {1:100}           +0.000%  +0.000%   +0.000  +0.000  +0.000  +0.000  +0.000
SHIPPED (L165)   {1:8, 2:8, 3:84}  +1.025%  +0.880%   +0.721  +0.415  -0.723  -1.980  -3.343
build on x1.15   {1:19,2:12,3:69}  +0.857%  +0.836%   +0.846  +0.846  +0.192  -1.057  -2.401
build on x1.00   {1:38, 2:9, 3:53} +0.607%  +0.622%   +0.614  +0.614  +0.614  -0.097  -1.369
build on x0.90   {1:52,2:18,3:30}  +0.415%  +0.445%   +0.430  +0.430  +0.430  +0.430  -0.260
build on x0.80   {1:69,2:17,3:14}  +0.251%  +0.212%   +0.232  +0.232  +0.232  +0.232  +0.232
```

`s_true` scales the new table — the honest way to say we do not know the FINAL
round's medians either. Each map is flat (RF cost exactly zero) down to its own
build scale and falls off a cliff below it.

**Shipped: x0.90.** Its worst case over x1.15..x0.80 is −0.26%, against the
shipped map's −3.34%. It is a strict SHALLOWING of the old map — **0 cases
deeper, 61 shallower** — so it cannot spend more wall than what ships today on
any median table whatsoever.

```
_L157_DEPTH  {1: 52, 2: 18, 3: 30}
OOS vs the L147 k=1 anchor:  s1 +0.4153% (116 moved / 18 worse)
                             s2 +0.4452% (116 moved / 20 worse)
```

### It is free, and that is measurable

```
                                 spends   free left  overspent  off floor   graded
beta package (no LP at all)       0.00s     14.72s      0.97s     18/100   0.926587
+ L147 tangent, k=1               3.52s     12.35s      2.12s     21/100   0.932722
+ the SHIPPED old map            14.54s      4.82s      5.61s     47/100   0.948344
+ the x0.90 map                   7.70s      8.17s      2.12s     21/100   0.932722
```

The x0.90 map lands on the same `graded` and the same 21 off-floor cases as
tangent-only: its entire quality gain costs **exactly zero** RF.

### Caveats, stated once and true of every row above

1. **My re-derivation is not a bit-exact reconstruction.** Running the same
   formula on the OLD medians reproduces the shipped map on 91 of 100 block
   counts, not 100. Eight of the nine disagreements are the shipped map being
   *shallower* than my formula, so the shipped derivation carried a constraint
   this one does not. The shipped map itself is used verbatim in every
   comparison, so the pricing of what ships today is unaffected.
2. **The first LP pass's seconds are missing from every number here** — and
   from every derivation before it. `t_beta` is the beta package, which had no
   shape LP at all. Including them pushes every case nearer the RF edge and
   makes the deep maps look worse, so this is conservative in the direction of
   the conclusion.
3. The final round's medians will move again. That is what the `s_true` grid is.

---

## 2. `requirements.txt`: the open question is answered, against us

`HANDOFF_2026-08-24.md` logged this as an open question worth 5.4%. The final
evaluation report §2(a) settles it:

> Your requirements.txt must list EVERY package your code imports, including
> transitive dependencies. Do not assume any package beyond the Python standard
> library is available. If your optimizer uses torch-geometric, torch-scatter,
> torch-sparse, **scipy**, or any other third-party package, it MUST appear in
> requirements.txt.

Our package shipped a **0-byte** `requirements.txt` while importing torch,
shapely, numpy and scipy.

The memory note "adding scipy blows up torch/shapely" describes the ALPHA
failure mode, which was a file listing *only the new* packages and so losing
torch/numpy. The fix is completeness, not emptiness. Shipped:

```
torch>=2.5.0        numpy>=1.24.0    shapely>=2.0.0    matplotlib>=3.7.0
tqdm>=4.60.0        requests>=2.28.0 scipy>=1.11.0
```

which is `iccad2026contest/requirements.txt` **verbatim** plus scipy, with torch
raised to the `>= 2.5.0` the report requires for Python 3.13. No pins — the
report's failure mode (b) is a pinned old torch.

**`make_submission.py` force-wrote this file to 0 bytes and asserted 0 bytes.**
A staged edit alone would have been reverted by the next restage and then failed
the gate. Both the writer and the gate are changed; the gate now asserts the
exact content and separately asserts that each of torch/numpy/shapely/scipy
appears, so an incomplete list cannot pass.

`vendor/` is unchanged and still only appended to `sys.path` when
`import scipy` fails, so a system scipy always wins. The fallback is inside a
broad `except Exception`, so an ABI-mismatched vendored `.so` degrades rather
than raising.

---

## 3. Three axes closed, one re-opened

| axis | verdict |
|---|---|
| quality-aware depth rule (pick k per n from which k helped) | 🔴 **RED** — held out it does not beat affordability-only: +0.4453% vs +0.4452% one way, **+0.3856% vs +0.4153%** the other. Training worse-counts 7-8 become 16-17 on test. Same shape as L127's tally fitting. `l172_greedy.py` |
| spending depth on cases already PAST the floor | 🔴 **RED** — the marginal cost of `R^0.3` past the floor is not small. Starved-cases→k=2 is NET **−0.929%**, →k=3 is **−2.353%**, against the x0.90 map's +0.430%. The 38 starved n carry 35.7% of the corpus weight. `l172_overspend.py` |
| `max_s`, the new CSV column | ⚪ not scoring-relevant — it is the slowest submission per case (p50 331.6s, max 2930.9s, p50 ratio 181x the median) |
| **LP solver speed** | 🟡 **RE-OPENED.** L155 priced a speedup at **0.0000%** because on the old table the deep map was free anyway. On the new table it is not, and the deep map's quality (+1.02% / +0.88%) is measured and now unreachable. `l172_lpspeed.py`: |

```
   LP X              depths   qual s1   qual s2  NET @x0.90
  1.00x  {1:52, 2:18, 3:30}  +0.4153%  +0.4452%   +0.4302%     <= what ships
  1.50x  {1:45,  2:9, 3:46}  +0.5761%  +0.5643%   +0.5702%     +0.14pp
  2.00x  {1:40,  2:7, 3:53}  +0.7898%  +0.7156%   +0.7527%     +0.32pp
   free  {1:30,       3:70}  +0.8793%  +0.7919%   +0.8356%     +0.41pp ceiling
```

2.0x is exactly the `f* = 2x` gate L155 could not reach, and L156 closed the
whole row-removal family behind it. But it is now worth **+0.32pp** instead of
nothing, so it should be re-read before the deadline rather than after. The
ceiling with the arms we hold is +0.41pp.

---

## 4. The budget every future verdict must use

`l172_budget.py`. Old table on the left is what every prior verdict in this
ledger was priced against.

```
uniform slowdown     OLD table      NEW table
   1.25x              -0.09%         ~ -1.4%
   1.50x              -0.70%          -4.54%
   2.00x                             -13.01%

flat +dt per case, NEW table:  +0.05s -0.134%   +0.10s -0.322%   +0.20s -1.156%
free budget:                   38.02s  ->  14.72s      (0.97s already overspent)
```

**Being faster is worth at most +0.229% of score** — we are 0.161% above a hard
floor of 0.7000 — and 74% of that sits on three cases (91/78/71, n=112/99/92),
all of which are pool-bound, not LP-bound. Runtime is a constraint now, not a
source of gain.

---

## 4b. Collision, and the merge

Between 10:24 and 11:38 a second agent was working in this same tree. At
11:37:15 it reverted `optimizer_constructive.py` to HEAD -- wiping the depth-map
change and its comment block -- and applied its own **L171**: the shape LP's
`hpwl_baseline` predictor, shipped by code default as `_LP_HB_K = "0.2994"`.

The two changes are independent (L171 rewrites one dict inside `_shape_lp`,
L172 rewrites the `_L157_DEPTH` table) and both are now in the tree. Their
version is preserved verbatim at
`_quarantine/optimizer_constructive.THEIRS-L171.py`.

⚠️ **L171's own justification quotes "19.79s of budget"**, which is the
2026-08-19 table. The real figure is 14.72 s and 0.97 s of it is already
overspent. L171 costs 0.4 grader-seconds so that particular verdict survives
the correction, but the reasoning chain used the stale denominator.

L171 measured OOS s1 +0.0807% / s2 +0.0513%; measured independently here
against the arm-mixed control it reads s1 +0.0928% / s2 +0.0346%. Both agree it
is ~+0.06-0.07% mean. It is also, by its own comment, **the first shipped
mechanism that makes cases worse** -- 74/75 of 234/233 movers regress. Kill
switch `ICCAD_LP_HB_PRED=0`.
⚠️ It was measured on top of the OLD depth map. The new map runs far fewer LP
passes, so its effect must be re-measured on the merged tree before it can be
quoted.

## 5. State, and what has NOT been done

`op_wrapper.py` md5 `ba9ec1be898b00ddbc41207f70bb2bc1` → `243dcf3897b296769026878e64475b13`
→ **`815d02dae4639b880c4985ca63827b33`** after the L171 merge (§4b).
`bin/constructive_linux` md5 `bc9912072cd97b45b47a03adec7170ce`, **unchanged** —
both changes are pure Python, the ELF was not rebuilt.
`make_submission.py stage` PASS (1424 files, 35504121 bytes).
Previous tarball preserved at `_quarantine/cadc1075.tar.gz.pre-L172`.

❌ **No in-set gate has run.** `l172_gates.sh` is written and ready (det1, det2,
a k=1 anchor on THIS tree, and the L147 kill switch), scored by
`l172_verdict.py`, whose G3 is new: it asserts the map ACTUALLY FIRED by
checking that no case spends more passes than the map allows and that the
histogram moved off the old map's `{3:66, 2:24, 1:10}`. A gate reporting PASS
because nothing ran is itself a failure in that script.

❌ **No Linux lane has run.**

⚠️ **The box was not ours.** Two concurrent copies of `l170_oos_hb.sh` (one of
them reading a script file that was edited mid-flight) put 30+ `constructive.exe`
on a 48-thread configuration; the binary probe timed out and s1 fell back to
python SA on **132 of 240 cases** — weighted cost 9.900044 against a healthy
1.4169, feasible 138/240. Quarantined with its post-mortem in
`_quarantine/README.txt`. `l171_oos_hb.sh` and `l172_gates.sh` both take a
lockfile now. A second agent is running jobs in this same tree; nothing here was
measured while it was busy.

---

## 6. The pool multiplier the map did not know about (L182)

`_L157_DEPTH` was built from `budget = 0.3046*M(n)*s_med - t_beta(n) - dt_tan/f`,
i.e. assuming our pool costs what M73's pool cost. L181 measures it, route A
off, one box, 100 cases: **M73 112.77 s, current 130.21 s = 1.155x**, for
**+1.66% of quality** (weighted cost 1.281457 -> 1.260247).

Priced on the new medians, the pool's own bill is **-1.0153%** at P = 1.155, so
the pool trade is NET about **+0.65%** — positive, but thinner than a naive
uniform-scaling read (-0.69%) suggests.

For the map itself, RF is taken against k=1 **at the same P**, so it isolates
the map rather than the pool:

```
   P        RF @med x1.0     @x0.90      @x0.80    NET @x0.90
 0.800        +0.0000%     +0.0000%    +0.0000%     +0.4302%
 1.000        +0.0000%     +0.0000%    -0.6906%     +0.4302%
 1.155        -0.0326%     -0.6156%    -1.4563%     -0.1854%

 re-derived with P folded in (s_med = 0.90):
 1.000   {1:52, 2:18, 3:30}   +0.4302%   == what ships
 1.155   {1:69, 2:15, 3:16}   +0.2315%
 0.800   {1:25, 2:14, 3:61}   +0.7989%
```

So the shipped map is exactly optimal at P = 1.00, leaves ~0.37pp on the table
if P < 1, and costs ~0.42pp against the P-aware map if P = 1.155. Its worst
case across the range is **-0.19%**.

**P cannot be pinned from here.** 1.155 is measured with route A OFF; the
shipped package runs route A ON at >=40 detected cores, route A has never run
on the grader (beta was M73, which lacks it), its -32.2% at 48 real cores is a
projection, and on this box's 16 physical cores it costs 2.9x. True P is
somewhere in ~[0.78, 1.155].

Not changed, and deliberately: the map is what the L177 gates are running
against right now, and a 0.42pp refinement in one corner of an unpinnable range
does not justify voiding a 50-minute gate cycle. Revisit if P is ever measured.
