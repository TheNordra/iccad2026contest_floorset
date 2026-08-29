# Handoff 2026-08-30 — research. Two questions, both now precisely posed.

**Read `HANDOFF_2026-08-29.md` for what was measured and closed.** This file is
only the forward-looking part, and it starts with what NOT to re-open, because
this session spent most of its time closing axes.

You are picking up two questions:

1. **Why can't L252's +1.50 % frame bound be collected?** — answered below, and
   the answer names a mechanism nobody has tried.
2. **The one unattacked path: a placer that decides where the big blocks go
   before the small ones fill the gaps.**

---

## 0. Shipping state — frozen, do not touch

| | |
|---|---|
| uploaded | `build_submission.D/cadc1075.tar.gz`, 408,795 B, Drive **Final** `1FDF1doINpSBKcr2OpL9PI19H04YLXyC4` |
| identity | `op_wrapper.py` md5 **`1c326784de7cd9246cd1f380e2842668`** |
| 48c Linux | **1.2264069637381392**, feasible 100/100 |
| projected | NET +5.224 % vs beta, graded **0.87818**, **rank 2**, margin over r2 **1.00 pp** |

`constructive.cpp` has not been modified in any of L252–L266. Every probe is a
separate `constructive_lXXX.exe` branched from md5 `e2c7b2f418ef2b70b6bff99f7adfbd37`.

## 1. The bound set — this is the project's answer to "why are we 17.6 % behind"

Every search-side decision now has a perfect-information bound:

| axis | bound | source |
|---|---|---|
| ordering | +0.005 % | M26 |
| seed | +0.001 % | M68 |
| shape | +0.099 % | M79 |
| selection over the pool | +0.0124 % | L250 |
| **frame** | **+1.50 %, and uncollectable — §2** | L252 + L264/265/266 |
| topology imitation | no usable gradient | L253 |
| relocation / repair | +1.22 % bound; built, no score | L255 / L256–L258 |
| eviction | geometry proven; no score | L259–L263 |

Against a generation deficit of **17.6 %** (hpwl +11.57 / area +9.18 / vrel
**−3.88, a surplus we already beat the label on**).

⚠️ Do not re-derive any of these. Each cost a probe with an identity gate.

---

## 2. Why the +1.50 % frame bound cannot be collected

L252 decomposed the frame axis on the heavy band:

    achieved                 +22.49 % of area over the label
      selector slack          -1.43 %   s_landed -> s_coarse   ( 9/40 cases)
      ladder grain            -2.38 %   s_coarse -> s_fine     (26/40 cases)
    tightest packable        +18.82 %   <- the cliff, unreachable
    label                      0.00 %

The **+1.50 % is the quality conversion of the reachable −3.81 % of area**, and
almost all of it is the **ladder grain**. To collect it you need rungs near the
per-case cliff. Three ways were tried and priced:

| ladder | quality (40 cases) | max-setter | NET (max-bound) | transfers? |
|---|---|---|---|---|
| dense, 26 rungs | +0.745 pp | **×1.48** | **−6.46 pp** | **yes** — positive on all 4 halves |
| coarse, 7 rungs | +0.359 pp | ×1.016 | +0.12 pp | no — flips (−0.188 / +1.148) |
| tuned, 4 rungs (same count as shipped) | +0.459 pp | ×1.012 | +0.28 pp | no — flips in **both** schemes |

🔑 **The dilemma, stated exactly:**

* **More rungs works but cannot be afforded.** Frames are tried in area order
  until `max_trials` *successes*, and L254 measured a *failing* pack as having
  already placed ~91 % of blocks. Every rung below `s_min` is paid in full, so
  26 rungs costs 1.48× on the max-setter — and the grader is max-bound at n=120
  (`d_max 3.204 s > sum/48 2.501 s`).
* **Fewer, better-placed rungs is affordable but is a fitted constant.**
  `s_min` is **per case**: L252 measured it across 1.0973 … 1.1971. A fixed
  4-rung ladder can only sit at the cliff for *some* cases; it wins there and
  loses elsewhere. That is why `tuned` flips sign in both split schemes and is
  ≈0 on the heaviest 20 cases. Same family as M56 / M79 / L258 —
  *case-idiosyncratic winner*, five times now.
* **And more tight frames degrade the selector.** With the dense ladder, 9/40
  cases get *worse* by >1 %: offering more over-tight candidates makes the frame
  chooser occasionally pick a worse one. M78's "adding candidates is harmful by
  default", at the frame level rather than the origin level.

### 2.1 🔑 The mechanism nobody has tried: adaptive frame search

Every attempt so far used a **fixed ladder**, which forces the choice between
"many rungs" (unaffordable) and "few fitted rungs" (won't transfer). But the
trial loop already knows, per case, which frames failed — and it throws that
information away.

**An adaptive search has no fitted constants and costs a constant number of
packs.** Bisect on `s`: try `s = 1.10`; if it packs, try tighter; if it fails,
try looser. Five probes locate the per-case cliff to ~0.01 — the same budget the
shipped ladder already spends, and *strictly better placed* because it adapts.

Why this is the right shape:

* it collects the **ladder grain** (the −2.38 %, i.e. most of the +1.50 %)
  without paying for rungs that were never going to pack;
* it has **no constants to fit**, so L266's failure mode cannot apply;
* the machinery is a ~30-line change to `run_pipeline`'s frame loop, and both
  gates already exist (`l252_identity.py --probe X --flags Y`, then
  `l263_quality.py --arms ...`, then `l264_wall.py`).

⚠️ Two things to watch. (a) The **clamp**: `frame_candidates()` floors every frame
at `max(pre_w, max_iw) + FRAME_EPS`; anything constructing a frame outside that
function must re-apply it, or it demands frames narrower than the widest block
and fails instantly while *looking* like a packer limit (this cost an hour in
L256). (b) Bisection changes which frames get compared, so `layout_score`'s
selection among them changes too — L263's 9 losing cases say that selection is
where tight frames go wrong.

**Honest expectation**: the ceiling is L252's +1.50 %, minus the cliff you cannot
cross, minus whatever the selector gives back. Call it +0.5…+1.0 pp if it works.
That clears the 0.30 % ship bar but is not the 2.2 % needed for rank 1.

---

## 3. The one path never attacked: decide the big blocks first

This is where the 2.2 % actually is.

### 3.1 What is known, and it is a lot

| | |
|---|---|
| L253 | our topology is **already** the label's — `d_hard` 6.8 %, and the label is not an outlier vs our own pool. Moving *closer* to it makes cost **worse** (the nearest candidate costs +13.7 %) |
| L252 | but we pack at **81.3 %** utilisation against the label's **96.6 %** |
| L254 | the jam is **not** lack of area: ≥ **10.45×** free at the moment it gives up, median 3 blocks left, jammed 3.4 pp below its own frame's allowance |
| **L259** | **it is contiguity.** At the jam the **largest** unplaced block has **ZERO** legal positions while smaller ones have thousands. The free space is shattered into pieces all smaller than the biggest thing left |
| L260 | displacing exactly **one** placed block opens a slot — 8/8 cases, median 0.75 % of the design |
| L261 | the cascade terminates: a greedy ejection chain completes **7/8** layouts at median **2** evictions. **Constructive** — a legal layout exists at a frame the shipped packer refuses |
| L255 | the prize curve: re-place ~2 % → +1.22 %; **~10 % → +3.23 %**; ~15 % → approaches the label |

🔑 **Read L259 + L261 together.** The greedy commits small and mid blocks into
positions that leave no home for a big one; but the *instance* admits a packing —
one eviction proves it. **The defect is the order of commitment, not the search.**

### 3.2 Why the cheap versions failed, so you don't repeat them

* **L256** shrank the frame and re-packed with the *same greedy* → only ~2 %
  deep, isolated −0.25 %, and its deployment was net negative.
* **L262** put eviction *inside* `pack_in_frame` → `s_min` genuinely fell
  (81.6 → 82.5 % util) and it **still produced no score**: portfolio +0.45 %,
  isolated 75 better / 91 worse. Area improves; hpwl and vrel pay it back.
* Both are repairs bolted onto a placer that has already made the wrong
  commitments. **A repair cannot fix an ordering defect.**

### 3.3 The shape of the thing to build

**Place the large blocks first, into a frame sized from the structural bound, and
let the small ones fill the residue.** Concretely, the pieces already exist:

* `ΣA / 0.968` is the **structural lower bound on area** and is already used and
  validated as a label-free baseline by L114's shape LP — that is your frame
  target, not a heuristic scale ladder.
* the **constraint-graph LP** (`build_and_solve`) legalises overlapping input and
  minimises exact HPWL. **L128 measured it holding the label's own arrangement at
  96.6 % density, feasible 100/100, scoring 1.083368** — so the legaliser is not
  the bottleneck, the *constructor* is.
* `l261_eject.py` is a working ejection chain; `l259_bruteforce.py` is a raster
  feasibility oracle. Both are offline but both are correct and reusable as the
  inner repair of a big-first constructor.

⚠️ **The one prior attempt**: L129 wrote a global placer from scratch (quadratic
placement → area-balanced bisection → cycle-free longest-path compaction → LP
polish) and reached **1.745 against the shipped 1.237**, winning 2/64 cases,
portfolio delta +0.010 %. Its own memory names the remaining work as **full
GORDIAN alternation** (solve → partition → re-solve under region constraints →
re-partition). Read `[[l129-global-placer-first-wins]]` before starting; it also
records that **hpwl work before the LP is not a lever** (the LP re-solves HPWL
exactly inside the topology), so effort belongs in *ordering and partitioning*,
not in wire heuristics.

### 3.4 The first probe, if you want a cheap gate before committing

**Big-first ordering, oracle-free**: re-run the shipped packer with the item order
forced to descending area (`ORDER_*` knobs exist; a probe binary can override the
sort key), and measure `s_min` and true cost. It is a one-constant change and it
tests the core claim — *is the commitment order the defect?* — for the price of
one arm.

M26 measured that injecting the **perfect** `fp_sol` order is worth +0.005 %, so
do not expect ordering alone to score. But M26 measured ordering *within the
existing frame regime*; the claim here is different — that big-first changes
which frames become **packable at all**, which is `s_min`, not cost. Measure
`s_min` first (`l262_smin.py` is the tool, point it at your probe binary).

---

## 4. Do NOT re-open

* Anything in `HANDOFF_2026-08-28_RESEARCH.md` §2, or §1 of this file's bound table.
* **Pool additions of any kind** — L248's curve is negative from the first
  profile, and L257 re-confirmed it for twins (NET −0.15 pp at K=1).
* **Twin deployments to rescue a weak mechanism** — L257/L258 showed the twin's
  ceiling is an identity (`argmin` over a union = `argmin` over per-pair
  `argmin`s), so it can never beat "keep the better of each pair", and the
  affordable gated version transferred at **0 %**.
* **Fixed frame ladders** — §2.
* fp_sol-supervised ML (user ruling 2026-08-05). Offline oracle probes that read
  labels are fine and are what L250–L253 are.
* `ICCAD_ANCHOR_W` sweeps; pool pruning.

## 5. Traps this session paid for — all of them silent

1. **`_l137_env()` is non-empty at ≥40 cores.** Building the binary's stdin with
   `gnn_hint=None` builds a *different case* than the 48c deployment path. Gate
   your input construction byte-for-byte against a spy on `_run_profile`.
2. **`env.update(env_over)` — the profile dict beats the ambient environment**
   (`optimizer_constructive.py:2178`). Setting `ICCAD_FRAME_SCALES` in the shell
   is a silent no-op for the 44 profiles that set it themselves.
3. **`m67_oos_probe` deletes every `ICCAD_*` at import.** Snapshot `os.environ`
   and `sys.argv` *before* the import; a probe of mine read its own `ITERS=40` as
   the default 6 and reported a "cap" that was not one.
4. **msys g++ needs `C:\msys64\ucrt64\bin` on PATH**, not just an absolute path —
   otherwise it exits 1 with **empty stderr**.
5. **Identity gates cut both ways.** For an *instrument* arm B must PASS
   (stderr-only). For a *mechanism* arm B must **FAIL** — a passing arm B means a
   silent no-op, and that is exactly how L256's frame-clamp bug was caught.
6. **`g46` and the running bbox are add-only.** Anything that removes a rect must
   rebuild both, or every later overlap test is against a stale index — a wrong
   layout, not a crash.
7. **A metric can be bounded by construction.** `free / (all unplaced area)` is
   `(tot·s² − placed)/(tot − placed)`, structurally > 1 for every `s > 1`. Ask
   whether your discriminator *could* have come out the other way.
8. **A weighted mean can improve while the count goes the wrong way** (L263:
   −0.52 % weighted, 75/91 on count). Report both.
9. **A transfer headline can read 100 % while your candidate flips sign**
   (L266). Read the per-arm per-half table.
10. **12 cases is not 40** (dense ladder: −1.68 % → −0.74 %).

## 6. Tooling — all built, all reusable, no changes needed

```
l252_identity.py --probe X --flags Y   two-arm byte-identity gate for ANY probe binary
l262_smin.py                           does a mechanism lower s_min? (ON/OFF, dense ladder)
l263_quality.py --arms ...             portfolio + isolated true cost, arbitrary arms
l264_wall.py                           same-batch min-of-N wall, all arms, max-setter
l266_splithalf.py                      free transfer check -- RUN THIS BEFORE ANY s2 CAPTURE
l257_twin.py                           exact offline pricing of any twin set
l258_maxsetter.py / l258_gate.py       per-profile timing; affordable-set + transfer
l259_bruteforce.py                     raster feasibility oracle for a jam state
l261_eject.py                          working ejection chain

l252_cache.pkl    40 cases x 51 profiles: baseline positions + full frame ladder
l257_cache.pkl    the same with ICCAD_L256=1, plus true costs for BOTH arms
l264_dense40.pkl / l265_tuned40.pkl / l264_wall.pkl / l260_mincut.pkl / l261_eject.pkl
```

`l252_cache.pkl` + `l257_cache.pkl` make any further decomposition on the heavy
band free, with no solving.

Reports: `L252` … `L266` in `L2xx_REPORT.md` / `L260_L261_REPORT.md` /
`L264_L265_REPORT.md`.

## 7. If you only do one thing

**§2.1 — adaptive frame search.** It is small, it has no fitted constants (so
L266's failure mode cannot apply), the gates are built, and it is the only way
left to collect any part of L252's +1.50 %.

**If you have days rather than hours: §3.** That is where the 2.2 % is, and L259
now says exactly what a new placer has to do differently — *commit the big blocks
first* — which is a much narrower brief than "write a global placer".
