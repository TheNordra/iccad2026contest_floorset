# L135 — a +0.36% boundary bug that a post-process cannot reach

Follow-on from L131. Submission untouched; everything here is read-only audit
plus offline simulation.

**Found: 10 preplaced blocks that can NEVER satisfy their boundary requirement,
worth up to +0.3566%.** Unlike L131 this one is **not** fixable from outside the
placer — three post-process attempts were built and all three fail for reasons
worth recording.

## 1. First, the ULP family is closed and had exactly one member

L131's bug was that `unary_union` is exact geometry with no tolerance. The other
two soft constraints are not exposed:

| constraint | check | ULP-exposed? |
|---|---|---|
| grouping | `unary_union`, exact | **YES** (L131) |
| boundary | `abs(bx - x_min_bb) < eps`, **eps = 1e-6** (`:527`) | no |
| MIB | compares `round(dim, 4)` (`:515`) | no |

So no further ULP wins exist. Anything left is a real violation.

## 2. The bug: MARGIN makes the outline 1e-4 too big for its own preplaced blocks

`constructive_l124.cpp:51` sets `MARGIN = 1e-4`, and the frame is sized

    w = max(w, max(pre_w + MARGIN, max_iw + MARGIN))        (:683, :1972)

i.e. **deliberately 1e-4 larger than the preplaced extent it must contain**.
Blocks packed against the frame land on `X.0001` and define the bbox. A preplaced
block sitting at the true `X.0` then misses the bbox edge by exactly MARGIN, and
the evaluator's threshold is 1e-6 — a hundred times tighter — so it scores a
boundary violation.

    case 54 blk 3   right edge 141.0    bbox xmax 141.0001   miss 1.0000e-04
    case 21 blk 39  right edge 128.0    bbox xmax 128.0001   miss 1.0000e-04
    case  7 blk  4  top   edge 100.0    bbox ymax 100.0001   miss 1.0000e-04

🔑 **The violation is unsatisfiable by construction.** The block's position is a
HARD constraint so it cannot move to the bbox, and the bbox is set by other
blocks 1e-4 beyond it. Every one of the 10 is preplaced — which is exactly why
they survived every boundary-repair pass the C++ already has (`snap_bnd`,
`final_group_boundary_nudge`, `final_single_edge_escape` all skip preplaced).

## 3. The prize

10 violations, priced with the **real `nsoft`** from the evaluator and the
multiplicative cost (the L131 §4 mistake, not repeated):

    weighted total now                1.236791669773
    if all 10 were removed            1.232381100018
    prize (UPPER BOUND)               +0.3566%

For scale: L131's shipped fix was +0.0758%, the in-set house bar is 0.05% and the
OOS ship bar is 0.30%. ⚠️ It is an **upper bound with real uncertainty** — it
assumes removing the violations changes nothing else, whereas any actual fix
changes the frame, and the frame is an input to the packing, so the layout moves.

## 4. 🚨 Three post-process attempts, and why each fails

**(a) Slide the violating block onto the edge.** Refused, correctly: all 10 are
preplaced and position is a HARD constraint. Moving them scores infeasible
(cost 10), which is far worse than one soft violation.

**(b) Snap anything near an edge** (`l135_bnd_verify.py`, first version). 423
blocks moved, **0 violations removed**, weighted total **−1.1401%**, and case 82
went `grouping_violations 0 → 2`.

🔑 The evaluator counts a block as touching within **1e-6**, so a block already
inside that band is not violating — and nudging it anyway is not free, because
the nudge is ULP-scale and **an ULP is exactly enough to break a cluster
abutment**. This is L131's mechanism firing in reverse, and handoff §3.1's rule
for the fourth time this session: a stage that optimises one term must not
quietly break another. Fixed by refusing to touch anything closer than the
evaluator's own threshold, and by re-running the L131 abutment repair afterwards.

**(c) Pull the bbox IN to meet the preplaced block** (`l135_shrink_verify.py`).
The only remaining direction, and it is geometrically blocked. On case 54, six
non-preplaced blocks define `xmax = 141.0001` and all six carry a RIGHT
requirement, so pulling them in by 1e-4 would keep them touching AND let the
preplaced block touch — except:

    blk 61 now overlaps blk 23 by (0.0001, 9.15)
    blk 68 now overlaps blk 52 by (0.0001, 15.1)
    blk 69 now overlaps blk 30 by (0.0001, 1.00)

**The packing is tightly abutted, so there is no slack to absorb the shrink.**
Absorbing it means propagating the 1e-4 through the whole constraint chain, which
is a re-compaction — not a post-process.

## 5. What a real fix would take, and why it was not built

The fix belongs where the bug is: **do not inflate the frame by MARGIN on a side
where a preplaced block carries the matching boundary requirement.** That is a
change to `constructive_l124.cpp:683` / `:1972`, which means:

* a **C++ rebuild**, so `bin/constructive_linux`'s md5 changes — the one thing
  L131's fix specifically avoided, and it re-opens the GLIBC 2.38 / static
  libstdc++ / x86-64 PIE analysis that the 08-15 handoff verified;
* MARGIN is a `static const`, so there is **no env knob** to A/B it — every
  measurement needs a rebuild;
* the frame is an input to the packing, so this is not a local repair: every case
  re-packs and the +0.3566% could come out anywhere.

With five days to the deadline, an uploaded and verified submission, and a
+0.0758% fix already built and waiting on a decision, this was **measured and
left**. It is the largest single quality item known on the shipped path and the
first thing to build if the submission is being revised at all.

## 6. Reproduce

```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l135_soft_audit.py results_L114_48c_lp_anchor.json
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l135_shrink_verify.py results_L114_48c_lp_anchor.json
```

Both are read-only. `l135_bnd_verify.py` is kept as the record of attempt (b) —
it now refuses to touch anything within the evaluator's 1e-6 band and reports
0 moves, which is the correct answer.
