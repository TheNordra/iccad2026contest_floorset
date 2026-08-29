# L277 — the violation axis on the graded shape: the prize is real, the cheap mechanism is not

L275 moved the target to the corpus the score is computed on, and there the
violation term is worth **−2.81 %** if driven to zero — 15 % of the total headroom,
and the one axis the L250–L274 arc never touched. L140 audited violations on
**OOS**, which L275 showed is the wrong corpus (6.3× the vrel).

This runs the same audit in-set, prices what is genuinely removable, and then
measures the obvious mechanism.

**Answer: removable is 12 of 81 violations (upper bound −0.4456 %), and the
post-hoc boundary snap that would collect it scores +0.0012 % — nothing.** The
prize sits on light cases; on the heavy ones the snap costs more hpwl than the
violation it removes.

---

## 1. The in-set inventory, on the CURRENT shipped code

`l135_soft_audit.py` on `results_L274_base_48c.json`. (CLAUDE.md's "16 grouping +
78 boundary" predates L114/L136/L137 and is stale.)

    boundary  59      grouping  22      MIB  0   (provable floor 0, recoverable 0)

      of the 59 boundary:
        23  are PREPLACED       <- position is a HARD constraint. Moving one
                                   breaks feasibility rather than fixing a
                                   violation. Counting these as recoverable is
                                   the obvious way to manufacture a fake prize.
        40  have a BLOCKED slide path
        11  are CLEAR and soft   <- the only removable ones
      of the 22 grouping:
         1  has a clear corridor

**Removable = 12 / 81 = 15 %.**

| what | weighted cost | vs base |
|---|---|---|
| base | 1.226325 | — |
| remove **all** violations | 1.191922 | **−2.8054 %** |
| remove every **CLEAR ∧ soft** one | 1.220860 | **−0.4456 %** |
| remove one per case where possible | 1.221231 | −0.4154 % |

🔑 **MIB is finished in-set: 0 violations, provable floor 0.** L124's bucket work
did its job. All of L140's OOS "MIB is 67–75 % of violations" is a property of
that corpus, not of the placer.

⚠️ The −0.4456 % is concentrated: **84 % of it is two cases** (98 at n=119 and 88
at n=109). Concentration has predicted transfer failure twice this week.

## 2. Why any CLEAR ∧ soft ones survive at all

`constructive.cpp:final_boundary_nudge` already snaps a boundary block to the edge
it must touch when nothing is in the way — but it skips
`blocks[i].cluster > 0 || blocks[i].is_preplaced`, and it runs **inside the frame
trial**, before compaction, `hpwl_push` and the shape LP move everything again.

So the natural repair is a snap on the **final** layout. That is pure
post-processing on positions — a change to `op_wrapper.py` alone, **no C++ change
and therefore no Linux ELF rebuild**, which is the only class of change that could
ship safely from here.

## 3. Measured, by the official scorer

`l277_snap.py` applies the snap to the saved positions and hands both arms to
`iccad2026_evaluate.py --score`. Guards: never move `preplaced` or `fixed`; exact
overlap test at the destination; the bounding box may not grow.

**Instrument check first.** The untouched control, re-scored through the *same*
solutions path, reproduces the original eval to **0.00e+00** — so the comparison is
between layouts, not between two scoring paths.

    control (identical positions)   1.22632513
    snapped                         1.22634001      +0.0012 %
    feasible 100/100 both;  7 blocks snapped across 6 cases

| case | n | cost | hpwl_gap | worth |
|---|---|---|---|---|
| 23 | 44 | 1.223097 → **1.151102** | 0.2413 → 0.2455 | −0.0008 % |
| 4 | 25 | 1.464153 → **1.371185** | 0.2569 → 0.2595 | −0.0002 % |
| 1 | 22 | 1.485789 → **1.374919** | 0.2431 → 0.2348 | −0.0002 % |
| 17 | 38 | 1.314570 → 1.317167 | 0.2622 → 0.2671 | +0.0000 % |
| 59 | 80 | 1.457863 → 1.459593 | 0.2011 → 0.2041 | +0.0004 % |
| **88** | **109** | 1.298342 → **1.299130** | 0.2543 → **0.2557** | **+0.0021 %** |

🔑 **The mechanism works and the weighting defeats it.** Removing a violation is
worth a *lot* of cost on a small case — 1.4858 → 1.3749 is −7.5 % — but `exp(n/12)`
gives n = 22 almost no weight. On the heavy cases that carry the weight, the snap
buys hpwl at 0.2543 → 0.2557 and the violation it removes does not pay for it.

That is the L267_L269 §2.3 exchange rate again, on a third pair of terms. Area vs
wire was 1.24; here it is **vrel vs wire**, and it is on the wrong side of the line
exactly where the weight is.

## 4. What this closes and what it leaves

**Closes:** the cheap violation mechanism — post-hoc boundary snapping — on the
graded shape. And it closes the "306/307 placeable but not placed" reading as a
guide to action: in-set, only 11/59 boundary violations are even CLEAR ∧ soft, and
snapping them nets zero.

**Does not close:** the −2.81 %. Most of it is locked behind two facts that are
properties of the *instance*, not of the placer:

* **23/59 boundary violations are on preplaced blocks.** Those positions are given
  by the problem. The only way to satisfy them is for the frame's bounding box to
  land where the preplaced block already is — i.e. it is a **frame** question, not
  a placement one. L136 already found and fixed one instance of exactly this
  (`MARGIN` 1e-4 vs the scorer's 1e-6 handing preplaced blocks impossible
  violations, +0.5972 %). Whether any of the remaining 23 are of that kind is
  **unmeasured and is the single cheapest thing left on this axis**.
* **40/59 have a blocked slide path**, i.e. removing them requires displacing
  another block — which is the topology move L276 says the corpus has no
  affordable route to.

## 5. Honest limits

1. The snap is one mechanism, the most obvious one. A smarter repair (displace the
   blocker, then snap) is the blocked-path case and belongs to L276's topology
   argument.
2. The `l277_vio_prize.py` upper bound assumes removal is free in hpwl and area.
   §3 measured that assumption and it is false — which is the point.
3. `N_soft` is back-derived per case as `V / vrel`; it agrees with the audit's own
   counts but is not read from the evaluator directly.

## 6. Files

```
l277_vio_prize.py     inventory + upper bound, CLEAR-and-soft only
l277_snap.py          the post-hoc snap + a control written through the same path
l277_snap_solutions.json  l277_ctrl_solutions.json
results_l277_{snap,ctrl}_solutions.json
```

Nothing shipped; `constructive.cpp` and `build_submission.D/` untouched.
