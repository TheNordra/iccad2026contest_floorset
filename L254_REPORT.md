# L254 — the 81.3 % ceiling is FRAGMENTATION, not geometry

**Verdict: L252's density ceiling is an artefact of an irrevocable greedy, not a
geometric fact.** At the tightest frame that fails, the packer has placed a median
of all-but-3 blocks, jammed **3.4 pp below the density its own frame allows**, and
is refusing an item that needs **at most 1/10th of the free space still in the
frame**. In 30/30 cases the room was there.

This is the one probe that could have *closed* the packer line by proof. It did
the opposite: it reopened it, and named the missing mechanism.

No shipping change. Offline probe, no labels read.

---

## 1. Instrument

`constructive_l254.exe`, branched from the pristine shipping `constructive.cpp`
(md5 `e2c7b2f4…`), carrying the L252 ladder emitters plus a failure record at all
three `pack_in_frame` bail-outs:

    L254FAIL <frame_idx> <kind> <ndone> <N> <nblk> <iarea> <placed_area> <fw> <fh>

`kind` ∈ PRE / PREOV (a preplaced block outside the frame or overlapping — the
frame is below the clamp, a structural miss) / SINGLE (a free-aspect single found
no origin) / ITEM (a cluster or block found no origin).

⚠️ Only the **primary** pack of each frame emits. `ORDER_SWAP`'s hill-climb and
`REFINE`'s guide passes both call `pack_in_frame` again, and their failures are
not cliff events — `L254_PRIMARY` gates that. Without it the record is polluted
by ordinary search noise that looks exactly like a cliff.

**Identity gate — 102/102 PASS on both arms** (flag off vs stock, flag on vs
stock), all 51 pool profiles on the two heaviest cases, compared on stdout bytes.

## 2. Result (OOS s1, n ≥ 101, dense 26-rung ladder, 30 cases with a cliff event)

The row measured is the **last failing frame below the first success** — the cliff
edge. Kinds: **ITEM 15 / SINGLE 15, zero PRE/PREOV** — every event is a genuine
packing failure, not a clamp artefact.

| | p10 | **p50** | p90 |
|---|---|---|---|
| blocks left unplaced | 1 | **3** | 35 |
| free area / area of the item that failed | 10.7× | **15.9×** | 26.6× |
| density reached when it jammed | 67.9 % | **80.2 %** | 83.2 % |

    minimum free/need across ALL 30 cases        10.45x
    density when it jammed (weighted)            77.4%   frame allowed 83.6%
    failing item area / mean block area          1.97x
    failed after >=90% of blocks placed          25/30

**The greedy stops 3.4 pp below the density its own frame permits, with ~3 blocks
to go, refusing an item of about twice the mean block size, while sitting on ten
times more free area than that item needs.** That is fragmentation.

### 2.1 A metric I designed that turned out to be worthless

I added `free / (area of ALL unplaced blocks)` as the "airtight" version and it
reads p50 6.79×, min 1.23×, 30/30 above 1. **It proves nothing**: it equals
`(tot·s² − placed)/(tot − placed)`, which is structurally > 1 for every `s > 1`.
It is a restatement of the frame scale, not evidence. Recorded so nobody quotes
it. The verdict rests on the four numbers above, none of which is bounded by
construction — the failing item could have been huge relative to the free space,
the jam could have coincided with the frame's own limit, and the pack could have
died early. None of those happened.

## 3. What it means

L252 established the ceiling and priced it (83.7 % of the area deficit).
L253 established that the topology is not the problem.
**L254 establishes that the ceiling is not geometric.** The three together:

    we are at the label's topology (L253, 6.8% edit distance)
    we cannot pack it densely (L252, 81.3% vs 96.6%)
    and the reason is not that the space is missing (L254, >=10x free at the jam)

⇒ **The missing capability is relocation.** The greedy places once and never moves
anything; at the cliff edge it needs to displace two or three already-placed
blocks to open a slot, and it structurally cannot.

🚨 **This is new evidence against a RED that was closed by argument, not by
measurement.** M27 closed "packer rewrite" on *"greedy is already on the (area,
HPWL) frontier"* — a statement about the layouts it **produces**, which L251
independently confirms. It says nothing about the **cliff edge**, and M64's
"single/few pair flips = 0 movers" explicitly assigned multi-pair coordinated
moves to *"M27's domain"*. That domain now has a number attached for the first
time.

## 4. What this does NOT prove

* It proves the **edge** is soft. It does **not** prove the whole way down to
  96.6 % is soft. Free space necessarily shrinks as the frame tightens; at the
  label's density only 3.4 % of the frame is void. The measured slack is at
  `s ≈ 1.07…1.11`, not at `s = 1.017`.
* It does not price anything. A repair pass costs wall on frames that currently
  fail — though note those frames are *already* paid for and thrown away, so the
  marginal cost is smaller than it looks.
* 30 of 40 cases produced a cliff event; the other 10 had their tightest
  candidate pack (L252's clamp cases), so they carry no information here.
* Sample s1 only.

## 5. Next probe, precisely posed

**How far down does the fragmentation regime extend?** Sweep the jam density
against frame scale: for each rung below `s_min`, record where the pack dies and
how much room it left. The scale at which "free at the jam" stops exceeding what
the remaining blocks need is the **true geometric floor of this packer** — and
the distance from 81.3 % to that floor is the exact prize for adding relocation.

Two outcomes, both worth having: a floor near 96.6 % says a repair pass is worth
real engineering; a floor near 82 % closes the packer line by proof and the
project's bound set is complete.

## 6. Files

```
l254_patch.py        pristine constructive.cpp -> constructive_l254.cpp (8 patches)
l254_anatomy.py      the sweep and the anatomy
l254_identity.log    the two-arm gate, 102/102 both arms
l254_anatomy.log
l254_rows.pkl        per-case edge records
l252_identity.py     now takes --probe/--flags, so it gates any probe binary
```
