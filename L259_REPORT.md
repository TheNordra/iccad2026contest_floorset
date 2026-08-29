# L259 — a better recreate cannot exist. The jam is the biggest block, not the space.

**Verdict: at the cliff edge, the largest unplaced block has ZERO legal positions,
while smaller unplaced blocks have hundreds.** The free space is shattered into
pieces all smaller than the biggest thing that still needs a home. So no recreate
— however clever — can repair the jam *given the greedy's own prefix*. The fix has
to change the prefix, which is full re-placement: M27 / L129 territory.

This closes "bolt a stronger recreate onto the greedy", which is what L256's
2 %-deep result was pointing at.

---

## 1. Why this probe

L256 ruined ~12 % of the design and re-placed it with the **same greedy that had
just jammed**, which is why it never went deeper than ~2 %. L253 says the topology
is already right and L254 says the jam has ≥10× the needed area free, so the
missing piece looked like a *recreate that can use fragmented space* — a small
combinatorial problem (median 3 items) nothing in the ledger had attacked.

Asymmetric by design: a YES opens a concrete mechanism worth L255's +3.23 %;
a NO does not close the axis absolutely (a different prefix might be placeable)
but does close the cheap version.

## 2. Instrument

`constructive_l259.exe`, branched from pristine `constructive.cpp`
(md5 `e2c7b2f4…`), emits the full jam state at the two real failure sites:

    L259JAM <frame_idx> <fw> <fh> <ndone> <N>
    L259P   <i> <x> <y> <w> <h>      every block already placed
    L259U   <i> <w> <h>              every block still unplaced

**Identity gate: 102/102 PASS on BOTH arms** — correct for an instrument
(stderr-only, no behaviour change).

## 3. Two wrong answers before the right one

Recorded because both looked like findings.

1. **Sparse candidate set.** The first backtracking solver used only
   per-rectangle corners. A block can take its x from one rectangle and its y
   from another, so the complete bottom-left candidate set is the **cross
   product** `{0} ∪ {right edges} × {0} ∪ {top edges}`. The sparse version
   returned NO after **1 node** on every case.
2. **An over-estimating diagnostic.** The "biggest free slot" I added computed
   max free *width* and max free *height* independently, so an L-shaped region
   reads as a rectangle. It reported a 71.9×39.8 slot for a 25.0×14.0 block and
   flatly contradicted the solver.

Neither was trustworthy, so the question was settled with no cleverness at all:
**rasterise the jam and test every position** (600×600 grid, integral image).
The raster validates itself — its occupancy matches the exact area occupancy to
0.1 pp on every case (77.8 vs 77.9 %, 83.2 vs 83.2 %, 80.8 vs 80.8 %, …).

## 4. The answer

| case | frame | placed | occupancy | biggest unplaced | free positions |
|---|---|---|---|---|---|
| n=120 | 120.7×324.0 | 108 | 77.8 % | **23.0 × 21.0** | **0** |
| | | | | 21.6 × 21.6 | 12 |
| | | | | 15.1 × 15.1 | 1539 |
| n=120 | 168.1×224.1 | 119 | 83.2 % | **25.0 × 14.0** | **0** |
| n=120 | 151.7×304.0 | 115 | 80.8 % | **19.6 × 26.5** | **0** |
| | | | | 18.6 × 25.1 | 0 |
| | | | | 15.8 × 21.3 | 0 |
| n=119 | 132.9×332.1 | 115 | 79.8 % | **24.4 × 24.4** | **0** |
| | | | | 19.4 × 19.4 | 35 |
| n=119 | 149.2×271.3 | 117 | 80.2 % | **21.0 × 16.0** | **0** |
| | | | | 8.5 × 8.5 | 6609 |

🔑 **In every case the largest remaining block has nowhere at all to go, and
smaller ones have plenty of room.** That is the precise shape of the jam, and it
sharpens L254: `free/need ≥ 10×` is true *by area*, but the largest contiguous
free rectangle is smaller than the largest remaining block. Area was never the
binding constraint — **contiguity** is.

The backtracking solver's NO (0 nodes: no candidate position for the first, and
largest, item) was therefore **correct**, for the right reason.

## 5. What this closes, and what it does not

**Closes: a stronger recreate.** Given the prefix the greedy has already
committed, no placement algorithm can finish the layout. Exact search, MIP,
skyline-with-backtracking — all of them face zero legal positions for the block
that matters. This is why L256's ruin-and-recreate stalled at ~2 % and why a
better acceptance rule (L258) could not help it.

**Does not close: coordinated re-placement.** The obstruction is *this* prefix.
A ruin set chosen to free a **contiguous** region large enough for the big blocks
— rather than L256's "overflow plus neighbours toward the far corner" — is still
untested, and that is precisely the M27 statement: multi-pair coordinated moves,
which M64 explicitly deferred to M27's domain and which L129 attacked from scratch
(reaching 1.745 against the shipped 1.237).

⇒ The measured target is unchanged (L255: re-place ~10 % → s ≈ 1.06 → **+3.23 %**)
and the mechanism that could reach it is unchanged: **a placer that decides where
the big blocks go before the small ones fill the gaps.** Which is a different
placer, not a repair pass on this one.

## 6. Honest limits

* 5 jams brute-forced, 15 solved by backtracking, all at the cliff edge of the
  proxy-winning profile, sample s1, heavy band.
* Nominal dims, no rotation, soft constraints ignored — all optimistic, so the
  NO is conservative and strong.
* The raster is 600×600; a block could in principle fit in a sliver the grid
  misses. Occupancy agreeing with exact area to 0.1 pp bounds that risk, and the
  positive rows (12, 35, 1539, 6609 free positions) show the raster is not simply
  saturated.

## 7. Files

```
l259_patch.py       pristine constructive.cpp -> constructive_l259.cpp (7 patches)
l259_feasible.py    the backtracking bound (fixed candidate set) + sanity
l259_bruteforce.py  the raster arbitration -- the one that is trustworthy
l259_feasible.pkl
```
