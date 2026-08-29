# L262 — eviction is implemented and it lowers the density ceiling, by ~1 % of area

**Verdict: the mechanism works. `s_min` falls in 6/12 cases, weighted utilisation
81.6 % → 82.5 %, area −1.04 %.** That is real — L252's `s_min` was a property of
the greedy rule and eviction moves it — but it is well short of what L261's
offline chain suggested (7/8 solved at median 2 evictions), and the gap is
explained below.

Not shipped. `ICCAD_L262` defaults off; `constructive.cpp` untouched.

---

## 1. What was built

When an item finds no origin, instead of failing the frame: find the anchor that
displaces the **fewest** already-placed items, evict them, and **re-queue both the
item and the evicted ones**.

🔑 **The item is deliberately NOT placed at the eviction anchor.** It goes back on
the worklist, so its real placement returns through `item_candidates()` and the
shipped scoring — wire, boundary, anchor and BP terms all still apply. Eviction
creates the opportunity; it never chooses the position. That is the difference
from L256, whose recreate was the same greedy that had just jammed.

Implementation notes that mattered:

* **The main loop became a worklist.** `for (const Item& it : items)` cannot
  re-queue. It is now an index loop over a growing `l262_work`.
* **Rect ownership.** `rect_own[]` parallels `rects`; **−1 = never evictable** —
  preplaced blocks, and movable members placed by the anchored first-pass (they
  are attached to preplaced walls). An anchor overlapping any −1 rect is
  disqualified outright.
* **Eviction at ITEM granularity, never block.** `pack_in_frame` skips any item
  that is *partially* placed, so evicting one block of a composite item would
  leave the rest orphaned and still return true.
* 🚨 **`g46` and the running bbox are add-only.** Evicting physically erases
  rects, so both are rebuilt from the survivors on every eviction. A stale grid
  would silently corrupt every subsequent overlap test — the failure would be a
  wrong layout, not a crash.
* The free-aspect single path falls through to the generic path on failure when
  the flag is on, instead of returning false.

**Gates** (`constructive_l262.exe`):

| arm | result | reading |
|---|---|---|
| A — flag OFF vs stock | **102/102 PASS** | the default path is the shipped placer |
| B — flag ON vs stock | **101/102** | live, but it fires on only 1 of 102 with the SHIPPED ladder |

## 2. Why arm B barely moves on the shipped ladder

The shipped ladder is `1.00, 1.05, 1.10, 1.20`. `s_min` sits around **1.11**, so
the rungs below it are 1.05 and 1.00 — **6 to 11 points too tight**, far more than
a handful of evictions can bridge. The ladder has no rung near the cliff edge,
which is L252's finding arriving from the other side.

So the measurement has to use the dense ladder, where rungs exist just below
`s_min`.

## 3. Result (dense 26-rung ladder, proxy-winning profile, 12 heavy cases)

| n | `s_min` OFF | `s_min` ON | |
|---|---|---|---|
| 120 | 1.1000 (82.6 %) | **1.0880 (84.5 %)** | tighter |
| 119 | 1.1046 (82.0 %) | **1.0839 (85.1 %)** | tighter |
| 118 | 1.1126 (80.8 %) | **1.0908 (84.0 %)** | tighter |
| 119 | 1.1200 (79.7 %) | **1.1100 (81.2 %)** | tighter |
| 118 | 1.0800 (85.7 %) | **1.0752 (86.5 %)** | tighter |
| 118 | 1.0912 (84.0 %) | **1.0900 (84.2 %)** | tighter |
| (6 others) | — | — | same |

    s_min OFF   1.1068   util 81.6%
    s_min ON    1.1010   util 82.5%     6/12 tighter
    area if the packer landed there:   -1.04%

Priced on L251's method (upper bound — assumes hpwl and vrel unchanged, which
L256 showed they are not): area_gap 0.2256 → 0.2128, QF 1.2511 → 1.2447,
**≈ +0.51 % of quality.**

## 4. Why it is smaller than L261 predicted

L261's offline chain solved 7/8 at median 2 evictions. Four differences, all of
which make the C++ version more conservative — and all of which are tunable:

1. **Footprint is the item's bounding box** (`it.w × it.h`), not the union of its
   member rectangles. For a composite item with offsets that is strictly larger
   than what actually needs to be free.
2. **Anchors overlapping any owner −1 rect are disqualified.** L261 could evict
   anything; the real one protects preplaced and anchored members.
3. **Re-queue, not place.** After eviction the item must still find a spot through
   `item_candidates()`, whose origins are abutment-derived — the freed hole is not
   guaranteed to generate one.
4. **Caps**: `l262_try >= 2` per item and `ICCAD_L262_MAX = 24` per pack. Neither
   has been swept.

(3) is the interesting one: it is exactly the property that keeps wire and
boundary respected, so relaxing it trades correctness of the scoring for depth.

## 5. What has NOT been measured

Everything that decides whether this is worth anything:

* **quality** — `s_min` falling is not score. L256's whole failure mode was area
  gained and hpwl lost, and the same test (`l256_score.py`, isolated + portfolio)
  has not been run on L262.
* **violations** — evicted blocks get re-placed by the shipped scorer, so boundary
  and cluster terms are *considered*, but L251 shows vrel is a **surplus** we
  would be spending.
* **wall** — the anchor search is O(|xs|·|ys|·|rects|) ≈ 240 × 240 × 120 per
  failure. Not measured. `l258_maxsetter.py` prices it.
* **selection** — L257/L258 established a per-profile gain must survive the pool
  proxy and the `hmin` coupling.
* OOS, s2, and the deployment form (the shipped ladder has no rung where this
  helps, so shipping it means changing `FRAME_SCALES` too — which is its own
  quality/wall change).

## 6. Files

```
l262_patch.py    pristine constructive.cpp -> constructive_l262.cpp (10 patches)
l262_smin.py     the ON/OFF s_min measurement -> l262_smin.pkl
constructive_l262.exe   arm A 102/102 PASS
```
