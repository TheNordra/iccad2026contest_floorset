# L260 / L261 — M27's minimum scale is an ejection chain of depth ~2, not a re-packer

**Verdict: the frames the shipped packer declares unpackable ARE packable, and
the move that unblocks them is tiny.** Displacing exactly **one** already-placed
block opens a slot for the blocker (8/8 cases); running the cascade to completion
finishes the whole layout in **7/8** cases at a median of **2 evictions**.

This is the answer to "how big is M27?": **a bounded ejection chain, not a
from-scratch global placer.**

⚠️ Geometric only. See §4 before treating this as score.

---

## 1. L260 — the minimum displacement

L259 showed the jam is contiguity, not area: the largest unplaced block has zero
legal positions while smaller ones have thousands. So: over every candidate anchor
(cross product of `{0} ∪ {right edges} ∪ {left − w}` × `{0} ∪ {top edges} ∪
{bottom − h}`, exact rectangle arithmetic, ~25 000 anchors per case), how many
placed blocks would the blocker's footprint overlap?

| n | left | blocker | must displace | their area |
|---|---|---|---|---|
| 120 | 12 | 23.0 × 21.0 | **1** | 0.91 % of design |
| 120 | 1 | 25.0 × 14.0 | **1** | 0.58 % |
| 120 | 5 | 19.6 × 26.5 | **1** | 0.61 % |
| 119 | 4 | 24.4 × 24.4 | **1** | 0.84 % |
| 119 | 2 | 21.0 × 16.0 | **1** | 0.93 % |
| 119 | 2 | 19.1 × 19.1 | **1** | 0.67 % |
| 118 | 3 | 23.7 × 23.7 | **1** | 0.18 % |
| 118 | 1 | 26.0 × 26.0 | **1** | 0.75 % |

**min 1, p50 1, max 1.** Median area 0.75 % of the design.

## 2. L261 — and the cascade terminates

A lower bound is worthless if the displaced block then displaces two more. So:
queue **every** unplaced block from the jam; repeatedly take the one needing a
home, place it at the anchor displacing the fewest blocks (0 if such an anchor
exists), evict those, and push them onto the queue. Greedy, no backtracking,
bounded in depth and evictions.

| n | left | result | depth | evictions | peak queue |
|---|---|---|---|---|---|
| 120 | 12 | **SOLVED** | 14 | 2 | 12 |
| 120 | 1 | **SOLVED** | 2 | 1 | 1 |
| 120 | 5 | **SOLVED** | 13 | 8 | 5 |
| 119 | 4 | **SOLVED** | 7 | 3 | 4 |
| 119 | 2 | **SOLVED** | 3 | 1 | 2 |
| 119 | 2 | no | 41 | 39 | 2 |
| 118 | 3 | **SOLVED** | 5 | 2 | 3 |
| 118 | 1 | **SOLVED** | 2 | 1 | 1 |

    completed the layout at the tighter frame   7/8
    evictions   min 1   p50 2   max 8
    peak queue  min 1   p50 3   max 12

🔑 **SOLVED is constructive** — an actual legal layout exists at a frame the
shipped packer refused. A "no" is not a proof of infeasibility (greedy, no
backtracking), so 7/8 is a lower bound on how often this works.

## 3. What it changes

L252 measured the packer's density ceiling at **81.3 %** utilisation and called it
`s_min`. That number is a property of **the greedy's placement rule**, not of the
instance: with one eviction, frames below `s_min` pack. So

* the L252 ceiling is not the reachable ceiling;
* the mechanism that crosses it is **depth ~2**, which is inside what a repair
  pass at the jam can afford — it is not L129's from-scratch global placer
  (1.745 against the shipped 1.237, days of engineering);
* it explains L256 exactly. L256's recreate was the *same greedy*, which by L259
  cannot place the blocker at all. One eviction is precisely the move it lacked.

L255's curve prices the target: re-placing ~10 % of the design reaches s ≈ 1.06
(89 % utilisation) and is worth **+3.23 %** of quality. The evictions measured here
are ~0.75 % of the design per step, so the budget is not the constraint.

## 4. What this is NOT, yet

🚨 **This is geometric feasibility only.** The chain ignores everything the packer
is actually scored on:

* **soft constraints** — boundary codes, cluster abutment, MIB shape agreement.
  A block evicted to an arbitrary free corner will very likely break its cluster
  or lose its boundary. L253/L251 show vrel is currently a **surplus** (−3.88 %
  vs the label); spending it is a real cost.
* **wire** — the chain places at the minimum-eviction anchor, not a wire-aware
  one. L256's whole failure mode was area gained and hpwl lost.
* **selection** — L256/L257/L258 established that a per-profile quality gain has
  to survive the pool proxy, the `hmin` coupling, and the wall.

So the honest claim is: **the geometric obstacle that stopped L256 is removable at
depth ~2.** Whether removing it produces *score* is exactly the question L256
answered NO for its own weaker mechanism, and it must be re-asked here.

Other limits: 8 jams, proxy-winning profile only, s1 heavy band, nominal dims,
no rotation.

## 5. The next step, precisely posed

Implement the eviction inside `pack_in_frame` — when an item finds no origin,
evict the minimum-count blocking set, place it, and re-queue the evicted — with
the shipped candidate scoring applied to every placement (so wire and boundary
are respected), then re-run L256's shrink loop on top of it. That converts
"geometrically packable" into "packable by our scorer", which is the only version
that can be priced.

Then the existing pipeline prices it end to end with no new tooling:
`l256_score.py` (isolated + portfolio), `l257_twin.py` (offline, any twin set),
`l258_maxsetter.py` (wall).

## 6. Files

```
l260_mincut.py    minimum displacement to open a slot -> l260_mincut.pkl
l261_eject.py     the ejection chain, depth/evictions -> l261_eject.pkl
```
